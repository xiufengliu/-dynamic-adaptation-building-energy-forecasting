"""Baseline forecasters: ARIMA, SVR, LSTM, GBM. Direct multi-step.

For each horizon h in {24, 168, 672}, we train models that predict y[t+h]
from features available at time t. This is the "direct" strategy (Hyndman).
"""
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

FEATS = ['lag_1', 'lag_24', 'lag_168',
         'airTemperature', 'dewTemperature', 'windSpeed',
         'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'month']


def make_direct(df, h):
    """Direct-strategy: features at t, target = y[t+h]."""
    df = df.copy().reset_index(drop=True)
    df['y_target'] = df['y'].shift(-h)
    df = df.dropna().reset_index(drop=True)
    X = df[FEATS].values.astype(np.float32)
    y = df['y_target'].values.astype(np.float32)
    return X, y


def gbm_forecast(train, val, test, h, params=None):
    Xtr, ytr = make_direct(train, h)
    Xva, yva = make_direct(val, h)
    Xte, yte = make_direct(test, h)
    p = dict(objective='regression', metric='mae', learning_rate=0.05,
             num_leaves=64, min_data_in_leaf=20, verbosity=-1, num_threads=4)
    if params: p.update(params)
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva)
    model = lgb.train(p, dtr, num_boost_round=500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    yhat = model.predict(Xte)
    return yte, yhat


def svr_forecast(train, val, test, h, n_subsample=4000):
    Xtr, ytr = make_direct(train, h)
    Xva, yva = make_direct(val, h)
    Xte, yte = make_direct(test, h)
    rng = np.random.default_rng(0)
    if len(Xtr) > n_subsample:
        idx = rng.choice(len(Xtr), n_subsample, replace=False)
        Xtr_s, ytr_s = Xtr[idx], ytr[idx]
    else:
        Xtr_s, ytr_s = Xtr, ytr
    sx = StandardScaler(); Xtr_sN = sx.fit_transform(Xtr_s); Xte_sN = sx.transform(Xte)
    sy = StandardScaler(); ytr_sN = sy.fit_transform(ytr_s.reshape(-1, 1)).ravel()
    m = SVR(kernel='rbf', C=1.0, gamma='scale')
    m.fit(Xtr_sN, ytr_sN)
    yhat = sy.inverse_transform(m.predict(Xte_sN).reshape(-1, 1)).ravel()
    return yte, yhat


def arima_forecast(train, val, test, h, order=(2, 1, 2)):
    y_tr = np.concatenate([train['y'].values, val['y'].values])
    y_te = test['y'].values
    try:
        model = ARIMA(y_tr, order=order).fit()
    except Exception:
        model = ARIMA(y_tr, order=(1, 1, 1)).fit()
    # forecast len(test) + h steps, align to direct horizon h
    steps = len(y_te)
    fc = model.forecast(steps=steps)
    # Direct-h alignment: prediction at test-index i should be for t+h; we use fc[i] as proxy
    # Because ARIMA forecast is rolled out from end of train, this gives h-step direct forecast only
    # for small h; for long h we apply naive mean-reversion by blending with train mean
    w = min(1.0, h / 672.0)
    yhat = (1 - w) * fc + w * y_tr.mean()
    # Compare against y_te shifted by h in the "direct" sense: we simply compare to y_te[:steps-h]
    # then pad. Cleaner: keep both same length for the common portion.
    if h > 0:
        yte_aligned = y_te[h:] if h < len(y_te) else y_te[-1:]
        yhat_aligned = yhat[:len(yte_aligned)]
    else:
        yte_aligned = y_te
        yhat_aligned = yhat
    return yte_aligned, yhat_aligned


class LSTMModel(nn.Module):
    def __init__(self, in_dim=5, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequence(df, h, lookback=168, feats=('y', 'airTemperature', 'dewTemperature', 'windSpeed', 'hour_sin')):
    """Build (N, lookback, F) tensors with target y[t+h]."""
    df = df.copy().reset_index(drop=True)
    X_cols = list(feats)
    arr = df[X_cols].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    N = len(df) - lookback - h
    if N <= 0:
        return np.zeros((0, lookback, len(X_cols)), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack([arr[i:i + lookback] for i in range(N)])
    Y = np.array([y[i + lookback + h - 1] for i in range(N)], dtype=np.float32)
    return X, Y


def lstm_forecast(train, val, test, h, epochs=25, batch=128, lookback=168,
                  lr=1e-3, hidden=64, layers=2, dropout=0.2, device=None,
                  return_history=False):
    device = device or DEVICE
    feats = ('y', 'airTemperature', 'dewTemperature', 'windSpeed', 'hour_sin')
    Xtr, ytr = make_sequence(train, h, lookback, feats)
    Xva, yva = make_sequence(val, h, lookback, feats)
    Xte, yte = make_sequence(test, h, lookback, feats)
    if len(Xtr) == 0 or len(Xte) == 0:
        return np.zeros(0), np.zeros(0), [] if return_history else None
    # normalize
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
    y_mu, y_sd = float(ytr.mean()), float(ytr.std() + 1e-6)
    ytr_n = (ytr - y_mu) / y_sd
    yva_n = (yva - y_mu) / y_sd

    model = LSTMModel(in_dim=len(feats), hidden=hidden, layers=layers,
                      dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr_n).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    yva_t = torch.from_numpy(yva_n).to(device)

    best_val = float('inf'); best_state = None; patience = 5; since = 0
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tr_losses = []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr_t[idx])
            loss = loss_fn(pred, ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            vl_n = loss_fn(pv, yva_t).item()
            # val MAE in original units
            vl_mae = float(np.mean(np.abs((pv.cpu().numpy() * y_sd + y_mu) - yva)))
        history.append(dict(epoch=ep, train_loss=float(np.mean(tr_losses)),
                            val_loss_norm=vl_n, val_mae=vl_mae))
        if vl_n < best_val:
            best_val = vl_n
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte_t = torch.from_numpy(Xte).to(device)
        yhat_n = model(Xte_t).cpu().numpy()
    yhat = yhat_n * y_sd + y_mu
    if return_history:
        return yte, yhat, history
    return yte, yhat

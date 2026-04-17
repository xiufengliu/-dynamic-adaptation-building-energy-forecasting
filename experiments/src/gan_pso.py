"""WGAN-GP forecaster with PSO hyperparameter adaptation."""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import make_sequence, DEVICE, FEATS
from . import metrics as M


def summarize_context(ctx):
    """Compact temporal summary for faster CPU training."""
    last = ctx[:, -1, :]
    tail24 = ctx[:, -24:, :] if ctx.size(1) >= 24 else ctx
    mean24 = tail24.mean(dim=1)
    std24 = tail24.std(dim=1, unbiased=False)
    mean_all = ctx.mean(dim=1)
    y_last = ctx[:, -1, :1]
    y_prev = ctx[:, -24, :1] if ctx.size(1) >= 24 else ctx[:, 0, :1]
    y_first = ctx[:, 0, :1]
    trend_short = y_last - y_prev
    trend_long = y_last - y_first
    return torch.cat([last, mean24, std24, mean_all, trend_short, trend_long], dim=-1)


class Generator(nn.Module):
    def __init__(self, in_ch=13, lookback=168, noise_dim=8, hidden=96, dropout=0.1):
        super().__init__()
        self.noise_dim = noise_dim
        self.summary_dim = in_ch * 4 + 2
        self.summary_proj = nn.Sequential(
            nn.Linear(self.summary_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.base = nn.Linear(self.summary_dim, 1)
        self.proj = nn.Sequential(
            nn.Linear(hidden + noise_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.noise_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, ctx, z=None):
        summary = summarize_context(ctx)
        if z is None:
            z = torch.zeros(ctx.size(0), self.noise_dim, device=ctx.device)
            if self.training and self.noise_dim > 0:
                z = torch.randn_like(z) * self.noise_scale.abs()
        hidden = self.summary_proj(summary)
        base = self.base(summary)
        delta = self.proj(torch.cat([hidden, z], dim=-1))
        return (base + delta).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, in_ch=13, lookback=168, hidden=96, dropout=0.1):
        super().__init__()
        summary_dim = in_ch * 4 + 2
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ctx, y):
        summary = self.summary_proj(summarize_context(ctx))
        return self.score(torch.cat([summary, y.unsqueeze(-1)], dim=-1)).squeeze(-1)


def gp_penalty(critic, ctx, y_real, y_fake):
    alpha = torch.rand(y_real.size(0), device=y_real.device)
    y_mix = (alpha * y_real + (1 - alpha) * y_fake).requires_grad_(True)
    score = critic(ctx, y_mix)
    grads = torch.autograd.grad(score.sum(), y_mix, create_graph=True)[0]
    return ((grads.norm(2, dim=-1) - 1) ** 2).mean()


GAN_FEATS = tuple(['y'] + FEATS)


def train_wgangp(train_df, val_df, h, hparams, epochs=6, lookback=168, batch=128,
                 device=None, return_history=False, verbose=False):
    device = device or DEVICE
    feats = GAN_FEATS
    Xtr, ytr = make_sequence(train_df, h, lookback, feats)
    Xva, yva = make_sequence(val_df, h, lookback, feats)
    if len(Xtr) == 0 or len(Xva) == 0:
        return None, [], float('inf')
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    y_mu, y_sd = float(ytr.mean()), float(ytr.std() + 1e-6)
    ytr_n = (ytr - y_mu) / y_sd
    yva_n = (yva - y_mu) / y_sd

    eta_G = float(hparams['eta_G']); eta_D = float(hparams['eta_D'])
    lam = float(hparams['lambda_gp']); beta1 = float(hparams['beta1'])
    n_critic = max(1, int(round(float(hparams['n_critic']))))
    dropout = float(hparams['dropout'])
    adv_weight = float(hparams.get('adv_weight', 0.05))
    pretrain_epochs = int(round(float(hparams.get('pretrain_epochs', max(2, epochs // 3)))))
    pretrain_epochs = min(max(pretrain_epochs, 0), max(epochs - 1, 0))

    G = Generator(in_ch=len(feats), lookback=lookback, dropout=dropout).to(device)
    D = Critic(in_ch=len(feats), lookback=lookback, dropout=dropout).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=eta_G, betas=(beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=eta_D, betas=(beta1, 0.999))

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr_n).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    history = []
    best_val = float('inf'); best_state = None
    patience = 6
    epochs_since_improve = 0
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t))
        losses_D, losses_G = [], []
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            ctx = Xtr_t[idx]; y_real = ytr_t[idx]
            d_loss_value = 0.0
            if ep >= pretrain_epochs:
                for _ in range(n_critic):
                    optD.zero_grad()
                    with torch.no_grad():
                        y_fake = G(ctx)
                    y_real_n = y_real + 0.02 * dropout * torch.randn_like(y_real)
                    d_real = D(ctx, y_real_n)
                    d_fake = D(ctx, y_fake)
                    gp = gp_penalty(D, ctx, y_real_n, y_fake)
                    d_loss = d_fake.mean() - d_real.mean() + lam * gp
                    d_loss.backward()
                    optD.step()
                    d_loss_value = float(d_loss.item())
            optG.zero_grad()
            y_fake = G(ctx)
            g_sup = F.l1_loss(y_fake, y_real)
            if ep >= pretrain_epochs:
                g_adv = -D(ctx, y_fake).mean()
                g_loss = g_sup + adv_weight * g_adv
            else:
                g_loss = g_sup
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            optG.step()
            losses_D.append(d_loss_value)
            losses_G.append(float(g_loss.item()))
        # val MAE in original units
        G.eval()
        with torch.no_grad():
            yhat_n = G(Xva_t).cpu().numpy()
        G.train()
        yhat = yhat_n * y_sd + y_mu
        vmae = float(np.mean(np.abs(yhat - yva)))
        history.append(dict(epoch=ep, val_mae=vmae,
                            d_loss=float(np.mean(losses_D)),
                            g_loss=float(np.mean(losses_G))))
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.detach().clone() for k, v in G.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if verbose:
            print(f'  ep{ep} vmae={vmae:.2f}')
        if epochs_since_improve >= patience:
            break

    if best_state is not None:
        G.load_state_dict(best_state)
    G.eval()
    bundle = dict(G=G, mu=mu, sd=sd, y_mu=y_mu, y_sd=y_sd)
    return bundle, history, best_val


def predict_wgangp(bundle, df, h, lookback=168, device=None):
    device = device or DEVICE
    feats = GAN_FEATS
    X, y = make_sequence(df, h, lookback, feats)
    if len(X) == 0: return np.zeros(0), np.zeros(0)
    Xn = (X - bundle['mu']) / bundle['sd']
    with torch.no_grad():
        yhat_n = bundle['G'](torch.from_numpy(Xn).to(device)).cpu().numpy()
    return y, yhat_n * bundle['y_sd'] + bundle['y_mu']


# ---------- PSO over hyperparameters ----------
BOUNDS = dict(
    eta_G=(5e-5, 5e-4),
    eta_D=(5e-5, 5e-4),
    lambda_gp=(1.0, 15.0),
    beta1=(0.0, 0.8),
    n_critic=(1, 4),
    dropout=(0.0, 0.2),
    adv_weight=(0.0, 0.12),
    pretrain_epochs=(1.0, 6.0),
)
DIM_ORDER = ['eta_G', 'eta_D', 'lambda_gp', 'beta1', 'n_critic', 'dropout', 'adv_weight', 'pretrain_epochs']


def _clip(x):
    for i, k in enumerate(DIM_ORDER):
        lo, hi = BOUNDS[k]
        x[i] = float(np.clip(x[i], lo, hi))
    return x


def _rand_particle(rng):
    return np.array([rng.uniform(*BOUNDS[k]) for k in DIM_ORDER], dtype=float)


def _to_hparams(vec):
    return {k: vec[i] for i, k in enumerate(DIM_ORDER)}


def fitness_fn(train, val, h, vec, datasets=None, w_acc=0.75, w_sust=0.15, w_dev=0.10,
               EF=0.417, c=0.12, k_reserve=1.5, epochs=3):
    hp = _to_hparams(vec)
    eval_sets = datasets if datasets is not None else [(train, val)]
    fit_vals = []
    mae_vals = []
    sigma_vals = []
    best_payload = None
    for tr_df, va_df in eval_sets:
        bundle, _, vmae = train_wgangp(tr_df, va_df, h, hp, epochs=epochs, verbose=False)
        if bundle is None:
            return float('inf'), None
        y_val, yhat_val = predict_wgangp(bundle, va_df, h)
        if len(y_val) == 0:
            return float('inf'), None
        acc = M.mae(y_val, yhat_val) / (y_val.std() + 1e-6)
        sigma_e = float(np.std(y_val - yhat_val))
        D_commit = np.clip(yhat_val + k_reserve * sigma_e, 0, None)
        sust = (EF * D_commit.sum() + c * D_commit.sum()) / (1.0 * len(y_val) * y_val.mean())
        dev = float(np.mean(y_val - yhat_val)) ** 2 / (y_val.mean() ** 2 + 1e-6)
        fit_vals.append(w_acc * acc + w_sust * sust + w_dev * dev)
        mae_vals.append(float(M.mae(y_val, yhat_val)))
        sigma_vals.append(sigma_e)
        best_payload = bundle, dict(val_mae=float(M.mae(y_val, yhat_val)),
                                    sigma_e=sigma_e, acc=acc, sust=sust, dev=dev)
    fit = float(np.mean(fit_vals))
    info = dict(val_mae=float(np.mean(mae_vals)),
                sigma_e=float(np.mean(sigma_vals)),
                fit=fit, n_eval_sets=len(eval_sets))
    return fit, (best_payload[0], info)


def pso(train=None, val=None, h=1, datasets=None, n_particles=10, n_iters=15, w=0.5, c1=1.5, c2=1.5,
        epochs_per_eval=3, seed=42, log_file=None):
    rng = np.random.default_rng(seed)
    X = np.stack([_rand_particle(rng) for _ in range(n_particles)])
    V = np.zeros_like(X)
    pbest = X.copy(); pbest_fit = np.full(n_particles, np.inf)
    pbest_info = [None] * n_particles
    gbest = X[0].copy(); gbest_fit = float('inf'); gbest_info = None
    history = []
    t0 = time.time()
    for it in range(n_iters):
        for i in range(n_particles):
            fit, payload = fitness_fn(train, val, h, X[i], datasets=datasets, epochs=epochs_per_eval)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit; pbest[i] = X[i].copy()
                pbest_info[i] = payload
            if fit < gbest_fit:
                gbest_fit = fit; gbest = X[i].copy()
                gbest_info = payload
        # update velocities and positions
        r1 = rng.random((n_particles, len(DIM_ORDER)))
        r2 = rng.random((n_particles, len(DIM_ORDER)))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        for i in range(n_particles):
            X[i] = _clip(X[i])
        elapsed = time.time() - t0
        history.append(dict(iter=it, gbest_fit=gbest_fit,
                            best_vmae=(gbest_info[1]['val_mae'] if gbest_info else None),
                            elapsed_s=elapsed))
        if log_file:
            with open(log_file, 'a') as fp:
                fp.write(f'iter={it} gbest_fit={gbest_fit:.4f} '
                         f'vmae={gbest_info[1]["val_mae"] if gbest_info else "NA"} '
                         f'elapsed={elapsed:.1f}s\n')
    return dict(gbest_vec=gbest, gbest_fit=float(gbest_fit),
                gbest_hparams=_to_hparams(gbest), gbest_bundle=gbest_info[0],
                gbest_info=gbest_info[1], history=history)

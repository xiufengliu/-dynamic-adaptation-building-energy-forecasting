"""Forecast evaluation metrics."""
import numpy as np


def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))
def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def mape(y, yhat, eps=1e-6):
    return float(100.0 * np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))))


def wmape(y, yhat):
    return float(100.0 * np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)))


def mmape(y, yhat, threshold=50.0):
    mask = np.abs(y) >= threshold
    if mask.sum() == 0: return float('nan')
    return float(100.0 * np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def all_metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return dict(MAE=mae(y, yhat), RMSE=rmse(y, yhat),
                MAPE=mape(y, yhat), WMAPE=wmape(y, yhat),
                mMAPE=mmape(y, yhat), R2=r2(y, yhat))


def sustainability(y_true, yhat_dict, k=1.5, EF=0.417, c=0.12):
    """Compute CO2/cost under D_t = yhat + k*sigma_e dispatch model.
    Returns dict of {model: {CO2_kg, cost_usd, CO2_reduction_pct, cost_reduction_pct}}.
    Baseline is a naive persistence model (y[t] = y[t-1])."""
    y = np.asarray(y_true, dtype=float)
    out = {}
    ref = None
    for name, yhat in yhat_dict.items():
        sigma = np.std(y - yhat, ddof=0)
        D = np.asarray(yhat, dtype=float) + k * sigma
        D = np.clip(D, 0, None)
        co2 = float(EF * D.sum())
        cost = float(c * D.sum())
        out[name] = dict(sigma_e=float(sigma), committed_kWh=float(D.sum()),
                         CO2_kg=co2, cost_usd=cost)
    # take worst-model as baseline for relative reductions
    baseline = max(out.items(), key=lambda kv: kv[1]['CO2_kg'])[0]
    base_co2 = out[baseline]['CO2_kg']
    base_cost = out[baseline]['cost_usd']
    for name in out:
        out[name]['CO2_reduction_pct'] = float(100.0 * (base_co2 - out[name]['CO2_kg']) / base_co2)
        out[name]['cost_reduction_pct'] = float(100.0 * (base_cost - out[name]['cost_usd']) / base_cost)
    out['_baseline'] = baseline
    return out

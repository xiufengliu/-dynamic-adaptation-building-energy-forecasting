"""Generate Figures 5 (validation MAE convergence) and 6 (PSO trace) from real logs."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from .data import load_site, build_features, splits
from .baselines import lstm_forecast, gbm_forecast
from .gan_pso import train_wgangp

RES = Path(__file__).resolve().parents[1] / 'results'
FIG_OUT = Path(__file__).resolve().parents[2] / 'figures'  # paper figures dir


def fig5_validation_mae():
    """Real LSTM, GBM training curves + GAN-PSO curve + ARIMA/SVR horizontal lines."""
    el, wx, md, keep = load_site(max_buildings=5)
    b = keep[0]
    s = build_features(el, wx, b); tr, va, te = splits(s)

    # LSTM with history
    _, _, lstm_hist = lstm_forecast(tr, va, te, h=1, epochs=30, return_history=True)

    # GBM is non-iterative per epoch, but lgb allows per-iteration valid metric. Re-implement manually.
    import lightgbm as lgb
    from .baselines import make_direct, FEATS
    Xtr, ytr = make_direct(tr, 1); Xva, yva = make_direct(va, 1)
    dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva)
    gbm_mae = []
    def _cb(env):
        try:
            # evaluate on val
            yhat = env.model.predict(Xva, num_iteration=env.iteration + 1)
            gbm_mae.append(float(np.mean(np.abs(yhat - yva))))
        except Exception:
            pass
    lgb.train(dict(objective='regression', metric='mae', learning_rate=0.05,
                   num_leaves=64, verbosity=-1, num_threads=4),
              dtr, num_boost_round=50, valid_sets=[dva], callbacks=[_cb, lgb.log_evaluation(0)])

    # Load best PSO hparams; train WGAN-GP with epoch history
    best_hp_path = RES / 'logs' / 'pso_history.json'
    if best_hp_path.exists():
        with open(best_hp_path) as fp:
            best = json.load(fp)
        hp = best['gbest_hparams']
    else:
        hp = dict(eta_G=2e-4, eta_D=2e-4, lambda_gp=10.0, beta1=0.5, n_critic=2, dropout=0.1)
    _, gan_hist, _ = train_wgangp(tr, va, h=1, hparams=hp, epochs=30)
    gan_mae = [h['val_mae'] for h in gan_hist]
    lstm_mae = [h['val_mae'] for h in lstm_hist]

    # ARIMA / SVR final MAE references
    from .baselines import arima_forecast, svr_forecast
    from . import metrics as M
    try:
        y_a, yhat_a = arima_forecast(tr, va, va, h=1); arima_ref = M.mae(y_a, yhat_a)
    except Exception:
        arima_ref = None
    try:
        y_s, yhat_s = svr_forecast(tr, va, va, h=1); svr_ref = M.mae(y_s, yhat_s)
    except Exception:
        svr_ref = None

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(range(1, len(lstm_mae) + 1), lstm_mae, color='#1f77b4', linewidth=1.8, label='LSTM (epochs)')
    if gbm_mae:
        ax.plot(range(1, len(gbm_mae) + 1), gbm_mae, color='#d62728', linewidth=1.8, label='GBM (boosting iter.)')
    ax.plot(range(1, len(gan_mae) + 1), gan_mae, color='#7e3fa6', linewidth=2.0, label='GAN-PSO (advers. steps)')
    if arima_ref is not None:
        ax.axhline(arima_ref, color='#ff7f0e', linestyle='--', linewidth=1.2,
                   label=f'ARIMA (final fit, MAE={arima_ref:.1f})')
    if svr_ref is not None:
        ax.axhline(svr_ref, color='#2ca02c', linestyle=':', linewidth=1.4,
                   label=f'SVR (final fit, MAE={svr_ref:.1f})')
    ax.set_xlabel('Training iteration / epoch')
    ax.set_ylabel('Validation MAE (kWh)')
    ax.set_title(f'Validation MAE convergence on {b}')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    ax.set_yscale('log')
    plt.tight_layout()
    out = FIG_OUT / 'training_convergence.pdf'
    plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print('Figure 5 saved:', out)
    # also save data
    with open(RES / 'logs' / 'fig5_data.json', 'w') as fp:
        json.dump(dict(lstm_mae=lstm_mae, gbm_mae=gbm_mae, gan_mae=gan_mae,
                       arima_ref=arima_ref, svr_ref=svr_ref, building=str(b)),
                  fp, indent=2, default=float)


def fig6_pso_trace():
    with open(RES / 'logs' / 'pso_history.json') as fp:
        h = json.load(fp)
    hist = h['history']
    iters = [e['iter'] + 1 for e in hist]
    fitness = [e['gbest_fit'] for e in hist]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(iters, fitness, color='#4B0082', linewidth=2.0, label='Global best fitness $F_g$')
    # plateau annotation — find iter where improvement < 1% of total improvement
    diffs = np.diff(fitness)
    total = (fitness[0] - fitness[-1]) or 1.0
    plateau = next((i for i in range(1, len(fitness)) if abs(diffs[i - 1]) < 0.01 * total), len(fitness))
    ax.axvline(x=plateau, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
               label=f'Plateau onset ($\\sim$iter {plateau})')
    ax.set_xlabel('PSO iteration')
    ax.set_ylabel('Global best fitness $F_g$')
    ax.set_title('PSO convergence trace during one adaptation cycle')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    ax.set_xlim(0, max(iters))
    plt.tight_layout()
    out = FIG_OUT / 'pso_convergence.pdf'
    plt.savefig(out, bbox_inches='tight', dpi=300); plt.close()
    print('Figure 6 saved:', out)


if __name__ == '__main__':
    import sys
    if 'fig5' in sys.argv or not sys.argv[1:]:
        fig5_validation_mae()
    if 'fig6' in sys.argv or not sys.argv[1:]:
        fig6_pso_trace()

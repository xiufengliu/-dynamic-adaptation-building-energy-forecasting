"""Produce LaTeX table snippets and a summary from results.json."""
from pathlib import Path
import json
import numpy as np

RES = Path(__file__).resolve().parents[1] / 'results'
TAB = RES / 'tables'


ORDER = ['ARIMA', 'LSTM', 'SVR', 'GBM', 'WGAN-GP-only', 'GAN-PSO']
PRETTY = {'ARIMA': 'ARIMA', 'LSTM': 'LSTM', 'SVR': 'SVR', 'GBM': 'GBM',
          'WGAN-GP-only': 'GAN only', 'GAN-PSO': 'GAN-PSO'}


def _fmt(v, fmt='{:.1f}'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return '--'
    try: return fmt.format(v)
    except Exception: return str(v)


def horizon_table(res, horizon, caption, label, include_ablation=False):
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{' + caption + r'}',
        r'\label{' + label + r'}',
        r'\begin{tabular}{l cccccc}',
        r'\toprule',
        r'Model & MAE (kWh) & RMSE (kWh) & MAPE (\%) & WMAPE (\%) & mMAPE (\%) & R$^2$ \\',
        r'\midrule',
    ]
    for m in ORDER:
        if m not in res: continue
        if (not include_ablation) and m == 'WGAN-GP-only': continue
        d = res[m]
        row = f"{PRETTY[m]} & {_fmt(d['MAE'])} & {_fmt(d['RMSE'])} & {_fmt(d['MAPE'])} & {_fmt(d['WMAPE'])} & {_fmt(d['mMAPE'])} & {_fmt(d['R2'], '{:.3f}')} \\\\"
        lines.append(row)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def ablation_table(res_h, caption, label):
    # WGAN-GP-only vs GAN-PSO at h=1
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{' + caption + r'}',
        r'\label{' + label + r'}',
        r'\begin{tabular}{l cccc}',
        r'\toprule',
        r'Configuration & MAE (kWh) & RMSE (kWh) & WMAPE (\%) & R$^2$ \\',
        r'\midrule',
    ]
    for m in ['WGAN-GP-only', 'GAN-PSO']:
        if m not in res_h: continue
        d = res_h[m]
        lines.append(f"{PRETTY[m]} & {_fmt(d['MAE'])} & {_fmt(d['RMSE'])} & {_fmt(d['WMAPE'])} & {_fmt(d['R2'], '{:.3f}')} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def sustainability_table(sust, caption, label):
    if not sust: return ''
    ordered = [k for k in ORDER if k in sust]
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{' + caption + r'}',
        r'\label{' + label + r'}',
        r'\begin{tabular}{l cccc}',
        r'\toprule',
        r'Model & $\sigma_e$ (kWh) & Committed (kWh) & CO$_2$ reduction (\%) & Cost reduction (\%) \\',
        r'\midrule',
    ]
    for m in ordered:
        d = sust[m]
        lines.append(f"{PRETTY[m]} & {_fmt(d['sigma_e'])} & {_fmt(d['committed_kWh'], '{:.0f}')} & {_fmt(d['CO2_reduction_pct'])} & {_fmt(d['cost_reduction_pct'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    with open(TAB / 'results.json') as fp:
        r = json.load(fp)
    res = r['results']
    n_b = r['n_buildings']

    out = []
    for h, caption, label in [
        (1, f'Forecasting accuracy (1-hour ahead; Site Panther, {n_b} buildings).', 'tab:h1'),
        (24, f'Forecasting accuracy for short-term horizon (24 hours ahead; {n_b} buildings).', 'tab:forecasting_accuracy'),
        (168, f'Forecasting accuracy for medium-term horizon (7 days ahead; {n_b} buildings).', 'tab:medium_term'),
        (672, f'Forecasting accuracy for long-term horizon (4 weeks ahead; {n_b} buildings).', 'tab:long_term'),
    ]:
        if str(h) not in res and h not in res: continue
        key = h if h in res else str(h)
        out.append(f'% === h = {h} ===')
        out.append(horizon_table(res[key], h, caption, label))
        out.append('')

    # Ablation table at h=24
    if 24 in res or '24' in res:
        key = 24 if 24 in res else '24'
        out.append('% === Ablation at h=24 ===')
        out.append(ablation_table(res[key], f'Ablation study: contribution of PSO-driven hyperparameter adaptation and multi-objective fitness ($h=24$).', 'tab:ablation'))
        out.append('')

    # Sustainability
    if r.get('sustainability'):
        out.append('% === Sustainability ===')
        out.append(sustainability_table({k: v for k, v in r['sustainability'].items() if k != '_baseline_model'},
                                         r'Sustainability impact under the dispatch model $D_t = \hat{Y}_t + k\sigma_e$ with $k=1.5$, $EF_{\mathrm{electricity}}=0.417$ kg CO$_2$/kWh, $c_{\mathrm{electricity}}=\$0.12$/kWh ($h=24$ short-term horizon).',
                                         'tab:sustainability'))
    tex = '\n'.join(out)
    with open(TAB / 'tables.tex', 'w') as fp:
        fp.write(tex)
    print('Tables saved:', TAB / 'tables.tex')
    print(tex)


if __name__ == '__main__':
    main()

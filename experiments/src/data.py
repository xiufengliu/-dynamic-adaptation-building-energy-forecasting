"""BDG2 data loader for a single site with weather merge and per-building splits."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def load_site(site='Panther', min_nonnull=12000, max_buildings=40, min_mean=1.0, seed=0):
    el = pd.read_csv(DATA / 'electricity_cleaned.csv', parse_dates=['timestamp'])
    md = pd.read_csv(DATA / 'metadata.csv')
    wx = pd.read_csv(DATA / 'weather.csv', parse_dates=['timestamp'])
    site_cols = [c for c in el.columns if c != 'timestamp' and c.startswith(site + '_')]
    nonnull = el[site_cols].notna().sum()
    means = el[site_cols].mean()
    keep = [c for c in site_cols if nonnull[c] >= min_nonnull and means[c] >= min_mean]
    rng = np.random.default_rng(seed)
    if len(keep) > max_buildings:
        keep = list(rng.choice(keep, size=max_buildings, replace=False))
    keep.sort()
    el_sub = el[['timestamp'] + keep].copy()
    wx_site = wx[wx['site_id'] == site].copy()
    md_sub = md[md['building_id'].isin(keep)].set_index('building_id')
    return el_sub, wx_site, md_sub, keep


def build_features(el_sub, wx_site, building):
    s = el_sub[['timestamp', building]].rename(columns={building: 'y'})
    s = s.merge(wx_site[['timestamp', 'airTemperature', 'dewTemperature', 'windSpeed']],
                on='timestamp', how='left')
    s['hour'] = s['timestamp'].dt.hour
    s['dow'] = s['timestamp'].dt.dayofweek
    s['month'] = s['timestamp'].dt.month
    s['is_weekend'] = (s['dow'] >= 5).astype(int)
    s['hour_sin'] = np.sin(2 * np.pi * s['hour'] / 24)
    s['hour_cos'] = np.cos(2 * np.pi * s['hour'] / 24)
    s['dow_sin'] = np.sin(2 * np.pi * s['dow'] / 7)
    s['dow_cos'] = np.cos(2 * np.pi * s['dow'] / 7)
    for lag in [1, 24, 168]:
        s[f'lag_{lag}'] = s['y'].shift(lag)
    s = s.dropna().reset_index(drop=True)
    for col in ['airTemperature', 'dewTemperature', 'windSpeed']:
        s[col] = s[col].fillna(s[col].median())
    return s


def splits(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    i1 = int(n * train_frac)
    i2 = int(n * (train_frac + val_frac))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


if __name__ == '__main__':
    el, wx, md, keep = load_site()
    print('buildings selected:', len(keep))
    print('first 5:', keep[:5])
    s = build_features(el, wx, keep[0])
    tr, va, te = splits(s)
    print('train/val/test sizes:', len(tr), len(va), len(te))
    print(s.head(3))

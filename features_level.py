"""Broadband power level per panel (raw log units, before normalisation): a continuum source such as a
scintillating star or a pulsar raises all three ON panels. Gain jumps between panels also move the
levels, and they are larger on the test telescope (spread of the OFF levels 1.88 vs 1.09), so the
training levels get extra random per-panel jumps to match. Writes features_level_{train,test}.pkl."""
import numpy as np, pandas as pd
from common import load, ON, OFF

EXTRA_GAIN_STD = 1.5  # sqrt(1.88^2 - 1.09^2): makes the training spread match the test telescope


def levels(X):
    out = []
    for s in range(0, len(X), 2000):
        x = np.asarray(X[s:s + 2000]).astype(np.float32); x[x == 0] = np.nan
        out.append(np.nanmedian(x, axis=(2, 3)))
    return np.concatenate(out)


def level_features(L):
    on, off = L[:, ON], L[:, OFF]
    spread = off.std(1)
    f = {'lvl_d': on.mean(1) - off.mean(1), 'lvl_dmin': on.min(1) - off.max(1),
         'lvl_off_spread': spread, 'lvl_on_spread': on.std(1)}
    f['lvl_dz'] = f['lvl_d'] / (spread + 0.5)
    adj = np.stack([L[:, 0] - L[:, 1], L[:, 2] - L[:, 1], L[:, 2] - L[:, 3], L[:, 4] - L[:, 3], L[:, 4] - L[:, 5]], 1)
    f['lvl_adj_min'] = adj.min(1); f['lvl_adj_mean'] = adj.mean(1)
    srt = np.sort(on - off.mean(1, keepdims=True), 1)
    f['lvl_on_lo'], f['lvl_on_mid'], f['lvl_on_hi'] = srt[:, 0], srt[:, 1], srt[:, 2]
    return pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()})


if __name__ == '__main__':
    import os
    for split in ['train', 'test']:
        path = f'cache_levels_{split}.npy'
        L = np.load(path) if os.path.exists(path) else levels(load(split)[0])
        if split == 'train':
            L = L + np.random.default_rng(0).normal(0, EXTRA_GAIN_STD, L.shape).astype(np.float32)
        F = level_features(L)
        F.to_pickle(f'features_level_{split}.pkl'); print(split, F.shape)

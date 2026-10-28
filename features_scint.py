"""Scintillation features at the scale of the blotches (4-16 rows x 16-64 channels): ON-minus-OFF spread and upper
tail of each smoothed panel, after clipping so narrow lines and glitches cannot dominate a broad patch.
Reads the normalised caches; writes features_scint_{train,deg_train,test}.pkl."""
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from common import ON, OFF

SCALES = [(4, 16), (8, 32), (16, 32), (16, 64)]


def scint_batch(z):
    zc = np.clip(z, -2.5, 2.5)
    f = {}
    for rt, rf in SCALES:
        v = uniform_filter(zc, size=(1, 1, rt, rf), mode='constant').reshape(len(z), 6, -1)
        st, hi = v.std(-1), np.quantile(v, 0.95, axis=-1)
        f[f'sc_std{rt}x{rf}_on_off'] = st[:, ON].mean(1) - st[:, OFF].mean(1)
        f[f'sc_std{rt}x{rf}_onmin_offmax'] = st[:, ON].min(1) - st[:, OFF].max(1)
        f[f'sc_q95_{rt}x{rf}_on_off'] = hi[:, ON].mean(1) - hi[:, OFF].mean(1)
    return pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()})


if __name__ == '__main__':
    for src, out in [('cache_z_train.npy', 'features_scint_train.pkl'), ('cache_zdeg_train.npy', 'features_scint_deg_train.pkl'),
                     ('cache_z_test.npy', 'features_scint_test.pkl')]:
        Z = np.load(src, mmap_mode='r')
        assert Z.shape[1:] == (6, 16, 128), Z.shape
        pd.concat([scint_batch(np.asarray(Z[a:a + 1000], np.float32)) for a in range(0, len(Z), 1000)],
                  ignore_index=True).to_pickle(out)
        print(out, Z.shape, flush=True)

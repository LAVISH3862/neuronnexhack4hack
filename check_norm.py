"""Compare bandpass strategies: does the ON-vs-OFF excess of scintillating stars survive normalisation?
Also measures how much the bandpass shape changes between panels of one cadence (train vs test)."""
import numpy as np, pandas as pd
from scipy.ndimage import median_filter, uniform_filter
from common import load, normalize

X, tr = load('train'); Xt, te = load('test')
idx = np.where(tr.label == 'scintillating_star')[0][:600]
nidx = np.where(tr.label == 'noise')[0][:600]


def norm_cadence(xb):
    """Bandpass SHAPE from the median of the 3 OFF panels (smoothed), per-panel level offset."""
    x = xb.astype(np.float32); valid = ~(xb == 0).all(-1); x[~valid] = np.nan
    spec = np.nanmedian(x, axis=2)
    lvl = np.nanmedian(spec, axis=-1, keepdims=True)
    shape = np.median(spec[:, [1, 3, 5]] - lvl[:, [1, 3, 5]], axis=1, keepdims=True)
    shape = median_filter(shape, size=(1, 1, 15), mode='nearest')
    z = x - (lvl + shape)[:, :, None, :]
    med = np.nanmedian(z, axis=(2, 3), keepdims=True)
    mad = np.nanmedian(np.abs(z - med), axis=(2, 3), keepdims=True) * 1.4826 + 1e-3
    z = (z - med) / mad; z[~valid] = 0
    return z


def patch_stat(z):
    sm = uniform_filter(z, size=(1, 1, 3, 9)) * np.sqrt(27)
    s = sm.std((2, 3))
    return s[:, [0, 2, 4]].mean(1) - s[:, [1, 3, 5]].mean(1)


for name, f in [('per-panel bp', lambda a: normalize(a, True)[0]), ('OFF-shape bp', norm_cadence)]:
    a = patch_stat(f(np.asarray(X[np.sort(idx)]))); b = patch_stat(f(np.asarray(X[np.sort(nidx)])))
    print(f'{name}: ON-OFF patch std  scint {a.mean():.3f}±{a.std():.3f}   noise {b.mean():.3f}±{b.std():.3f}   '
          f'separation {(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2):.2f}')

for nm, A in [('train', X), ('test', Xt)]:
    x = np.asarray(A[:1000]).astype(np.float32); x[x == 0] = np.nan
    spec = np.nanmedian(x, axis=2); spec -= np.nanmedian(spec, -1, keepdims=True)
    sm = median_filter(spec, size=(1, 1, 15), mode='nearest')
    dev = sm - np.median(sm, axis=1, keepdims=True)
    print(f'{nm}: panel-to-panel bandpass shape deviation (smoothed) rms {np.sqrt((dev[..., 8:-8] ** 2).mean()):.2f}, '
          f'edges {np.sqrt((dev[..., :8] ** 2).mean()):.2f}; noise MAD ~{np.nanmedian(np.abs(x - np.nanmedian(x, (2, 3), keepdims=True))) * 1.48:.1f}')

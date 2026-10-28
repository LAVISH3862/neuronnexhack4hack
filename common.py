"""Shared loading and telescope-independent normalisation for Signal Hunters."""
import numpy as np, pandas as pd, warnings
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = 'signal-hunters-round2'
LABELS = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
          'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']
LAB2ID = {l: i for i, l in enumerate(LABELS)}
ON, OFF = [0, 2, 4], [1, 3, 5]


def load(split):
    X = np.load(f'{DATA}/{split}_cadences.npy', mmap_mode='r')
    meta = pd.read_csv(f'{DATA}/{split}_meta.csv')
    return X, meta


def normalize(xb, gbt):
    """uint8 (b, 6, 16, 128) -> z-scored float32 of the same shape, plus a (b, 6, 16) valid-row mask.

    The bandpass shape is shared by all panels of a cadence, so it is estimated from the OFF panels
    only (median, then a running median across frequency so narrow lines survive) and each panel
    keeps its own level (gain). ON-only emission such as scintillation therefore survives.
    GBT's centre-channel spike is removed, then each panel is divided by its robust noise scale.
    Dropped rows become 0 (the noise mean)."""
    x = xb.astype(np.float32)
    valid = ~(xb == 0).all(-1)
    x[~valid] = np.nan
    spec = np.nanmedian(x, axis=2)
    spec = np.where(np.isnan(spec), 96.0, spec)
    lvl = np.median(spec, axis=-1, keepdims=True)
    raw = np.median(spec[:, OFF] - lvl[:, OFF], axis=1, keepdims=True)  # (b, 1, 128)
    shape = median_filter(raw, size=(1, 1, 15), mode='nearest')
    if gbt:
        shape[..., 64] += raw[..., 64] - 0.5 * (raw[..., 63] + raw[..., 65])
    z = x - (lvl + shape)[:, :, None, :]
    med = np.nanmedian(z, axis=(2, 3), keepdims=True)
    mad = np.nanmedian(np.abs(z - med), axis=(2, 3), keepdims=True) * 1.4826 + 1e-3
    z = (z - med) / mad
    z[~valid] = 0.0
    return np.clip(z, -6, 30).astype(np.float32), valid


def iter_normalized(X, meta, batch=1000):
    gbt = meta.telescope.iloc[0] == 'GBT'
    for s in range(0, len(X), batch):
        z, valid = normalize(np.asarray(X[s:s + batch]), gbt)
        yield s, z, valid, meta.slew_gap_rows.values[s:s + batch]

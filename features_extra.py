"""Extra scintillation features: broad (16-32 channel) bumps in the time-averaged spectrum that appear
in the ON panels only. Lines and glitches are clipped first so they don't dominate.
Writes features_extra_{train,test}.pkl."""
import numpy as np, pandas as pd, sys
from scipy.ndimage import uniform_filter1d
from common import load, iter_normalized, ON, OFF
from features import padded, shifted_sum

DRIFTS = np.round(np.arange(-1.2, 1.2001, 0.04), 3).astype(np.float32)


def extra_batch(z, valid, gap):
    zc = np.clip(z, -4, 4)
    n = np.maximum(valid.sum(-1), 1)[..., None].astype(np.float32)  # (b, 6, 1)
    spec = zc.sum(2) / n  # time-averaged spectrum (b, 6, 128)
    f = {}
    for k in [8, 16, 24, 32, 48]:
        sm = uniform_filter1d(spec, k, axis=-1, mode='nearest')
        s = sm[..., k // 2:-k // 2].std(-1)  # avoid edges
        mx = sm[..., k // 2:-k // 2].max(-1)
        f[f'sc{k}_on_mean'] = s[:, ON].mean(1); f[f'sc{k}_off_mean'] = s[:, OFF].mean(1)
        f[f'sc{k}_diff'] = f[f'sc{k}_on_mean'] - f[f'sc{k}_off_mean']
        f[f'sc{k}_minon_maxoff'] = s[:, ON].min(1) - s[:, OFF].max(1)
        f[f'sc{k}_max_diff'] = mx[:, ON].mean(1) - mx[:, OFF].mean(1)
        if k in (16, 32):
            # ON spectrum minus mean OFF spectrum: bumps that are ON-only
            d = sm[:, ON] - sm[:, OFF].mean(1, keepdims=True)
            f[f'sc{k}_onoff_std'] = d[..., k // 2:-k // 2].std(-1).mean(1)
            f[f'sc{k}_onoff_max'] = d[..., k // 2:-k // 2].max(-1).mean(1)
            # do the ON panels share one pattern (correlated) or twinkle independently?
            a = sm[:, ON] - sm[:, ON].mean(-1, keepdims=True)
            c = lambda i, j: (a[:, i] * a[:, j]).sum(-1) / np.sqrt((a[:, i] ** 2).sum(-1) * (a[:, j] ** 2).sum(-1) + 1e-6)
            f[f'sc{k}_on_corr'] = (c(0, 1) + c(1, 2) + c(0, 2)) / 3
    # time-variability of broad emission within ON panels (twinkling)
    band = uniform_filter1d(zc, 16, axis=-1, mode='nearest')  # (b, 6, 16, 128)
    tv = band.std(2).mean(-1)  # std over time, averaged over channels
    f['tv_on_mean'] = tv[:, ON].mean(1); f['tv_diff'] = tv[:, ON].mean(1) - tv[:, OFF].mean(1)
    f.update(connect_features(z, valid, gap))
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def connect_features(z, valid, gap):
    """Best line in each ON panel found independently (drift and start channel), then: do the three
    lines connect into one straight track in global time? ETI: yes. Same-drift traps: no."""
    z3 = uniform_filter1d(z[:, ON], 3, axis=-1, mode='constant') * np.sqrt(3)
    zp = padded(z3)
    nv = np.sqrt(np.maximum(valid[:, ON].sum(-1), 1)).astype(np.float32)[..., None]
    best = np.full(z3.shape[:2], -1e9, np.float32); bd = np.zeros_like(best); bc = np.zeros_like(best)
    for d in DRIFTS:
        S = shifted_sum(zp, d, 0) / nv  # (b, 3, 128): line starting at channel c in row 0
        m = S.max(-1); upd = m > best
        best = np.where(upd, m, best); bd = np.where(upd, d, bd); bc = np.where(upd, S.argmax(-1), bc)
    L = (16 + gap).astype(np.float32)
    f = {}
    f['cn_min_snr'] = best.min(1); f['cn_mean_d'] = bd.mean(1); f['cn_d_spread'] = bd.max(1) - bd.min(1)
    f['cn_abs_mean_d'] = np.abs(bd.mean(1))
    f['cn_mid_resid'] = np.abs(bc[:, 1] - 0.5 * (bc[:, 0] + bc[:, 2]))  # ON panels are equally spaced in time
    f['cn_slope_mismatch'] = np.abs((bc[:, 2] - bc[:, 0]) / (4 * L) - bd.mean(1))
    f['cn_pred_err'] = np.abs(bc[:, 0] + bd[:, 0] * 2 * L - bc[:, 1]) + np.abs(bc[:, 1] + bd[:, 1] * 2 * L - bc[:, 2])
    return f


if __name__ == '__main__':
    for split in (sys.argv[1:] or ['train', 'test']):
        X, meta = load(split)
        F = pd.concat([extra_batch(z, v, g) for s, z, v, g in iter_normalized(X, meta, 1000)], ignore_index=True)
        F.to_pickle(f'features_extra_{split}.pkl'); print(split, F.shape)

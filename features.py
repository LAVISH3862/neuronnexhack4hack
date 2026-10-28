"""Physics features for every cadence: drift-line search, ON/OFF track consistency in global time,
dedispersed broadband search and scintillation texture. Writes features_{train,test}.pkl."""
import numpy as np, pandas as pd, time, sys
from scipy.ndimage import uniform_filter, uniform_filter1d
from common import load, iter_normalized, ON, OFF

LOCAL_D = np.array([0, .1, .2, .35, .5, .75, 1, 1.5, 2, 3, 4, 5, 6, 8], np.float32)
LOCAL_D = np.concatenate([LOCAL_D, -LOCAL_D[1:]])
TRACK_D = np.round(np.arange(-1.2, 1.2001, 0.04), 3).astype(np.float32)
TRACK_D = TRACK_D[TRACK_D != 0]  # zero drift handled separately (a non-drifting ON-only line is a trap)
DISP_K = [0, 2, 4, 6, 9, 12, 16, 22]
R = np.arange(16)


OFFS = 160


def padded(z3):
    zp = np.zeros(z3.shape[:3] + (OFFS + 128 + OFFS,), np.float32)
    zp[..., OFFS:OFFS + 128] = z3
    return zp


def shifted_sum(zp, d, pad):
    """Sum over rows of z3[..., r, c + round(d*r)] for c in [-pad, 128+pad); zp = padded(z3)."""
    W = 128 + 2 * pad
    out = np.zeros(zp.shape[:2] + (W,), np.float32)
    for r in R:
        s = OFFS - pad + int(np.round(d * r))
        out += zp[:, :, r, s:s + W]
    return out


def local_search(zp, nvalid):
    """Per panel line SNR for every drift in LOCAL_D, maximised over start channel -> (b, 6, nd)."""
    res = np.stack([shifted_sum(zp, d, 0).max(-1) for d in LOCAL_D], -1)
    return res / np.sqrt(np.maximum(nvalid, 1))[..., None]


def track_features(zp, gap, nvalid):
    """Search straight tracks in global time t = p*(16+gap)+r. For each drift and start channel,
    sum each panel along the track; return ON-only scores."""
    b = zp.shape[0]
    L = 16 + gap  # (b,)
    norm = np.sqrt(np.maximum(nvalid, 1))  # (b, 6)
    pad = 24
    c0 = np.arange(-140, 270)
    best = {k: np.full(b, -1e9, np.float32) for k in ['on_sum', 'on_min_minus_off', 'track_score']}
    at = {k: np.zeros(b, np.float32) for k in ['d', 'off_max', 'on_min', 'on_sum_of_best']}
    for d in TRACK_D:
        P = shifted_sum(zp, d, pad) / norm[..., None]  # (b, 6, 128+2pad), index c+pad
        panels = []
        for p in range(6):
            cp = c0[None, :] + np.round(d * p * L)[:, None].astype(int) + pad  # (b, nc)
            ok = (cp >= 0) & (cp < P.shape[-1])
            v = np.take_along_axis(P[:, p], np.clip(cp, 0, P.shape[-1] - 1), axis=1)
            panels.append(np.where(ok, v, 0.0))
        T = np.stack(panels, 1)  # (b, 6, nc)
        on, off = T[:, ON], T[:, OFF]
        on_min, off_max = on.min(1), off.max(1)
        score = on_min - np.maximum(off_max, 0)
        on_sum = on.sum(1) / np.sqrt(3)
        i = score.argmax(1)
        sc = score[np.arange(b), i]
        upd = sc > best['track_score']
        best['track_score'] = np.where(upd, sc, best['track_score'])
        at['d'] = np.where(upd, d, at['d'])
        at['off_max'] = np.where(upd, off_max[np.arange(b), i], at['off_max'])
        at['on_min'] = np.where(upd, on_min[np.arange(b), i], at['on_min'])
        at['on_sum_of_best'] = np.where(upd, on_sum[np.arange(b), i], at['on_sum_of_best'])
        best['on_sum'] = np.maximum(best['on_sum'], on_sum.max(1))
        best['on_min_minus_off'] = np.maximum(best['on_min_minus_off'], (on_sum - off.sum(1) / np.sqrt(3)).max(1))
    f = {f'trk_{k}': v for k, v in best.items()}
    f.update({f'trk_at_{k}': v for k, v in at.items()})
    f['trk_at_absd'] = np.abs(at['d'])
    return f


def disp_delays(k):
    x = 0.75 + 0.25 * np.arange(128) / 127
    shape = (x ** -2 - 1) / (0.75 ** -2 - 1)
    return np.round(k * shape).astype(int)  # low frequency (small c) arrives later


def dedisperse(z, valid):
    """Dedispersed broadband time series max per panel for each K -> (b, 6, nK) and pulse counts."""
    b = z.shape[0]
    maxes, npk = [], []
    vf = valid.astype(np.float32)
    for k in DISP_K:
        dl = disp_delays(k)
        t = np.arange(-k, 16)
        rows = t[:, None] + dl[None, :]  # (T, 128)
        ok = (rows >= 0) & (rows < 16)
        rc = np.clip(rows, 0, 15)
        g = z[:, :, rc, np.arange(128)[None, :]] * ok  # (b, 6, T, 128)
        cnt = (vf[:, :, rc] * ok).sum(-1)  # (b, 6, T)
        ts = g.sum(-1) / np.sqrt(np.maximum(cnt, 1))
        ts = np.where(cnt >= 32, ts, 0)
        maxes.append(ts.max(-1))
        npk.append((ts > 5).sum(-1))
    return np.stack(maxes, -1), np.stack(npk, -1)


def features_batch(z, valid, gap):
    b = z.shape[0]
    nvalid = valid.sum(-1).astype(np.float32)  # (b, 6)
    z3 = uniform_filter1d(z, 3, axis=-1, mode='constant') * np.sqrt(3)
    zp = padded(z3)
    f = {}

    loc = local_search(zp, nvalid)  # (b, 6, nd)
    ad = np.abs(LOCAL_D)
    groups = {'l0': ad == 0, 'lslow': (ad > 0) & (ad <= .5), 'lmed': (ad > .5) & (ad <= 2), 'lsteep': ad > 2}
    for g, m in groups.items():
        v = loc[..., m].max(-1)
        for p in range(6): f[f'p{p}_{g}'] = v[:, p]
        f[f'on_mean_{g}'] = v[:, ON].mean(1); f[f'off_mean_{g}'] = v[:, OFF].mean(1)
        f[f'on_min_{g}'] = v[:, ON].min(1); f[f'off_max_{g}'] = v[:, OFF].max(1)
        f[f'off_min_{g}'] = v[:, OFF].min(1); f[f'on_max_{g}'] = v[:, ON].max(1)
        f[f'all_min_{g}'] = v.min(1); f[f'n_panels_{g}_gt6'] = (v > 6).sum(1)
    bd = LOCAL_D[loc.argmax(-1)]
    for p in range(6): f[f'p{p}_bestd'] = bd[:, p]

    # Zero-drift lines per channel across panels: persistent vs ON-only (non-drifting trap)
    S0 = z3.sum(2) / np.sqrt(np.maximum(nvalid, 1))[..., None]  # (b, 6, 128)
    f['zd_all_min'] = S0.min(1).max(-1)
    f['zd_on_only'] = (S0[:, ON].min(1) - np.maximum(S0[:, OFF].max(1), 0)).max(-1)
    f['zd_on_and_off'] = np.minimum(S0[:, ON].max(1), S0[:, OFF].max(1)).max(-1)
    f['zd_n_lines'] = (S0.max(1) > 6).sum(-1)
    f['zd_5of6'] = np.sort(S0, axis=1)[:, 1].max(-1)  # line in at least 5 of 6 panels (fade in one)

    f.update(track_features(zp, gap, nvalid))
    del zp

    dm, npk = dedisperse(z, valid)
    for p in range(6):
        f[f'p{p}_bb0'] = dm[:, p, 0]
        f[f'p{p}_disp'] = dm[:, p, 1:].max(-1)
        f[f'p{p}_dispk'] = np.array(DISP_K[1:])[dm[:, p, 1:].argmax(-1)]
        f[f'p{p}_npk'] = npk[:, p, 1:].max(-1)
    for k in ['bb0', 'disp', 'npk']:
        v = np.stack([f[f'p{p}_{k}'] for p in range(6)], 1)
        f[f'on_mean_{k}'] = v[:, ON].mean(1); f[f'off_mean_{k}'] = v[:, OFF].mean(1)
        f[f'on_max_{k}'] = v[:, ON].max(1); f[f'off_max_{k}'] = v[:, OFF].max(1)
        f[f'on_min_{k}'] = v[:, ON].min(1)
        f[f'n_on_{k}_gt6'] = (v[:, ON] > 6).sum(1); f[f'n_off_{k}_gt6'] = (v[:, OFF] > 6).sum(1)

    # Scintillation texture: blotchy, many channels; smooth over (3 rows, 9 channels)
    sm = uniform_filter(z, size=(1, 1, 3, 9), mode='constant') * np.sqrt(27)
    patch_std = sm.std((2, 3)); patch_max = sm.max((2, 3)); mean = z.mean((2, 3))
    q90 = np.quantile(sm.reshape(b, 6, -1), 0.9, axis=-1)
    for nm, v in [('patch_std', patch_std), ('patch_max', patch_max), ('patch_q90', q90), ('mean', mean)]:
        for p in range(6): f[f'p{p}_{nm}'] = v[:, p]
        f[f'on_mean_{nm}'] = v[:, ON].mean(1); f[f'off_mean_{nm}'] = v[:, OFF].mean(1)
        f[f'on_min_{nm}'] = v[:, ON].min(1); f[f'off_max_{nm}'] = v[:, OFF].max(1)
    f['n_dropped_rows'] = (~valid).sum((1, 2))
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in f.items()})


def build(split, limit=None):
    X, meta = load(split)
    if limit: X, meta = X[:limit], meta.iloc[:limit]
    out, t0 = [], time.time()
    for s, z, valid, gap in iter_normalized(X, meta):
        out.append(features_batch(z, valid, gap))
        print(f'{split} {s + len(z)}/{len(X)}  {time.time() - t0:.0f}s', flush=True)
    F = pd.concat(out, ignore_index=True)
    return F


if __name__ == '__main__':
    splits = [sys.argv[1]] if len(sys.argv) > 1 else ['train', 'test']
    for split in splits:
        F = build(split)
        F.to_pickle(f'features_{split}.pkl')
        print(split, F.shape)

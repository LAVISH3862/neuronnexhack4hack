"""2D periodogram features: radar chirps are sweeps that repeat in a sawtooth, which gives a sharp peak
in each panel's 2D power spectrum away from the axes; a single satellite line gives a ridge, persistent
RFI (vertical lines) sits on the ky = 0 row. Computed on clean train, noise-degraded train and test.
Writes features_fft_{train,deg_train,test}.pkl."""
import numpy as np, pandas as pd
from common import ON, OFF


def fft_features(Z, batch=2000):
    out = []
    for a in range(0, len(Z), batch):
        z = np.clip(np.asarray(Z[a:a + batch], np.float32), -4, 8)
        P = np.abs(np.fft.rfft2(z, axes=(2, 3))) ** 2  # (b, 6, 16, 65): time-frequency x channel-frequency
        P[:, :, 0, :] = 0; P[:, :, :, 0] = 0  # drop pure-time and pure-frequency structure (RFI, glitches)
        P[:, :, :, 1:3] = 0; P[:, :, 1, :] = 0  # and the very lowest (bandpass residual, slow drifts)
        flat = P.reshape(len(z), 6, -1)
        mean = flat.mean(-1) + 1e-6
        top = np.sort(flat, -1)[..., -3:]
        peak = top[..., -1] / mean; top3 = top.sum(-1) / mean
        f = {}
        for nm, v in [('fft_peak', peak), ('fft_top3', top3)]:
            for p in range(6): f[f'p{p}_{nm}'] = v[:, p]
            f[f'on_mean_{nm}'] = v[:, ON].mean(1); f[f'off_mean_{nm}'] = v[:, OFF].mean(1)
            f[f'all_min_{nm}'] = v.min(1); f[f'all_med_{nm}'] = np.median(v, 1)
            f[f'n_{nm}_gt'] = (v > np.median(v) * 0 + 12).sum(1)
        out.append(pd.DataFrame({k: np.asarray(x, np.float32) for k, x in f.items()}))
    return pd.concat(out, ignore_index=True)


if __name__ == '__main__':
    for name, path in [('train', 'cache_z_train.npy'), ('deg_train', 'cache_zdeg_train.npy'), ('test', 'cache_z_test.npy')]:
        F = fft_features(np.load(path, mmap_mode='r'))
        F.to_pickle(f'features_fft_{name}.pkl'); print(name, F.shape, flush=True)

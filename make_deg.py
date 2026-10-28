"""Noise-degraded copy of the normalised training cadences, matching the test telescope's lower
signal-to-noise (test signals are ~0.75x as strong relative to the noise). Per cadence:
z' = (z + s*N(0,1)) / sqrt(1+s^2) with s ~ U(0.6, 1.1), i.e. SNR scaled by 0.67-0.86, noise stays unit.
Writes cache_zdeg_train.npy (float16)."""
import sys, numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
name = sys.argv[2] if len(sys.argv) > 2 else 'zdeg'
lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.6, 1.1)

Z = np.load('cache_z_train.npy'); V = np.load('cache_valid_train.npy')
rng = np.random.default_rng(seed)
s = rng.uniform(lo, hi, len(Z)).astype(np.float32)
out = np.zeros_like(Z)
for a in range(0, len(Z), 2000):
    z = Z[a:a + 2000].astype(np.float32)
    n = rng.standard_normal(z.shape, dtype=np.float32)
    zz = (z + s[a:a + 2000, None, None, None] * n) / np.sqrt(1 + s[a:a + 2000, None, None, None] ** 2)
    zz *= V[a:a + 2000][..., None]  # dropped rows stay 0
    out[a:a + 2000] = np.clip(zz, -6, 30)
np.save(f'cache_{name}_train.npy', out); np.save(f'cache_{name}sigma_train.npy', s)
print('done', out.shape, 'SNR factor range', round(float((1 / np.sqrt(1 + s ** 2)).min()), 2), round(float((1 / np.sqrt(1 + s ** 2)).max()), 2))

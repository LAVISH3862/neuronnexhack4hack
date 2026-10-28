"""Normalise all cadences once and cache them as float16 for CNN training."""
import numpy as np, time
from common import load, iter_normalized

for split in ['train', 'test']:
    X, meta = load(split)
    Z = np.zeros(X.shape, np.float16)
    V = np.zeros(X.shape[:3], bool)
    t0 = time.time()
    for s, z, valid, gap in iter_normalized(X, meta, batch=2000):
        Z[s:s + len(z)] = z
        V[s:s + len(z)] = valid
    np.save(f'cache_z_{split}.npy', Z)
    np.save(f'cache_valid_{split}.npy', V)
    print(split, Z.shape, f'{time.time() - t0:.0f}s', flush=True)

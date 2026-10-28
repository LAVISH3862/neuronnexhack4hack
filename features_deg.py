"""Physics features (base + extra) computed on the noise-degraded training cadences, with the same
column names as features_train.pkl + features_extra_train.pkl. usage: python features_deg.py PART NPARTS"""
import sys, time, numpy as np, pandas as pd
from common import load
from features import features_batch
from features_extra import extra_batch

part, nparts = int(sys.argv[1]), int(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else 'zdeg'
out_name = 'features_deg_train' if name == 'zdeg' else f'features_{name}_train'
_, tr = load('train')
Z = np.load(f'cache_{name}_train.npy', mmap_mode='r'); V = np.load('cache_valid_train.npy'); G = tr.slew_gap_rows.values
idx = np.array_split(np.arange(len(Z)), nparts)[part]
out, t0 = [], time.time()
for a in range(idx[0], idx[-1] + 1, 1000):
    b = min(a + 1000, idx[-1] + 1)
    z = np.asarray(Z[a:b], np.float32); v = V[a:b]; g = G[a:b]
    out.append(pd.concat([features_batch(z, v, g), extra_batch(z, v, g)], axis=1))
    print(f'part {part}: {b - idx[0]}/{len(idx)} {time.time() - t0:.0f}s', flush=True)
pd.concat(out, ignore_index=True).to_pickle(f'{out_name}_p{part}.pkl')

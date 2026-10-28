"""Final candidate: geometric average of the cross-fitted stacks (different test splits, pseudo-label teachers and
model sets), so only changes they agree on survive. Reports out-of-fold validation of the average and label changes.
usage: python final_avg.py OUT.csv STACK [STACK ...]   (e.g. stackdeg_fmcv_C stackdeg_fm2cv0_C ...)"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import load, LAB2ID
from blend import check_and_write

out, names = sys.argv[1], sys.argv[2:]
_, tr = load('train'); y = tr.label.map(LAB2ID).values
geo = lambda Ps: (lambda q: q / q.sum(1, keepdims=True))(np.exp(np.mean([np.log(np.clip(P, 1e-6, 1)) for P in Ps], 0)))
for kind in ['oofclean', 'oofdeg']:
    P = geo([np.load(f'preds/{n}_{kind}.npy') for n in names])
    print(f'{kind}: average of {len(names)} stacks, macro {f1_score(y, P.argmax(1), average="macro"):.4f}')
T = geo([np.load(f'preds/{n}_test.npy') for n in names])
check_and_write(T.argmax(1), out); np.save(out.replace('.csv', '_probs.npy'), T)
for ref in ['stackdeg_fm_C', 'stackdeg_fmcv_C']:
    print(f'labels changed vs {ref}: {int((T.argmax(1) != np.load(f"preds/{ref}_test.npy").argmax(1)).sum())}')

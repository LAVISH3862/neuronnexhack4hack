"""FixMatch-heavy final: geometric 50/50 blend of a final stack with the FixMatch students' 5-fold average, so the
students' own (anchor-free) test decisions carry real weight. Reports the same blend's out-of-fold validation.
usage: python make_heavy.py STACKNAME OUT.csv   (STACKNAME e.g. stackdeg_fm -> preds/stackdeg_fm_C_*.npy)"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import load, LABELS, LAB2ID
from blend import check_and_write

stack, out = sys.argv[1], sys.argv[2]
_, tr = load('train'); y = tr.label.map(LAB2ID).values
geo = lambda a, b: (lambda q: q / q.sum(1, keepdims=True))(np.exp(0.5 * np.log(np.clip(a, 1e-6, 1)) + 0.5 * np.log(np.clip(b, 1e-6, 1))))
F = {'clean': np.zeros((len(y), 9)), 'deg': np.zeros((len(y), 9))}; T = 0
for k in range(5):
    i = np.load(f'preds/prefmstu_f{k}_s0_val_idx.npy')
    F['clean'][i] = np.load(f'preds/prefmstu_f{k}_s0_val.npy'); F['deg'][i] = np.load(f'preds/prefmstu_f{k}_s0_valdeg.npy')
    T = T + np.load(f'preds/prefmstu_f{k}_s0_test.npy') / 5
S = {'clean': np.load(f'preds/{stack}_C_oofclean.npy'), 'deg': np.load(f'preds/{stack}_C_oofdeg.npy')}
f1 = lambda p: f1_score(y, p.argmax(1), average='macro')
print(f'stack alone:       clean {f1(S["clean"]):.4f}  test-like {f1(S["deg"]):.4f}')
print(f'FixMatch students: clean {f1(F["clean"]):.4f}  test-like {f1(F["deg"]):.4f}')
print(f'50/50 heavy blend: clean {f1(geo(S["clean"], F["clean"])):.4f}  test-like {f1(geo(S["deg"], F["deg"])):.4f}')
st = np.load(f'preds/{stack}_C_test.npy'); P = geo(st, T)
check_and_write(P.argmax(1), out); np.save(out.replace('.csv', '_probs.npy'), P)
print('changed vs stack alone:', int((P.argmax(1) != st.argmax(1)).sum()))

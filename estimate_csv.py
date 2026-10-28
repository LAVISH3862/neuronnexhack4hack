"""Rough public-score estimate for any submission CSV: checks the format, scores it against our best
model's probabilities (soft macro-F1), and calibrates that against our uploads with known public scores.
Biased toward agreeing with our model, so treat it as a guide. usage: python estimate_csv.py FILE.csv"""
import sys, numpy as np, pandas as pd
from common import LABELS, load

REF = 'submissions/sub6_probs.npy'
KNOWN = [('submissions/sub2_cnn_gbm_blend.csv', 0.8235), ('submissions/sub3.csv', 0.8708), ('submissions/sub4.csv', 0.8839),
         ('submissions/sub6.csv', 0.8933), ('teammate/mate_submission1.csv', 0.8655)]
L = {l: i for i, l in enumerate(LABELS)}
_, te = load('test')
P = np.load(REF)


def labels(path):
    return pd.read_csv(path).set_index('id').loc[te.id].label.map(L).values


def soft_f1(pred):
    f = []
    for k in range(9):
        tp = P[pred == k, k].sum(); fp = (1 - P[pred == k, k]).sum(); fn = P[pred != k, k].sum()
        f.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f)


s = pd.read_csv(sys.argv[1])
ok = list(s.columns) == ['id', 'label'] and len(s) == 10000 and s.id.is_unique and set(s.id) == set(te.id) and s.label.isin(LABELS).all()
print('format valid:', ok)
if not ok: sys.exit()
lab = labels(sys.argv[1])
x, y = np.array([(soft_f1(labels(p)), lb) for p, lb in KNOWN]).T
a, b = np.polyfit(x, y, 1)
print('label counts', pd.Series(np.array(LABELS)[lab]).value_counts().to_dict())
print(f'agreement with sub5 {(lab == P.argmax(1)).mean():.3f} | with sub4 {(lab == labels("submissions/sub4.csv")).mean():.3f}')
print(f'estimated public score {a * soft_f1(lab) + b:.3f}')
ct = pd.crosstab(pd.Series(np.array(LABELS)[lab], name='file'), pd.Series(np.array(LABELS)[P.argmax(1)], name='ours'))
diff = [(ct.loc[r, c], r, c) for r in ct.index for c in ct.columns if r != c and ct.loc[r, c] >= 15]
for n, r, c in sorted(diff, reverse=True): print(f'  {n:4d} cadences: file says {r}, we say {c}')

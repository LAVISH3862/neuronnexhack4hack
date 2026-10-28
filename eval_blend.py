"""Evaluate blends on held-out data, and test whether EM label-mix adaptation helps when the label mix
shifts (as it does on PKS). usage: python eval_blend.py cnn_f0_s0 [gbm_oof_name]"""
import sys, numpy as np
from sklearn.metrics import f1_score
from common import load, LAB2ID
from blend import adapt

_, tr = load('train'); y = tr.label.map(LAB2ID).values
cnn = sys.argv[1]; gbm = sys.argv[2] if len(sys.argv) > 2 else 'gbm'
idx = np.load(f'preds/{cnn}_val_idx.npy'); pc = np.load(f'preds/{cnn}_val.npy')
pg = np.load(f'preds/{gbm}_oof.npy')[idx]; yv = y[idx]
f1 = lambda p, yy=yv: f1_score(yy, p.argmax(1), average='macro')


def geo(ps, ws):
    l = sum(w * np.log(np.clip(p, 1e-6, 1)) for p, w in zip(ps, ws)) / sum(ws)
    q = np.exp(l); return q / q.sum(1, keepdims=True)


print(f'CNN {f1(pc):.4f}   GBM {f1(pg):.4f}')
best = max(((f1(geo([pc, pg], [1, w])), w) for w in [0, .25, .5, .75, 1, 1.5, 2]))
for w in [0, .25, .5, .75, 1, 1.5, 2]: print(f'  blend gbm weight {w}: {f1(geo([pc, pg], [1, w])):.4f}')
pb = geo([pc, pg], [1, best[1]])

rng = np.random.default_rng(0)
res = {a: [] for a in [0, .5, 1]}
for t in range(30):
    pri = rng.dirichlet(np.ones(9) * 3)
    pri[0] = max(pri[0], 0.01)
    n = 3000
    sel = np.concatenate([rng.choice(np.where(yv == k)[0], max(int(pri[k] * n), 5)) for k in range(9)])
    for a in res:
        q = pb[sel] if a == 0 else adapt(pb[sel], np.full(9, 1 / 9), a)[0]
        res[a].append(f1(q, yv[sel]))
for a, v in res.items(): print(f'shifted label mix, EM alpha {a}: mean F1 {np.mean(v):.4f}  min {np.min(v):.4f}')

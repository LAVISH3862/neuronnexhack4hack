"""Pick the final combination on the full out-of-fold validation (clean and test-like), with blend weights
chosen by cross-fitting (tune on one half of the cadences, score on the other) so scores are honest.
Candidates: the stack with ResNet inside, and geometric blends of a stack with the 5-fold ResNet.
usage: python eval_final.py OUT.csv [resnet_tag ...]"""
import os, sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from common import load, LABELS, LAB2ID
from blend import check_and_write

out = sys.argv[1]; rtags = sys.argv[2:] or ['pre']
_, tr = load('train'); y = tr.label.map(LAB2ID).values
f1 = lambda p, m=slice(None): f1_score(y[m], p[m].argmax(1), average='macro')


def stack(name):
    return {k: np.load(f'preds/{name}_C_{k}.npy') for k in ['oofclean', 'oofdeg', 'test']}


def cnn(tag):
    d = {'oofclean': np.zeros((len(y), 9)), 'oofdeg': np.zeros((len(y), 9)), 'test': 0}
    for k in range(5):
        i = np.load(f'preds/{tag}_f{k}_s0_val_idx.npy')
        d['oofclean'][i] = np.load(f'preds/{tag}_f{k}_s0_val.npy'); d['oofdeg'][i] = np.load(f'preds/{tag}_f{k}_s0_valdeg.npy')
        d['test'] = d['test'] + np.load(f'preds/{tag}_f{k}_s0_test.npy') / 5
    return d


def geo(a, b, w):
    q = np.exp((1 - w) * np.log(np.clip(a, 1e-6, 1)) + w * np.log(np.clip(b, 1e-6, 1)))
    return q / q.sum(1, keepdims=True)


fams = [cnn(t) for t in rtags]  # several ResNet families: equal-weight average
R = {k: sum(f[k] for f in fams) / len(fams) for k in fams[0]}
NEW, OLD = os.environ.get('STACK_NEW', 'stackdeg_pre'), os.environ.get('STACK_OLD', 'stackdeg_v4s1')
cands = {'stack with ResNet inside': stack(NEW)}
W = [0.3, 0.4, 0.5, 0.6, 0.7]
rng = np.random.default_rng(0); half = rng.random(len(y)) < 0.5
for sname, S in [('old stack + ResNet blend', stack(OLD)), ('new stack + ResNet blend', stack(NEW))]:
    # cross-fitted weight: choose on one half, score on the other
    pred = np.zeros(len(y), int)
    for a in [half, ~half]:
        w = max(W, key=lambda w: f1(geo(S['oofdeg'], R['oofdeg'], w), a))
        pred[~a] = geo(S['oofdeg'], R['oofdeg'], w)[~a].argmax(1)
    w_all = max(W, key=lambda w: f1(geo(S['oofdeg'], R['oofdeg'], w)))
    cands[sname] = {k: geo(S[k], R[k], w_all) for k in S}
    cands[sname]['crossfit_deg'] = f1_score(y, pred, average='macro'); cands[sname]['w'] = w_all
print(f'ResNet alone (5-fold OOF): clean {f1(R["oofclean"]):.4f}  test-like {f1(R["oofdeg"]):.4f}')
best = None
for n, c in cands.items():
    cf = c.get('crossfit_deg', f1(c['oofdeg']))
    print(f'{n:28s} clean {f1(c["oofclean"]):.4f}  test-like {f1(c["oofdeg"]):.4f}  (cross-fitted test-like {cf:.4f})'
          + (f'  weight {c["w"]}' if 'w' in c else ''))
    if best is None or cf > best[1]: best = (n, cf)
print('BEST:', best)
check_and_write(cands[best[0]]['test'].argmax(1), out)
np.save(out.replace('.csv', '_probs.npy'), cands[best[0]]['test'])

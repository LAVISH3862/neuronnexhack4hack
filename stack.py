"""Combine the 5-fold CNN with the feature models, evaluated on full out-of-fold predictions.

A) geometric blend of CNN and LightGBM probabilities, B) a LightGBM stacker on
features + CNN out-of-fold probabilities. Per-class logit offsets (e.g. an ETI boost) are tuned with
cross-fitting so the reported score is honest. Writes preds/<name>_test.npy for blend.py.

usage: python stack.py cnn            # CNN tag: preds/cnn_f{0..4}_s0_{val,test}.npy
"""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from common import load, LABELS, LAB2ID

tag = sys.argv[1] if len(sys.argv) > 1 else 'cnn'
_, tr = load('train'); y = tr.label.map(LAB2ID).values
f1 = lambda p, yy=y: f1_score(yy, p.argmax(1), average='macro')

# CNN out-of-fold and test (average of fold models)
cnn_oof = np.zeros((len(y), 9), np.float32); cnn_te = 0; nf = 0
for k in range(5):
    if not os.path.exists(f'preds/{tag}_f{k}_s0_val.npy'): continue
    cnn_oof[np.load(f'preds/{tag}_f{k}_s0_val_idx.npy')] = np.load(f'preds/{tag}_f{k}_s0_val.npy')
    cnn_te = cnn_te + np.load(f'preds/{tag}_f{k}_s0_test.npy'); nf += 1
cnn_te /= nf
have = cnn_oof.sum(1) > 0
print(f'CNN folds: {nf}, OOF coverage {have.mean():.2f}, CNN OOF F1 {f1(cnn_oof[have], y[have]):.4f}')

gbm_oof = np.load('preds/gbm_oof.npy'); gbm_te = np.load('preds/gbm_test.npy')
print(f'GBM OOF F1 {f1(gbm_oof):.4f}')


def geo(ps, ws):
    l = sum(w * np.log(np.clip(p, 1e-6, 1)) for p, w in zip(ps, ws)) / sum(ws)
    q = np.exp(l); return q / q.sum(1, keepdims=True)


TRAIN_PRIOR = np.bincount(y, minlength=9) / len(y)
# Test label mix estimated with BBSE (confusion-matrix inversion of the sub2 predictions)
TEST_PRIOR = np.array([1.7, 6.9, 9.7, 10.9, 7.8, 7.1, 19.8, 12.2, 23.9]) / 100
SW = None  # per-class sample weights that re-weight validation data to the test mix


def mf1(yy, pred):
    return f1_score(yy, pred, average='macro', sample_weight=None if SW is None else SW[yy])


def tune_offsets(lp, yy, iters=2):
    """Coordinate ascent on per-class log-prob offsets to maximise macro-F1."""
    b = np.zeros(9)
    for _ in range(iters):
        for k in range(9):
            best = (mf1(yy, (lp + b).argmax(1)), b[k])
            for v in np.linspace(-1.5, 2.0, 15):
                bb = b.copy(); bb[k] = v
                s = mf1(yy, (lp + bb).argmax(1))
                if s > best[0] + 1e-4: best = (s, v)
            b[k] = best[1]
    return b


def crossfit_offsets(p, yy):
    """Honest estimate: tune offsets on half the data, score on the other half, and vice versa."""
    lp = np.log(np.clip(p, 1e-6, 1)); rng = np.random.default_rng(0)
    m = rng.random(len(yy)) < 0.5; pred = np.zeros(len(yy), int)
    for a in [m, ~m]:
        b = tune_offsets(lp[a], yy[a]); pred[~a] = (lp[~a] + b).argmax(1)
    return mf1(yy, pred), tune_offsets(lp, yy)


H = have
res = {}
for w in [0.25, 0.5, 0.75, 1.0]:
    res[f'blend_w{w}'] = f1(geo([cnn_oof[H], gbm_oof[H]], [1, w]), y[H])
print('geometric blends:', {k: round(v, 4) for k, v in res.items()})
wbest = max([0.25, 0.5, 0.75, 1.0], key=lambda w: res[f'blend_w{w}'])
pb_oof = geo([cnn_oof, gbm_oof], [1, wbest]); pb_te = geo([cnn_te, gbm_te], [1, wbest])
s, b_blend = crossfit_offsets(pb_oof[H], y[H])
print(f'blend w={wbest}: {f1(pb_oof[H], y[H]):.4f}  with cross-fitted class offsets {s:.4f}  offsets {np.round(b_blend, 2)}')

# Stacker: LightGBM on features + CNN OOF probs
Ftr = pd.concat([pd.read_pickle('features_train.pkl'), pd.read_pickle('features_extra_train.pkl')], axis=1)
Fte = pd.concat([pd.read_pickle('features_test.pkl'), pd.read_pickle('features_extra_test.pkl')], axis=1)
for k, l in enumerate(LABELS):
    Ftr[f'cnn_{l}'] = cnn_oof[:, k]; Fte[f'cnn_{l}'] = cnn_te[:, k]
Ftr, ytr = Ftr[H].reset_index(drop=True), y[H]
st_oof = np.zeros((len(ytr), 9), np.float32); st_te = np.zeros((len(Fte), 9), np.float32)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(Ftr, ytr)):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
                           class_weight='balanced', verbose=-1, n_jobs=8)
    m.fit(Ftr.iloc[a], ytr[a])
    st_oof[b] = m.predict_proba(Ftr.iloc[b]); st_te += m.predict_proba(Fte) / 5
s2, b_stack = crossfit_offsets(st_oof, ytr)
print(f'stacker: {f1(st_oof, ytr):.4f}  with cross-fitted class offsets {s2:.4f}  offsets {np.round(b_stack, 2)}')
print(classification_report(ytr, st_oof.argmax(1), target_names=LABELS, digits=3))

# Same, but scored on validation data re-weighted to the estimated test label mix
SW = TEST_PRIOR / TRAIN_PRIOR
s3, b_shift = crossfit_offsets(st_oof, ytr)
print(f'stacker under test label mix: plain {mf1(ytr, st_oof.argmax(1)):.4f}  with cross-fitted offsets {s3:.4f}  '
      f'offsets {np.round(b_shift, 2)}')
SW = None

def softmax_off(p, b):
    q = np.exp(np.log(np.clip(p, 1e-6, 1)) + b); return q / q.sum(1, keepdims=True)


np.save(f'preds/blend_{tag}_test.npy', pb_te)
np.save(f'preds/blendoff_{tag}_test.npy', softmax_off(pb_te, b_blend))
np.save(f'preds/stack_{tag}_test.npy', st_te)
np.save(f'preds/stackoff_{tag}_test.npy', softmax_off(st_te, b_stack))
np.save(f'preds/stackshift_{tag}_test.npy', softmax_off(st_te, b_shift))
np.save(f'preds/stack_{tag}_oof.npy', st_oof)
for nm, p in [('blend', pb_te), ('stack', st_te), ('stackshift', softmax_off(st_te, b_shift))]:
    print(nm, 'test dist', pd.Series(np.array(LABELS)[p.argmax(1)]).value_counts().to_dict())

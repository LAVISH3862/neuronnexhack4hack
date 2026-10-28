"""Stacker with pseudo-labels: LightGBM on features + CNN probabilities, trained on the training
cadences plus the test cadences that a previous model labelled with high confidence, so it also learns
how the features look on the test telescope. Out-of-fold scores on train stay honest (the pseudo-labelled
test cadences are only ever added to the training side).

usage: python stack_pl.py CNN_TAG PL_SOURCE_NAME [THR] [OUT_NAME]
       e.g. python stack_pl.py cnn stack_cnn 0.9   (pseudo-labels from preds/stack_cnn_test.npy)
"""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from common import load, LABELS, LAB2ID

tags = sys.argv[1].split('+'); src = sys.argv[2]
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
out = sys.argv[4] if len(sys.argv) > 4 else f'stackpl_{sys.argv[1]}'
nseeds = int(sys.argv[5]) if len(sys.argv) > 5 else 3  # bag LightGBMs with different seeds
_, tr = load('train'); y = tr.label.map(LAB2ID).values

Ftr = pd.concat([pd.read_pickle('features_train.pkl'), pd.read_pickle('features_extra_train.pkl')], axis=1)
Fte = pd.concat([pd.read_pickle('features_test.pkl'), pd.read_pickle('features_extra_test.pkl')], axis=1)
if os.environ.get('LEVEL', '1') == '1' and os.path.exists('features_level_train.pkl'):  # broadband ON/OFF power levels
    Ftr = pd.concat([Ftr, pd.read_pickle('features_level_train.pkl')], axis=1)
    Fte = pd.concat([Fte, pd.read_pickle('features_level_test.pkl')], axis=1)
for tag in tags:  # out-of-fold CNN probabilities on train, fold-average on test
    oof = np.zeros((len(y), 9), np.float32); te = 0
    for k in range(5):
        oof[np.load(f'preds/{tag}_f{k}_s0_val_idx.npy')] = np.load(f'preds/{tag}_f{k}_s0_val.npy')
        te = te + np.load(f'preds/{tag}_f{k}_s0_test.npy') / 5
    assert (oof.sum(1) > 0).all(), f'{tag}: missing folds'
    for k, l in enumerate(LABELS):
        Ftr[f'{tag}_{l}'] = oof[:, k]; Fte[f'{tag}_{l}'] = te[:, k]

pp = np.load(f'preds/{src}_test.npy')
pl = np.where(pp.max(1) >= thr)[0]; ypl = pp.argmax(1)[pl]
print(f'pseudo-labels from {src}: {len(pl)} of {len(pp)} test cadences at thr {thr}:',
      pd.Series(np.array(LABELS)[ypl]).value_counts().to_dict(), flush=True)


def model(seed=0):
    return lgb.LGBMClassifier(random_state=seed, n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
                              class_weight='balanced', verbose=-1, n_jobs=8)


oof = np.zeros((len(y), 9), np.float32); pt = np.zeros((len(Fte), 9), np.float32)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(Ftr, y)):
    X = pd.concat([Ftr.iloc[a], Fte.iloc[pl]], ignore_index=True); yy = np.r_[y[a], ypl]
    for s in range(nseeds):
        m = model(s).fit(X, yy)
        oof[b] += m.predict_proba(Ftr.iloc[b]) / nseeds; pt += m.predict_proba(Fte) / (5 * nseeds)
print(f'OOF macro-F1 (train cadences) {f1_score(y, oof.argmax(1), average="macro"):.4f}')
print(classification_report(y, oof.argmax(1), target_names=LABELS, digits=3))
np.save(f'preds/{out}_test.npy', pt); np.save(f'preds/{out}_oof.npy', oof)
print(f'test: agreement with pseudo-label source {(pt.argmax(1) == pp.argmax(1)).mean():.3f}')
print('test dist', pd.Series(np.array(LABELS)[pt.argmax(1)]).value_counts().to_dict())

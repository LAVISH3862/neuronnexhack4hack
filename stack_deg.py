"""Stacker trained at the test telescope's signal-to-noise. Compares three training sets, all scored on
the noise-degraded validation (our proxy for the test telescope) and on clean validation:
  A clean train (as sub4), B degraded train, C clean + degraded train; each plus pseudo-labelled test.
usage: python stack_deg.py cnn+cnnpl PL_SOURCE THR OUT [NSEEDS]  -> preds/<OUT>_{A,B,C}_test.npy"""
import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from common import load, LABELS, LAB2ID

tags = sys.argv[1].split('+'); src = sys.argv[2]; thr = float(sys.argv[3]); out = sys.argv[4]
nseeds = int(sys.argv[5]) if len(sys.argv) > 5 else 1
_, tr = load('train'); y = tr.label.map(LAB2ID).values

lev_tr = pd.read_pickle('features_level_train.pkl'); lev_te = pd.read_pickle('features_level_test.pkl')
Fc = pd.concat([pd.read_pickle('features_train.pkl'), pd.read_pickle('features_extra_train.pkl'), lev_tr], axis=1)
Fd = pd.concat([pd.read_pickle('features_deg_train.pkl'), lev_tr], axis=1)[Fc.columns]
Ft = pd.concat([pd.read_pickle('features_test.pkl'), pd.read_pickle('features_extra_test.pkl'), lev_te], axis=1)
DEG2 = os.environ.get('DEG2', '0') == '1'  # second noise-degraded copy (different noise draw, wider range)
if DEG2:
    from features_level import level_features, EXTRA_GAIN_STD
    L0 = np.load('cache_levels_train.npy')
    lev2 = level_features(L0 + np.random.default_rng(1).normal(0, EXTRA_GAIN_STD, L0.shape).astype(np.float32))
    Fd2 = pd.concat([pd.read_pickle('features_zdeg2_train.pkl'), lev2], axis=1)[Fc.columns]
if os.environ.get('FFT', '1') == '1' and os.path.exists('features_fft_train.pkl'):  # 2D periodogram (radar)
    Fc = pd.concat([Fc, pd.read_pickle('features_fft_train.pkl')], axis=1)
    Fd = pd.concat([Fd, pd.read_pickle('features_fft_deg_train.pkl')], axis=1)
    Ft = pd.concat([Ft, pd.read_pickle('features_fft_test.pkl')], axis=1)
    if DEG2:
        if not os.path.exists('features_fft_zdeg2_train.pkl'):
            from features_fft import fft_features
            fft_features(np.load('cache_zdeg2_train.npy', mmap_mode='r')).to_pickle('features_fft_zdeg2_train.pkl')
        Fd2 = pd.concat([Fd2, pd.read_pickle('features_fft_zdeg2_train.pkl')], axis=1)
def add_fit_features(F):
    """ETI vs trap: one straight track through the ON panels vs the best free line in each ON panel."""
    best = lambda p: F[[f'p{p}_l0', f'p{p}_lslow', f'p{p}_lmed']].max(1)
    indep = (best(0) + best(2) + best(4)) / np.sqrt(3)
    F['fit_indep'] = indep; F['fit_gap'] = indep - F['trk_on_sum']
    F['fit_ratio'] = F['trk_on_sum'] / (indep.abs() + 1)
    F['fit_off_indep'] = (best(1) + best(3) + best(5)) / np.sqrt(3)
    return F


if os.environ.get('FIT', '0') == '1':
    Fc, Fd, Ft = add_fit_features(Fc), add_fit_features(Fd), add_fit_features(Ft)
    if DEG2: Fd2 = add_fit_features(Fd2)
for tag in tags:
    oc = np.zeros((len(y), 9), np.float32); od = np.zeros((len(y), 9), np.float32); te = 0
    for k in range(5):
        i = np.load(f'preds/{tag}_f{k}_s0_val_idx.npy')
        oc[i] = np.load(f'preds/{tag}_f{k}_s0_val.npy'); od[i] = np.load(f'preds/{tag}_f{k}_s0_valdeg.npy')
        if DEG2:
            if k == 0: od2 = np.zeros((len(y), 9), np.float32)
            od2[i] = np.load(f'preds/{tag}_f{k}_s0_valdeg2.npy')
        te = te + np.load(f'preds/{tag}_f{k}_s0_test.npy') / 5
    for k, l in enumerate(LABELS):
        Fc[f'{tag}_{l}'] = oc[:, k]; Fd[f'{tag}_{l}'] = od[:, k]; Ft[f'{tag}_{l}'] = te[:, k]
        if DEG2: Fd2[f'{tag}_{l}'] = od2[:, k]

pp = np.load(f'preds/{src}_test.npy'); pl = np.where(pp.max(1) >= thr)[0]; ypl = pp.argmax(1)[pl]
print(f'pseudo-labels: {len(pl)} at thr {thr}', flush=True)


import json
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0)
PARAMS.update(json.loads(os.environ.get('LGB_PARAMS', '{}')))


def model(seed):
    return lgb.LGBMClassifier(random_state=seed, class_weight='balanced', verbose=-1, n_jobs=8, **PARAMS)


res = {}
for mode in os.environ.get('MODES', 'ABC'):
    oof_c = np.zeros((len(y), 9)); oof_d = np.zeros((len(y), 9)); pt = np.zeros((len(Ft), 9))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Fc, y):
        parts = {'A': [Fc.iloc[a]], 'B': [Fd.iloc[a]], 'C': [Fc.iloc[a], Fd.iloc[a]],
                 'D': [Fc.iloc[a], Fd.iloc[a]] + ([Fd2.iloc[a]] if DEG2 else [])}[mode]
        X = pd.concat(parts + [Ft.iloc[pl]], ignore_index=True)
        yy = np.concatenate([y[a]] * len(parts) + [ypl])
        for s in range(nseeds):
            m = model(s).fit(X, yy)
            oof_c[b] += m.predict_proba(Fc.iloc[b]) / nseeds; oof_d[b] += m.predict_proba(Fd.iloc[b]) / nseeds
            pt += m.predict_proba(Ft) / (5 * nseeds)
    fc = f1_score(y, oof_c.argmax(1), average='macro'); fd = f1_score(y, oof_d.argmax(1), average='macro')
    pcd = f1_score(y, oof_d.argmax(1), average=None)
    print(f'{mode}: clean val {fc:.4f} | degraded (test-like) val {fd:.4f}   per-class degraded: '
          + ' '.join(f'{l[:5]} {v:.2f}' for l, v in zip(LABELS, pcd)), flush=True)
    np.save(f'preds/{out}_{mode}_test.npy', pt)
    np.save(f'preds/{out}_{mode}_oofclean.npy', oof_c); np.save(f'preds/{out}_{mode}_oofdeg.npy', oof_d)
    print('   test dist', pd.Series(np.array(LABELS)[pt.argmax(1)]).value_counts().to_dict(), flush=True)

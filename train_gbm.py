"""5-fold gradient boosting on the physics features. Saves out-of-fold and test probabilities."""
import os, numpy as np, pandas as pd, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from common import load, LABELS, LAB2ID

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from sklearn.ensemble import HistGradientBoostingClassifier

_, tr = load('train')
Ftr = pd.read_pickle('features_train.pkl'); Fte = pd.read_pickle('features_test.pkl')
if os.path.exists('features_extra_train.pkl'):
    Ftr = pd.concat([Ftr, pd.read_pickle('features_extra_train.pkl')], axis=1)
    Fte = pd.concat([Fte, pd.read_pickle('features_extra_test.pkl')], axis=1)
if os.path.exists('features_level_train.pkl'):  # broadband ON/OFF power levels
    Ftr = pd.concat([Ftr, pd.read_pickle('features_level_train.pkl')], axis=1)
    Fte = pd.concat([Fte, pd.read_pickle('features_level_test.pkl')], axis=1)
drop = [c for c in sys.argv[1:]]
Ftr = Ftr.drop(columns=drop, errors='ignore'); Fte = Fte.drop(columns=drop, errors='ignore')
y = tr.label.map(LAB2ID).values


def make():
    if lgb:
        return lgb.LGBMClassifier(n_estimators=700, learning_rate=0.04, num_leaves=31, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                  class_weight='balanced', verbose=-1, n_jobs=8)
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, class_weight='balanced',
                                          l2_regularization=1.0, random_state=0)


oof = np.zeros((len(y), 9), np.float32); pt = np.zeros((len(Fte), 9), np.float32)
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Ftr, y)):
    m = make().fit(Ftr.iloc[a], y[a])
    oof[b] = m.predict_proba(Ftr.iloc[b]); pt += m.predict_proba(Fte) / 5
    print(f'fold {k}: {f1_score(y[b], oof[b].argmax(1), average="macro"):.4f}', flush=True)

print('OOF macro-F1', round(f1_score(y, oof.argmax(1), average='macro'), 4))
print(classification_report(y, oof.argmax(1), target_names=LABELS, digits=3))
print(pd.DataFrame(confusion_matrix(y, oof.argmax(1)), index=LABELS, columns=[l[:6] for l in LABELS]))
np.save('preds/gbm_oof.npy', oof); np.save('preds/gbm_test.npy', pt)
print('test pred dist', pd.Series(np.array(LABELS)[pt.argmax(1)]).value_counts().to_dict())
if lgb:
    imp = pd.Series(m.feature_importances_, Ftr.columns).sort_values(ascending=False)
    print(imp.head(25).to_dict())

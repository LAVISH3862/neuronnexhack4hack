"""Adversarial validation: which features tell GBT (train) from PKS (test)? Those are domain-shift risks."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

Ftr = pd.read_pickle('features_train.pkl'); Fte = pd.read_pickle('features_test.pkl')
X = pd.concat([Ftr, Fte], ignore_index=True); d = np.r_[np.zeros(len(Ftr)), np.ones(len(Fte))]
m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
p = cross_val_predict(m, X, d, cv=3, method='predict_proba')[:, 1]
print('train-vs-test AUC', round(roc_auc_score(d, p), 4))
m.fit(X, d)
sub = np.random.default_rng(0).choice(len(X), 6000, replace=False)
pi = permutation_importance(m, X.iloc[sub], d[sub], n_repeats=3, random_state=0, scoring='roc_auc')
print(pd.Series(pi.importances_mean, X.columns).sort_values(ascending=False).head(20).round(4).to_dict())
for c in ['p0_mean', 'on_mean_patch_std', 'n_dropped_rows', 'on_mean_bb0', 'on_mean_l0']:
    print(c, 'train', Ftr[c].describe()[['mean', '50%', 'std']].round(3).to_dict(), 'test', Fte[c].describe()[['mean', '50%', 'std']].round(3).to_dict())

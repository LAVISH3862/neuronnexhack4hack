"""Class-prior correction for the test telescope's label mix: estimate the PKS class proportions without labels
(invert the confusion matrix of the test-like out-of-fold predictions), then reweight each class's probability by
estimated test share / training share. usage: python prior_correct.py STACK OUT.csv  (STACK e.g. stackdeg_fm)"""
import sys, numpy as np
from scipy.optimize import nnls
from common import load, LAB2ID
from blend import check_and_write

stack, out = sys.argv[1], sys.argv[2]
_, tr = load('train'); y = tr.label.map(LAB2ID).values
O, T = np.load(f'preds/{stack}_C_oofdeg.npy'), np.load(f'preds/{stack}_C_test.npy')
p = O.argmax(1)
C = np.stack([np.bincount(p[y == j], minlength=9) / (y == j).sum() for j in range(9)], 1)  # C[i, j] = P(pred i | true j)
q = nnls(C, np.bincount(T.argmax(1), minlength=9) / len(T))[0]; q /= q.sum()
P = T * (q / (np.bincount(y, minlength=9) / len(y)))
check_and_write(P.argmax(1), out); np.save(out.replace('.csv', '_probs.npy'), P / P.sum(1, keepdims=True))
print('estimated test mix', np.round(q, 3), '| labels changed', int((P.argmax(1) != T.argmax(1)).sum()))

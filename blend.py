"""Blend model probabilities, optionally adapt to the test label mix (EM prior estimation),
and write a checked submission CSV.

usage: python blend.py OUT.csv gbm:1 cnn_f-1_s0:1 [--em 0.5]
       model names refer to preds/<name>_test.npy (gbm -> preds/gbm_test.npy)
"""
import sys, numpy as np, pandas as pd
from common import load, LABELS

TRAIN_PRIOR = None


def em_prior(p, prior_src, iters=100):
    """Saerens et al. 2002: estimate the test label prior from predicted probabilities."""
    pt = prior_src.copy()
    for _ in range(iters):
        q = p * (pt / prior_src); q /= q.sum(1, keepdims=True)
        pt = q.mean(0)
    return pt


def adapt(p, prior_src, alpha):
    pt = em_prior(p, prior_src)
    q = p * (pt / prior_src) ** alpha
    return q / q.sum(1, keepdims=True), pt


def check_and_write(pred, out):
    _, te = load('test')
    sub = pd.DataFrame({'id': te.id, 'label': np.array(LABELS)[pred]})
    ss = pd.read_csv('signal-hunters-round2/sample_submission.csv')
    assert len(sub) == 10000 and sub.id.is_unique and set(sub.id) == set(ss.id)
    assert sub.label.isin(LABELS).all()
    sub.to_csv(out, index=False)
    print('wrote', out, sub.label.value_counts().to_dict())


if __name__ == '__main__':
    out = sys.argv[1]
    alpha = 0.0
    args = sys.argv[2:]
    if '--em' in args:
        i = args.index('--em'); alpha = float(args[i + 1]); args = args[:i] + args[i + 2:]
    P, W = 0, 0
    for a in args:
        name, w = a.split(':'); w = float(w)
        p = np.load(f'preds/{name}_test.npy')
        P = P + w * np.log(np.clip(p, 1e-6, 1)); W += w
    P = np.exp(P / W); P /= P.sum(1, keepdims=True)
    if alpha > 0:
        src = P.mean(0) * 0 + 1 / 9  # models are trained class-balanced, so treat their prior as uniform
        P, pt = adapt(P, src, alpha)
        print('estimated test prior', dict(zip(LABELS, pt.round(3))))
    check_and_write(P.argmax(1), out)
    np.save(out.replace('.csv', '_probs.npy'), P)

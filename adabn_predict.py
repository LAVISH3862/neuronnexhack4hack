"""AdaBN: re-estimate the CNN's BatchNorm statistics on the target data (no labels needed), then predict.
Checks the effect on held-out validation data with PKS-like corruptions, then writes test predictions.

usage: python adabn_predict.py cnn 0 1 2 3 4     # tag and folds -> preds/<tag>ada_f{k}_s0_test.npy
"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
tag, folds = sys.argv[1], [int(f) for f in sys.argv[2:]]
sys.argv = ['x', '--fold', '0']
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
exec(open('train_cnn.py').read().split('torch.backends.cudnn.benchmark')[0])


def adapt_bn(model, split, idx, aug=0.0):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None  # cumulative average over the target data
    model.train()
    torch.manual_seed(1); np.random.seed(1)
    with torch.no_grad():
        for i, z, v, g in batches(split, idx, 256, False):
            if aug > 0: z, v = augment(z, v, aug)
            with torch.autocast('cuda', dtype=torch.float16):
                model(to_tall(z, v, g)); model(to_tall(z.flip(-1), v, g))
    model.eval()


skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))
for k in folds:
    va = skf[k][1]
    path = f'models/{tag}_f{k}_s0.pt'
    m = Net(args.width).to(dev); m.load_state_dict(torch.load(path)); m.eval()
    f = lambda p: f1_score(y_all[va], p.argmax(1), average='macro')
    base_clean = f(predict(m, 'train', va)); base_pks = f(predict(m, 'train', va, aug_strength=1.5, seed=123))
    adapt_bn(m, 'train', va, aug=1.5)  # adapt to the corrupted validation set itself
    ada_pks = f(predict(m, 'train', va, aug_strength=1.5, seed=123))
    print(f'fold {k}: clean {base_clean:.4f} | PKS-like val: plain {base_pks:.4f} -> AdaBN {ada_pks:.4f}', flush=True)
    m.load_state_dict(torch.load(path)); adapt_bn(m, 'test', np.arange(len(te)))
    pt = predict(m, 'test', np.arange(len(te)))
    np.save(f'preds/{tag}ada_f{k}_s0_test.npy', pt)
    p0 = np.load(f'preds/{tag}_f{k}_s0_test.npy')
    print(f'   test: agreement with plain model {(pt.argmax(1) == p0.argmax(1)).mean():.3f}, '
          f'mean max-prob plain {p0.max(1).mean():.3f} -> AdaBN {pt.max(1).mean():.3f}', flush=True)

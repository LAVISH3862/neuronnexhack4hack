"""AdaBN check for the pretrained ResNet (CPU, so the GPU training job is undisturbed): re-estimate the
BatchNorm statistics on the target data and compare test-like validation macro-F1 with and without.
usage: python adabn_pre.py TAG FOLD"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
tag, fold = sys.argv[1], int(sys.argv[2])
sys.argv = ['x', '--fold', str(fold), '--level', '--no-rank']
import numpy as np, torch, torch.nn as nn
src = open('train_pre.py').read().split('torch.backends.cudnn.benchmark')[0].replace("dev = 'cuda'", "dev = 'cpu'")
exec(src)
torch.set_num_threads(8)
from sklearn.metrics import f1_score
Z['train'] = np.load('cache_zdeg_train.npy')  # test-like copy
m = Net().to(dev); m.load_state_dict(torch.load(f'models/{tag}_f{fold}_s0.pt', map_location='cpu')); m.eval()
F1 = lambda p: f1_score(y_all[va_idx], p.argmax(1), average='macro')
base = predict(m, 'train', va_idx, gain=True)
print(f'{tag} fold {fold} test-like: plain {F1(base):.4f}', flush=True)
for mod in m.modules():
    if isinstance(mod, nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
m.train()
with torch.no_grad():
    for i, z, v, g in batches('train', va_idx, 128, False):
        set_levels(i, np.ones(len(i))); m(to_tall(z, v, g))
m.eval()
ada = predict(m, 'train', va_idx, gain=True)
print(f'{tag} fold {fold} test-like: AdaBN {F1(ada):.4f}', flush=True)

"""Out-of-fold CNN predictions on the noise-degraded training cadences (cache_zdeg_train.npy), so the
stacker can be trained at the test telescope's signal-to-noise. Fold-k model predicts fold k's
validation cadences. usage: python predict_deg.py cnn cnnpl ...  -> preds/<tag>_f{k}_s0_valdeg.npy"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
tags = sys.argv[1:]
sys.argv = ['x', '--fold', '0']
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold
exec(open('train_cnn.py').read().split('torch.backends.cudnn.benchmark')[0])
dname = os.environ.get('DEG_NAME', 'zdeg'); suffix = 'deg' if dname == 'zdeg' else dname.replace('z', '', 1)
Z['train'] = np.load(f'cache_{dname}_train.npy')
skf = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y_all)), y_all))
for tag in tags:
    for k in range(5):
        va = skf[k][1]
        assert (np.load(f'preds/{tag}_f{k}_s0_val_idx.npy') == va).all()
        m = Net(args.width).to(dev); m.load_state_dict(torch.load(f'models/{tag}_f{k}_s0.pt')); m.eval()
        p = predict(m, 'train', va)
        np.save(f'preds/{tag}_f{k}_s0_val{suffix}.npy', p)
        f = lambda q: f1_score(y_all[va], q.argmax(1), average='macro')
        print(f'{tag} fold {k}: clean {f(np.load(f"preds/{tag}_f{k}_s0_val.npy")):.4f} -> degraded {f(p):.4f}', flush=True)

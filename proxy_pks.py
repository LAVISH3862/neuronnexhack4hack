"""Realistic test-telescope proxy for fold 0: each fold-0 training cadence gets a real, confidently
noise-only PKS test cadence mixed in, z = (z_train + z_pks) / sqrt(2): signals at ~0.7x SNR on top of the
test telescope's real noise texture, glitches, bandpass residue, dropped rows and panel gain jumps.
CPU only. Modes:
  python proxy_pks.py make            -> cache_pks0.npz + features_pks0.pkl
  python proxy_pks.py cnn TAG         -> preds/TAG_f0_s0_valpks.npy  (train_cnn.py-style models)
  python proxy_pks.py pre TAG         -> preds/TAG_f0_s0_valpks.npy  (train_pre.py ResNet with levels)
  python proxy_pks.py eval CNNTAGS RESNETTAG
"""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd

mode = sys.argv[1]
va = np.load('preds/pre_f0_s0_val_idx.npy')


def mixed_arrays():
    d = np.load('cache_pks0.npz')
    Zp = np.load('cache_z_train.npy'); Vp = np.load('cache_valid_train.npy')
    Zp[va] = d['z'].astype(np.float16); Vp[va] = d['v']
    return Zp, Vp, d


if mode == 'make':
    from common import load
    from features import features_batch
    from features_extra import extra_batch
    from features_level import level_features
    from features_fft import fft_features
    _, tr = load('train')
    Ztr = np.load('cache_z_train.npy', mmap_mode='r'); Zte = np.load('cache_z_test.npy', mmap_mode='r')
    Vtr = np.load('cache_valid_train.npy'); Vte = np.load('cache_valid_test.npy')
    noise = np.load('cache_pks_noise_idx.npy'); j = np.random.default_rng(7).choice(noise, len(va))
    z = (np.asarray(Ztr[va], np.float32) + np.asarray(Zte[j], np.float32)) / np.sqrt(2)
    v = Vtr[va] & Vte[j]; z *= v[..., None]; z = np.clip(z, -6, 30).astype(np.float32)
    L = np.load('cache_levels_train.npy')[va]; Lt = np.load('cache_levels_test.npy')[j]
    Lm = np.nan_to_num(L + (Lt - np.nanmean(Lt, 1, keepdims=True)), nan=96.0).astype(np.float32)
    g = tr.slew_gap_rows.values[va]
    np.savez('cache_pks0.npz', z=z, v=v, L=Lm, j=j)
    F = pd.concat([pd.concat([features_batch(z[a:a + 600], v[a:a + 600], g[a:a + 600]),
                              extra_batch(z[a:a + 600], v[a:a + 600], g[a:a + 600])], axis=1)
                   for a in range(0, len(va), 600)], ignore_index=True)
    F = pd.concat([F, level_features(Lm), fft_features(z)], axis=1)
    F.to_pickle('features_pks0.pkl'); print('made', z.shape, F.shape)

elif mode in ('cnn', 'pre', 'cnn2'):
    import torch
    tag, extra = sys.argv[2], sys.argv[3:]  # extra: model options for 'pre', e.g. --upscale 3, or --rank
    script = {'cnn': 'train_cnn.py', 'pre': 'train_pre.py', 'cnn2': 'train_cnn2.py'}[mode]  # cnn2: rank-Gaussian input
    rank = '--rank' in extra; extra = [e for e in extra if e != '--rank']
    sys.argv = ['x', '--fold', '0'] + (['--level'] + ([] if rank else ['--no-rank']) + extra if mode == 'pre' else [])
    src = open(script).read().split('torch.backends.cudnn.benchmark')[0].replace("dev = 'cuda'", "dev = 'cpu'")
    exec(src)
    torch.set_num_threads(6)
    Zp, Vp, d = mixed_arrays(); Z['train'] = Zp; V['train'] = Vp
    if mode == 'pre':
        LEV_ALL[torch.as_tensor(va)] = torch.from_numpy(d['L'])
    m = Net(args.width).to(dev) if mode in ('cnn', 'cnn2') else Net().to(dev)
    m.load_state_dict(torch.load(f'models/{tag}_f0_s0.pt', map_location='cpu')); m.eval()
    p = predict(m, 'train', va)
    np.save(f'preds/{tag}_f0_s0_valpks.npy', p)
    from sklearn.metrics import f1_score
    print(f'{tag} fold 0: clean {f1_score(y_all[va], np.load(f"preds/{tag}_f0_s0_val.npy").argmax(1), average="macro"):.4f}'
          f'  real-PKS-noise proxy {f1_score(y_all[va], p.argmax(1), average="macro"):.4f}', flush=True)

elif mode == 'eval':
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    from common import load, LABELS, LAB2ID
    tags = sys.argv[2].split('+'); rtag = sys.argv[3]
    _, tr = load('train'); y = tr.label.map(LAB2ID).values
    lev = pd.read_pickle('features_level_train.pkl'); fft_c = pd.read_pickle('features_fft_train.pkl'); fft_d = pd.read_pickle('features_fft_deg_train.pkl')
    Fc = pd.concat([pd.read_pickle('features_train.pkl'), pd.read_pickle('features_extra_train.pkl'), lev, fft_c], axis=1)
    Fd = pd.concat([pd.read_pickle('features_deg_train.pkl'), lev, fft_d], axis=1)[Fc.columns]
    Ft = pd.concat([pd.read_pickle('features_test.pkl'), pd.read_pickle('features_extra_test.pkl'),
                    pd.read_pickle('features_level_test.pkl'), pd.read_pickle('features_fft_test.pkl')], axis=1)
    Fp = pd.read_pickle('features_pks0.pkl')[Fc.columns]
    for t in tags:
        oc = np.zeros((len(y), 9)); od = np.zeros((len(y), 9)); te = 0
        for k in range(5):
            i = np.load(f'preds/{t}_f{k}_s0_val_idx.npy')
            oc[i] = np.load(f'preds/{t}_f{k}_s0_val.npy'); od[i] = np.load(f'preds/{t}_f{k}_s0_valdeg.npy')
            te = te + np.load(f'preds/{t}_f{k}_s0_test.npy') / 5
        pp = np.load(f'preds/{t}_f0_s0_valpks.npy')
        for k, l in enumerate(LABELS):
            Fc[f'{t}_{l}'] = oc[:, k]; Fd[f'{t}_{l}'] = od[:, k]; Ft[f'{t}_{l}'] = te[:, k]; Fp[f'{t}_{l}'] = pp[:, k]
    src = np.load('preds/stackdeg6_C_test.npy'); pl = np.where(src.max(1) >= 0.9)[0]
    rest = np.setdiff1d(np.arange(len(y)), va)  # stacker never sees fold-0 cadences
    X = pd.concat([Fc.iloc[rest], Fd.iloc[rest], Ft.iloc[pl]], ignore_index=True)
    yy = np.r_[y[rest], y[rest], src.argmax(1)[pl]]
    S = 0
    for s in range(2):
        m = lgb.LGBMClassifier(random_state=s, n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2.0,
                               class_weight='balanced', verbose=-1, n_jobs=6).fit(X, yy)
        S = S + m.predict_proba(Fp) / 2
    R = np.load(f'preds/{rtag}_f0_s0_valpks.npy'); yv = y[va]
    f = lambda p: f1_score(yv, p.argmax(1), average='macro')
    geo = lambda a, b, w: (lambda q: q / q.sum(1, keepdims=True))(np.exp((1 - w) * np.log(np.clip(a, 1e-6, 1)) + w * np.log(np.clip(b, 1e-6, 1))))
    print(f'real-PKS-noise proxy, fold 0: stack {f(S):.4f}   ResNet {f(R):.4f}')
    for w in [0.2, 0.3, 0.4, 0.5, 0.6]:
        print(f'  blend ResNet weight {w}: {f(geo(S, R, w)):.4f}')
    np.save('preds/proxy_pks0_stack.npy', S)

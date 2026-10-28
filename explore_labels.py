"""Plot example cadences per label (train) and a few test cadences, to see what each class looks like."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA = 'signal-hunters-round2'
os.makedirs('figures', exist_ok=True)
Xtr = np.load(f'{DATA}/train_cadences.npy', mmap_mode='r')
Xte = np.load(f'{DATA}/test_cadences.npy', mmap_mode='r')
tr = pd.read_csv(f'{DATA}/train_meta.csv')

ORDER = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
         'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']


def flatten(cad):
    """Treat 0 as missing, remove each panel's smooth bandpass and scale by its MAD.
    The bandpass is a running median across frequency, so narrow (1-3 channel) lines survive."""
    from scipy.ndimage import median_filter
    x = cad.astype(np.float32)
    x[x == 0] = np.nan
    spec = np.nanmedian(x, axis=1)                                   # (6, 128) time-median spectrum
    bp = median_filter(np.nan_to_num(spec, nan=96.0), size=(1, 15), mode='nearest')
    x = x - bp[:, None, :]
    mad = np.nanmedian(np.abs(x), axis=(1, 2), keepdims=True) * 1.4826 + 1e-6
    return np.nan_to_num(x / mad, nan=0.0)


def gallery(X, idx, names, path, flat=True):
    fig, axes = plt.subplots(len(idx), 6, figsize=(13, 1.5 * len(idx)))
    for r, (i, name) in enumerate(zip(idx, names)):
        cad = flatten(X[i]) if flat else X[i].astype(np.float32)
        lim = (-1, 5) if flat else (40, 170)
        for p in range(6):
            ax = axes[r, p]
            ax.imshow(cad[p], aspect='auto', cmap='magma', vmin=lim[0], vmax=lim[1], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(['ON', 'OFF'][p % 2], fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=7, rotation=0, ha='right', va='center')
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close()


rng = np.random.default_rng(0)
idx, names = [], []
for lab in ORDER:
    for i in rng.choice(np.where(tr.label == lab)[0], 2, replace=False):
        idx.append(i); names.append(lab)
gallery(Xtr, idx, names, 'figures/labels_flattened.png')

# Domain shift: raw train (GBT) vs raw test (PKS)
ti = rng.choice(len(Xtr), 4, replace=False); si = rng.choice(len(Xte), 4, replace=False)
fig, axes = plt.subplots(8, 6, figsize=(13, 12))
for r, (X, i, tag) in enumerate([(Xtr, i, 'GBT train') for i in ti] + [(Xte, i, 'PKS test') for i in si]):
    for p in range(6):
        axes[r, p].imshow(X[i, p], aspect='auto', cmap='magma', vmin=40, vmax=170, interpolation='nearest')
        axes[r, p].set_xticks([]); axes[r, p].set_yticks([])
    axes[r, 0].set_ylabel(tag, fontsize=7, rotation=0, ha='right', va='center')
plt.tight_layout(); plt.savefig('figures/gbt_vs_pks_raw.png', dpi=80); plt.close()

# Quick global stats per telescope
for name, X in [('train', Xtr), ('test', Xte)]:
    s = np.asarray(X[:2000])
    z = (s == 0).mean()
    v = s[s > 0].astype(np.float32)
    print(f'{name}: zero frac {z:.4f}  median {np.median(v):.1f}  p1 {np.percentile(v, 1):.1f}  p99 {np.percentile(v, 99):.1f}')
    prof = np.where(s > 0, s, np.nan).astype(np.float32)
    bp = np.nanmedian(prof, axis=(0, 1, 2))
    print(f'  bandpass min/max {np.nanmin(bp):.1f}/{np.nanmax(bp):.1f}, centre ch 63/64: {bp[63]:.1f}/{bp[64]:.1f}')

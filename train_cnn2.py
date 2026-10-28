"""CNN v2: like train_cnn.py, but each panel is rank-Gaussianised (values replaced by the normal quantile
of their rank within the panel), so the network cannot depend on the telescope's noise distribution.
Optionally adds pseudo-labelled test cadences (--pseudo FILE --pl-thr T).

CNN on the 'tall' cadence image: the 6 panels placed at their global time rows
(p * (16 + slew_gap) + r), so an ETI track is one straight line through the ON panels.

Channels: [normalised power, ON/OFF indicator (+1 ON row, -1 OFF row, 0 gap or dropped)].
Augmentations imitate the PKS test telescope: extra noise, dropped rows, broadband glitches,
bandpass ripple/tilt residuals, plus frequency shifts and flips.

usage: python train_cnn.py --fold 0          # 5-fold split, validates on fold 0
       python train_cnn.py --fold -1 --seed 1 # train on all data
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Anaconda's MKL and torch each ship an OpenMP runtime
import argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from common import load, LABELS, LAB2ID

ap = argparse.ArgumentParser()
ap.add_argument('--fold', type=int, default=0)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--bs', type=int, default=96)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--width', type=int, default=32)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--tag', default='cnn2')
ap.add_argument('--pseudo', default='')
ap.add_argument('--no-rank', action='store_true')
ap.add_argument('--pl-thr', type=float, default=0.9)
args = ap.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'

_, tr = load('train'); _, te = load('test')
y_all = tr.label.map(LAB2ID).values
Z = {'train': np.load('cache_z_train.npy'), 'test': np.load('cache_z_test.npy', mmap_mode='r')}
V = {s: np.load(f'cache_valid_{s}.npy') for s in ['train', 'test']}
G = {'train': tr.slew_gap_rows.values, 'test': te.slew_gap_rows.values}

if args.fold >= 0:
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    tr_idx, va_idx = list(skf.split(np.zeros(len(y_all)), y_all))[args.fold]
else:
    tr_idx, va_idx = np.arange(len(y_all)), np.array([], int)

NTR = len(y_all)
if args.pseudo:
    pp = np.load(args.pseudo)
    pl_idx = np.where(pp.max(1) >= args.pl_thr)[0]; y_pl = pp.argmax(1)
    Z['test'] = np.load('cache_z_test.npy')
    print(f'pseudo-labels: {len(pl_idx)} test cadences (thr {args.pl_thr})',
          pd.Series(np.array(LABELS)[y_pl[pl_idx]]).value_counts().to_dict(), flush=True)
else:
    pl_idx = np.array([], int); y_pl = np.zeros(len(te), int)
fit_idx = np.concatenate([tr_idx, pl_idx + NTR])  # indices >= NTR are test cadences
y_comb = np.concatenate([y_all, y_pl])

SIGN = torch.tensor([1., -1., 1., -1., 1., -1.], device=dev)


def rank_gauss(z, valid):
    """Per panel: replace each value by the standard-normal quantile of its rank (noise shape invariant)."""
    B = z.shape[0]
    x = z.float().reshape(B * 6, -1)
    r = x.argsort(1).argsort(1).float()
    g = torch.special.ndtri((r + 0.5) / x.shape[1]).reshape(z.shape)
    return g * valid[..., None].float()


def to_tall(z, valid, gap):
    """z (B,6,16,128), valid (B,6,16) bool, gap (B,) -> (B, 2, 128, 128)."""
    if not args.no_rank: z = rank_gauss(z, valid)
    B = z.shape[0]
    out = torch.zeros(B, 2, 128, 128, device=dev, dtype=z.dtype)
    rows = (torch.arange(6, device=dev)[None, :, None] * (16 + gap[:, None, None])
            + torch.arange(16, device=dev)[None, None, :])
    b = torch.arange(B, device=dev)[:, None, None].expand(B, 6, 16)
    vf = valid.to(z.dtype)
    out[b, 0, rows] = z * vf[..., None]
    out[b, 1, rows] = (SIGN[None, :, None] * vf)[..., None].expand(B, 6, 16, 128)
    return out


def augment(z, valid, strength=1.0):
    B = z.shape[0]
    # frequency flip and shift (same for all panels)
    flip = torch.rand(B, device=dev) < 0.5
    z = torch.where(flip[:, None, None, None], z.flip(-1), z)
    shift = int(np.random.randint(-24, 25))
    z = torch.roll(z, shift, dims=-1)
    # lower the SNR: the PKS receiver is noisier (its noise MAD is ~1.45x GBT's in raw log units)
    sig = torch.rand(B, 1, 1, 1, device=dev) * 1.1 * strength
    z = (z + sig * torch.randn_like(z)) / torch.sqrt(1 + sig ** 2)
    # residual bandpass ripple and tilt per panel
    c = torch.linspace(0, 1, 128, device=dev)
    per = 0.08 + 0.4 * torch.rand(B, 6, 1, 1, device=dev)
    amp = 0.4 * strength * torch.rand(B, 6, 1, 1, device=dev) * (torch.rand(B, 1, 1, 1, device=dev) < 0.6)
    tilt = 0.6 * strength * (torch.rand(B, 6, 1, 1, device=dev) - 0.5)
    z = z + amp * torch.sin(2 * np.pi * c / per + 6.28 * torch.rand(B, 6, 1, 1, device=dev)) + tilt * (c - 0.5)
    # broadband impulsive rows (in any panel, so they appear in OFF too)
    glitch = (torch.rand(B, 6, 16, 1, device=dev) < 0.012 * strength).to(z.dtype)
    z = z + glitch * (1 + 5 * torch.rand(B, 6, 16, 1, device=dev))
    # dropped rows
    valid = valid & (torch.rand(B, 6, 16, device=dev) > 0.01 * strength)
    # per-panel residual gain/scale mismatch
    z = z * (1 + 0.15 * strength * (torch.rand(B, 6, 1, 1, device=dev) - 0.5))
    return z, valid


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.sk(x))


class Net(nn.Module):
    """Two full-resolution convs to catch 1-channel lines, then one residual block per scale."""
    def __init__(self, w=32, n=9):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                  nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                  nn.MaxPool2d(2))
        chans = [w * 3 // 2, w * 3, w * 6, w * 8]
        layers, cin = [], w
        for i, c in enumerate(chans):
            layers.append(Block(cin, c))
            if i < 3: layers.append(nn.MaxPool2d(2))
            cin = c
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * cin, n))

    def forward(self, x):
        h = self.body(self.stem(x))
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(h)


def batches(split, idx, bs, train):
    order = np.random.permutation(idx) if train else idx
    for s in range(0, len(order), bs):
        i = np.sort(order[s:s + bs])
        z = torch.from_numpy(np.asarray(Z[split][i], np.float32)).to(dev)
        v = torch.from_numpy(V[split][i]).to(dev)
        g = torch.from_numpy(G[split][i]).to(dev)
        yield i, z, v, g


def batches_mixed(idx, bs):
    order = np.random.permutation(idx)
    for s in range(0, len(order), bs):
        i = np.sort(order[s:s + bs]); a = i[i < NTR]; b = i[i >= NTR] - NTR
        z = np.concatenate([np.asarray(Z['train'][a], np.float32), np.asarray(Z['test'][b], np.float32)])
        v = np.concatenate([V['train'][a], V['test'][b]]); g = np.concatenate([G['train'][a], G['test'][b]])
        yield i, torch.from_numpy(z).to(dev), torch.from_numpy(v).to(dev), torch.from_numpy(g).to(dev)


@torch.no_grad()
def predict(model, split, idx, aug_strength=0.0, seed=0):
    model.eval()
    out = np.zeros((len(idx), 9), np.float32)
    pos = {k: n for n, k in enumerate(idx)}
    torch.manual_seed(seed); np.random.seed(seed)
    for i, z, v, g in batches(split, idx, 256, False):
        if aug_strength > 0: z, v = augment(z, v, aug_strength)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(to_tall(z, v, g)).float().softmax(-1)
            p = p + model(to_tall(z.flip(-1), v, g)).float().softmax(-1)  # TTA: frequency flip
        out[[pos[k] for k in i]] = (p / 2).cpu().numpy()
    return out


torch.backends.cudnn.benchmark = True
model = Net(args.width).to(dev)  # channels_last is ~4x slower on this GPU
counts = np.bincount(y_all[tr_idx], minlength=9)
cw = torch.tensor((counts.sum() / counts / 9) ** 0.5, dtype=torch.float32, device=dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
steps = args.epochs * int(np.ceil(len(fit_idx) / args.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.15)
scaler = torch.amp.GradScaler()
yt = torch.from_numpy(y_comb).to(dev)

for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot = 0
    for i, z, v, g in batches_mixed(fit_idx, args.bs):
        z, v = augment(z, v)
        x = to_tall(z, v, g)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(model(x).float(), yt[i], weight=cw, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(i)
    msg = f'ep {ep} loss {tot / len(fit_idx):.4f} {time.time() - t0:.0f}s'
    if len(va_idx) and (ep % 4 == 3 or ep == args.epochs - 1):
        pv = predict(model, 'train', va_idx)
        msg += f'  val F1 {f1_score(y_all[va_idx], pv.argmax(1), average="macro"):.4f}'
    print(msg, flush=True)

name = f'{args.tag}_f{args.fold}_s{args.seed}'
print('saving', name, flush=True)
torch.save(model.state_dict(), f'models/{name}.pt')
if len(va_idx):
    pv = predict(model, 'train', va_idx)
    pva = predict(model, 'train', va_idx, aug_strength=1.5, seed=123)
    np.save(f'preds/{name}_val.npy', pv); np.save(f'preds/{name}_val_idx.npy', va_idx)
    print('clean val macro-F1', round(f1_score(y_all[va_idx], pv.argmax(1), average='macro'), 4))
    print('PKS-like val macro-F1', round(f1_score(y_all[va_idx], pva.argmax(1), average='macro'), 4))
    print(classification_report(y_all[va_idx], pv.argmax(1), target_names=LABELS, digits=3))
pt = predict(model, 'test', np.arange(len(te)))
np.save(f'preds/{name}_test.npy', pt)
print('test pred dist', pd.Series(np.array(LABELS)[pt.argmax(1)]).value_counts().to_dict())

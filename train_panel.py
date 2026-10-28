"""Panel model (idea C): shared per-panel encoder + explicit ON-vs-OFF comparison + 1D frequency network.
Same training machinery as CNN v4 (pseudo-labels, test-like noise, panel levels).

CNN v4: CNN v3 plus an optional third input channel with each panel's broadband power level relative to
the OFF panels (--level), with test-telescope-like random gain jumps added to the training levels.
Also writes out-of-fold predictions on the noise-degraded training copy (test-like validation).

CNN v3: CNN v2 plus random frequency-axis zoom (0.75-1.3x) during training, which rescales
slopes, line widths, dispersion sweeps and patch sizes (robustness to a new telescope).

CNN v2: like train_cnn.py, but each panel is rank-Gaussianised (values replaced by the normal quantile
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
ap.add_argument('--tag', default='panel')
ap.add_argument('--zoom-p', type=float, default=0.0)  # CNN v3 (cnn3pl) was trained with --zoom-p 0.5
ap.add_argument('--no-pl-balance', action='store_true')
ap.add_argument('--noise-lo', type=float, default=0.0)  # extra-noise sigma range (test telescope is noisier)
ap.add_argument('--noise-hi', type=float, default=1.1)
ap.add_argument('--level', action='store_true')
ap.add_argument('--shift', type=int, default=0)  # max frequency shift; no wrap-around (it broke ETI tracks)
ap.add_argument('--trap-p', type=float, default=0.0)  # synthetic traps: misalign one ON panel of an ETI
ap.add_argument('--gain-std', type=float, default=1.5)
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
LEV_ALL = torch.from_numpy(np.nan_to_num(np.concatenate([np.load('cache_levels_train.npy'),
                                                         np.load('cache_levels_test.npy')]).astype(np.float32), nan=96.0)).to(dev)
LV_BATCH = None  # (B, 6) relative panel levels for the batch being built, set before to_tall


def set_levels(i_all, gain_mask=None, gen=None):
    """i_all: indices into train+test (test offset by NTR). Optional extra gain jumps on masked rows."""
    global LV_BATCH
    raw = LEV_ALL[torch.as_tensor(i_all, device=dev)]
    if gain_mask is not None:
        noise = torch.randn(raw.shape, device=dev, generator=gen) * args.gain_std
        raw = raw + noise * torch.as_tensor(gain_mask, device=dev, dtype=raw.dtype)[:, None]
    LV_BATCH = (raw - raw[:, [1, 3, 5]].mean(1, keepdim=True)) / 3.0
if args.pseudo:
    pp = np.load(args.pseudo); conf = pp.max(1); y_pl = pp.argmax(1)
    sel = conf >= args.pl_thr
    if not args.no_pl_balance:
        # Self-training under-selects hard classes (scintillation), which teaches the model to under-call
        # them. Top each class up to 80% of its estimated test share (BBSE), using predictions with
        # confidence >= 0.6, most confident first.
        test_prior = np.array([2.3, 7.0, 9.7, 7.2, 8.2, 8.2, 19.7, 12.7, 24.8]) / 100
        n_tot = sel.sum()
        for k in range(9):
            want, have = int(0.8 * test_prior[k] * n_tot), int((sel & (y_pl == k)).sum())
            if have < want:
                cand = np.where(~sel & (y_pl == k) & (conf >= 0.6))[0]
                sel[cand[np.argsort(-conf[cand])][:want - have]] = True
    pl_idx = np.where(sel)[0]
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
    """Panel model input: (B, 6, 3, 16, 128) = [power, panel level vs OFF, slew gap], per panel."""
    if not args.no_rank: z = rank_gauss(z, valid)
    B = z.shape[0]
    vf = valid.to(z.dtype)[..., None]  # (B, 6, 16, 1)
    lv = (LV_BATCH.to(z.dtype) if LV_BATCH is not None else torch.zeros(B, 6, device=dev, dtype=z.dtype))
    gp = ((gap.to(z.dtype) - 4.5) / 1.5)[:, None, None, None].expand(B, 6, 16, 128)
    return torch.stack([z * vf, lv[:, :, None, None] * vf.expand(B, 6, 16, 128), gp], 2)


def freq_zoom(z, p, lo=0.75, hi=1.3):
    """Per-cadence frequency rescale by s (output channel c samples source 63.5 + (c-63.5)/s).
    Linear interpolation, with fresh noise added so the noise keeps unit variance; channels that
    fall outside the source band become pure noise."""
    B = z.shape[0]
    s = torch.ones(B, device=dev)
    m = torch.rand(B, device=dev) < p
    s[m] = lo * (hi / lo) ** torch.rand(int(m.sum()), device=dev)
    c = torch.arange(128, device=dev, dtype=torch.float32)
    x = 63.5 + (c[None, :] - 63.5) / s[:, None]
    x0 = torch.floor(x); w = (x - x0)[:, None, None, :]
    i0 = x0.long(); i1 = i0 + 1
    in0 = ((i0 >= 0) & (i0 <= 127)).float()[:, None, None, :]; in1 = ((i1 >= 0) & (i1 <= 127)).float()[:, None, None, :]
    g0 = torch.gather(z, -1, i0.clamp(0, 127)[:, None, None, :].expand(B, 6, 16, 128))
    g1 = torch.gather(z, -1, i1.clamp(0, 127)[:, None, None, :].expand(B, 6, 16, 128))
    out = g0 * (1 - w) * in0 + g1 * w * in1
    var = (1 - w) ** 2 * in0 + w ** 2 * in1
    return out + torch.randn_like(out) * torch.sqrt(torch.clamp(1 - var, min=0))


def noise_shift(z, k):
    """Shift along frequency by k channels WITHOUT wrap-around; vacated channels get fresh unit noise."""
    if k == 0: return z
    out = torch.randn_like(z)
    if k > 0: out[..., k:] = z[..., :-k]
    else: out[..., :k] = z[..., -k:]
    return out


def inject_traps(z, yb, i):
    """Turn some real ETI cadences into 'ON-only lines that do not connect' traps: shift one ON panel by
    6-30 channels so the track breaks, and relabel rfi_intermittent. Training cadences only."""
    sel = (yb == LAB2ID['eti_candidate']) & torch.as_tensor(i < NTR, device=dev) & (torch.rand(len(yb), device=dev) < args.trap_p)
    for b in torch.where(sel)[0].tolist():
        p = int(np.random.choice([0, 2, 4])); k = int(np.random.randint(6, 31)) * int(np.random.choice([-1, 1]))
        z[b, p] = noise_shift(z[b, p], k)
    yb = yb.clone(); yb[sel] = LAB2ID['rfi_intermittent']
    return z, yb


def augment(z, valid, strength=1.0):
    B = z.shape[0]
    if args.zoom_p > 0 and strength == 1.0: z = freq_zoom(z, args.zoom_p)
    # frequency flip and shift (same for all panels)
    flip = torch.rand(B, device=dev) < 0.5
    z = torch.where(flip[:, None, None, None], z.flip(-1), z)
    if args.shift > 0:
        z = noise_shift(z, int(np.random.randint(-args.shift, args.shift + 1)))
    # lower the SNR: the PKS receiver is noisier (its noise MAD is ~1.45x GBT's in raw log units)
    sig = (args.noise_lo + torch.rand(B, 1, 1, 1, device=dev) * (args.noise_hi - args.noise_lo)) * strength
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
    """Shared per-panel encoder (keeps 32 frequency bands), explicit ON-vs-OFF comparison per band,
    then a dilated 1D network across frequency so drifting tracks can be matched between panels."""
    def __init__(self, w=32, n=9):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                 nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(),
                                 nn.MaxPool2d(2), Block(w, 2 * w), nn.MaxPool2d(2), Block(2 * w, 3 * w))
        c = 6 * w  # mean+max over time of 3w channels
        self.mix = nn.Sequential(nn.Conv1d(10 * c, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU())
        layers = []
        for d in [1, 2, 4, 8, 1]:
            layers += [nn.Conv1d(256, 256, 3, padding=d, dilation=d, bias=False), nn.BatchNorm1d(256), nn.ReLU()]
        self.freq = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, n))

    def forward(self, x):
        B = x.shape[0]
        h = self.enc(x.reshape(B * 6, 3, 16, 128))  # (B*6, 3w, 4, 32)
        h = torch.cat([h.mean(2), h.amax(2)], 1).reshape(B, 6, -1, 32)  # (B, 6, c, 32)
        on, off = h[:, [0, 2, 4]], h[:, [1, 3, 5]]
        cmp = [on.mean(1) - off.mean(1), on.amin(1) - off.amax(1), on.mean(1), off.mean(1)]
        u = torch.cat([h.reshape(B, -1, 32)] + cmp, 1)  # (B, 10c, 32)
        u = self.freq(self.mix(u))
        return self.head(torch.cat([u.mean(2), u.amax(2)], 1))


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
def predict(model, split, idx, aug_strength=0.0, seed=0, gain=False):
    model.eval()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    out = np.zeros((len(idx), 9), np.float32)
    pos = {k: n for n, k in enumerate(idx)}
    torch.manual_seed(seed); np.random.seed(seed)
    for i, z, v, g in batches(split, idx, 256, False):
        if aug_strength > 0: z, v = augment(z, v, aug_strength)
        if args.level:
            set_levels(i + (NTR if split == 'test' else 0), np.ones(len(i)) if gain else None, gen)
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
        yb = yt[i]
        if args.trap_p > 0: z, yb = inject_traps(z, yb, i)
        if args.level: set_levels(i, (i < NTR).astype(np.float32))  # gain jumps on training cadences only
        x = to_tall(z, v, g)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(model(x).float(), yb, weight=cw, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() * len(i)
    msg = f'ep {ep} loss {tot / len(fit_idx):.4f} {time.time() - t0:.0f}s'
    if len(va_idx) and (ep % 4 == 3 or ep == args.epochs - 1):
        pv = predict(model, 'train', va_idx)
        msg += f'  val F1 {f1_score(y_all[va_idx], pv.argmax(1), average="macro"):.4f}'
    print(msg, flush=True)

name = f'{args.tag}_f{args.fold}_s0'  # seed is encoded in the tag
print('saving', name, flush=True)
torch.save(model.state_dict(), f'models/{name}.pt')
if len(va_idx):
    pv = predict(model, 'train', va_idx)
    pva = predict(model, 'train', va_idx, aug_strength=1.5, seed=123, gain=True)
    Zc = Z['train']; Z['train'] = np.load('cache_zdeg_train.npy')  # test-like copy
    pdg = predict(model, 'train', va_idx, gain=True); Z['train'] = Zc
    np.save(f'preds/{name}_valdeg.npy', pdg)
    print('test-like (degraded) val macro-F1', round(f1_score(y_all[va_idx], pdg.argmax(1), average='macro'), 4))
    np.save(f'preds/{name}_val.npy', pv); np.save(f'preds/{name}_val_idx.npy', va_idx)
    print('clean val macro-F1', round(f1_score(y_all[va_idx], pv.argmax(1), average='macro'), 4))
    print('PKS-like val macro-F1', round(f1_score(y_all[va_idx], pva.argmax(1), average='macro'), 4))
    print(classification_report(y_all[va_idx], pv.argmax(1), target_names=LABELS, digits=3))
pt = predict(model, 'test', np.arange(len(te)))
np.save(f'preds/{name}_test.npy', pt)
print('test pred dist', pd.Series(np.array(LABELS)[pt.argmax(1)]).value_counts().to_dict())

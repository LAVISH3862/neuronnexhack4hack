"""Time the CNN pieces: data loading, to_tall, augment, forward/backward (channels_last vs not, cudnn.benchmark)."""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.argv = ['x', '--fold', '0', '--epochs', '1']
import torch, numpy as np
src = open('train_cnn.py').read().split('model = Net(')[0]
exec(src)


def t(f, n=10):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t0) / n


i = np.sort(np.random.choice(tr_idx, 96, replace=False))
print('load batch', t(lambda: torch.from_numpy(np.asarray(Z['train'][i], np.float32)).to(dev)))
z = torch.from_numpy(np.asarray(Z['train'][i], np.float32)).to(dev); v = torch.from_numpy(V['train'][i]).to(dev)
g = torch.from_numpy(G['train'][i]).to(dev)
print('augment', t(lambda: augment(z, v)))
print('to_tall', t(lambda: to_tall(z, v, g)))
x = to_tall(z, v, g); yy = torch.from_numpy(y_all[i]).to(dev)
for bench in [False, True]:
    torch.backends.cudnn.benchmark = bench
    for cl in [False, True]:
        m = Net(32).to(dev)
        xx = x
        if cl: m = m.to(memory_format=torch.channels_last); xx = x.contiguous(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(m.parameters())
        sc = torch.amp.GradScaler()

        def step():
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.cross_entropy(m(xx).float(), yy)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
        step(); step()
        s = t(step, 5)
        print(f'cudnn.benchmark={bench} channels_last={cl}: {s * 1000:.0f} ms/step -> {s * len(tr_idx) / 96:.0f} s/epoch')

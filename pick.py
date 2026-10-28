"""Rank fold-0 test configurations by the mean of the synthetic test-like score (from the training log) and
the real-PKS-noise proxy score (from proxy_<tag>.log). Prints tags best first.
usage: python pick.py TAG [TAG ...]"""
import re, sys


def score(path, pattern):
    try:
        m = re.search(pattern, open(path).read()); return float(m.group(1)) if m else 0.0
    except OSError:
        return 0.0


rows = []
for tag in sys.argv[1:]:
    d = score(f'{tag}_f0.log', r'test-like \(degraded\) val macro-F1 ([0-9.]+)')
    p = score(f'proxy_{tag}.log', r'real-PKS-noise proxy ([0-9.]+)')
    rows.append(((d + p) / 2, tag, d, p))
rows.sort(reverse=True)
for s, tag, d, p in rows:
    print(f'# {tag}: combined {s:.4f} (synthetic test-like {d:.4f}, real-PKS-noise {p:.4f})', file=sys.stderr)
print(' '.join(tag for _, tag, _, _ in rows))

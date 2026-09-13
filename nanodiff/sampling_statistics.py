"""Batch-level sampling statistics; uncertainty is conditional on fixed checkpoints."""
import math
import numpy as np


def counts(tokens, vocab, order=1, lag=1):
    """Never form a tuple across sequence boundaries."""
    x = np.asarray(tokens, dtype=np.int64)
    if x.ndim != 2 or not 1 <= order <= 3 or lag < 1:
        raise ValueError('Expected sequences, order 1..3, and positive lag')
    width = x.shape[1] - (order - 1) * lag
    if width <= 0 or x.size == 0 or np.any(x < 0) or np.any(x >= vocab):
        raise ValueError('Invalid tokens or tuple length')
    ids = np.zeros((len(x), width), dtype=np.int64)
    for i in range(order):
        ids = ids * vocab + x[:, i*lag:i*lag+width]
    return np.bincount(ids.ravel(), minlength=vocab**order)


def js_counts(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or min(a.sum(), b.sum()) <= 0 or np.any(a < 0) or np.any(b < 0):
        raise ValueError('Nonempty nonnegative count arrays of same shape required')
    p, q = a/a.sum(), b/b.sum()
    m = (p+q)/2
    return float(.5 * (np.sum(p[p>0]*np.log2(p[p>0]/m[p>0])) +
                       np.sum(q[q>0]*np.log2(q[q>0]/m[q>0]))))


def bootstrap_fixed_checkpoints(values, draws=10000, seed=7103):
    """Resample generation batches within each fixed checkpoint, NOT tokens.

    Input shape: checkpoint, independent generation replicate. For paired
    contrasts, input the replicate-wise differences. Training uncertainty and
    reference-corpus uncertainty are deliberately not estimated.
    """
    x = np.asarray(values, float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all() or draws < 1:
        raise ValueError('Expected finite checkpoint x replicate matrix')
    rng = np.random.default_rng(seed)
    means = np.zeros(draws)
    for row in x:
        means += row[rng.integers(len(row), size=(draws, len(row)))].mean(-1)/len(x)
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(x.mean()), ci95=[float(low), float(high)],
                checkpoint_means=x.mean(-1).tolist(), replicates_per_checkpoint=x.shape[1])


def paired_sign_test(differences):
    """Exact two-sided sign test. Ties removed. Null: P(delta<0)=1/2
    independently for each retained pair, conditional on fixed checkpoints.
    This tests signs, not the mean magnitude; pairing is by RNG replicate.
    """
    x = np.asarray(differences, float).ravel()
    if not np.isfinite(x).all():
        raise ValueError('Finite differences required')
    negative, positive = int((x < 0).sum()), int((x > 0).sum())
    n = negative + positive
    p = min(1., 2*sum(math.comb(n,k) for k in range(min(negative,positive)+1))/2**n) if n else 1.
    return dict(negative=negative, positive=positive, ties=int((x==0).sum()), p_two_sided=p)

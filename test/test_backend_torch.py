"""TensorFlow-free tests for the PyTorch backend (pytest).

    pytest test/test_backend_torch.py

Covers the pieces that ``test/unittests.py`` (which imports TensorFlow at module
level) does not: the batched exchangeability matrix, the eigendecomposition-based
``expm_gtr`` and its custom backward (incl. the degenerate Jukes-Cantor spectrum),
``expm_ntr``, the log-space ``traverse_branch``, and Felsenstein pruning on a deep
tree against an exact float64 log-space reference (the case that underflowed in
the old exp -> matmul -> log path).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import scipy.linalg
import torch
from scipy.special import logsumexp

from evoten import model, set_backend, substitution_models
from evoten.backend import backend
from evoten.backend_pytorch import BackendTorch
from evoten.expm_torch import expm_gtr, expm_ntr
from evoten.tree_handler import TreeHandler


@pytest.fixture(autouse=True)
def _use_torch_backend():
    set_backend("pytorch")


def _random_reversible(d, M, rng, dtype=torch.float64):
    """Random normalised GTR rate matrices (M, d, d) and their equilibria (M, d)."""
    R = rng.exponential(1.0, size=(M, d, d))
    R = 0.5 * (R + R.transpose(0, 2, 1))
    for m in range(M):
        np.fill_diagonal(R[m], 0.0)
    pi = rng.dirichlet(np.ones(d), size=M)
    R_t = torch.tensor(R, dtype=dtype)
    pi_t = torch.tensor(pi, dtype=dtype)
    return BackendTorch().make_rate_matrix(R_t, pi_t), pi_t


# ---------------------------------------------------------------------------
# kernels -> matrices
# ---------------------------------------------------------------------------
def test_make_symmetric_pos_semidefinite_batched():
    kernel = torch.randn(3, 5, 5, dtype=torch.float64)
    R = BackendTorch().make_symmetric_pos_semidefinite(kernel)
    assert R.shape == (3, 5, 5)
    torch.testing.assert_close(R, R.transpose(-1, -2))
    assert torch.all(torch.diagonal(R, dim1=-2, dim2=-1) == 0)
    assert torch.all(R >= 0)


def test_make_rate_matrix_normalized_jukes_cantor():
    R, pi = substitution_models.jukes_cantor(mue=[1., 2., 5.], d=3)
    Q = BackendTorch().make_rate_matrix(R, pi).numpy()
    ref = np.array([[[-1., .5, .5], [.5, -1, .5], [.5, .5, -1]]] * 3)
    np.testing.assert_allclose(Q, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# matrix exponentials
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d,M", [(4, 1), (4, 3), (20, 2)])
def test_expm_gtr_matches_matrix_exp(d, M):
    rng = np.random.default_rng(0)
    Q, pi = _random_reversible(d, M, rng)
    t = torch.tensor(np.concatenate([[0.0, 1e-13, 1e-9, 1e-3], np.logspace(-2, 1.5, 12)]),
                     dtype=torch.float64)[:, None]                       # (B, 1)
    P = expm_gtr(Q, t, pi)                                               # (B, M, d, d)
    ref = torch.linalg.matrix_exp(Q[None] * t[..., None, None])
    assert P.shape == (t.shape[0], M, d, d)
    torch.testing.assert_close(P, ref, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(P.sum(-1), torch.ones(t.shape[0], M, d, dtype=P.dtype))
    eye = torch.eye(d, dtype=P.dtype).expand(M, d, d)
    assert torch.equal(P[0], eye) and torch.equal(P[1], eye)             # identity at t ~ 0


def test_expm_gtr_per_model_times_and_float32():
    rng = np.random.default_rng(3)
    Q, pi = _random_reversible(4, 3, rng)
    t = torch.tensor(rng.uniform(0.01, 2.0, size=(5, 3)), dtype=torch.float64)   # (B, M)
    P = expm_gtr(Q, t, pi)
    ref = torch.linalg.matrix_exp(Q[None] * t[..., None, None])
    torch.testing.assert_close(P, ref, atol=1e-9, rtol=1e-9)
    P32 = expm_gtr(Q.float(), t.float(), pi.float())
    assert P32.dtype == torch.float32
    torch.testing.assert_close(P32.double(), ref, atol=2e-6, rtol=1e-5)


def test_expm_gtr_gradcheck():
    rng = np.random.default_rng(1)
    d, M, B = 4, 2, 3
    b = BackendTorch()
    R_k = torch.tensor(rng.normal(size=(M, d, d)), dtype=torch.float64, requires_grad=True)
    pi_k = torch.tensor(rng.normal(size=(M, d)), dtype=torch.float64, requires_grad=True)
    t = torch.tensor(rng.uniform(0.05, 2.0, size=(B, 1)), dtype=torch.float64, requires_grad=True)

    def f(R_k, pi_k, t):
        pi = b.make_equilibrium(pi_k)
        Q = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(R_k), pi)
        return expm_gtr(Q, t, pi)

    assert torch.autograd.gradcheck(f, (R_k, pi_k, t), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_expm_gtr_degenerate_spectrum_gradient():
    """Jukes-Cantor has an eigenvalue of multiplicity d-1: back-propagating through
    torch.linalg.eigh would give inf/NaN, the divided-difference backward must match
    the exact autograd of torch.linalg.matrix_exp."""
    # The eigen-route (like TF's) differentiates on the reversible manifold, so the
    # perturbation must enter through the symmetric kernel as in training: a constant
    # kernel gives exactly Jukes-Cantor exchangeabilities after softplus.
    b = BackendTorch()
    d = 4
    K = torch.full((1, d, d), float(b.inverse_softplus(torch.tensor(4. / 3))), dtype=torch.float64)
    pi_t = torch.full((1, d), 0.25, dtype=torch.float64)
    t = torch.tensor([[0.3], [1.2], [4.0]], dtype=torch.float64)
    G = torch.randn(3, 1, d, d, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    K1 = K.clone().requires_grad_(True)
    t1 = t.clone().requires_grad_(True)
    Q1 = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(K1), pi_t)
    ev = torch.linalg.eigvalsh(Q1[0].detach())
    assert torch.isclose(ev[:-1], ev[0]).all()                          # multiplicity d-1
    P1 = expm_gtr(Q1, t1, pi_t)
    (P1 * G).sum().backward()

    K2 = K.clone().requires_grad_(True)
    t2 = t.clone().requires_grad_(True)
    Q2 = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(K2), pi_t)
    P2 = torch.linalg.matrix_exp(Q2[None] * t2[..., None, None])
    (P2 * G).sum().backward()

    assert torch.isfinite(K1.grad).all() and torch.isfinite(t1.grad).all()
    torch.testing.assert_close(K1.grad, K2.grad, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(t1.grad, t2.grad, atol=1e-8, rtol=1e-6)


def test_expm_ntr_matches_matrix_exp_and_gtr():
    rng = np.random.default_rng(5)
    Q, pi = _random_reversible(6, 2, rng)
    t = torch.logspace(-2, 1, 10, dtype=torch.float64)[:, None]
    torch.testing.assert_close(expm_gtr(Q, t, pi), expm_ntr(Q, t), atol=1e-9, rtol=1e-9)
    # a genuinely non-reversible rate matrix
    A = torch.tensor(rng.exponential(size=(1, 5, 5)), dtype=torch.float64)
    A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
    Q_ntr = (A - torch.diag_embed(A.sum(-1))).requires_grad_(True)
    P = expm_ntr(Q_ntr, t)
    torch.testing.assert_close(P, torch.linalg.matrix_exp(Q_ntr[None] * t[..., None, None]),
                               atol=1e-9, rtol=1e-9)
    P.sum().backward()
    assert torch.isfinite(Q_ntr.grad).all()


# ---------------------------------------------------------------------------
# traverse_branch
# ---------------------------------------------------------------------------
def _small_traverse_case():
    P = np.array([[[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]]]], dtype=np.float32)  # (1,1,3,3)
    x = np.array([[[[0.1, 0.3, 0.2], [0.1, 0.8, 0.1]]]], dtype=np.float32)                   # (1,1,2,3)
    return P, x


def test_traverse_branch_values():
    b = BackendTorch()
    P, x = _small_traverse_case()
    y = x @ np.transpose(P, (0, 1, 3, 2))
    np.testing.assert_allclose(b.traverse_branch(x, P, logarithmic=False).numpy(), y, atol=1e-6)
    np.testing.assert_allclose(b.traverse_branch(np.log(x), P, logarithmic=True).numpy(),
                               np.log(y), atol=1e-6)
    y_t = x @ P
    np.testing.assert_allclose(
        b.traverse_branch(np.log(x), P, logarithmic=True, transposed=True).numpy(),
        np.log(y_t), atol=1e-6)


def test_traverse_branch_position_specific_matrices():
    b = BackendTorch()
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(3), size=(2, 1, 4, 3)).astype(np.float32)    # (nodes, M, L, d, d)
    x = rng.dirichlet(np.ones(3), size=(2, 1, 4)).astype(np.float32)       # (nodes, M, L, d)
    y = np.einsum("nmld,nmlzd->nmlz", x, P)
    np.testing.assert_allclose(b.traverse_branch(np.log(x), P).numpy(), np.log(y), atol=1e-6)


def test_traverse_branch_no_underflow():
    """Log partials of deep subtrees are far below log(float32 tiny) ~ -87; the
    stable path must be exact there, the old exp -> matmul -> log path was not."""
    b = BackendTorch()
    X = torch.full((2, 1, 3, 4), -300.0)                    # (nodes, models, L, d) log partials
    P = torch.eye(4).expand(2, 1, 1, 4, 4)                  # (nodes, models, L*=1, d, d)
    torch.testing.assert_close(b.traverse_branch(X, P), X)
    assert (b.traverse_branch(X, P, stable=False) == -1e3).all()


# ---------------------------------------------------------------------------
# Felsenstein pruning on a deep tree vs an exact float64 log-space reference
# ---------------------------------------------------------------------------
def _balanced_newick(n_leaves, blen):
    def rec(lo, hi):
        if hi - lo == 1:
            return f"L{lo}:{blen}"
        mid = (lo + hi) // 2
        return f"({rec(lo, mid)},{rec(mid, hi)}):{blen}"
    return rec(0, n_leaves) + ";"


def _reference_loglik(tree, names, leaves, Q, pi):
    """Exact log-space pruning in float64 with scipy's expm, postorder over the Bio tree."""
    name2idx = {nm: i for i, nm in enumerate(names)}
    L, d = leaves.shape[-2], leaves.shape[-1]
    lv = leaves[:, 0].astype(np.float64)

    def rec(clade):
        if clade.is_terminal():
            with np.errstate(divide="ignore"):
                return np.log(lv[name2idx[clade.name]])
        acc = np.zeros((L, d))
        for ch in clade.clades:
            lp = rec(ch)
            with np.errstate(divide="ignore"):
                logP = np.log(np.maximum(scipy.linalg.expm(Q * ch.branch_length), 0.0))
            acc += logsumexp(lp[:, None, :] + logP[None, :, :], axis=-1)
        return acc

    return logsumexp(rec(tree.bio_tree.root) + np.log(pi)[None, :], axis=-1)


def test_deep_tree_loglik_matches_exact_reference(tmp_path):
    n, d, L = 256, 4, 5
    nwk = tmp_path / "balanced.nwk"
    nwk.write_text(_balanced_newick(n, 0.05))
    tree = TreeHandler.read(str(nwk))
    names = [t.name for t in tree.bio_tree.get_terminals(order="preorder")]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, d, size=(n, L))
    leaves = np.eye(d, dtype=np.float32)[idx][:, None]                 # (n, 1, L, d)
    leaves[rng.random(n) < 0.1] = 1.0                                   # missing taxa
    R, pi = substitution_models.jukes_cantor(d=d)

    Q = backend.make_rate_matrix(torch.tensor(R), torch.tensor(pi))
    tp = backend.make_transition_probs(Q, torch.tensor(np.asarray(tree.branch_lengths, np.float32)),
                                       torch.tensor(pi))
    lp = torch.log(torch.tensor(pi))[:, None, :]
    ll = model.loglik(torch.tensor(leaves), tree, tp, lp, leaf_names=names,
                      leaves_are_probabilities=True)                    # (1, L)
    ref = _reference_loglik(tree, names, leaves, Q[0].double().numpy(), pi[0].astype(np.float64))
    assert ref.max() < -250                                             # far below float32 underflow
    np.testing.assert_allclose(ll.numpy()[0], ref, rtol=1e-4, atol=1e-2)

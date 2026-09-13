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
from evoten import expm_torch
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


def _near_jukes_cantor(d, M, jitter, seed=0):
    """Symmetric kernels + equilibria of the kind trained codon components end up with
    (Agrippa sweep 5): uniform pi and exchangeabilities equal up to ``jitter``, i.e. one
    eigenvalue of multiplicity d-1, split by ~jitter."""
    g = torch.Generator().manual_seed(seed)
    k0 = float(BackendTorch().inverse_softplus(torch.tensor(1.0)))
    K = k0 + jitter * torch.randn(M, d, d, dtype=torch.float64, generator=g)
    return K, torch.full((M, d), 1.0 / d, dtype=torch.float64)


def _flaky_eigh(fail):
    """A torch.linalg.eigh that raises like non-converging cuSOLVER whenever fail(A)."""
    real_eigh = torch.linalg.eigh

    def eigh(A, *args, **kwargs):
        if fail(A):
            raise torch.linalg.LinAlgError("linalg.eigh: The algorithm failed to converge (simulated)")
        return real_eigh(A, *args, **kwargs)
    return eigh


def test_expm_gtr_float32_near_jukes_cantor_matches_reference():
    """The eigen-route is accurate on the (d-1)-fold degenerate spectrum in float32:
    only the eigenbasis of the cluster is ill-determined, not the exponential."""
    b = BackendTorch()
    d, M = 64, 2
    K, pi = _near_jukes_cantor(d, M, jitter=1e-9)
    Q = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(K), pi)
    ev = torch.linalg.eigvalsh(Q[0])
    assert ev[d - 2] - ev[0] < 1e-6                                      # multiplicity d-1
    t = torch.tensor([[1e-3], [0.1], [1.0], [5.0]], dtype=torch.float64)
    ref = torch.linalg.matrix_exp(Q[None] * t[..., None, None])
    P32 = expm_gtr(Q.float(), t.float(), pi.float())
    torch.testing.assert_close(P32.double(), ref, atol=2e-6, rtol=1e-5)


def test_expm_gtr_eigh_fallback_on_degenerate_spectrum(monkeypatch):
    """Agrippa sweep 5 died when float32 cuSOLVER could not decompose a near-Jukes-Cantor
    codon matrix (eigenvalue of multiplicity 63). With that failure simulated for float32
    input, expm_gtr must redo the decomposition in float64 and still return the exact
    transition matrices and the float64 route's gradients."""
    b = BackendTorch()
    d, M = 64, 2
    K, pi = _near_jukes_cantor(d, M, jitter=1e-9)
    t = torch.tensor([[1e-3], [0.1], [1.0], [5.0]], dtype=torch.float64)
    G = torch.randn(4, M, d, d, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    K64 = K.clone().requires_grad_(True)                                 # reference: float64 route
    Q64 = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(K64), pi)
    P64 = expm_gtr(Q64, t, pi)
    (P64 * G).sum().backward()
    ref = torch.linalg.matrix_exp(Q64.detach()[None] * t[..., None, None])

    monkeypatch.setattr(expm_torch, "eigh_fallbacks", 0)
    monkeypatch.setattr(torch.linalg, "eigh", _flaky_eigh(lambda A: A.dtype == torch.float32))
    K32 = K.float().requires_grad_(True)
    with pytest.warns(RuntimeWarning, match="recomputed in float64"):
        Q32 = b.make_rate_matrix(b.make_symmetric_pos_semidefinite(K32), pi.float())
        P32 = expm_gtr(Q32, t.float(), pi.float())
    assert expm_torch.eigh_fallbacks == 1
    assert P32.dtype == torch.float32
    torch.testing.assert_close(P32.double(), ref, atol=2e-6, rtol=1e-5)
    (P32 * G.float()).sum().backward()
    assert torch.isfinite(K32.grad).all()
    torch.testing.assert_close(K32.grad.double(), K64.grad, atol=1e-4, rtol=1e-3)


def test_expm_gtr_eigh_failure_on_non_finite_input_is_reported(monkeypatch):
    """Diverged (non-finite) rate parameters are not retried but named as such."""
    Q, pi = _random_reversible(4, 2, np.random.default_rng(2))
    Q = Q.clone()
    Q[1, 0, 1] = float("nan")
    t = torch.tensor([[0.5]], dtype=torch.float64)
    monkeypatch.setattr(expm_torch, "eigh_fallbacks", 0)
    monkeypatch.setattr(torch.linalg, "eigh", _flaky_eigh(lambda A: True))
    with pytest.raises(torch.linalg.LinAlgError, match=r"non-finite .*\[\[1\]\]"):
        expm_gtr(Q, t, pi)
    assert expm_torch.eigh_fallbacks == 0


def test_expm_gtr_eigh_failure_in_float64_too_propagates(monkeypatch):
    """A solver failure that float64 cannot fix either is raised, never swallowed."""
    Q, pi = _random_reversible(4, 2, np.random.default_rng(4))
    t = torch.tensor([[0.5]], dtype=torch.float64)
    monkeypatch.setattr(expm_torch, "eigh_fallbacks", 0)
    monkeypatch.setattr(torch.linalg, "eigh", _flaky_eigh(lambda A: True))
    with pytest.warns(RuntimeWarning), pytest.raises(torch.linalg.LinAlgError, match="simulated"):
        expm_gtr(Q.float(), t.float(), pi.float())
    assert expm_torch.eigh_fallbacks == 1


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


# ---------------------------------------------------------------------------
# level-routed pruning: routing, node layout, gradients, strata, torch.compile
# ---------------------------------------------------------------------------
_DATA = os.path.join(os.path.dirname(__file__), "data")


def _caterpillar_newick(n_leaves, rng):
    """(((L0,L1),L2),L3)...: height n_leaves-1, one leaf and one internal child per level."""
    s = f"L0:{rng.uniform(0.05, 0.5):.3f}"
    for i in range(1, n_leaves):
        s = f"({s},L{i}:{rng.uniform(0.05, 0.5):.3f}):{rng.uniform(0.05, 0.5):.3f}"
    return s + ";"


def _random_newick(n_leaves, rng, max_children=4):
    """Random multifurcating tree; the children of a node have different heights."""
    nodes = [f"L{i}:{rng.uniform(0.05, 0.5):.3f}" for i in range(n_leaves)]
    while len(nodes) > 1:
        k = min(len(nodes), int(rng.integers(2, max_children + 1)))
        pick = set(rng.choice(len(nodes), size=k, replace=False).tolist())
        merged = "(" + ",".join(nodes[i] for i in sorted(pick)) + f"):{rng.uniform(0.05, 0.5):.3f}"
        nodes = [nd for i, nd in enumerate(nodes) if i not in pick] + [merged]
    return nodes[0] + ";"


def _read_tree(tmp_path, newick, name="t.nwk"):
    path = tmp_path / name
    path.write_text(newick)
    tree = TreeHandler.read(str(path))
    names = [t.name for t in tree.bio_tree.get_terminals(order="preorder")]
    return tree, names


def _reference_partials(tree, names, leaves, Q):
    """Exact float64 log partials log P(leaves below u | u) of every internal node
    (dict name -> (L, d)), postorder over the Bio tree with scipy's expm."""
    name2idx = {nm: i for i, nm in enumerate(names)}
    lv = np.asarray(leaves, np.float64)
    out = {}

    def rec(clade):
        if clade.is_terminal():
            with np.errstate(divide="ignore"):
                return np.log(lv[name2idx[clade.name]])
        acc = 0.0
        for ch in clade.clades:
            lp = rec(ch)
            blen = float(np.float32(ch.branch_length))   # TreeHandler stores float32 lengths
            with np.errstate(divide="ignore"):
                logP = np.log(np.maximum(scipy.linalg.expm(Q * blen), 0.0))
            acc = acc + logsumexp(lp[:, None, :] + logP[None, :, :], axis=-1)
        out[clade.name] = acc
        return acc

    rec(tree.bio_tree.root)
    return out


def test_level_routing_simple3():
    """simple3.tree: (((A,B)G,C)I,(D,(E,F)H)J)K; layers [A..F], [G,H], [I,J], [K]."""
    t = TreeHandler.read(os.path.join(_DATA, "simple3.tree"))
    routing = t.level_routing()
    assert len(routing) == t.height == 3
    assert [g[0] for g in routing[0]] == [1, 2]
    l, src, dst, src_id, dst_id = routing[0][0]                # A,B -> G ; E,F -> H
    np.testing.assert_array_equal(src, [0, 1, 4, 5]); np.testing.assert_array_equal(dst, [0, 0, 1, 1])
    assert (src_id, dst_id) == (False, False)
    l, src, dst, src_id, dst_id = routing[0][1]                # C -> I ; D -> J (one each: no aggregation)
    np.testing.assert_array_equal(src, [2, 3]); np.testing.assert_array_equal(dst, [0, 1])
    assert (src_id, dst_id) == (False, True)
    assert routing[1] == [(2, *routing[1][0][1:3], True, True)]  # G -> I, H -> J: pure pass-through
    l, src, dst, src_id, dst_id = routing[2][0]                # I, J -> K
    np.testing.assert_array_equal(dst, [0, 0]); assert (src_id, dst_id) == (True, False)


def test_level_routing_covers_every_node_once(tmp_path):
    rng = np.random.default_rng(1)
    tree, _ = _read_tree(tmp_path, _random_newick(40, rng))
    routing = tree.level_routing()
    assert len(routing) == tree.height
    for h, groups in enumerate(routing):
        start_h = tree.cum_layer_sizes[h] - tree.layer_sizes[h]
        seen = np.concatenate([g[1] for g in groups])
        assert sorted(seen.tolist()) == list(range(tree.layer_size(h)))
        assert [g[0] for g in groups] == sorted({g[0] for g in groups})
        for l, src, dst, src_id, dst_id in groups:
            assert l > h
            start_l = tree.cum_layer_sizes[l] - tree.layer_sizes[l]
            np.testing.assert_array_equal(tree.parent_indices[start_h + src], start_l + dst)
            assert src_id == (len(src) == tree.layer_size(h) and np.array_equal(src, np.arange(len(src))))
            assert dst_id == (len(dst) == tree.layer_size(l) and np.array_equal(dst, np.arange(len(dst))))


def test_device_index_cache_and_reset(tmp_path):
    tree, names = _read_tree(tmp_path, _random_newick(8, np.random.default_rng(2)))
    like = torch.zeros(3)
    a = tree.device_index(("x",), np.array([2, 0, 1]), like)
    assert a.dtype == torch.int64 and a.tolist() == [2, 0, 1]
    assert tree.device_index(("x",), lambda: 1 / 0, like) is a          # cache hit: values not evaluated
    r1 = tree.routing_for(like)
    assert tree.routing_for(like) is r1 and torch.is_tensor(r1[0][0][1])
    tree.update()                                                        # any tree change drops the caches
    assert tree.device_index(("x",), np.array([1]), like) is not a


@pytest.mark.parametrize("kind,d", [("caterpillar", 4), ("caterpillar", 64), ("random", 4), ("random", 20)])
def test_pruning_matches_reference_all_nodes(tmp_path, kind, d):
    """Values of every internal node (layout (num_anc, M, L, d) in node-index order) and the
    log-likelihood, with permuted leaf names and two models, against the exact recursion."""
    rng = np.random.default_rng(7)
    n, L, M = (41, 3, 2) if kind == "caterpillar" else (60, 3, 2)
    tree, names = _read_tree(tmp_path, _caterpillar_newick(n, rng) if kind == "caterpillar"
                             else _random_newick(n, rng))
    if kind == "caterpillar":
        assert tree.height == n - 1
    perm = rng.permutation(n)
    names_in = [names[i] for i in perm]
    leaves = rng.dirichlet(np.ones(d), size=(n, L))
    leaves[rng.random(n) < 0.15] = 1.0                                   # missing taxa
    Q, pi = _random_reversible(d, M, rng)
    tp = expm_gtr(Q, torch.tensor(np.asarray(tree.branch_lengths, np.float64)), pi)   # (B, M, d, d)
    lp = torch.log(pi)[:, None, :]
    leaves_t = torch.tensor(leaves[perm])[:, None]                       # (n, 1, L, d) float64

    anc = model.compute_ancestral_probabilities(leaves_t, tree, tp, leaf_names=names_in,
                                                leaves_are_probabilities=True)
    assert anc.shape == (tree.num_anc, M, L, d)
    ll = model.loglik(leaves_t, tree, tp, lp, leaf_names=names_in, leaves_are_probabilities=True)
    for m in range(M):
        ref = _reference_partials(tree, names, leaves, Q[m].numpy())
        for name, part in ref.items():
            i = tree.nodes[name].index - tree.num_leaves
            np.testing.assert_allclose(anc[i, m].numpy(), part, rtol=1e-9, atol=1e-9, err_msg=name)
        root = ref[tree.bio_tree.root.name]
        np.testing.assert_allclose(ll[m].numpy(), logsumexp(root + np.log(pi[m].numpy()), axis=-1),
                                   rtol=1e-9, atol=1e-9)


def test_pruning_gradcheck_without_recompute(tmp_path):
    """Backward through the pruning under plain autograd. The former in-place accumulation
    raised 'modified by an inplace operation' on trees of height >= 2, so training could only
    run under activation checkpointing."""
    rng = np.random.default_rng(3)
    tree, names = _read_tree(tmp_path, _random_newick(9, rng))
    assert tree.height >= 2
    d, M, L = 3, 1, 2
    Q, pi = _random_reversible(d, M, rng)
    tp = expm_gtr(Q, torch.tensor(np.asarray(tree.branch_lengths, np.float64)), pi).detach().requires_grad_(True)
    logits = torch.tensor(np.log(rng.dirichlet(np.ones(d), size=(9, L))))[:, None].requires_grad_(True)
    lp = torch.log(pi)[:, None, :]

    def f(tp, logits):
        return model.loglik(logits, tree, tp, lp, leaf_names=names, leaves_are_probabilities=False)

    assert torch.autograd.gradcheck(f, (tp, logits), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_stratified_compact_kernel_matches_full(tmp_path):
    rng = np.random.default_rng(5)
    tree, names = _read_tree(tmp_path, _random_newick(30, rng))
    tree.stratify_branchlen(6)
    assert tree.is_stratified
    d, M, L = 4, 2, 3
    Q, pi = _random_reversible(d, M, rng)
    tp_compact = expm_gtr(Q, torch.tensor(np.asarray(tree.strata_blen_values, np.float64)), pi)  # (K, M, d, d)
    leaves = torch.tensor(rng.dirichlet(np.ones(d), size=(30, L)))[:, None]
    lp = torch.log(pi)[:, None, :]
    ll_compact = model.loglik(leaves, tree, tp_compact, lp, leaf_names=names, leaves_are_probabilities=True)
    tp_full = tp_compact[torch.as_tensor(tree.branch_to_stratum, dtype=torch.int64)]      # (B, M, d, d)
    tree.is_stratified = False
    ll_full = model.loglik(leaves, tree, tp_full, lp, leaf_names=names, leaves_are_probabilities=True)
    tree.is_stratified = True
    torch.testing.assert_close(ll_compact, ll_full)


def test_torch_compile_captures_pruning_in_one_graph(tmp_path):
    """dynamo + AOT autograd (backend aot_eager: no inductor, no compiler needed) capture the
    whole pruning as one graph without breaks or data-dependent guards, values and gradients
    equal eager, and a different column count reuses the dynamic-shape graph. Runs with
    capture_scalar_outputs=True, which hidten switches on globally at import."""
    dynamo = pytest.importorskip("torch._dynamo")
    rng = np.random.default_rng(11)
    tree, names = _read_tree(tmp_path, _caterpillar_newick(12, rng))
    d, M = 4, 2
    Q, pi = _random_reversible(d, M, rng, dtype=torch.float32)
    tp = expm_gtr(Q, torch.tensor(np.asarray(tree.branch_lengths, np.float32)), pi).detach().requires_grad_(True)
    lp = torch.log(pi)[:, None, :]

    def leaves_for(L):
        return torch.tensor(rng.dirichlet(np.ones(d), size=(12, L)).astype(np.float32))[:, None]

    def f(leaves, tp):
        return model.loglik(leaves, tree, tp, lp, leaf_names=names, leaves_are_probabilities=True)

    leaves, leaves2 = leaves_for(5), leaves_for(7)
    eager = f(leaves, tp)
    (g_eager,) = torch.autograd.grad(eager.sum(), [tp])
    dynamo.reset()
    try:
        with dynamo.config.patch(capture_scalar_outputs=True):
            ex = dynamo.explain(f)(leaves, tp)
            assert (ex.graph_count, ex.graph_break_count) == (1, 0), ex.break_reasons
            cf = torch.compile(f, backend="aot_eager", dynamic=True)
            out = cf(leaves, tp)
            (g,) = torch.autograd.grad(out.sum(), [tp])
            torch.testing.assert_close(out, eager)
            torch.testing.assert_close(g, g_eager)
            with dynamo.config.patch(error_on_recompile=True):
                out2 = cf(leaves2, tp)
            torch.testing.assert_close(out2, f(leaves2, tp))
    finally:
        dynamo.reset()

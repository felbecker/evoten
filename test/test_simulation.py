"""Tests for evoten.simulation (pytest, PyTorch backend).

    pytest test/test_simulation.py

Sampled columns are compared by chi-square goodness-of-fit tests (fixed seeds)
with exact probabilities from scipy's expm or evoten's pruning.
"""
import itertools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import scipy.linalg
import scipy.stats
import torch

from evoten import model, set_backend, simulation
from evoten.backend_pytorch import BackendTorch
from evoten.tree_handler import TreeHandler

MIN_P_VALUE = 1e-6


@pytest.fixture(autouse=True)
def _use_torch_backend():
    set_backend("pytorch")


def _random_gtr(d, M, rng):
    """Random normalized reversible rate matrices (M, d, d) and equilibria (M, d), float64."""
    R = rng.exponential(1.0, size=(M, d, d))
    R = 0.5 * (R + R.transpose(0, 2, 1))
    for m in range(M):
        np.fill_diagonal(R[m], 0.0)
    pi = rng.dirichlet(3.0 * np.ones(d), size=M)
    Q = BackendTorch().make_rate_matrix(torch.tensor(R), torch.tensor(pi))
    return Q, torch.tensor(pi)


def _random_ntr(d, M, rng):
    """Random non-reversible rate matrices (M, d, d), float64."""
    A = rng.exponential(0.5, size=(M, d, d))
    for m in range(M):
        np.fill_diagonal(A[m], 0.0)
    return torch.tensor(A - A.sum(axis=-1)[:, :, None] * np.eye(d))


def _chi2_p_value(observed, probs, min_expected=5.0):
    """Pearson chi-square goodness-of-fit p-value; cells with an expected count
    below min_expected are pooled."""
    observed = np.asarray(observed, dtype=np.float64).ravel()
    expected = np.asarray(probs, dtype=np.float64).ravel() * observed.sum()
    small = expected < min_expected
    obs = np.append(observed[~small], observed[small].sum())
    exp = np.append(expected[~small], expected[small].sum())
    if exp[-1] == 0:
        assert obs[-1] == 0, "observed an outcome with probability 0"
        obs, exp = obs[:-1], exp[:-1]
    statistic = np.sum((obs - exp) ** 2 / exp)
    return scipy.stats.chi2.sf(statistic, len(obs) - 1)


def _pattern_codes(states, d):
    """Index of each column's leaf pattern in itertools.product(range(d), repeat=num_leaves) order."""
    place_values = d ** np.arange(states.shape[0] - 1, -1, -1)
    return (states.astype(np.int64) * place_values[:, None]).sum(axis=0)


def test_labels_follow_weights_and_identity_branches_copy_states():
    tree = TreeHandler.from_newick("(A:0.1,B:0.2)R;")
    d, weights = 3, np.array([0.2, 0.5, 0.3])
    P = torch.eye(d, dtype=torch.float64).expand(tree.num_nodes - 1, 3, d, d)
    root = torch.full((3, d), 1.0 / d, dtype=torch.float64)
    s = simulation.sample_mixture(tree, P, root, weights, 100_000, seed=0, chunk_size=30_000)
    assert s.leaf_states.shape == (2, 100_000) and s.labels.shape == (100_000,)
    assert s.leaf_names == ["A", "B"]
    assert _chi2_p_value(np.bincount(s.labels, minlength=3), weights) > MIN_P_VALUE
    np.testing.assert_array_equal(s.leaf_states[0], s.leaf_states[1])
    assert _chi2_p_value(np.bincount(s.leaf_states[0], minlength=d), root[0].numpy()) > MIN_P_VALUE


@pytest.mark.parametrize("reversible", [True, False])
def test_two_leaf_joint_distribution(reversible):
    rng = np.random.default_rng(1)
    d = 4
    tree = TreeHandler.from_newick("(A:0.3,B:0.7)R;")
    if reversible:
        Q, root = _random_gtr(d, 1, rng)
    else:
        Q = _random_ntr(d, 1, rng)
        root = torch.tensor(rng.dirichlet(np.ones(d)))[None]
    s = simulation.sample_mixture_ctmc(tree, Q, root, [1.0], 200_000, reversible=reversible, seed=3)
    a, b = s.leaf_names.index("A"), s.leaf_names.index("B")
    observed = np.zeros((d, d))
    np.add.at(observed, (s.leaf_states[a], s.leaf_states[b]), 1)
    Qn, r = Q[0].numpy(), root[0].numpy()
    PA, PB = scipy.linalg.expm(0.3 * Qn), scipy.linalg.expm(0.7 * Qn)
    expected = np.einsum("r,ra,rb->ab", r, PA, PB)
    assert _chi2_p_value(observed, expected) > MIN_P_VALUE
    if reversible:
        # pulley principle: the joint distribution depends only on the path length
        np.testing.assert_allclose(expected, r[:, None] * scipy.linalg.expm(1.0 * Qn), atol=1e-12)
    else:
        # the root prior is used at the root, not the stationary distribution of Q
        eigval, eigvec = np.linalg.eig(Qn.T)
        stationary = np.real(eigvec[:, np.argmin(np.abs(eigval))])
        stationary /= stationary.sum()
        expected_stationary = np.einsum("r,ra,rb->ab", stationary, PA, PB)
        assert _chi2_p_value(observed, expected_stationary) < MIN_P_VALUE


def test_site_pattern_frequencies_match_mixture_likelihood():
    rng = np.random.default_rng(7)
    d, n = 3, 300_000
    tree = TreeHandler.from_newick("((A:0.2,B:0.5)X:0.3,(C:0.1,(D:0.4,E:0.25)Y:0.15)Z:0.35)R;")
    Q_gtr, pi_gtr = _random_gtr(d, 1, rng)
    Q_ntr = _random_ntr(d, 1, rng)
    root_ntr = torch.tensor(rng.dirichlet(2.0 * np.ones(d)))[None]
    blen = torch.tensor(np.asarray(tree.branch_lengths, dtype=np.float64))
    # component 0: reversible with speed 1.3; component 1: non-reversible with its own branch lengths
    P0 = simulation.transition_matrices(1.3 * Q_gtr, blen, pi_gtr)
    own_blen = blen * torch.tensor(rng.uniform(0.5, 2.0, size=(blen.shape[0], 1)))
    P1 = simulation.transition_matrices(Q_ntr, own_blen, reversible=False)
    P = torch.cat([P0, P1], dim=1)                                       # (B, 2, d, d)
    root = torch.cat([pi_gtr, root_ntr], dim=0)                          # (2, d)
    weights = np.array([0.35, 0.65])
    s = simulation.sample_mixture(tree, P, root, weights, n, seed=11, chunk_size=128_000)

    num_leaves = tree.num_leaves
    patterns = np.array(list(itertools.product(range(d), repeat=num_leaves))).T   # (L, d**L)
    leaves = torch.tensor(np.eye(d)[patterns][:, None])                            # (L, 1, d**L, d)
    loglik = model.loglik(leaves, tree, P, torch.log(root)[:, None, :], leaf_names=s.leaf_names)
    probs = np.exp(loglik.numpy())                                                  # (2, d**L)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    codes = _pattern_codes(s.leaf_states, d)
    counts = np.bincount(codes, minlength=d ** num_leaves)
    assert _chi2_p_value(counts, weights @ probs) > MIN_P_VALUE
    for m in range(2):
        per_label = np.bincount(codes[s.labels == m], minlength=d ** num_leaves)
        assert _chi2_p_value(per_label, probs[m]) > MIN_P_VALUE


def test_zero_length_branch_copies_the_parent_state():
    tree = TreeHandler.from_newick("((A:0.0,B:0.4)X:0.2,C:0.3)R;")
    Q, pi = _random_gtr(4, 1, np.random.default_rng(5))
    s = simulation.sample_mixture_ctmc(tree, Q, pi, [1.0], 5000, seed=2, return_internal=True)
    assert s.internal_states.shape == (tree.num_nodes - tree.num_leaves, 5000)
    assert tree.node_names[-1] == "R"
    x = tree.node_names.index("X") - tree.num_leaves
    np.testing.assert_array_equal(s.leaf_states[s.leaf_names.index("A")], s.internal_states[x])
    assert not np.array_equal(s.leaf_states[s.leaf_names.index("B")], s.internal_states[x])


def test_states_with_probability_zero_are_never_sampled():
    tree = TreeHandler.from_newick("(A:0.5,B:1.5)R;")
    rates = np.array([[0, 1, 2, 0], [1, 0, 1, 0], [2, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float64)
    Q = torch.tensor(rates - np.diag(rates.sum(axis=1)))[None]          # no inflow into state 3
    root = torch.tensor([[0.5, 0.3, 0.2, 0.0]], dtype=torch.float64)
    s = simulation.sample_mixture_ctmc(tree, Q, root, [1.0], 50_000, reversible=False, seed=0)
    assert s.leaf_states.max() < 3


def test_inverse_cdf_edge_cases():
    probs = torch.full((10,), 0.1, dtype=torch.float64)
    assert torch.cumsum(probs, 0)[-1] < 1.0                               # float rounding
    cdf = simulation._cdf(probs)
    assert cdf[-1] == 1.0
    u = torch.tensor([0.0, 0.55, 1.0 - 2.0 ** -53], dtype=torch.float64)
    assert simulation._sample_rows(cdf, u).tolist() == [0, 5, 9]
    cdf = simulation._cdf(torch.tensor([[0.0, 0.5, 0.0, 0.5]] * 3, dtype=torch.float64))
    u = torch.tensor([0.0, 0.5, 0.25], dtype=torch.float64)
    assert simulation._sample_rows(cdf, u).tolist() == [1, 3, 1]


def test_same_seed_gives_same_columns():
    tree = TreeHandler.from_newick("((A:0.1,B:0.2)X:0.3,C:0.4)R;")
    Q, pi = _random_gtr(4, 2, np.random.default_rng(0))

    def run(**kwargs):
        return simulation.sample_mixture_ctmc(tree, Q, pi, [0.4, 0.6], 2500, chunk_size=1000, **kwargs)

    a, b, c = run(seed=123), run(seed=123), run(seed=124)
    np.testing.assert_array_equal(a.leaf_states, b.leaf_states)
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.array_equal(a.leaf_states, c.leaf_states)
    generator = torch.Generator().manual_seed(123)
    np.testing.assert_array_equal(run(generator=generator).leaf_states, a.leaf_states)


def test_speed_is_equivalent_to_scaled_branch_lengths():
    tree = TreeHandler.from_newick("((A:0.1,B:0.2)X:0.3,C:0.4)R;")
    Q, pi = _random_gtr(4, 2, np.random.default_rng(4))
    rho = torch.tensor([0.5, 2.0], dtype=torch.float64)
    t = torch.tensor(np.asarray(tree.branch_lengths, dtype=np.float64))
    P_rho = simulation.transition_matrices(rho[:, None, None] * Q, t, pi)
    P_t = simulation.transition_matrices(Q, t * rho[None, :], pi)
    torch.testing.assert_close(P_rho, P_t)
    a = simulation.sample_mixture_ctmc(tree, Q, pi, [0.5, 0.5], 1000, rho=rho, seed=9)
    b = simulation.sample_mixture_ctmc(tree, Q, pi, [0.5, 0.5], 1000, branch_lengths=t * rho[None, :], seed=9)
    np.testing.assert_array_equal(a.leaf_states, b.leaf_states)


def test_invalid_arguments_raise():
    rng = np.random.default_rng(6)
    tree = TreeHandler.from_newick("((A:0.1,B:0.2)X:0.3,C:0.4)R;")
    Q, pi = _random_gtr(4, 2, rng)
    P = simulation.transition_matrices(Q, tree.branch_lengths, pi)
    with pytest.raises(ValueError):
        simulation.sample_mixture(tree, P[:-1], pi, [0.5, 0.5], 10)        # branches
    with pytest.raises(ValueError):
        simulation.sample_mixture(tree, P, pi, [0.2, 0.3, 0.5], 10)        # number of models
    with pytest.raises(ValueError):
        simulation.sample_mixture(tree, P, pi, [-0.5, 1.5], 10)            # negative weight
    with pytest.raises(ValueError):                                         # not reversible
        simulation.transition_matrices(_random_ntr(4, 1, rng), tree.branch_lengths, torch.full((4,), 0.25))
    pi_zero = pi.clone()
    pi_zero[:, 0] = 0.0
    pi_zero /= pi_zero.sum(-1, keepdim=True)
    with pytest.raises(ValueError):                                         # frequency 0
        simulation.transition_matrices(Q, tree.branch_lengths, pi_zero)

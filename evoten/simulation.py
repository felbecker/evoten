"""Simulation of alignment columns under a mixture of substitution models along
a tree (PyTorch).

Each column independently draws a mixture component m from the mixture
weights, a root state from the root distribution of m and then, from the root
to the leaves, the state of every node from the row of its branch's transition
matrix P[branch, m] that belongs to the state of its parent. This is the
generative model of evoten's mixture likelihoods (model.loglik with one model
per mixture component).

Torch-only and, like evoten.expm_gtr_torch, not imported by the evoten package:
``from evoten import simulation``.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from evoten.expm_gtr_torch import expm_gtr, expm_ntr
from evoten.tree_handler import TreeHandler


@dataclass
class MixtureSample:
    """ Simulated alignment columns.

    Attributes:
        leaf_states: Array (num_leaves, n) of leaf states, rows in the order
            of leaf_names.
        labels: Array (n,) with the mixture component of each column.
        leaf_names: Leaf names in TreeHandler order, i.e.
            tree.node_names[:tree.num_leaves] (the preorder of the terminals).
        internal_states: None or array (num_nodes - num_leaves, n) with the
            states of the ancestral nodes in TreeHandler index order (root
            last).
    """
    leaf_states: np.ndarray
    labels: np.ndarray
    leaf_names: list[str]
    internal_states: np.ndarray | None = None


def transition_matrices(
    rate_matrices,
    branch_lengths,
    equilibrium=None,
    *,
    reversible: bool = True,
    dtype: torch.dtype = torch.float64,
    device=None,
    tol: float = 1e-6
) -> torch.Tensor:
    """ Computes the transition matrices P = exp(Q t) of all branches and
    models.

    Args:
        rate_matrices: Rate matrices of shape (M, d, d) with non-negative
            off-diagonal entries and zero row sums, already multiplied by any
            per-model speed.
        branch_lengths: Evolutionary times of shape (B, 1) or (B, M); row b
            belongs to the branch above tree node b (see
            TreeHandler.branch_lengths).
        equilibrium: Stationary distributions of shape (M, d) or (d,).
            Required if reversible.
        reversible: If True, the rate matrices must fulfill detailed balance
            with respect to the strictly positive equilibrium and the
            symmetric eigendecomposition (expm_gtr_torch.expm_gtr) is used,
            otherwise the Pade approximation (expm_gtr_torch.expm_ntr).
        dtype: Floating point type of the computation.
        device: Device of the computation, default that of rate_matrices.
        tol: Relative tolerance of the input checks.

    Returns:
        Row-stochastic matrices of shape (B, M, d, d) with
        P[b, m, i, j] = P(state j at node b | state i at its parent; model m).
    """
    Q = torch.as_tensor(rate_matrices).to(dtype=dtype, device=device)
    t = torch.as_tensor(branch_lengths).to(dtype=dtype, device=Q.device)
    if Q.dim() != 3 or Q.shape[-1] != Q.shape[-2]:
        raise ValueError(
            f"rate_matrices must have shape (M, d, d), got {tuple(Q.shape)}."
        )
    M, d = Q.shape[0], Q.shape[-1]
    if t.dim() != 2 or t.shape[1] not in (1, M):
        raise ValueError(
            f"branch_lengths must have shape (B, 1) or (B, {M}), "
            f"got {tuple(t.shape)}."
        )
    if not torch.isfinite(t).all() or torch.any(t < 0):
        raise ValueError("branch_lengths must be finite and non-negative.")
    _check_rate_matrices(Q, tol)
    if not reversible:
        return expm_ntr(Q, t)
    if equilibrium is None:
        raise ValueError("equilibrium is required for reversible models.")
    pi = torch.as_tensor(equilibrium).to(dtype=dtype, device=Q.device)
    if pi.dim() == 1:
        pi = pi[None]
    if pi.dim() != 2 or pi.shape[-1] != d or pi.shape[0] not in (1, M):
        raise ValueError(
            f"equilibrium must have shape ({M}, {d}) or ({d},), "
            f"got {tuple(pi.shape)}."
        )
    pi = pi.expand(M, d)
    if torch.any(pi <= 1e-12) or torch.any((pi.sum(-1) - 1).abs() > tol):
        raise ValueError(
            "equilibrium must consist of strictly positive distributions; "
            "use reversible=False for states with frequency 0."
        )
    flux = pi[:, :, None] * Q
    scale = flux.abs().amax(dim=(-2, -1), keepdim=True)
    if torch.any((flux - flux.transpose(-1, -2)).abs() > tol * scale):
        raise ValueError(
            "rate_matrices violate detailed balance with respect to "
            "equilibrium; use reversible=False for non-reversible models."
        )
    return expm_gtr(Q, t, pi)


def iter_sample_mixture(
    tree: TreeHandler,
    transition_probs,
    root_probs,
    weights,
    n_columns: int,
    *,
    chunk_size: int = 65536,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    device=None,
    return_internal: bool = False
) -> Iterator[MixtureSample]:
    """ Samples independent alignment columns from a mixture of substitution
    models along a tree, chunk by chunk.

    Each column draws a component m ~ Categorical(weights), a root state from
    root_probs[m] and the state of every other node from the row of
    transition_probs[node, m] that belongs to the state of its parent. Nodes
    are visited in decreasing TreeHandler index, so every parent (larger
    subtree height, larger index) is sampled before its children. Sampling
    inverts float64 cumulative distributions whose last entry is exactly 1,
    so states with probability 0 are never drawn.

    Args:
        tree: TreeHandler (not stratified: one transition matrix per branch).
        transition_probs: Row-stochastic matrices of shape
            (num_nodes-1, M, d, d) or (num_nodes-1, 1, d, d) where row b
            belongs to the branch above node b, e.g. from
            transition_matrices.
        root_probs: Root state distributions of shape (M, d), (1, d) or (d,):
            the equilibrium distributions of reversible models, the root
            priors otherwise.
        weights: Non-negative mixture weights of shape (M,), normalized
            internally.
        n_columns: Total number of columns.
        chunk_size: Maximum number of columns per chunk. Memory use is
            O(chunk_size * (num_nodes + d)).
        generator: Random number generator on the sampling device.
        seed: Seed of a new generator, used if generator is None. Without
            both, torch's global random number generator is used.
        device: Sampling device, default that of transition_probs.
        return_internal: Whether to return the states of the ancestral nodes.

    Yields:
        One MixtureSample per chunk. The columns are reproducible for the
        same seed, chunk_size and device type.
    """
    if n_columns < 1 or chunk_size < 1:
        raise ValueError("n_columns and chunk_size must be positive.")
    P = torch.as_tensor(transition_probs)
    device = P.device if device is None else torch.device(device)
    P = P.to(dtype=torch.float64, device=device)
    if P.dim() != 4 or P.shape[-1] != P.shape[-2]:
        raise ValueError(
            "transition_probs must have shape (num_nodes-1, M, d, d), "
            f"got {tuple(P.shape)}."
        )
    num_branches, M_P, d = P.shape[0], P.shape[1], P.shape[-1]
    if tree.height < 1:
        raise ValueError("The tree must have at least one internal node.")
    if num_branches != tree.num_nodes - 1:
        raise ValueError(
            f"transition_probs has {num_branches} branches, "
            f"the tree has {tree.num_nodes - 1}."
        )
    parents = [int(p) for p in tree.parent_indices]
    if any(p <= c for c, p in enumerate(parents)):
        raise ValueError("Every parent must have a larger node index than "
                         "its children.")
    root = torch.as_tensor(root_probs).to(dtype=torch.float64, device=device)
    if root.dim() == 1:
        root = root[None]
    w = torch.as_tensor(weights).to(dtype=torch.float64, device=device)
    if (w.dim() != 1 or not torch.isfinite(w).all() or torch.any(w < 0)
            or w.sum() <= 0):
        raise ValueError("weights must be a vector of finite, non-negative "
                         "values that are not all 0.")
    M = w.shape[0]
    if root.dim() != 2 or root.shape[-1] != d:
        raise ValueError(
            f"root_probs must have shape ({M}, {d}), (1, {d}) or ({d},), "
            f"got {tuple(root.shape)}."
        )
    for name, models in (("transition_probs", M_P),
                         ("root_probs", root.shape[0])):
        if models not in (1, M):
            raise ValueError(f"{name} has {models} models, weights has {M}.")
    _check_distributions(P, "transition_probs")
    _check_distributions(root, "root_probs")

    cdf_P = _cdf(P).reshape(num_branches, M_P * d, d)
    cdf_root = _cdf(root)
    cdf_w = _cdf(w / w.sum())
    if generator is None and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    num_nodes, num_leaves = tree.num_nodes, tree.num_leaves
    leaf_names = list(tree.node_names[:num_leaves])
    state_dtype = torch.int16 if d <= 2**15 else torch.int32
    np_state_dtype = np.uint8 if d <= 256 else np.int32
    np_label_dtype = np.uint8 if M <= 256 else np.int32

    for start in range(0, n_columns, chunk_size):
        C = min(chunk_size, n_columns - start)
        z = _sample_rows(cdf_w, _uniform(C, generator, device))
        root_cdf = cdf_root[z] if root.shape[0] > 1 else cdf_root[0]
        X = torch.empty((num_nodes, C), dtype=state_dtype, device=device)
        X[-1] = _sample_rows(root_cdf, _uniform(C, generator, device))
        offset = z * d if M_P > 1 else None
        for node in range(num_nodes - 2, -1, -1):
            row_index = X[parents[node]].long()
            if offset is not None:
                row_index = row_index + offset
            rows = cdf_P[node].index_select(0, row_index)
            X[node] = _sample_rows(rows, _uniform(C, generator, device))
        internal = (X[num_leaves:].cpu().numpy().astype(np_state_dtype)
                    if return_internal else None)
        yield MixtureSample(
            leaf_states=X[:num_leaves].cpu().numpy().astype(np_state_dtype),
            labels=z.cpu().numpy().astype(np_label_dtype),
            leaf_names=leaf_names,
            internal_states=internal,
        )


def sample_mixture(
    tree: TreeHandler,
    transition_probs,
    root_probs,
    weights,
    n_columns: int,
    **kwargs
) -> MixtureSample:
    """ Like iter_sample_mixture, but returns all columns as one
    MixtureSample. """
    chunks = list(iter_sample_mixture(
        tree, transition_probs, root_probs, weights, n_columns, **kwargs
    ))
    internal = None
    if chunks[0].internal_states is not None:
        internal = np.concatenate([c.internal_states for c in chunks], axis=1)
    return MixtureSample(
        leaf_states=np.concatenate([c.leaf_states for c in chunks], axis=1),
        labels=np.concatenate([c.labels for c in chunks]),
        leaf_names=chunks[0].leaf_names,
        internal_states=internal,
    )


def sample_mixture_ctmc(
    tree: TreeHandler,
    rate_matrices,
    equilibrium,
    weights,
    n_columns: int,
    *,
    rho=None,
    branch_lengths=None,
    reversible: bool = True,
    **kwargs
) -> MixtureSample:
    """ Samples columns from a mixture of CTMCs in which component m has the
    rate matrix rho[m] * rate_matrices[m] and the root distribution
    equilibrium[m].

    Args:
        tree: TreeHandler.
        rate_matrices: Rate matrices of shape (M, d, d), e.g. from
            Backend.make_rate_matrix.
        equilibrium: Root distributions of shape (M, d) or (d,): the
            stationary distributions if reversible, else the root priors.
        weights: Mixture weights of shape (M,).
        n_columns: Number of columns.
        rho: Optional per-component speeds of shape (M,).
        branch_lengths: Branch lengths of shape (num_nodes-1, 1) or
            (num_nodes-1, M), default tree.branch_lengths.
        reversible: See transition_matrices.
        kwargs: Passed to iter_sample_mixture (chunk_size, generator, seed,
            device, return_internal).
    """
    Q = torch.as_tensor(rate_matrices).to(
        dtype=torch.float64, device=kwargs.get("device")
    )
    if rho is not None:
        rho = torch.as_tensor(rho).to(dtype=torch.float64, device=Q.device)
        Q = Q * rho[:, None, None]
    if branch_lengths is None:
        branch_lengths = tree.branch_lengths
    if isinstance(branch_lengths, torch.Tensor):
        branch_lengths = branch_lengths.detach()
    P = transition_matrices(
        Q, branch_lengths, equilibrium if reversible else None,
        reversible=reversible, device=Q.device
    )
    return sample_mixture(tree, P, equilibrium, weights, n_columns, **kwargs)


def _check_rate_matrices(Q, tol):
    if not torch.isfinite(Q).all():
        raise ValueError("rate_matrices contain non-finite values.")
    eye = torch.eye(Q.shape[-1], dtype=torch.bool, device=Q.device)
    scale = Q.abs().amax(dim=(-2, -1), keepdim=True)
    if torch.any(Q.masked_fill(eye, 0.0) < -tol * scale):
        raise ValueError("rate_matrices have negative off-diagonal entries.")
    if torch.any(Q.sum(-1).abs() > tol * scale[..., 0]):
        raise ValueError("The rows of rate_matrices must sum to 0.")


def _check_distributions(probs, name, tol=1e-6):
    if not torch.isfinite(probs).all() or torch.any(probs < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    deviation = float((probs.sum(-1) - 1).abs().max())
    if deviation > tol:
        raise ValueError(f"The rows of {name} must sum to 1 "
                         f"(maximal deviation {deviation:.3g}).")


def _cdf(probs):
    """ Cumulative distributions of the last axis, divided by their last entry
    so that it is exactly 1. """
    cdf = torch.cumsum(probs, dim=-1)
    return cdf / cdf[..., -1:]


def _uniform(n, generator, device):
    return torch.rand(n, dtype=torch.float64, device=device,
                      generator=generator)


def _sample_rows(cdf, u):
    """ Inverse transform sampling: for every u in [0, 1) the first index
    whose cumulative probability exceeds u. cdf has shape (K,) or
    (len(u), K). """
    if cdf.dim() == 1:
        idx = torch.searchsorted(cdf, u, right=True)
    else:
        idx = torch.searchsorted(cdf, u[:, None], right=True)[:, 0]
    return idx.clamp_(max=cdf.shape[-1] - 1)

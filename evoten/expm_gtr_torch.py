"""PyTorch counterpart of :mod:`evoten.expm_gtr`.

A direct translation, function for function and tolerance for tolerance, so
that the two backends compute the same matrix exponential for reversible rate
matrices. Kept in a separate module rather than behind the :class:`Backend`
facade because callers use the three-step form -- precompute the
eigendecomposition once, then exponentiate it for many branch lengths -- which
the facade does not expose.
"""

import warnings
from typing import NamedTuple

import torch


class GTRDecomp(NamedTuple):
    """Holds precomputed eigendecomposition for GTR matrix exponentiation."""

    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    sqrt_pi: torch.Tensor
    inv_sqrt_pi: torch.Tensor


def _exp_clip_min(dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return -745.0
    return -87.0


def _eig_positive_tol(dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return 1e-12
    return 1e-6


def _t_zero_tol(dtype: torch.dtype) -> float:
    if dtype == torch.float64:
        return 1e-12
    return 1e-6


#: Set once a CUDA ``eigh`` has failed, so later calls skip straight to the CPU
#: instead of paying a failed cuSOLVER attempt per forward pass.
_CUSOLVER_FAILED = False


def _eigh(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh``, falling back to the CPU when cuSOLVER fails.

    cuSOLVER allocates its handle outside torch's caching allocator, so
    ``cusolverDnCreate`` can fail under device memory pressure even though the
    matrices themselves would fit easily. They are small enough here that the
    CPU is a cheap fallback.
    """
    global _CUSOLVER_FAILED
    if S.device.type == "cuda" and not _CUSOLVER_FAILED:
        try:
            return torch.linalg.eigh(S)
        except RuntimeError as e:
            _CUSOLVER_FAILED = True
            warnings.warn(f"CUDA eigh failed ({e}); falling back to the CPU.")
    eigvals, eigvecs = torch.linalg.eigh(S.cpu())
    return eigvals.to(S.device), eigvecs.to(S.device)


def precompute_gtr(
    Q: torch.Tensor, pi: torch.Tensor, epsilon: float = 1e-16
) -> GTRDecomp:
    """Precompute eigendecomposition for GTR matrix exponentiation. Supports
    broadcasting over leading dimensions of Q and pi.

    Args:
        Q: Tensor of shape (..., d, d), batch of square rate matrices.
        pi: Tensor of shape (..., d) or (d,), stationary distributions.
        epsilon: Small constant for numerical stability.

    Returns:
        GTRDecomp object.
    """
    d = Q.shape[-1]
    pi = pi.expand(*Q.shape[:-2], d)

    pi_safe = pi.clamp_min(epsilon)
    pi_safe = pi_safe / pi_safe.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    sqrt_pi = torch.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / sqrt_pi.clamp_min(epsilon)

    # Symmetric matrix
    S = Q * inv_sqrt_pi.unsqueeze(-2)
    S = S * sqrt_pi.unsqueeze(-1)
    S = 0.5 * (S + S.transpose(-2, -1))

    eigvals, eigvecs = _eigh(S)

    return GTRDecomp(eigvals, eigvecs, sqrt_pi, inv_sqrt_pi)


def expm_gtr_from_decomp(
    decomp: GTRDecomp, t: torch.Tensor
) -> torch.Tensor:
    """Compute matrix exponential for GTR using precomputed eigendecomposition.
    Supports broadcasting over leading dimensions of t.

    Args:
        decomp: GTRDecomp object with precomputed eigendecomposition.
        t: Tensor of shape (...,), evolutionary times.

    Returns:
        Tensor of shape (..., d, d) equal to expm(Q) computed via symmetric
        eigendecomposition.
    """
    t = torch.as_tensor(
        t, dtype=decomp.eigvals.dtype, device=decomp.eigvals.device
    )
    t = t.clamp_min(0.0)

    # Broadcast t if necessary
    batch_shape = torch.broadcast_shapes(decomp.eigvals.shape[:-1], t.shape)
    t_b = t.expand(batch_shape)
    t_b_exp = t_b.unsqueeze(-1)

    # For a proper rate matrix, eigvals should be <= 0 up to floating-point
    # noise. Bound positives away to avoid explosive exponentials.
    eig_tol = _eig_positive_tol(decomp.eigvals.dtype)
    safe_eigvals = torch.where(
        decomp.eigvals > eig_tol,
        torch.zeros_like(decomp.eigvals),
        decomp.eigvals,
    )

    # exp(t * D) with bounded exponent range
    exp_arg = safe_eigvals * t_b_exp
    exp_arg = exp_arg.clamp(_exp_clip_min(exp_arg.dtype), 0.0)
    exp_eigs = torch.exp(exp_arg)

    # U * exp(D)
    U_times_exp = decomp.eigvecs * exp_eigs.unsqueeze(-2)

    # Reconstruct M = U exp(D) U^T
    M = torch.matmul(U_times_exp, decomp.eigvecs.transpose(-2, -1))

    # Transform back: Pi^{-1/2} M Pi^{1/2}
    inv_sqrt_pi = decomp.inv_sqrt_pi.unsqueeze(-1)
    sqrt_pi = decomp.sqrt_pi.unsqueeze(-2)
    expQ = inv_sqrt_pi * M * sqrt_pi

    # Reproject numerically to a row-stochastic matrix.
    expQ = expQ.clamp_min(0.0)
    expQ = expQ / expQ.sum(dim=-1, keepdim=True).clamp_min(1e-16)

    # Guarantee identity for near-zero branch lengths.
    eye = torch.eye(
        expQ.shape[-1], dtype=expQ.dtype, device=expQ.device
    ).expand_as(expQ)
    near_zero_t = t_b.abs() <= _t_zero_tol(expQ.dtype)
    near_zero_t = near_zero_t.unsqueeze(-1).unsqueeze(-1)
    expQ = torch.where(near_zero_t, eye, expQ)

    return expQ


def expm_gtr(
    Q: torch.Tensor,
    t: torch.Tensor,
    pi: torch.Tensor,
    epsilon: float = 1e-16,
) -> torch.Tensor:
    """Compute matrix exponential for *reversible* rate matrices Q.

    Assumes detailed balance: pi_i * Q_{i,j} = pi_j * Q_{j,i}, pi_i > 0.

    Args:
        Q: Tensor of shape (..., d, d), batch of square rate matrices.
        t: Scalar tensor or 1-D tensor of shape (...,), evolutionary times.
        pi: Tensor of shape (..., d) or (d,), stationary distributions.
        epsilon: Small constant for numerical stability.

    Returns:
        Tensor of shape (..., d, d) equal to expm(Q) computed via symmetric
        eigendecomposition.
    """
    decomp = precompute_gtr(Q, pi, epsilon=epsilon)
    return expm_gtr_from_decomp(decomp, t)

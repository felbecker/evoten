"""Matrix exponentials for the PyTorch backend.

Torch twins of :mod:`evoten.expm_gtr` (TensorFlow): the same symmetric
eigendecomposition route for time-reversible (GTR) rate matrices, the same
row-stochastic re-projection and identity-at-``t=0`` guards, and a Pade route
(``torch.linalg.matrix_exp``) for general (non-reversible, NTR) rate matrices.

The GTR route does NOT back-propagate through ``torch.linalg.eigh``: its
backward divides by eigenvalue gaps and returns inf/NaN for (near-)degenerate
spectra, and a Jukes-Cantor matrix -- the usual initialisation -- has one
eigenvalue of multiplicity ``d-1``. Instead :class:`_SymExpm` differentiates
the symmetric matrix exponential with the Daleckii-Krein (divided differences)
formula, which is well defined for repeated eigenvalues. The eigendecomposition
is computed once per rate matrix and reused for every evolutionary time ``t``
it is broadcast against (branches x models), as in the TF implementation.
"""
import torch


def _exp_clip_min(dtype: torch.dtype) -> float:
    return -745.0 if dtype == torch.float64 else -87.0


def _eig_positive_tol(dtype: torch.dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-6


def _t_zero_tol(dtype: torch.dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-6


def _divided_difference_tol(dtype: torch.dtype) -> float:
    return 1e-10 if dtype == torch.float64 else 1e-5


class _SymExpm(torch.autograd.Function):
    """``exp(S * t)`` for a symmetric ``S`` via eigendecomposition.

    ``S`` has shape ``(*sb, d, d)``, ``t`` shape ``(*tb)``; the result has
    shape ``(*bb, d, d)`` with ``bb = broadcast(sb, tb)``. Eigenvalues above
    ``_eig_positive_tol`` are treated as zero (a proper rate matrix has none)
    and the exponent is clipped from below, exactly as in the TF version.

    Backward: with ``S = U diag(lam) U^T`` and ``g_k = exp(lam_k t)`` the
    derivative of ``U diag(g) U^T`` along a symmetric perturbation ``dS`` is
    ``U (F o (U^T dS U)) U^T`` with the divided differences
    ``F_ij = (g_i - g_j) / (lam_i - lam_j)`` and ``F_ii = t g_i``; near-equal
    eigenvalues use the derivative instead of the quotient. The cotangent
    w.r.t. ``t`` is ``sum_k lam_k g_k (U^T G U)_kk`` (``d/dt exp(St) = S exp(St)``).
    """

    @staticmethod
    def forward(ctx, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(S)                # (*sb, d), (*sb, d, d)
        dtype = S.dtype
        lam = torch.where(eigvals > _eig_positive_tol(dtype),
                          torch.zeros_like(eigvals), eigvals)
        bb = torch.broadcast_shapes(lam.shape[:-1], t.shape)
        t_b = t.broadcast_to(bb).unsqueeze(-1)                 # (*bb, 1)
        arg = lam * t_b                                        # (*bb, d)
        clip = _exp_clip_min(dtype)
        clipped = arg < clip
        g = torch.exp(arg.clamp(min=clip, max=0.0))            # (*bb, d)
        M = (eigvecs * g.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        ctx.save_for_backward(lam, eigvecs, t_b, g, clipped)
        ctx.s_shape = S.shape
        ctx.t_shape = t.shape
        return M

    @staticmethod
    def backward(ctx, G: torch.Tensor):
        lam, U, t_b, g, clipped = ctx.saved_tensors
        Ut = U.transpose(-1, -2)
        W = Ut @ G @ U                                         # (*bb, d, d) cotangent in the eigenbasis
        # d g_k / d lam_k = t g_k, zero where the exponent was clipped.
        deriv = torch.where(clipped, torch.zeros_like(g), t_b * g)      # (*bb, d)
        dl = lam.unsqueeze(-1) - lam.unsqueeze(-2)             # (*sb, d, d)
        dg = g.unsqueeze(-1) - g.unsqueeze(-2)                 # (*bb, d, d)
        small = dl.abs() < _divided_difference_tol(g.dtype)
        small = small | torch.zeros_like(dg, dtype=torch.bool)  # broadcast to (*bb, d, d)
        safe_dl = torch.where(small, torch.ones_like(dg), dl)
        F = torch.where(small,
                        0.5 * (deriv.unsqueeze(-1) + deriv.unsqueeze(-2)),
                        dg / safe_dl)
        gS = U @ (F * W) @ Ut                                  # (*bb, d, d)
        gS = 0.5 * (gS + gS.transpose(-1, -2))                 # S is symmetric
        grad_S = gS.sum_to_size(ctx.s_shape) if ctx.needs_input_grad[0] else None
        grad_t = None
        if ctx.needs_input_grad[1]:
            diagW = torch.diagonal(W, dim1=-2, dim2=-1)        # (*bb, d)
            dt = torch.where(clipped, torch.zeros_like(g), lam * g) * diagW
            grad_t = dt.sum(-1).sum_to_size(ctx.t_shape)
        return grad_S, grad_t


def _finish(P: torch.Tensor, t_b: torch.Tensor) -> torch.Tensor:
    """Row-stochastic re-projection and identity for near-zero branch lengths
    (shared by the GTR and NTR routes; ``t_b`` is broadcast to ``P.shape[:-2]``)."""
    P = P.clamp(min=0.0)
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-16)
    d = P.shape[-1]
    eye = torch.eye(d, dtype=P.dtype, device=P.device)
    near_zero_t = (t_b.abs() <= _t_zero_tol(P.dtype))[..., None, None]
    return torch.where(near_zero_t, eye, P)


def expm_gtr(Q, t, pi, epsilon: float = 1e-16) -> torch.Tensor:
    """``exp(Q * t)`` for *reversible* rate matrices ``Q`` (detailed balance
    ``pi_i Q_ij = pi_j Q_ji``, ``pi_i > 0``) via the symmetric eigendecomposition
    of ``Pi^{1/2} Q Pi^{-1/2}``.

    Args:
        Q: ``(..., d, d)`` rate matrices.
        t: evolutionary times, broadcastable against ``Q``'s leading dims
            (e.g. ``Q (M, d, d)`` and ``t (B, 1)`` give ``(B, M, d, d)``).
        pi: ``(..., d)`` or ``(d,)`` stationary distributions.
        epsilon: numerical floor for ``pi``.

    Returns:
        ``(broadcast(Q.shape[:-2], t.shape), d, d)`` row-stochastic matrices.
    """
    Q = torch.as_tensor(Q)
    pi = torch.as_tensor(pi, dtype=Q.dtype, device=Q.device)
    t = torch.as_tensor(t, dtype=Q.dtype, device=Q.device).clamp(min=0.0)
    d = Q.shape[-1]
    pi = torch.broadcast_to(pi, (*Q.shape[:-2], d))
    pi_safe = pi.clamp(min=epsilon)
    pi_safe = pi_safe / pi_safe.sum(dim=-1, keepdim=True).clamp(min=epsilon)
    sqrt_pi = torch.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / sqrt_pi.clamp(min=epsilon)
    S = Q * inv_sqrt_pi.unsqueeze(-2) * sqrt_pi.unsqueeze(-1)  # S_ij = sqrt(pi_i) Q_ij / sqrt(pi_j)
    S = 0.5 * (S + S.transpose(-1, -2))
    M = _SymExpm.apply(S, t)                                    # (*bb, d, d)
    expQ = inv_sqrt_pi.unsqueeze(-1) * M * sqrt_pi.unsqueeze(-2)
    t_b = torch.broadcast_to(t, expQ.shape[:-2])
    return _finish(expQ, t_b)


def expm_ntr(Q, t) -> torch.Tensor:
    """``exp(Q * t)`` for a general rate matrix via ``torch.linalg.matrix_exp``
    (Pade approximation with exact autograd). Same broadcasting as :func:`expm_gtr`."""
    Q = torch.as_tensor(Q)
    t = torch.as_tensor(t, dtype=Q.dtype, device=Q.device).clamp(min=0.0)
    P = torch.linalg.matrix_exp(Q * t[..., None, None])
    t_b = torch.broadcast_to(t, P.shape[:-2])
    return _finish(P, t_b)

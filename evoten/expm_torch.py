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

``torch.linalg.eigh`` raises when its solver does not converge, and trained
components drift towards Jukes-Cantor, whose (d-1)-fold eigenvalue float32
cuSOLVER occasionally fails to decompose; :func:`_eigh_robust` retries such a
decomposition in float64 instead of aborting the run.
"""
import warnings

import torch


def _exp_clip_min(dtype: torch.dtype) -> float:
    return -745.0 if dtype == torch.float64 else -87.0


def _eig_positive_tol(dtype: torch.dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-6


def _t_zero_tol(dtype: torch.dtype) -> float:
    return 1e-12 if dtype == torch.float64 else 1e-6


def _divided_difference_tol(dtype: torch.dtype) -> float:
    return 1e-10 if dtype == torch.float64 else 1e-5


#: How often the fast ``torch.linalg.eigh`` failed and was redone in float64 (diagnostic).
eigh_fallbacks = 0


def _eigh_robust(S: torch.Tensor):
    """``torch.linalg.eigh(S)`` with a float64 retry when the solver does not converge.

    Trained GTR components drift towards Jukes-Cantor: in Agrippa sweep 5, 6 of the 8
    codon-stack components (d=64) ended with an eigenvalue of multiplicity 63 and gaps
    down to 1e-10, and float32 cuSOLVER eventually raised "failed to converge ... too
    many repeated eigenvalues", killing a run in its last epoch. The matrix exponential
    is perfectly well conditioned there -- only the eigenbasis of the cluster is
    ill-determined, and any orthonormal one gives the same ``U diag(g) U^T`` -- so the
    failed call is redone in float64 on the same device and, should that fail too, with
    LAPACK on the CPU; the result is cast back to ``S``'s dtype and device.

    The fast path is unchanged, so results stay bit-identical whenever it succeeds, and
    ``torch.linalg.eigh`` already synchronises with the device to check its status, so
    the retry costs nothing until it is needed. Non-finite input means diverged rate
    parameters, not a solver problem, and is reported as such.
    """
    global eigh_fallbacks
    try:
        return torch.linalg.eigh(S)
    except torch.linalg.LinAlgError as err:
        failure = err
    finite = torch.isfinite(S).flatten(-2).all(-1)            # (*sb,)
    if not bool(finite.all()):
        bad = (~finite).nonzero().tolist()
        raise torch.linalg.LinAlgError(
            f"expm_gtr: non-finite symmetrised rate matrix at batch element(s) {bad}; "
            f"the rate parameters have diverged, this is not an eigensolver failure"
        ) from failure
    eigh_fallbacks += 1
    if eigh_fallbacks == 1 or eigh_fallbacks % 1000 == 0:     # visible, never floods a log
        warnings.warn(
            f"expm_gtr: torch.linalg.eigh ({S.dtype}, {S.device}) did not converge on a "
            f"near-degenerate spectrum; recomputed in float64 (fallback #{eigh_fallbacks}): "
            f"{failure}", RuntimeWarning, stacklevel=3)
    try:
        lam, U = torch.linalg.eigh(S.to(torch.float64))
    except torch.linalg.LinAlgError:
        if S.device.type == "cpu":
            raise
        lam, U = torch.linalg.eigh(S.detach().to("cpu", torch.float64))
    return lam.to(S.device, S.dtype), U.to(S.device, S.dtype)


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
        eigvals, eigvecs = _eigh_robust(S)                     # (*sb, d), (*sb, d, d)
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

import tensorflow as tf
from typing import NamedTuple


class GTRDecomp(NamedTuple):
    """Holds precomputed eigendecomposition for GTR matrix exponentiation."""
    eigvals: tf.Tensor
    eigvecs: tf.Tensor
    sqrt_pi: tf.Tensor
    inv_sqrt_pi: tf.Tensor


def _exp_clip_min(dtype: tf.dtypes.DType) -> float:
    if dtype == tf.float64:
        return -745.0
    return -87.0


def _eig_positive_tol(dtype: tf.dtypes.DType) -> float:
    if dtype == tf.float64:
        return 1e-12
    return 1e-6


def _t_zero_tol(dtype: tf.dtypes.DType) -> float:
    if dtype == tf.float64:
        return 1e-12
    return 1e-6

def precompute_gtr(
    Q: tf.Tensor, pi: tf.Tensor, epsilon: float = 1e-16
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

    # Broadcast pi
    d = tf.shape(Q)[-1]
    pi = tf.broadcast_to(pi, tf.concat([tf.shape(Q)[:-2], [d]], axis=0))

    eps = tf.cast(epsilon, Q.dtype)
    pi_safe = tf.maximum(pi, eps)
    pi_safe /= tf.maximum(tf.reduce_sum(pi_safe, axis=-1, keepdims=True), eps)
    sqrt_pi = tf.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / tf.maximum(sqrt_pi, eps)

    # Symmetric matrix
    S = Q * tf.expand_dims(inv_sqrt_pi, -2)
    S = S * tf.expand_dims(sqrt_pi, -1)
    S = 0.5 * (S + tf.linalg.matrix_transpose(S))

    eigvals, eigvecs = tf.linalg.eigh(S)

    return GTRDecomp(eigvals, eigvecs, sqrt_pi, inv_sqrt_pi)

def expm_gtr_from_decomp(decomp: GTRDecomp, t: tf.Tensor) -> tf.Tensor:
    """Compute matrix exponential for GTR using precomputed eigendecomposition.
    Supports broadcasting over leading dimensions of t.

    Args:
        decomp: GTRDecomp object with precomputed eigendecomposition.
        t: Tensor of shape (...,), evolutionary times.

    Returns:
        Tensor of shape (..., d, d) equal to expm(Q) computed via symmetric
        eigendecomposition.
    """

    t = tf.convert_to_tensor(t, dtype=decomp.eigvals.dtype)
    t = tf.maximum(t, tf.cast(0.0, t.dtype))

    # Broadcast t if necessary
    t_shape = tf.shape(t)
    eig_shape = tf.shape(decomp.eigvals)[:-1]
    batch_shape = tf.broadcast_dynamic_shape(eig_shape, t_shape)
    t_b = tf.broadcast_to(t, batch_shape)
    t_b_exp = tf.expand_dims(t_b, axis=-1)

    # For a proper rate matrix, eigvals should be <= 0 up to floating-point
    # noise. Bound positives away to avoid explosive exponentials.
    eig_tol = tf.cast(_eig_positive_tol(decomp.eigvals.dtype), decomp.eigvals.dtype)
    safe_eigvals = tf.where(
        decomp.eigvals > eig_tol,
        tf.cast(0.0, decomp.eigvals.dtype),
        decomp.eigvals,
    )

    # exp(t * D) with bounded exponent range
    exp_arg = safe_eigvals * t_b_exp
    exp_arg = tf.clip_by_value(
        exp_arg,
        tf.cast(_exp_clip_min(exp_arg.dtype), exp_arg.dtype),
        tf.cast(0.0, exp_arg.dtype),
    )
    exp_eigs = tf.exp(exp_arg)

    # U * exp(D)
    U_times_exp = decomp.eigvecs * tf.expand_dims(exp_eigs, axis=-2)

    # Reconstruct M = U exp(D) U^T
    M = tf.matmul(U_times_exp, decomp.eigvecs, transpose_b=True)

    # Transform back: Pi^{-1/2} M Pi^{1/2}
    inv_sqrt_pi = tf.expand_dims(decomp.inv_sqrt_pi, axis=-1)
    sqrt_pi = tf.expand_dims(decomp.sqrt_pi, axis=-2)
    expQ = inv_sqrt_pi * M * sqrt_pi

    # Reproject numerically to a row-stochastic matrix.
    expQ = tf.maximum(expQ, tf.cast(0.0, expQ.dtype))
    row_sums = tf.reduce_sum(expQ, axis=-1, keepdims=True)
    row_sums = tf.maximum(row_sums, tf.cast(1e-16, expQ.dtype))
    expQ = expQ / row_sums

    # Guarantee identity for near-zero branch lengths.
    eye = tf.eye(tf.shape(expQ)[-1], batch_shape=tf.shape(expQ)[:-2], dtype=expQ.dtype)
    zero_tol = tf.cast(_t_zero_tol(expQ.dtype), expQ.dtype)
    near_zero_t = tf.abs(t_b) <= zero_tol
    near_zero_t = tf.expand_dims(tf.expand_dims(near_zero_t, -1), -1)
    expQ = tf.where(near_zero_t, eye, expQ)

    return expQ

def expm_gtr(
    Q: tf.Tensor, t: tf.Tensor, pi: tf.Tensor, epsilon: float = 1e-16
) -> tf.Tensor:
    """Compute matrix exponential for *reversible* rate matrices Q using TF.

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
    M = expm_gtr_from_decomp(decomp, t)
    return M

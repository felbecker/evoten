import tensorflow as tf
from typing import NamedTuple


class GTRDecomp(NamedTuple):
    """Holds precomputed eigendecomposition for GTR matrix exponentiation."""
    eigvals: tf.Tensor
    eigvecs: tf.Tensor
    sqrt_pi: tf.Tensor
    inv_sqrt_pi: tf.Tensor

def precompute_gtr(Q: tf.Tensor, pi: tf.Tensor) -> GTRDecomp:
    """Precompute eigendecomposition for GTR matrix exponentiation. Supports
    broadcasting over leading dimensions of Q and pi.

    Args:
        Q: Tensor of shape (..., d, d), batch of square rate matrices.
        pi: Tensor of shape (..., d) or (d,), stationary distributions.

    Returns:
        GTRDecomp object.
    """

    # Broadcast pi
    d = tf.shape(Q)[-1]
    pi = tf.broadcast_to(pi, tf.concat([tf.shape(Q)[:-2], [d]], axis=0))

    sqrt_pi = tf.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    # Symmetric matrix
    S = Q * tf.expand_dims(inv_sqrt_pi, -2)
    S = S * tf.expand_dims(sqrt_pi, -1)

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

    # Broadcast t if necessary
    t_shape = tf.shape(t)
    eig_shape = tf.shape(decomp.eigvals)[:-1]
    batch_shape = tf.broadcast_dynamic_shape(eig_shape, t_shape)
    t_b = tf.broadcast_to(t, batch_shape)
    t_b_exp = tf.expand_dims(t_b, axis=-1)

    # exp(t * D)
    exp_eigs = tf.exp(decomp.eigvals * t_b_exp)

    # U * exp(D)
    U_times_exp = decomp.eigvecs * tf.expand_dims(exp_eigs, axis=-2)

    # Reconstruct M = U exp(D) U^T
    M = tf.matmul(U_times_exp, decomp.eigvecs, transpose_b=True)

    # Transform back: Pi^{-1/2} M Pi^{1/2}
    inv_sqrt_pi = tf.expand_dims(decomp.inv_sqrt_pi, axis=-1)
    sqrt_pi = tf.expand_dims(decomp.sqrt_pi, axis=-2)
    expQ = inv_sqrt_pi * M * sqrt_pi
    return expQ

def expm_gtr(Q: tf.Tensor, t: tf.Tensor, pi: tf.Tensor) -> tf.Tensor:
    """Compute matrix exponential for *reversible* rate matrices Q using TF.

    Assumes detailed balance: pi_i * Q_{i,j} = pi_j * Q_{j,i}, pi_i > 0.

    Args:
        Q: Tensor of shape (..., d, d), batch of square rate matrices.
        t: Scalar tensor or 1-D tensor of shape (...,), evolutionary times.
        pi: Tensor of shape (..., d) or (d,), stationary distributions.

    Returns:
        Tensor of shape (..., d, d) equal to expm(Q) computed via symmetric
        eigendecomposition.
    """
    decomp = precompute_gtr(Q, pi)
    M = expm_gtr_from_decomp(decomp, t)
    return M

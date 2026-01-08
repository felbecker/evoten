import tensorflow as tf
from typing import NamedTuple


class GTRDecomp(NamedTuple):
    eigvals: tf.Tensor
    eigvecs: tf.Tensor
    sqrt_pi: tf.Tensor
    inv_sqrt_pi: tf.Tensor

def precompute_gtr(Q: tf.Tensor, pi: tf.Tensor) -> GTRDecomp:

    Q = tf.convert_to_tensor(Q)
    pi = tf.convert_to_tensor(pi)

    # Broadcast pi
    N = tf.shape(Q)[-1]
    pi = tf.broadcast_to(pi, tf.concat([tf.shape(Q)[:-2], [N]], axis=0))

    sqrt_pi = tf.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    # Symmetric matrix
    S = Q * tf.expand_dims(inv_sqrt_pi, -2)
    S = S * tf.expand_dims(sqrt_pi, -1)

    eigvals, eigvecs = tf.linalg.eigh(S)

    return GTRDecomp(eigvals, eigvecs, sqrt_pi, inv_sqrt_pi)

def expm_gtr_from_decomp(decomp: GTRDecomp, t: tf.Tensor) -> tf.Tensor:
    eigvals = decomp.eigvals
    eigvecs = decomp.eigvecs
    sqrt_pi = decomp.sqrt_pi
    inv_sqrt_pi = decomp.inv_sqrt_pi

    # Broadcast t if necessary
    t_shape = tf.shape(t)
    eig_shape = tf.shape(eigvals)[:-1]
    batch_shape = tf.broadcast_dynamic_shape(eig_shape, t_shape)
    t_b = tf.broadcast_to(t, batch_shape)
    t_b_exp = tf.expand_dims(t_b, axis=-1)

    # exp(t * D)
    exp_eigs = tf.exp(eigvals * t_b_exp)

    # U * exp(D)
    U_times_exp = eigvecs * tf.expand_dims(exp_eigs, axis=-2)

    # Reconstruct M = U exp(D) U^T
    M = tf.matmul(U_times_exp, eigvecs, transpose_b=True)

    # Transform back: Pi^{-1/2} M Pi^{1/2}
    expQ = inv_sqrt_pi[..., tf.newaxis] * M * sqrt_pi[..., tf.newaxis, :]
    return expQ



def expm_gtr(Q: tf.Tensor, t: tf.Tensor, pi: tf.Tensor) -> tf.Tensor:
    """Compute matrix exponential for *reversible* rate matrices Q using TF.

    Assumes detailed balance: pi_i * Q_{i,j} = pi_j * Q_{j,i}, pi_i > 0.

    Args:
        Q: Tensor of shape (..., N, N), batch of square rate matrices.
        t: Scalar tensor or 1-D tensor of shape (...,), evolutionary times.
        pi: Tensor of shape (..., N) or (N,), stationary distributions (positive).

    Returns:
        Tensor of shape (..., N, N) equal to expm(Q) computed via symmetric eigendecomposition.
    """
    decomp = precompute_gtr(Q, pi)
    M = expm_gtr_from_decomp(decomp, t)
    return M


# def expm_gtr_V1(Q: tf.Tensor, t: tf.Tensor, pi: tf.Tensor) -> tf.Tensor:
#     """Compute matrix exponential for *reversible* rate matrices Q using TF.

#     Assumes detailed balance: pi_i * Q_{i,j} = pi_j * Q_{j,i}, pi_i > 0.

#     Note: Works and is almost as precise as tf.linalg.expm, but causes OOM errors
#     much sooner.

#     Args:
#         Q: Tensor of shape (..., N, N), batch of square rate matrices.
#         t: Scalar tensor or 1-D tensor of shape (K,), evolutionary times.
#         pi: Tensor of shape (..., N) or (N,), stationary distributions (positive).

#     Returns:
#         Tensor of shape (..., N, N) equal to expm(Q) computed via symmetric eigendecomposition.
#     """
#     # Validate shapes (to the extent possible in graph)
#     Q = tf.convert_to_tensor(Q)
#     t = tf.convert_to_tensor(t, dtype=Q.dtype)
#     pi = tf.convert_to_tensor(pi, dtype=Q.dtype)

#     Q *= t

#     # Ensure pi is shape (..., N)
#     # Allow pi shape (N,) or batch-shape compatible with Q's leading dims
#     q_shape = tf.shape(Q)
#     N = q_shape[-1]

#     # Broadcast pi to batch shape of Q's leading dims
#     # If pi has shape (N,), expand to (..., N)
#     pi_shape = tf.shape(pi)
#     if pi_shape.shape[0] == 1 and pi_shape[0] == N:
#         # pi is (N,) - expand to (..., N) via tf.broadcast_to
#         batch_shape = q_shape[:-2]
#         pi = tf.broadcast_to(pi, tf.concat([batch_shape, [N]], axis=0))
#     else:
#         # Try to broadcast pi to batch dims; tf will error if incompatible
#         target_shape = tf.concat([q_shape[:-2], [N]], axis=0)
#         pi = tf.broadcast_to(pi, target_shape)

#     # Small positive clamp for numerical stability
#     eps = tf.constant(1e-30, dtype=Q.dtype)
#     pi = tf.maximum(pi, eps)

#     # compute sqrt and inv-sqrt of pi with proper broadcasting for matrix multiplies
#     sqrt_pi = tf.sqrt(pi)                               # shape (..., N)
#     inv_sqrt_pi = 1.0 / sqrt_pi                         # shape (..., N)

#     # Build S = Pi^{1/2} Q Pi^{-1/2}
#     # Equivalent to: S_{ij} = sqrt(pi_i) * Q_{ij} / sqrt(pi_j)
#     # We implement via broadcasting multiplies
#     # shape manipulations: expand dims so that multiplication broadcasts across matrix axes
#     left = tf.expand_dims(sqrt_pi, axis=-1)             # (..., N, 1)
#     right = tf.expand_dims(inv_sqrt_pi, axis=-2)        # (..., 1, N)
#     S = left * Q * right                                # (..., N, N)
#     # S should be symmetric (within numerical precision) if detailed balance holds.

#     # Eigendecompose S (symmetric) -> eigenvalues ( ..., N), eigenvectors (..., N, N)
#     # Use tf.linalg.eigh which is differentiable and XLA-friendly
#     eigvals, eigvecs = tf.linalg.eigh(S)                # eigvecs are orthonormal: U

#     # Compute U * exp(diag(eigvals)) by multiplying columns of eigvecs by exp(eigvals)
#     exp_eig = tf.exp(eigvals)                           # (..., N)
#     # multiply columns: expand dims so exp_eig aligns with columns (axis -1)
#     U_times_exp = eigvecs * tf.expand_dims(exp_eig, axis=-2)  # (..., N, N)

#     # reconstruct M = U * exp(D) * U^T
#     M = tf.matmul(U_times_exp, eigvecs, transpose_b=True)    # (..., N, N)

#     # Now transform back: exp(Q) = Pi^{-1/2} * M * Pi^{1/2}
#     left_back = tf.expand_dims(inv_sqrt_pi, axis=-1)     # (..., N, 1)
#     right_back = tf.expand_dims(sqrt_pi, axis=-2)       # (..., 1, N)
#     expQ = left_back * M * right_back                   # (..., N, N)

#     return expQ
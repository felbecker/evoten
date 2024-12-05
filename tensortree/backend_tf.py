import tensorflow as tf
import numpy as np
from Bio.Phylo import BaseTree
from functools import partial
from tensortree import util




"""Constructs a stack of normalized rate matrices, i.e. 1 time unit = 1 expected mutation per site.
Args:
    exchangeabilities: Symmetric, positive-semidefinite exchangeability matrices with zero diagonal. Shape: (k, d, d) 
    equilibrium: A vector of relative frequencies. Shape: (k, d) 
    epsilon: Small value to avoid division by zero.
    normalized: If True, the rate matrices are normalized.
Returns:
    Normalized rate matrices. Output shape: (k, d, d) 
"""
def make_rate_matrix(exchangeabilities, equilibrium, epsilon=1e-16, normalized=True):
    Q = exchangeabilities *  tf.expand_dims(equilibrium, 1)
    diag = tf.reduce_sum(Q, axis=-1, keepdims=True)
    eye = tf.eye(tf.shape(diag)[1], batch_shape=tf.shape(diag)[:1], dtype=diag.dtype)
    Q -= diag * eye
    # normalize
    if normalized:
        mue = tf.expand_dims(equilibrium, -1) * diag
        mue = tf.reduce_sum(mue, axis=-2, keepdims=True)
        Q /= tf.maximum(mue, epsilon)
    return Q


"""Constructs a probability matrix of mutating or copying one an input to another over a given amount of evolutionary time.
Args:
    rate_matrix: Rate matrix of shape (k, d, d).
    distances: Evolutionary times of shape (n, k) 
Returns:
    Stack of probability matrices of shape (n, k, d, d)
"""
def make_transition_probs(rate_matrix, distances):
    rate_matrix = tf.expand_dims(rate_matrix, 0)
    distances = tf.expand_dims(tf.expand_dims(distances, -1), -1)
    P = tf.linalg.expm(rate_matrix * distances) # P[b,m,i,j] = P(X(tau_b) = j | X(0) = i; model m))
    return P


"""
Converts a kernel of parameters to positive branch lengths.
"""
def make_branch_lengths(kernel):
    return tf.math.softplus(kernel)


"""Constructs a stack of symmetric, positive-semidefinite matrices with zero diagonal from a parameter kernel.
    Note: Uses an overparameterized d x d kernel for speed of computation.
Args:
    kernel: Tensor of shape (k, d, d).
"""
def make_symmetric_pos_semidefinite(kernel):
    R = 0.5 * (kernel + tf.transpose(kernel, [0,2,1])) #make symmetric
    R = tf.math.softplus(R)
    R -= tf.linalg.diag(tf.linalg.diag_part(R)) #zero diagonal
    return R


"""Constructs a stack of equilibrium distributions from a parameter kernel.
"""
def make_equilibrium(kernel):
    return tf.nn.softmax(kernel)


"""
Computes the probabilities after traversing a branch when starting with distributions X.
Args:
    X: tensor with logits of shape (n, k, L, d). 
    branch_probabilities: tensor of shape (n, k, d, d). 
    transposed: If True, a transposed transition matrix (X is present, T is future) will be used.
Returns:
    logits of shape (n, k, L, d)
"""
def traverse_branch(X, branch_probabilities, transposed=False, logarithmic=True):

    if True:
        #fast matmul version, but requires conversion, might be numerically unstable
        if logarithmic:
            X = probs_from_logits(X)
        X = tf.matmul(X, branch_probabilities, transpose_b=not transposed)
        if logarithmic:
            X = logits_from_probs(X)
    else:
        if not logarithmic:
            raise ValueError("Non-logarithmic traversal is not supported.")
        # slower (no matmul) but more stable? 
        X = tf.math.reduce_logsumexp(X[..., tf.newaxis, :] + tf.math.log(branch_probabilities[:,:,tf.newaxis]), axis=-1)
    return X


r"""
Aggregates the partial log-likelihoods of child nodes.
Args:
    X: tensor of shape (n, ...)
    parent_map: A list-like object of shape (n) that contains the index of parent nodes.
    num_ancestral: Total number of ancestral nodes.
Returns:
    tensor of shape (num_ancestral, ...).
"""
def aggregate_children_log_probs(X, parent_map, num_ancestral):
    return tf.math.unsorted_segment_sum(X, parent_map, num_ancestral)

"""
Computes log likelihoods given root logits and equilibrium distributions.
Args:
    root_logits: Logits at the root node of shape (k, L, d)
    equilibrium_logits: Equilibrium distribution logits of shape (k, d)
Returns:
    Log-likelihoods of shape (k, L)
"""
def loglik_from_root_logits(root_logits, equilibrium_logits):
    return tf.math.reduce_logsumexp(root_logits + equilibrium_logits[:,tf.newaxis], axis=-1)


""" Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
"""
def logits_from_probs(probs, log_zero_val=-1e3):
    epsilon = tf.constant(np.finfo(util.default_dtype).tiny, dtype=probs.dtype)
    logits = tf.math.log(tf.maximum(probs, epsilon))
    zero_mask = tf.cast(tf.equal(probs, 0), dtype=logits.dtype)
    logits = (1-zero_mask) * logits + zero_mask * log_zero_val
    return logits


""" Computes element-wise exp
"""
def probs_from_logits(logits):
    return tf.math.exp(logits)


def reorder(tensor, permutation, axis=0):
    return tf.gather(tensor, permutation, axis=axis)


def concat(tensors, axis=0):
    return tf.concat(tensors, axis=axis)


"""Initializes the ancestral logits tensor with zeros."""
def get_ancestral_logits_init_tensor(leaves, models, num_ancestral):
    _, L, _, d = tf.unstack(tf.shape(leaves), 4)
    return tf.zeros((num_ancestral, models, L, d), dtype=leaves.dtype)
import torch
import numpy as np
from Bio.Phylo import BaseTree
from functools import partial



def _ensure_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, list):
        return torch.tensor(x)
    return x

"""Constructs a stack of normalized rate matrices, i.e. 1 time unit = 1 expected mutation per site.
Args:
    exchangeabilities: Symmetric, positive-semidefinite exchangeability matrices with zero diagonal. Shape: (k, d, d) 
    equilibrium_kernel: A vector of relative frequencies. Shape: (k, d) 
    epsilon: Small value to avoid division by zero.
    normalized: If True, normalize the rate matrix by the expected mutation rate.
Returns:
    Normalized rate matrices. Output shape: (k, d, d) 
""" 
def make_rate_matrix(exchangeabilities, equilibrium, epsilon=1e-16, normalized=True):
    exchangeabilities = _ensure_tensor(exchangeabilities)
    equilibrium = _ensure_tensor(equilibrium)
    Q = torch.mul(exchangeabilities, equilibrium[:,None])
    diag = torch.sum(Q, -1, True)
    eye = torch.eye(diag.shape[1], dtype=diag.dtype, device=diag.device)
    eye = eye[None]
    eye = eye.repeat(diag.shape[0], 1, 1)
    Q -= diag * eye
    # normalize
    if normalized:
        mue = equilibrium[..., None] * diag
        mue = torch.sum(mue, dim=-2, keepdim=True)
        Q /= torch.maximum(mue, torch.tensor(epsilon))
    return Q


"""Constructs a probability matrix of mutating or copying one an input to another over a given amount of evolutionary time.
Args:
    rate_matrix: Rate matrix of shape (k, d, d).
    distances: Evolutionary times of shape (n, k) 
Returns:
    Stack of probability matrices of shape (n, k, d, d)
"""
def make_transition_probs(rate_matrix, distances):
    rate_matrix = _ensure_tensor(rate_matrix)
    distances = _ensure_tensor(distances)
    rate_matrix = rate_matrix[None]
    distances = distances[..., None, None]
    P = torch.linalg.matrix_exp(rate_matrix * distances) # P[b,m,i,j] = P(X(tau_b) = j | X(0) = i; model m))
    return P


"""
Converts a kernel of parameters to positive branch lengths.
"""
def make_branch_lengths(kernel):
    kernel = _ensure_tensor(kernel)
    return torch.nn.functional.softplus(kernel)


"""Constructs a stack of symmetric, positive-semidefinite matrices with zero diagonal from a parameter kernel.
    Note: Uses an overparameterized d x d kernel for speed of computation.
Args:
    kernel: Tensor of shape (k, d, d).
"""
def make_symmetric_pos_semidefinite(kernel):
    R = 0.5 * (kernel + kernel.transpose([0,2,1])) #make symmetric
    R = torch.nn.functional.softplus(R)
    R -= torch.diag(torch.diagonal(R)) #zero diagonal
    return R

"""Constructs a stack of equilibrium distributions from a parameter kernel.
"""
def make_equilibrium(kernel):
    return torch.nn.functional.softmax(kernel, dim=-1)


"""
Computes the probabilities after traversing a branch when starting with distributions X.
Args:
    X: tensor with logits of shape (n, k, L, d). 
    branch_probabilities: tensor of shape (n, k, d, d). 
Returns:
    logits of shape (n, k, L, d)
"""
def traverse_branch(X, branch_probabilities):
    X = probs_from_logits(X)
    X = torch.matmul(X, branch_probabilities)
    X = logits_from_probs(X)
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
    parent_map = _ensure_tensor(parent_map).to(X.device)
    #return tf.math.unsorted_segment_sum(X, parent_map, num_ancestral)
    Y = torch.zeros(num_ancestral, *X.shape[1:], dtype=X.dtype, device=X.device)
    I = parent_map[:,None,None,None].expand(X.shape)
    Y = Y.scatter_add(0, I, X)
    return Y

"""
Computes log likelihoods given root logits and equilibrium distributions.
Args:
    root_logits: Logits at the root node of shape (k, L, d)
    equilibrium_logits: Equilibrium distribution logits of shape (k, d)
Returns:
    Log-likelihoods of shape (k, L)
"""
def loglik_from_root_logits(root_logits, equilibrium_logits):
    return torch.logsumexp(root_logits + equilibrium_logits[:,None], dim=-1)


""" Computes element-wise logarithm with output_i=log_zero_val where x_i=0.
"""
def logits_from_probs(probs, log_zero_val=-1e3):
    probs = _ensure_tensor(probs)
    epsilon = torch.tensor(np.finfo(np.float32).tiny)
    logits = torch.log(torch.maximum(probs, epsilon))
    zero_mask = (probs == 0).to(logits.dtype)
    logits = (1-zero_mask) * logits + zero_mask * log_zero_val
    return logits


""" Computes element-wise exp
"""
def probs_from_logits(logits):
    return torch.exp(logits)


def reorder(tensor, permutation, axis=0):
    tensor = _ensure_tensor(tensor)
    permutation = _ensure_tensor(permutation)
    return tensor[permutation]


def concat(tensors, axis=0):
    return torch.cat(tensors, axis=axis)


"""Initializes the ancestral logits tensor with zeros."""
def get_ancestral_logits_init_tensor(leaves, models, num_ancestral):
    return torch.zeros((num_ancestral, models, leaves.shape[-2], leaves.shape[-1]), 
                       dtype=leaves.dtype, device=leaves.device)
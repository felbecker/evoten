import tensorflow as tf
import numpy as np
from Bio.Phylo import BaseTree
from functools import partial
from tensortree import util


# Documentation in the base class. Hover over the method name to see the docstring.
class BackendTF(util.Backend):

    def make_rate_matrix(self, exchangeabilities, equilibrium, epsilon=1e-16, normalized=True):
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


    def make_transition_probs(self, rate_matrix, distances):
        rate_matrix = tf.expand_dims(rate_matrix, 0)
        distances = tf.expand_dims(tf.expand_dims(distances, -1), -1)
        P = tf.linalg.expm(rate_matrix * distances) # P[b,m,i,j] = P(X(tau_b) = j | X(0) = i; model m))
        return P



    def make_branch_lengths(self, kernel):
        return tf.math.softplus(kernel)


    def make_symmetric_pos_semidefinite(self, kernel):
        R = 0.5 * (kernel + tf.transpose(kernel, [0,2,1])) #make symmetric
        R = tf.math.softplus(R)
        R -= tf.linalg.diag(tf.linalg.diag_part(R)) #zero diagonal
        return R


    def make_equilibrium(self, kernel):
        return tf.nn.softmax(kernel)


    def traverse_branch(self, X, branch_probabilities, transposed=False, logarithmic=True):
        if True:
            #fast matmul version, but requires conversion, might be numerically unstable
            if logarithmic:
                X = self.probs_from_logits(X)
            X = tf.matmul(X, branch_probabilities, transpose_b=not transposed)
            if logarithmic:
                X = self.logits_from_probs(X)
        else:
            if not logarithmic:
                raise ValueError("Non-logarithmic traversal is not supported.")
            # slower (no matmul) but more stable? 
            X = tf.math.reduce_logsumexp(X[..., tf.newaxis, :] + tf.math.log(branch_probabilities[:,:,tf.newaxis]), axis=-1)
        return X


    def aggregate_children_log_probs(self, X, parent_map, num_ancestral):
        return tf.math.unsorted_segment_sum(X, parent_map, num_ancestral)


    def loglik_from_root_logits(self, root_logits, equilibrium_logits):
        return tf.math.reduce_logsumexp(root_logits + equilibrium_logits[:,tf.newaxis], axis=-1)
    

    def marginals_from_beliefs(self, beliefs):
        loglik = tf.math.reduce_logsumexp(beliefs[-1:], axis=-1, keepdims=True)
        return beliefs - loglik


    def logits_from_probs(self, probs, log_zero_val=-1e3):
        epsilon = tf.constant(np.finfo(util.default_dtype).tiny, dtype=probs.dtype)
        logits = tf.math.log(tf.maximum(probs, epsilon))
        zero_mask = tf.cast(tf.equal(probs, 0), dtype=logits.dtype)
        logits = (1-zero_mask) * logits + zero_mask * log_zero_val
        return logits


    def probs_from_logits(self, logits):
        return tf.math.exp(logits)


    def reorder(self, tensor, permutation, axis=0):
        return tf.gather(tensor, permutation, axis=axis)


    def concat(self, tensors, axis=0):
        return tf.concat(tensors, axis=axis)


    def make_zeros(self, leaves, models, num_nodes):
        _, _, L, d = tf.unstack(tf.shape(leaves), 4)
        return tf.zeros((num_nodes, models, L, d), dtype=leaves.dtype)
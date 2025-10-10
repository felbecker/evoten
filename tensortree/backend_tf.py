from functools import partial

import numpy as np
import tensorflow as tf
from Bio.Phylo import BaseTree

from tensortree import util


# Documentation in the base class. Hover over the method name to see the
# docstring.
class BackendTF(util.Backend):

    def make_rate_matrix(
            self,
            exchangeabilities,
            equilibrium,
            epsilon=1e-16,
            normalized=True
        ):
        Q = exchangeabilities * tf.expand_dims(equilibrium, -2)
        diag = tf.reduce_sum(Q, axis=-1, keepdims=True)
        eye = tf.eye(tf.shape(diag)[-2], dtype=diag.dtype)
        Q -= diag * eye
        # normalize
        if normalized:
            mue = tf.expand_dims(equilibrium, -1) * diag
            mue = tf.reduce_sum(mue, axis=-2, keepdims=True)
            Q /= tf.maximum(mue, epsilon)
        return Q


    def make_transition_probs(self, rate_matrix, distances):
        distances = tf.expand_dims(tf.expand_dims(distances, -1), -1)
        # P[b,m,i,j] = P(X(tau_b) = j | X(0) = i; model m))
        P = tf.linalg.expm(rate_matrix * distances)
        return P


    def make_branch_lengths(self, kernel):
        return tf.math.softplus(kernel)


    def inverse_softplus(self, branch_lengths):
        epsilon=1e-16
        # Cast to float64 to prevent overflow of large entries
        branch_lengths64 = tf.cast(branch_lengths, tf.float64)
        result = tf.math.log(tf.math.expm1(branch_lengths64) + epsilon)
        # Cast back to the original data type of `features`
        return tf.cast(result, branch_lengths.dtype)


    def make_symmetric_pos_semidefinite(self, kernel):
        kernel_shape = tf.shape(kernel)
        kernel = tf.reshape(kernel, (-1, kernel_shape[-2], kernel_shape[-1]))
        R = 0.5 * (kernel + tf.transpose(kernel, [0,2,1])) #make symmetric
        R = tf.math.softplus(R)
        R -= tf.linalg.diag(tf.linalg.diag_part(R)) #zero diagonal
        R = tf.reshape(R, kernel_shape)
        return R


    def make_equilibrium(self, kernel):
        return tf.nn.softmax(kernel)


    def traverse_branch(
            self,
            X,
            transition_probs,
            transposed=False,
            logarithmic=True
        ):
        # fast matmul version, but requires conversion, might be numerically
        # unstable
        if logarithmic:
            X = self.probs_from_logits(X)

        if transition_probs.shape[-3] == 1:
            # broadcasting  in L is required
            # it is most efficient to let the matmul op handle this
            transition_probs = transition_probs[..., 0, :, :]
            X = tf.matmul(X, transition_probs, transpose_b=not transposed)
        else:
            # the user has provided transition matrices for all positions
            # add a dummy dimension to X for the matmul op
            X = tf.expand_dims(X, axis=-2)
            X = tf.matmul(X, transition_probs, transpose_b=not transposed)
            # strip the dummy dimension
            X = X[..., 0, :]

        if logarithmic:
            X = self.logits_from_probs(X)
        return X


    def aggregate_children_log_probs(self, X, parent_map, num_ancestral):
        return tf.math.unsorted_segment_sum(X, parent_map, num_ancestral)


    def loglik_from_root_logits(self, root_logits, equilibrium_logits):
        return tf.math.reduce_logsumexp(
            root_logits + equilibrium_logits, axis=-1
        )


    def marginals_from_beliefs(self, beliefs, same_loglik=True):
        loglik = tf.math.reduce_logsumexp(
            beliefs[-1:] if same_loglik else beliefs,
            axis=-1,
            keepdims=True
        )
        return beliefs - loglik


    def logits_from_probs(self, probs, log_zero_val=-1e3):
        epsilon = tf.constant(
            np.finfo(util.default_dtype).tiny, dtype=probs.dtype
        )
        logits = tf.math.log(tf.maximum(probs, epsilon))
        zero_mask = tf.cast(tf.equal(probs, 0), dtype=logits.dtype)
        logits = (1-zero_mask) * logits + zero_mask * log_zero_val
        return logits


    def probs_from_logits(self, logits):
        return tf.math.exp(logits)


    def gather(self, tensor, indices, axis=0):
        return tf.gather(tensor, indices, axis=axis)


    def concat(self, tensors, axis=0):
        return tf.concat(tensors, axis=axis)


    def make_zeros(self, leaves, models, num_nodes):
        _, _, L, d = tf.unstack(tf.shape(leaves), 4)
        return tf.zeros((num_nodes, models, L, d), dtype=leaves.dtype)
from tensortree.tree import Tree
import tensortree.util
import os

# load a computational backend
DEFAULT_BACKEND = "tensorflow"
backend_name = os.environ.get("tensortree_backend", DEFAULT_BACKEND)
util._validate_backend(backend_name)
if backend_name == "tensorflow":
    import tensortree.backend_tf as backend


"""
Computes the log-likelihood P(X | tree).
Uses a vectorized implementation of Felsenstein's pruning algorithm
that treats sequences and tree layers in parallel.
Args:
    X: tensor of shape (batch_size, seq_len, d)
    tree: TensorTree object
Returns:
    log-likelihood tensor of shape (batch_size, seq_len)
"""
@tf.function
def loglik(X, tree : Tree):
    pass
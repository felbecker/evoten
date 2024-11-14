from tensortree.tree import Tree


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
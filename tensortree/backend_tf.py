import tensorflow as tf
from Bio.Phylo import BaseTree



"""
Returns a Variable representing the branch lengths for a tree.
"""
def make_branch_lengths(tree : BaseTree, num_models=1):
    pass


"""
"""
def make_rate_matrix():
    pass


"""
Args:
    rate_matrix: Rate matrix of shape (num_models, d, d) or (1, d, d).
    branch_lengths: Tensor of shape (num_nodes, num_models) representing the branch lengths/evolutionary times.
Returns:
    Tensor of shape (num_nodes, num_models, d, d) representing the transition probabilities along each branch.
"""
def make_branch_probabilities(rate_matrix, branch_lengths):
    pass


"""
Computes the probabilities after traversing a branch when starting with distributions X.
Args:
    X: tensor of shape (k, num_models, d)
    branch_probabilities: tensor of shape (k, num_models, d, d)
"""
def traverse_branch(X, branch_probabilities):
    pass


"""
Aggregates the log-probabilities of child nodes.
Args:
    X: tensor of shape (batch_size, num_children, num_models, d)
    parent_map: A list-like object of shape (batch_size, num_parents) that contains the number of children for each parent node.
                The computation will rely on the correct order of the children according to parent_map.
                Example:
                parent_map = [[2,1,3]]
                Stands for this topology and assumes that the children are ordered D E F G H I:
                A   B   C
                |\  |  /|\
                D E F G H I
Returns:
    tensor of shape (batch_size, num_parents, num_models, d) representing the aggregated probabilities of the parent nodes.
"""
def aggregate_children_log_probs(X, parent_map):
    pass
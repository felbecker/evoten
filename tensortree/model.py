from tensortree.tree_handler import TreeHandler
import tensortree.util as util
import numpy as np

backend = util.load_backend()


"""
Computes the ancestral logits at all internal (ancestral) nodes in the tree 
given logits at the leaves and a tree topology.
Supports multiple models (parameter sets that share the same tree topology).
Uses a vectorized implementation of Felsenstein's pruning algorithm
that treats models, sequence positions and all nodes within a tree layer in parallel.
Args:
    leaves: Logits of all symbols at all leaves of shape (num_leaves, L, models, d) or (num_leaves, L, 1, d). 
    leaf_names: Names of the leaves (list-like of length num_leaves). Used to reorder correctly.
    tree_handler: TreeHandler object
    rate_matrix: Rate matrix of shape (models, d, d)
    return_only_root: If True, only the root node logits are returned.
    leaves_are_probabilities: If True, leaves are assumed to be probabilities or one-hot encoded.
    return_probabilities: If True, return probabilities instead of logits.
Returns:
    Ancestral logits of shape (L, models, d) if return_only_root 
    else shape (num_ancestral_nodes, L, models, d)
"""
#@backend.decorator
def compute_ancestral_probabilities(leaves, leaf_names,
                                    tree_handler : TreeHandler, rate_matrix, 
                                    return_only_root = False, leaves_are_probabilities = False,
                                    return_probabilities = False):
    
    if leaves_are_probabilities:
        leaves = backend.logits_from_probs(leaves)

    if not return_only_root:
        anc_probs = backend.Cache(size = tree_handler.depth, dtype=leaves.dtype, infer_shape=False)

    # initialize 
    leaves = tree_handler.reorder(leaves, leaf_names)
    proc_leaves = tree_handler.layer_sizes[tree_handler.depth]
    X = leaves[:proc_leaves]
    
    for depth in range(tree_handler.depth, 0, -1):
        
        # traverse all edges {u, parent(u)} for all u of same depth
        B = tree_handler.get_branch_lengths_by_depth(depth)
        P = backend.make_transition_probs(rate_matrix, B)
        T = backend.traverse_branch(X, P)

        if depth == 2:
            print(P[2])
            print(backend.probs_from_logits(T[2]))

        num_leaves_above = tree_handler.layer_leaves[depth-1]

        # aggregate over child nodes
        child_counts = tree_handler.get_child_counts_by_depth(depth-1)[num_leaves_above:]
        X_anc = backend.aggregate_children_log_probs(T, child_counts)
        if depth > 1:
            if num_leaves_above > 0:
                X_leaves = leaves[proc_leaves:proc_leaves+num_leaves_above]
                proc_leaves += num_leaves_above
                X = backend.concat([X_leaves, X_anc])
            else:
                X = X_anc

        if not return_only_root:
            anc_probs = anc_probs.write(tree_handler.depth - depth, X_anc)

    if return_only_root:
        logits = X_anc[-1]
    else:
        logits = anc_probs.concat()

    return backend.probs_from_logits(logits) if return_probabilities else logits


"""
Computes log P(leaves | tree, rate_matrix).
Args:
    leaves: Logits of all symbols at all leaves of shape (num_leaves, L, models, d).
    tree_handler: TreeHandler object
    rate_matrix: Rate matrix of shape (models, d, d)
    equilibrium_logits: Equilibrium distribution logits of shape (models, d).
    leaves_are_probabilities: If True, leaves are assumed to be probabilities or one-hot encoded.
Returns:
    Log-likelihoods of shape (L, models).
"""
@backend.decorator
def loglik(leaves, leaf_names, tree_handler : TreeHandler, rate_matrix, equilibrium_logits, 
           leaves_are_probabilities=False):
    root_logits = compute_ancestral_probabilities(leaves, leaf_names, tree_handler, rate_matrix, 
                                                  return_only_root = True, leaves_are_probabilities=leaves_are_probabilities, 
                                                  return_probabilities=False)
    return backend.loglik_from_root_logits(root_logits, equilibrium_logits)
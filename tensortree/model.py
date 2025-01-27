from tensortree.tree_handler import TreeHandler
from tensortree.util import backend
import numpy as np




def traverse_branches(inputs, rate_matrix, branch_lengths, transposed=False, logarithmic=True):
    """
    Traverse n branches (u, inputs) in parallel and compute P(inputs | u).
    Per default, we assume that u is further in the past than inputs and that the branch_length > 0 between them
    indicates evolutionary time.

    Args:
        inputs: Log-likelihoods of shape (n, models, L, d)
        branch_lengths: Branch lengths of shape (n, models)
        rate_matrix: Rate matrix of shape (models, d, d)
        transposed: If transposed=False, compute P(v | u) (default).
                    If transposed=True, compute P(u | v).
    """
    P = backend.make_transition_probs(rate_matrix, branch_lengths)
    U = backend.traverse_branch(inputs, P, transposed, logarithmic=logarithmic)
    return U

def compute_ancestral_probabilities(leaves,
                                    tree_handler : TreeHandler, 
                                    rate_matrix, 
                                    branch_lengths, 
                                    leaf_names=None,
                                    return_only_root = False, 
                                    leaves_are_probabilities = True,
                                    return_probabilities = False):
    """
    Computes all (partial) log-likelihoods at all internal (ancestral) nodes u in the given tree,
    that is P(leaves below u | u, tree) for all u that are not leaves.
    Supports multiple, parallel models with broadcasting for rates and leaves in the model dimension.
    Uses a vectorized implementation of Felsenstein's pruning algorithm
    that treats models, sequence positions and all nodes within a tree layer in parallel.

    Args:
        leaves: Logits of all symbols at all leaves of shape (num_leaves, models, L, d). 
        tree_handler: TreeHandler object
        rate_matrix: Rate matrix of shape (models, d, d)
        branch_lengths: Branch lengths of shape (num_nodes-1, models)
        leaf_names: Names of the leaves (list-like of length num_leaves). Used to reorder correctly.
        return_only_root: If True, only the root node logliks are returned.
        leaves_are_probabilities: If True, leaves are assumed to be probabilities or one-hot encoded.
        return_probabilities: If True, return probabilities instead of logliks.

    Returns:
        Ancestral logliks of shape (models, L, d) if return_only_root 
        else shape (num_ancestral_nodes, models, L, d)
    """

    assert tree_handler.height > 0, "Tree must have at least one internal node."
    
    if leaves_are_probabilities:
        leaves = backend.logits_from_probs(leaves)

    # allocate a single chunk of memory for all internal nodes (no concat or masking required)
    # add results as layers are processed
    anc_logliks = backend.make_zeros(leaves, tree_handler.num_models, tree_handler.num_anc)

    # reorder the leaves to match the tree handler's internal order
    if leaf_names is None:
        X = leaves
    else:
        X = tree_handler.reorder(leaves, leaf_names)
    
    for height in range(tree_handler.height):
        
        # traverse all edges (parent(X), X) for all nodes X of same height in parallel
        B = tree_handler.get_values_by_height(branch_lengths, height)
        T = traverse_branches(X, rate_matrix, B)

        # aggregate over child nodes and add to anc_logliks
        parent_indices = tree_handler.get_parent_indices_by_height(height)
        Z = backend.aggregate_children_log_probs(T, parent_indices-tree_handler.num_leaves, tree_handler.num_anc)
        anc_logliks += Z

        if height < tree_handler.height-1:
            X = tree_handler.get_values_by_height(anc_logliks, height+1, leaves_included=False)

    if return_only_root:
        anc_logliks = anc_logliks[-1]

    return backend.probs_from_logits(anc_logliks) if return_probabilities else anc_logliks


def loglik(leaves, 
           tree_handler : TreeHandler, 
           rate_matrix, 
           branch_lengths, 
           equilibrium_logits, 
           leaf_names=None,
           leaves_are_probabilities=True):
    """
    Computes log P(leaves | tree, rate_matrix).

    Args:
        leaves: Logits of all symbols at all leaves of shape (num_leaves, models, L, d).
        tree_handler: TreeHandler object
        rate_matrix: Rate matrix of shape (models, d, d)
        branch_lengths: Branch lengths of shape (num_nodes-1, models)
        equilibrium_logits: Equilibrium distribution logits of shape (models, d).
        leaf_names: Names of the leaves (list-like of length num_leaves). Used to reorder correctly.
        leaves_are_probabilities: If True, leaves are assumed to be probabilities or one-hot encoded.

    Returns:
        Log-likelihoods of shape (models, L).
    """
    # handle the edge case where the input tree consists of a single node
    if tree_handler.height == 0:
        if leaves_are_probabilities:
            leaves = backend.logits_from_probs(leaves)[0,:]
        return backend.loglik_from_root_logits(leaves, equilibrium_logits)
    else:
        root_logits = compute_ancestral_probabilities(leaves, tree_handler, rate_matrix, branch_lengths, leaf_names,
                                                    return_only_root = True, leaves_are_probabilities=leaves_are_probabilities, 
                                                    return_probabilities=False)
        return backend.loglik_from_root_logits(root_logits, equilibrium_logits)
    

def compute_ancestral_marginals(leaves,
                                tree_handler : TreeHandler, 
                                rate_matrix, 
                                branch_lengths, 
                                equilibrium_logits, 
                                leaf_names=None,
                                leaves_are_probabilities = True,
                                return_probabilities = False):
    """
    Compute all marginal distributions at internal (ancestral) nodes u in the given leave data 
    and the tree. Formally, the method computes P(u | leaves, tree) for all u that are not leaves.

    Args:
        leaves: Logits of all symbols at all leaves of shape (num_leaves, models, L, d).
        tree_handler: TreeHandler object
        rate_matrix: Rate matrix of shape (models, d, d)
        branch_lengths: Branch lengths of shape (num_nodes-1, models)
        equilibrium_logits: Equilibrium distribution logits of shape (models, d).
        leaf_names: Names of the leaves (list-like of length num_leaves). Used to reorder correctly.
        leaves_are_probabilities: If True, leaves are assumed to be probabilities or one-hot encoded.
        return_probabilities: If True, return probabilities instead of logliks.

    Returns:
        Ancestral marginals of shape (num_ancestral_nodes, models, L, d).
    """
    
    assert tree_handler.height > 0, "Tree must have at least one internal node."
    
    if leaves_are_probabilities:
        leaves = backend.logits_from_probs(leaves)

    # Precompute the transition matrices, they'll be reused multiple times
    P_full = backend.make_transition_probs(rate_matrix, branch_lengths) 

    # throughput the method we compute:
    # beliefs: P(leaves, u | tree) for all ancestral nodes u
    # upward_messages: P(leaves below u | parent(u), tree) for all nodes u except root
    # downward_messages: P(leaves not below u, parent(u) | tree) for all nodes u except root

    # allocate single chunks of memory to work with
    beliefs = backend.make_zeros(leaves, tree_handler.num_models, tree_handler.num_anc)

    # reorder the leaves to match the tree handler's internal order
    if leaf_names is None:
        X = leaves
    else:
        X = tree_handler.reorder(leaves, leaf_names)

    # upward pass
    upward_messages = []
    for height in range(tree_handler.height):

        # traverse all edges (parent(u), u) for all nodes u of same height
        P = tree_handler.get_values_by_height(P_full, height)
        M = backend.traverse_branch(X, P) 
        upward_messages.append(M)

        # include equilibrium distribution when passing messages to the root, so
        # we don't have to add it later
        if height == tree_handler.height-1:
            M += equilibrium_logits / tree_handler.layer_sizes[height]

        # aggregate over child nodes and add to beliefs
        parent_indices = tree_handler.get_parent_indices_by_height(height)
        Z = backend.aggregate_children_log_probs(M, 
                                                 parent_indices-tree_handler.num_leaves, 
                                                 tree_handler.num_anc)
        beliefs += Z

        if height < tree_handler.height-1:
            X = tree_handler.get_values_by_height(beliefs, height+1, leaves_included=False)

    # note that now belief[x] is identical to the partial log-likelihood of node x
    # computed by Felsenstein's pruning algorithm (compute_ancestral_probabilities)
    # except that the root already includes the equilibrium distribution

    # initialize X for the downward pass, for each child of the root with 
    # height=tree.height-1, we compute a message that summarizes the likelihood of all 
    # other children of the root
    downward_messages = beliefs[-1:] - upward_messages[-1]

    # downward pass
    belief_updates = [backend.make_zeros(leaves, tree_handler.num_models, 1)] # add zeros for root 
    for height in range(tree_handler.height-1, 0, -1):
        
        # compute updates for the beliefs 
        P = tree_handler.get_values_by_height(P_full, height)
        U = backend.traverse_branch(downward_messages, P, transposed=True)
        belief_updates.insert(0, U)

        if height > 1:
            # compute downward messages
            layer_beliefs = tree_handler.get_values_by_height(beliefs, height, leaves_included=False)
            downward_messages = U + layer_beliefs - upward_messages[height-1]

    beliefs += backend.concat(belief_updates, axis=0)
    marginals = backend.marginals_from_beliefs(beliefs)

    return backend.probs_from_logits(marginals) if return_probabilities else marginals
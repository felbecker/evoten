import numpy as np

from tensortree.tree_handler import TreeHandler
from tensortree.util import backend



def compute_ancestral_probabilities(
    leaves,
    tree_handler : TreeHandler,
    transition_probs,
    leaf_names=None,
    return_only_root = False,
    leaves_are_probabilities = True,
    return_probabilities = False
):
    """
    Computes all (partial) log-likelihoods at all internal (ancestral) nodes u
    in the given tree, that is P(leaves below u | u, tree) for all u that are
    not leaves.
    Supports multiple, parallel models. Uses a vectorized implementation of
    Felsenstein's pruning algorithm that treats models, sequence positions and
    all nodes within a tree layer in parallel.

    * Broadcasting is supported for this dimension.

    Args:
        leaves: Logits of all symbols at all leaves of shape
            (num_leaves, models*, L, d).
        tree_handler: TreeHandler object
        transition_probs: Probabilistic transition matrices of shape
            (num_nodes-1, models, L*, d, d).
        leaf_names: Names of the leaves (list-like of length num_leaves).
            Used to reorder correctly.
        return_only_root: If True, only the root node logliks are returned.
        leaves_are_probabilities: If True, leaves are assumed to be
            probabilities or one-hot encoded.
        return_probabilities: If True, return probabilities instead of logliks.

    Returns:
        Ancestral logliks of shape (models, L, d) if return_only_root
        else shape (num_ancestral_nodes, models, L, d)
    """

    assert tree_handler.height > 0, "Tree must have at least one internal node."

    if leaves_are_probabilities:
        leaves = backend.logits_from_probs(leaves)

    # allocate a single chunk of memory for all internal nodes (no concat or
    # masking required) add results as layers are processed
    num_models = transition_probs.shape[1]
    anc_logliks = backend.make_zeros(
        leaves, num_models, tree_handler.num_anc
    )

    # reorder the leaves to match the tree handler's internal order
    if leaf_names is None:
        X = leaves
    else:
        X = tree_handler.reorder(leaves, leaf_names)

    for height in range(tree_handler.height):

        # traverse all edges (parent(X), X) for all nodes X of same height in
        # parallel
        P = tree_handler.get_values_by_height(transition_probs, height)
        T = backend.traverse_branch(X, P)

        # aggregate over child nodes and add to anc_logliks
        parent_indices = tree_handler.get_parent_indices_by_height(height)
        Z = backend.aggregate_children_log_probs(
            T,
            parent_indices-tree_handler.num_leaves,
            tree_handler.num_anc
        )
        anc_logliks += Z

        if height < tree_handler.height-1:
            X = tree_handler.get_values_by_height(
                anc_logliks, height+1, leaves_included=False
            )

    if return_only_root:
        anc_logliks = anc_logliks[-1]

    if return_probabilities:
        return backend.probs_from_logits(anc_logliks)
    else:
        return anc_logliks


def loglik(
    leaves,
    tree_handler : TreeHandler,
    transition_probs,
    equilibrium_logits,
    leaf_names=None,
    leaves_are_probabilities=True
):
    """
    Computes log P(leaves | tree, rate_matrix).

    * Broadcasting is supported for this dimension.

    Args:
        leaves: Logits of all symbols at all leaves of shape
            (num_leaves, models*, L, d).
        tree_handler: TreeHandler object
        transition_probs: Probabilistic transition matrices of shape
            (num_nodes-1, models, L*, d, d).
        equilibrium_logits: Equilibrium distribution logits of shape
            (models, L*, d).
        leaf_names: Names of the leaves (list-like of length num_leaves).
            Used to reorder correctly.
        leaves_are_probabilities: If True, leaves are assumed to be
            probabilities or one-hot encoded.

    Returns:
        Log-likelihoods of shape (models, L).
    """
    # handle the edge case where the input tree consists of a single node
    if tree_handler.height == 0:
        if leaves_are_probabilities:
            leaves = backend.logits_from_probs(leaves)[0,:]
        return backend.loglik_from_root_logits(leaves, equilibrium_logits)
    else:
        root_logits = compute_ancestral_probabilities(
            leaves,
            tree_handler,
            transition_probs,
            leaf_names,
            return_only_root = True,
            leaves_are_probabilities=leaves_are_probabilities,
            return_probabilities=False
        )
        return backend.loglik_from_root_logits(root_logits, equilibrium_logits)


def propagate(
    root,
    tree_handler : TreeHandler,
    transition_probs
):
    """
    Propagates a root distribution along the tree topology. The method computes
    P(u | root, tree) for all nodes u in the tree.

    * Broadcasting is supported for this dimension.

    Args:
        root: Probabilities of all symbols at the root node of shape
            (1, models*, L, d).
        tree_handler: TreeHandler object
        transition_probs: Probabilistic transition matrices of shape
            (num_nodes-1, models, L*, d, d).
    """

    assert tree_handler.height > 0, "Tree must have at least one internal node."

    dist, cur_dist = root, root

    for height in range(tree_handler.height-1, -1, -1):

        # traverse all edges (parent(X), X) for all nodes X of same height in
        # parallel
        P = tree_handler.get_values_by_height(transition_probs, height)
        U = backend.traverse_branch(
            cur_dist, P, transposed=True, logarithmic=False
        )
        dist = backend.concat([U, dist], axis=0)

        if height > 0:
            # compute downward messages
            parent_indices = tree_handler.get_parent_indices_by_height(height-1)
            cur_dist = backend.gather(
                dist, parent_indices-tree_handler.cum_layer_sizes[height-1]
            )

    return dist



def compute_ancestral_marginals(
    leaves,
    tree_handler : TreeHandler,
    transition_probs,
    equilibrium_logits,
    leaf_names=None,
    leaves_are_probabilities = True,
    return_probabilities = False,
    return_upward_messages = False,
    return_downward_messages = False
):
    """
    Compute all marginal distributions at internal (ancestral) nodes u in the
    given leave data and the tree. Formally, the method computes
    sP(u | leaves, tree) for all u that are not leaves.

    * Broadcasting is supported for this dimension.

    Args:
        leaves: Logits of all symbols at all leaves of shape
            (num_leaves, models*, L, d).
        tree_handler: TreeHandler object
        transition_probs: Probabilistic transition matrices of shape
            (num_nodes-1, models, L*, d, d).
        equilibrium_logits: Equilibrium distribution logits of shape
            (models, d).
        leaf_names: Names of the leaves (list-like of length num_leaves). Used
            to reorder correctly.
        leaves_are_probabilities: If True, leaves are assumed to be
            probabilities or one-hot encoded.
        return_probabilities: If True, return probabilities instead of logliks.
        return_upward_messages: If True, also returns upward messages of shape
            (num_nodes-1, models).
        return_downward_messages: If True, also returns downward messages of
            shape (num_nodes-1, models).

    Returns:
        Ancestral marginals of shape (num_ancestral_nodes, models, L, d).
    """

    assert tree_handler.height > 0, "Tree must have at least one internal node."

    if leaves_are_probabilities:
        leaves = backend.logits_from_probs(leaves)

    # throughout the method we compute:
    # beliefs: P(leaves, u | tree) for all ancestral nodes u
    # upward_messages: P(leaves below u | parent(u), tree) for all nodes u
    # except root downward_messages: P(leaves not below u, parent(u) | tree)
    # for all nodes u except root

    # allocate single chunk of memory to work with
    beliefs = backend.make_zeros(
        leaves, tree_handler.num_models, tree_handler.num_anc
    )

    # reorder the leaves to match the tree handler's internal order
    if leaf_names is None:
        X = leaves
    else:
        X = tree_handler.reorder(leaves, leaf_names)

    # upward pass
    upward_messages = []
    for height in range(tree_handler.height):

        # traverse upwards all edges (parent(u), u) for all nodes u of same
        # height
        P = tree_handler.get_values_by_height(transition_probs, height)
        M = backend.traverse_branch(X, P)
        upward_messages.append(M)

        # include equilibrium distribution when passing messages to the root, so
        # we don't have to add it later
        if height == tree_handler.height-1:
            M += equilibrium_logits / tree_handler.layer_sizes[height]

        # aggregate over child nodes and add to beliefs
        parent_indices = tree_handler.get_parent_indices_by_height(height)
        Z = backend.aggregate_children_log_probs(
            M,
            parent_indices-tree_handler.num_leaves,
            tree_handler.num_anc
        )
        beliefs += Z

        if height < tree_handler.height-1:
            X = tree_handler.get_values_by_height(
                beliefs, height+1, leaves_included=False
            )

    # note that now belief[x] is identical to the partial log-likelihood of
    # node x computed by Felsenstein's pruning algorithm
    # (compute_ancestral_probabilities)
    # except that the root already includes the equilibrium distribution

    # initialize X for the downward pass, for each child of the root with
    # height=tree.height-1, we compute a message that summarizes the likelihood
    # of all other children of the root
    downward_messages_by_cur_height = beliefs[-1:] - upward_messages[-1]

    # downward pass
    # add zeros for root
    belief_updates = backend.make_zeros(leaves, tree_handler.num_models, 1)

    if return_downward_messages:
        downward_messages = [downward_messages_by_cur_height]

    for height in range(tree_handler.height-1, 0, -1):

        # compute updates for the beliefs
        P = tree_handler.get_values_by_height(transition_probs, height)
        U = backend.traverse_branch(
            downward_messages_by_cur_height, P, transposed=True
        )
        belief_updates = backend.concat([U, belief_updates], axis=0)

        # compute the downward messages for the next layer, unless height==1
        # and not return_downward_messages
        if height > 1 or return_downward_messages:

            # compute downward messages
            parent_indices = tree_handler.get_parent_indices_by_height(height-1)
            layer_beliefs = backend.gather(
                beliefs, parent_indices-tree_handler.num_leaves
            )
            parent_updates = backend.gather(
                belief_updates,
                parent_indices-tree_handler.cum_layer_sizes[height-1]
            )
            downward_messages_by_cur_height = parent_updates + layer_beliefs
            downward_messages_by_cur_height -= upward_messages[height-1]

            if return_downward_messages:
                downward_messages.append(downward_messages_by_cur_height)


    beliefs += belief_updates
    marginals = backend.marginals_from_beliefs(beliefs)

    results = [marginals]

    if return_upward_messages:
        results.append(backend.concat(upward_messages, axis=0))

    if return_downward_messages:
        results.append(backend.concat(downward_messages[::-1], axis=0))

    if return_probabilities:
        results = [backend.probs_from_logits(x) for x in results]

    return tuple(results) if len(results) > 1 else results[0]


def compute_leaf_out_marginals(
    leaves,
    tree_handler : TreeHandler,
    transition_probs,
    equilibrium_logits,
    leaf_names=None,
    leaves_are_probabilities = True,
    return_probabilities = False
):
    """
    Computes the marginal distributions of the leaves given all other leaves,
    the tree topology and the rate matrix. Formally, the method computes
    P(u | leaves_except_u, tree, rates) for all leaves u.

    * Broadcasting is supported for this dimension.

    Args:
        leaves: Logits of all symbols at all leaves of shape
            (num_leaves, models*, L, d).
        tree_handler: TreeHandler object
        transition_probs: Probabilistic transition matrices of shape
            (num_nodes-1, models, L*, d, d).
        equilibrium_logits: Equilibrium distribution logits of shape
            (models, d).
        leaf_names: Names of the leaves (list-like of length num_leaves). Used
            to reorder correctly.
        leaves_are_probabilities: If True, leaves are assumed to be
            probabilities or one-hot encoded.
        return_probabilities: If True, return probabilities instead of logliks.

    Returns:
        Leaf-out marginals of shape (num_leaves, models, L, d).
    """
    _, downward_messages = compute_ancestral_marginals(
        leaves,
        tree_handler,
        transition_probs,
        equilibrium_logits,
        leaf_names,
        leaves_are_probabilities=leaves_are_probabilities,
        return_probabilities=False,
        return_upward_messages=False,
        return_downward_messages=True
    )

    # gather downward messages to leaves
    downward_messages_to_leaves = downward_messages[:tree_handler.num_leaves]

    # compute the transition matrices for all leaf edges
    P_leaf_edges = transition_probs[:tree_handler.num_leaves]

    # compute P(u, leaves_except_u | tree, rates)
    leaf_out = backend.traverse_branch(
        downward_messages_to_leaves, P_leaf_edges, transposed=True
    )

    # compzute P(u | leaves_except_u, tree, rates)
    leaf_out = backend.marginals_from_beliefs(leaf_out, same_loglik=False)

    if return_probabilities:
        return backend.probs_from_logits(leaf_out)
    else:
        return leaf_out

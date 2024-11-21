from Bio import Phylo 
import numpy as np
import tensortree.util as util


backend = util.load_backend()

"""
Wraps a rooted tree and provides utility functions useful for a
depth-wise processing of the tree.
"""
class TreeHandler():
    """
    Args:
        tree: Bio.Phylo tree object that will we wrapped by this class.
    """
    def __init__(self, tree : Phylo.BaseTree):
        self.bio_tree = tree
        self.update_datastructures()
        self.setup_init_branch_lengths()
        # initially, get_branch_lengths will return branch lengths
        # parsed from the tree file
        # calling set_branch_lengths later will overwrite these values
        self.set_branch_lengths(self.init_branch_lengths)


    """ Reads a tree from a file. 
    Args:
        filename: handle or filepath
        fmt: Format of the tree file. Supports all formats supported by Bio.Phylo.
    """
    @classmethod
    def read(cls, file, fmt="newick"):
        bio_tree = Phylo.read(file, fmt)
        return cls(bio_tree)
    
    """ Get indices for a list of node names (strings).
    """
    def get_indices(self, node_names):
        return [self.node_indices[name] for name in node_names]
    

    """ Sets the branch lengths of the tree.
    Args:
        branch_lengths: A tensor of shape (num_nodes-1, ...) representing the branch lengths of each node to its parent.
    """
    def set_branch_lengths(self, branch_lengths):
        self.branch_lengths = branch_lengths


    """ Retrieves a vector with the branch lengths for each node in the given depth.
    Args:
        depth: Path length from the root.
    """
    def get_branch_lengths_by_depth(self, depth):
        return self.get_values_by_depth(self.branch_lengths, depth, no_root=True)


    """ Retrieves all values from the leftmost axis of a tensor corresponding to all nodes of a given depth.
    Args:
        kernel: A tensor of shape (num_nodes-1, ...) if no_root else (num_nodes, ...),
        representing all branch lengths ordered by tree depth starting with the root.
        depth: A path length from the root.
        only_leaves: If True, kernel is assumed to be a tensor of shape (num_leaves, ...).
        no_root:    If True, the root node is excluded from the result.
    Returns:
        Tensor of shape (layer_size, ...) representing the branch lengths for the tree layer.
    """
    def get_values_by_depth(self, kernel, depth, only_leaves=False, no_root=False):
        k = self.cum_layer_sizes[depth]-int(no_root)
        s = self.layer_sizes[depth]
        return kernel[-k:self.num_nodes-k+s-int(no_root)]


    r""" Retrieves a vector with the number of child nodes for each node in the given layer.
        Example:
        For the tree
           ROOT
        /   |   \
        A   B   C
        |\  |  /|\
        D E F G H I
        and layer 1, the function will return [2,1,3].
    Args:
        layer: Layer index, starting from 0 for the leaves ending at num_layers-1 for the root.
    """
    def get_child_counts_by_depth(self, depth):
        return self.get_values_by_depth(self.child_counts, depth)
    

    """ Reorders the tensor along the given axis to be sorted in a way
        compatible with the tree. 
    Args:
        tensor: A tensor of shape (..., k, ...).
        node_names: List-like of k node names in the order as they appear in the tensor.
        axis: The axis along which the tensor should be reordered.
    """
    def reorder(self, tensor, node_names, axis=0):
        permutation = np.argsort(self.get_indices(node_names))
        return backend.reorder(tensor, permutation, 0)
    

    """ Initializes or updates utility datastructures for the tree.
    """
    def update_datastructures(self):
        # name nodes if they are not named
        for idx, clade in enumerate(self.bio_tree.find_clades()):
            if not clade.name:
                clade.name = str(idx)

        # maps each node to the path length from the root
        self.node_depths = self.bio_tree.depths(unit_branch_lengths=True) 
        self.num_nodes = len(self.node_depths)

        # compute the number of nodes in each depth-layer of the tree, starting from the root
        self.depth = max(self.node_depths.values())
        self.layer_sizes = np.zeros(self.depth+1, dtype=int)
        for node,d in self.node_depths.items():
            self.layer_sizes[d] += 1
        self.cum_layer_sizes = np.cumsum(self.layer_sizes)

        # compute a unique index for each tree node sorted descending by depth (i.e. root last)
        # among nodes with the same depth leaves come first 
        # finally, leaves and internal nodes are sorted left-to-right
        nodes_per_depth = [[] for _ in range(self.depth+1)]
        self.layer_leaves = np.zeros(self.depth+1, dtype=int) #number of leaves per layer
        for node, d in self.node_depths.items():
            if node.is_terminal():
                nodes_per_depth[self.depth-d].insert(self.layer_leaves[d], node)
                self.layer_leaves[d] += 1
            else:
                nodes_per_depth[self.depth-d].append(node)
        self.node_indices = {node.name: i for i, node in enumerate(node for nodes in nodes_per_depth for node in nodes)}

        # compute the number of children for each node
        self.child_counts = np.zeros(self.num_nodes, dtype=int)
        for clade in self.bio_tree.find_clades(order="level"):
            self.child_counts[self.node_indices[clade.name]] = len(clade)


    """ Initializes the branch lengths of the tree.
    """
    def setup_init_branch_lengths(self):
        self.init_branch_lengths = np.zeros((self.num_nodes-1, 1)) 
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                self.init_branch_lengths[self.node_indices[child.name]] = self.bio_tree.distance(clade, child)
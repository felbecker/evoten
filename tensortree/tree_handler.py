from Bio import Phylo 
import numpy as np

class TreeHandler():
    """
    Args:
        tree: Bio.Phylo tree object that will we wrapped by this class.
    """
    def __init__(self, tree : Phylo.BaseTree):
        self.bio_tree = tree
        self.update_datastructures()
        self.setup_branch_lengths()


    """ Reads a tree from a file. 
    Args:
        filename: handle or filepath
        fmt: Format of the tree file. Supports all formats supported by Bio.Phylo.
    """
    @classmethod
    def read(cls, file, fmt="newick"):
        bio_tree = Phylo.read(file, fmt)
        return cls(bio_tree)

    """ Returns the index of a given leaf or internal node.
    """
    def get_index(self, node_name):
        pass


    """ Retrieves all values for a given tree layer.
        For example, let t be a (num_nodes, 1) tensor of branch lengths of all nodes to their parent.
        Then get_values(t, 0) will return the branch lengths of the leaves with maximum distance to the root.
    Args:
        kernel: A tensor of shape (num_nodes, ...) representing all branch lengths ordered by tree level starting with the leaves.
        layer: Layer index, starting from 0 for the leaves ending at num_layers-1 for the root.
    Returns:
        Tensor of shape (layer_size, ...) representing the branch lengths for the tree layer.
    """
    def get_values(self, kernel, layer):
        k = self.cum_layer_sizes[layer]
        s = self.layer_sizes[layer]
        return kernel[k-s:k]

    """ Retrieves a vector with the branch lengths for each node in the given layer.
    """
    def get_branch_lengths(self, layer):
        return self.get_values(self.branch_lengths, layer)


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
    def get_child_counts(self, layer):
        k = self.cum_layer_sizes[layer]
        s = self.layer_sizes[layer]
        return self.child_counts[k-s:k]
    

    """ Initializes or updates utility datastructures for the tree.
    """
    def update_datastructures(self):
        # name nodes if they are not named
        for idx, clade in enumerate(self.bio_tree.find_clades()):
            if not clade.name:
                clade.name = str(idx)

        # maps each internal node to its depth
        self.depths = self.bio_tree.depths(unit_branch_lengths=True) 
        self.num_nodes = len(self.depths)

        # compute the number of nodes in each layer of the tree, starting from the leaves
        max_depth = max(self.depths.values())
        self.layer_sizes = np.zeros(max_depth+1, dtype=int)
        for _,d in self.depths.items():
            self.layer_sizes[max_depth-d] += 1

        # compute a unique index for each tree node as well as the number of children for each node
        # the indices have bottom-up order starting with the leaves
        self.node_indices = {}
        self.child_counts = np.zeros(self.num_nodes, dtype=int)
        for i, clade in enumerate(self.bio_tree.find_clades(order="level")):
            j = self.num_nodes - i - 1
            self.node_indices[clade.name] = j
            self.child_counts[j] = len(clade)

        # compute the number of nodes in each layer of the tree, starting from the leaves
        self.num_layers = self.layer_sizes.size
        self.cum_layer_sizes = np.cumsum(self.layer_sizes)


    """ Initializes the branch lengths of the tree.
    """
    def setup_branch_lengths(self):
        self.branch_lengths = np.zeros(self.num_nodes)
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                self.branch_lengths[self.node_indices[child.name]] = self.bio_tree.distance(clade, child)
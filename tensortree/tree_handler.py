from Bio import Phylo 
import numpy as np
import tensortree.util as util
from dataclasses import dataclass


backend = util.load_backend()



@dataclass
class NodeData:
    node: Phylo.BaseTree.Clade
    parent: Phylo.BaseTree.Clade
    height: int = -1
    index: int = -1
    finished: bool = False


"""
Wraps a rooted tree and provides utility functions useful for a
height-wise processing of the tree.
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
        return [self.nodes[name].index for name in node_names]
    

    """ Sets the branch lengths of the tree.
    Args:
        branch_lengths: A tensor of shape (num_nodes-1, ...) representing the branch lengths of each node to its parent.
    """
    def set_branch_lengths(self, branch_lengths):
        self.branch_lengths = branch_lengths


    """ Retrieves a vector with the branch lengths for each node with the given height.
    Args:
        height: Height of the subtree rooted at a node.
    """
    def get_branch_lengths_by_height(self, height):
        return self.get_values_by_height(self.branch_lengths, height)
    

    """ Retrieves a vector with the index of the parent for each node in a height layer."""
    def get_parent_indices_by_height(self, height):
        return self.get_values_by_height(self.parent_indices, height)


    """ Retrieves all values from the leftmost axis of a tensor corresponding to all nodes with a given height.
    Args:
        kernel: A tensor of shape (num_nodes-1, ...) or (num_nodes, ...) (root might be excluded),
        representing all branch lengths ordered by tree height and left-to-right (starting with leaves).
        height: Height of the subtree rooted at a node.
        leaves_included: If False, the method will assume a kernel if shape (num_nodes-num_leaves, ...) and height=0 is invalid.
    Returns:
        Tensor of shape (layer_size, ...) representing the branch lengths for the tree layer.
    """
    def get_values_by_height(self, kernel, height, leaves_included=True):
        k = self.cum_layer_sizes[height] 
        s = self.layer_sizes[height]
        n = int(not leaves_included) * self.layer_sizes[0]
        return kernel[k-s-n:k-n]


    r""" Retrieves a vector with the number of child nodes that are leafs for each node in the given layer.
        Example:
        For the tree
           ROOT
        /   |   \
        A   B   C
        |\  |  /|\
        D E F G H I
        |       | |
        x       y z
        and height=1, the function will return [1,0,2].
    Args:
        height: Height of the subtree rooted at a node.
    """
    def get_leaf_counts_by_height(self, height):
        return self.get_values_by_height(self.leaf_counts, height)
    

    r""" Retrieves a vector with the number of child nodes that are internal for each node in the given layer.
        Example:
        For the tree
           ROOT
        /   |   \
        A   B   C
        |\  |  /|\
        D E F G H I
        |       | |
        x       y z
        and height=1, the function will return [1,2,1].
    Args:
        height: Height of the subtree rooted at a node.
    """
    def get_internal_counts_by_height(self, height):
        return self.get_values_by_height(self.internal_counts, height)
    

    """ Reorders the tensor along the given axis to be sorted in a way
        compatible with the tree. 
        This method is meant to be statically compiled in the compute graph 
        (leaf order is always the same).
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
    def update_datastructures(self, unnamed_node_keyword="tensortree_node"):
        # name nodes if they are not named, make sure all nodes have a unique names
        provided_names = set()
        for idx, clade in enumerate(self.bio_tree.find_clades()):
            if clade.name:
                assert(unnamed_node_keyword not in clade.name, 
                    f"The keyword '{unnamed_node_keyword}' is reserved for internal use.")
            else:
                assert(clade.name not in provided_names, "All nodes must have unique names.")
                provided_names.add(clade.name)
                clade.name = unnamed_node_keyword+'_'+str(idx)

        # create a node-to-parent map
        self.nodes = {self.bio_tree.root.name: NodeData(self.bio_tree.root, parent=None)}
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                self.nodes[child.name] = NodeData(child, parent=clade)
                
        self.num_nodes = len(self.nodes)

        # map each node to the height of its subtree
        node_queue = self.bio_tree.get_terminals(order="preorder")
        self.num_leaves = len(node_queue)
        while node_queue:
            u = node_queue.pop(0) # leftmost unfinished node
            u_data = self.nodes[u.name]
            if u_data.finished:
                continue
            if all(self.nodes[child.name].finished for child in u):
                u_data.finished = True
                if u.is_terminal():
                    u_data.height = 0
                else:
                    u_data.height = 1 + max(self.nodes[child.name].height for child in u)
            if u != self.bio_tree.root:
                node_queue.append(u_data.parent)

        self.height = self.nodes[self.bio_tree.root.name].height

        # compute the number of nodes in each height-layer of the tree
        self.layer_sizes = np.zeros(self.height+1, dtype=int)
        for _,data in self.nodes.items():
            self.layer_sizes[data.height] += 1
        self.cum_layer_sizes = np.cumsum(self.layer_sizes)

        # compute a unique index for each node  
        # sorted by height and left-to-right within each layer
        index_struct = [[] for _ in range(self.height+1)]
        for clade in self.bio_tree.find_clades(order="postorder"):
            index_struct[self.nodes[clade.name].height].append(clade)
        i = 0
        for node_list in index_struct:
            for node in node_list:
                self.nodes[node.name].index = i
                i += 1

        # compute the number of leaf/internal children for each node as well as parent indices
        self.leaf_counts = np.zeros(self.num_nodes, dtype=int)
        self.internal_counts = np.zeros(self.num_nodes, dtype=int)
        self.parent_indices = np.zeros(self.num_nodes-1, dtype=int)
        for clade in self.bio_tree.find_clades(order="level"):
            num_leaf_children = len([child for child in clade if child.is_terminal()])
            num_internal_children = len(clade) - num_leaf_children
            self.leaf_counts[self.nodes[clade.name].index] = num_leaf_children
            self.internal_counts[self.nodes[clade.name].index] = num_internal_children
            if clade != self.bio_tree.root:
                self.parent_indices[self.nodes[clade.name].index] = self.nodes[self.nodes[clade.name].parent.name].index


    """ Initializes the branch lengths of the tree.
    """
    def setup_init_branch_lengths(self):
        self.init_branch_lengths = np.zeros((self.num_nodes-1, 1)) 
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                self.init_branch_lengths[self.nodes[child.name].index] = self.bio_tree.distance(clade, child)
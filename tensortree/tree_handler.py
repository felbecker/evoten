import copy
import uuid
from dataclasses import dataclass
from io import StringIO

import numpy as np
from Bio import Phylo

from tensortree.util import backend, default_dtype


@dataclass
class NodeData:
    node: Phylo.BaseTree.Clade
    parent: Phylo.BaseTree.Clade
    height: int = -1
    index: int = -1
    finished: bool = False



class TreeHandler():
    """
    Wraps a rooted tree and provides utility functions useful for a
    height-wise processing of the tree.

    Args:
        tree: Bio.Phylo tree object that will we wrapped by this class.
            If None, a tree with only a root node will be created.
        root_name: Name of the root node when creating a new tree.
    """
    def __init__(self, tree : Phylo.BaseTree = None, root_name=None):
        if tree is None:
            tree = Phylo.BaseTree.Clade()
            tree.name = root_name if root_name is not None else "ROOT"
        self.root_name = tree.name
        self.bio_tree = tree
        self.collapse_queue = []
        self.split_queue = []
        self.update(force_reset_init_lengths=True)


    @classmethod
    def read(cls, file, fmt="newick"):
        """ Reads a tree from a file.

        Args:
            filename: handle or filepath
            fmt: Format of the tree file. Supports all formats supported by
                Bio.Phylo.
        """
        bio_tree = Phylo.read(file, fmt)
        return cls(bio_tree)


    @classmethod
    def from_newick(cls, newick_str):
        """ Reads a tree from a newick string.

        Args:
            filename: handle or filepath
            fmt: Format of the tree file. Supports all formats supported by
                Bio.Phylo.
        """
        handle = StringIO(newick_str)
        bio_tree = Phylo.read(handle, "newick")
        return cls(bio_tree)


    @classmethod
    def copy(cls, tree_handler):
        """ Copies another tree handler.

        Args:
            tree_handler: TreeHandler object to copy.
        """
        treedata = tree_handler.to_newick()
        handle = StringIO(treedata)
        tree = Phylo.read(handle, "newick")
        return cls(tree, tree_handler.root_name)


    def get_indices(self, node_names):
        """ Get indices for a list of node names (strings).
        """
        return [self.nodes[name].index for name in node_names]


    def get_index(self, node_name):
        return self.nodes[node_name].index


    def set_branch_lengths(self, branch_lengths, update_phylo_tree=True):
        """ Sets the branch lengths of the tree.

        Args:
            branch_lengths: A tensor of shape (num_nodes-1, k) representing the
                branch lengths of each node to its parent.
                            k is the number of models
        """
        self.branch_lengths = branch_lengths
        self.num_models = branch_lengths.shape[1]
        if update_phylo_tree:
            assert branch_lengths.shape[-1] == 1, "Branch lengths must be of " \
                "shape (num_nodes-1, 1) when updating the phylogenetic tree."
            for clade in self.bio_tree.find_clades(order="level"):
                for child in clade:
                    i = self.nodes[child.name].index
                    child.branch_length = branch_lengths[i, 0]


    def get_branch_lengths_by_height(self, height):
        """ Retrieves a vector with the branch lengths for each node with the
        given height.

        Args:
            height: Height of the subtree rooted at a node.
        """
        return self.get_values_by_height(self.branch_lengths, height)


    def get_parent_indices_by_height(self, height):
        """ Retrieves a vector with the index of the parent for each node in a
        height layer."""
        return self.get_values_by_height(self.parent_indices, height)


    def get_values_by_height(self, kernel, height, leaves_included=True):
        """ Retrieves all values from the leftmost axis of a tensor
        corresponding to all nodes with a given height.

        Args:
            kernel: A tensor of shape (num_nodes-1, ...) or (num_nodes, ...)
                (root last; might be excluded),
            representing all branch lengths ordered by tree height and
                left-to-right (starting with leaves).
            height: Height of the subtree rooted at a node.
            leaves_included: If False, the method will assume a kernel if shape
                (num_nodes-num_leaves, ...) and height=0 is invalid.

        Returns:
            Tensor of shape (layer_size, ...) representing the branch lengths
            for the tree layer.
        """
        k = self.cum_layer_sizes[height]
        s = self.layer_sizes[height]
        n = int(not leaves_included) * self.layer_sizes[0]
        return kernel[k-s-n:k-n]


    def get_leaf_counts_by_height(self, height):
        r""" Retrieves a vector with the number of child nodes that are leafs
        for each node in the given layer.

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
        return self.get_values_by_height(self.leaf_counts, height)


    def get_internal_counts_by_height(self, height):
        r""" Retrieves a vector with the number of child nodes that are
        internal for each node in the given layer.

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
        return self.get_values_by_height(self.internal_counts, height)


    def reorder(self, tensor, node_names, axis=0):
        """ Reorders the tensor along the given axis to be sorted in a way
            compatible with the tree.
            This method is meant to be statically compiled in the compute graph
            (leaf order is always the same).

        Args:
            tensor: A tensor of shape (..., k, ...).
            node_names: List-like of k node names in the order as they appear
                in the tensor.
            axis: The axis along which the tensor should be reordered.
        """
        permutation = np.argsort(self.get_indices(node_names))
        return backend.gather(tensor, permutation, 0)


    def update(
            self,
            unnamed_node_keyword="tensortree_node",
            force_reset_init_lengths=False
        ):
        """ Initializes or updates utility datastructures for the tree.
        """

        #apply any queued modifications
        mods_applied = self._apply_mods(unnamed_node_keyword)

        # name nodes if they are not named, make sure all nodes have a unique names
        provided_names = set()
        for idx, clade in enumerate(self.bio_tree.find_clades()):
            if clade.name:
                assert clade.name not in provided_names, f"All nodes must have" \
                    "unique names. {clade.name} is not unique."
                provided_names.add(clade.name)
            else:
                clade.name = unnamed_node_keyword+'_'+uuid.uuid4().hex

        # create a node-to-parent map
        self.nodes = {
            self.bio_tree.root.name: NodeData(self.bio_tree.root, parent=None)
        }
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                self.nodes[child.name] = NodeData(child, parent=clade)

        self.num_nodes = len(self.nodes)

        # map each node to the height of its subtree
        node_queue = self.bio_tree.get_terminals(order="preorder")
        self.num_leaves = len(node_queue)
        self.num_anc = self.num_nodes - self.num_leaves
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
                    m = max(self.nodes[child.name].height for child in u)
                    u_data.height = 1 + m
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
        self.node_names = []
        for node_list in index_struct:
            for node in node_list:
                self.nodes[node.name].index = i
                self.node_names.append(node.name)
                i += 1

        # compute the number of leaf/internal children for each node as well as
        # parent indices
        self.leaf_counts = np.zeros(self.num_nodes, dtype=int)
        self.internal_counts = np.zeros(self.num_nodes, dtype=int)
        self.parent_indices = np.zeros(self.num_nodes-1, dtype=int)
        for clade in self.bio_tree.find_clades(order="level"):
            leaf_childs = len(
                [child for child in clade if child.is_terminal()]
            )
            int_childs = len(clade) - leaf_childs
            self.leaf_counts[self.nodes[clade.name].index] = leaf_childs
            self.internal_counts[self.nodes[clade.name].index] = int_childs
            if clade != self.bio_tree.root:
                i = self.nodes[self.nodes[clade.name].parent.name].index
                self.parent_indices[self.nodes[clade.name].index] = i

        if mods_applied or force_reset_init_lengths:
            self.setup_init_branch_lengths()
            # initially, get_branch_lengths will return branch lengths
            # parsed from the tree file
            # calling set_branch_lengths later will overwrite these values
            self.set_branch_lengths(
                self.init_branch_lengths, update_phylo_tree=False
            )


    def setup_init_branch_lengths(self):
        """ Initializes the branch lengths of the tree.
        """
        self.init_branch_lengths = np.zeros(
            (self.num_nodes-1, 1), dtype=default_dtype
        )
        for clade in self.bio_tree.find_clades(order="level"):
            for child in clade:
                L = child.branch_length
                self.init_branch_lengths[self.nodes[child.name].index] = L


    def collapse(self, node_name):
        """ Collapses a node in the tree. Call update() after all tree
        modifications are done.
        """
        self.collapse_queue.append(self.nodes[node_name].node)


    def split(self, node_name, n=2, branch_length=1.0, names=None):
        """ Generates n new descendants for a node. Call update() after all
        tree modifications are done.
        """
        node = self.nodes[node_name].node
        self.split_queue.append((node, n, branch_length, names))


    def prune(self):
        """ Prunes the tree by removing all leaves, i.e. strips the lowest
        height layer.
            Call update() after all tree modifications are done.
        """
        for node in self.bio_tree.get_terminals():
            self.collapse(node.name)


    def change_root(self, new_root_name):
        """ Rotates the tree such that a different node becomes the root.
            Calls update() automatically.
        Args:
            new_root_name: Name of the new root node, can be any internal node
            in the tree.
        """
        self.bio_tree.root_with_outgroup(new_root_name)
        self.root_name = new_root_name
        self.update(force_reset_init_lengths=True)


    # applies queued modifications to the tree before update
    def _apply_mods(self, unnamed_node_keyword="tensortree_node"):
        mods_applied = len(self.collapse_queue) > 0 or len(self.split_queue) > 0
        # collapse
        for node in self.collapse_queue:
            self.bio_tree.collapse(node)
        self.collapse_queue.clear()
        # split
        for node, n, branch_length, names in self.split_queue:
            node.split(n, branch_length)
            if names is not None:
                for child, name in zip(node, names):
                    child.name = name
        self.split_queue.clear()
        return mods_applied




    def to_newick(self, no_internal_names=True):
        """ Returns the newick string representation of the tree.
        """
        if no_internal_names:
            #remove internal names
            internal_names = []
            for clade in self.bio_tree.find_clades():
                if not clade.is_terminal():
                    internal_names.append(clade.name)
                    clade.name = None
        handle = StringIO()
        Phylo.write(self.bio_tree, handle, "newick")
        if no_internal_names:
            #reset the names
            for name in internal_names:
                clade = self.nodes[name].node
                clade.name = name
        return handle.getvalue()


    def draw(self, no_labels=False, axes=None, do_show=True):
        """ Plots the tree. """
        # Flip branches so deeper clades are displayed at top
        self.bio_tree.ladderize()
        result = Phylo.draw(
            self.bio_tree,
            label_func=lambda x: "" if no_labels else x.name,
            axes=axes,
            do_show=do_show
        )
        return result
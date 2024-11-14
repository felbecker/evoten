import unittest
import numpy as np
from tensortree.tree import TreeHandler


class TestTree(unittest.TestCase):

    def test_layer_sizes(self):
        layer_sizes1 = TreeHandler.read("test/data/simple.tree").layer_sizes
        layer_sizes2 = TreeHandler.read("test/data/simple2.tree").layer_sizes
        self.assertTrue(np.all(layer_sizes1 == np.array([2, 3, 1])))
        self.assertTrue(np.all(layer_sizes2 == np.array([2, 2, 3, 1])))

    def test_child_counts(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/star.tree")
        self.assertTrue(np.all(t1.child_counts == np.array([0,0,2,0,0,3])))
        self.assertTrue(np.all(t2.child_counts == np.array([0,0,2,0,2,0,0,3])))
        self.assertTrue(np.all(t3.child_counts == np.array([0,0,0,0,4])))
        self.assertTrue(np.all(t1.get_child_counts(0) == np.array([0,0])))
        self.assertTrue(np.all(t1.get_child_counts(1) == np.array([2,0,0])))
        self.assertTrue(np.all(t1.get_child_counts(2) == np.array([3])))
        self.assertTrue(np.all(t2.get_child_counts(0) == np.array([0,0])))
        self.assertTrue(np.all(t2.get_child_counts(1) == np.array([2,0])))
        self.assertTrue(np.all(t2.get_child_counts(2) == np.array([2,0,0])))
        self.assertTrue(np.all(t2.get_child_counts(3) == np.array([3])))
        self.assertTrue(np.all(t3.get_child_counts(0) == np.array([0,0,0,0])))
        self.assertTrue(np.all(t3.get_child_counts(1) == np.array([4])))

    def test_get_values(self):
        t = TreeHandler.read("test/data/simple.tree")
        branch_lens = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
        self.assertTrue(np.all(t.get_values(branch_lens, 0) == np.array([0.1, 0.5])))
        self.assertTrue(np.all(t.get_values(branch_lens, 1) == np.array([0.3, 0.7, 0.2])))

    def test_branch_lengths(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/star.tree")
        self.assertTrue(np.all(t1.branch_lengths == np.array([0.4, 0.3, 0.5, 0.2, 0.1, 0.])))
        self.assertTrue(np.all(t2.branch_lengths == np.array([0.1, 0.1, 0.4, 0.3, 0.5, 0.2, 0.1, 0.])))
        self.assertTrue(np.all(t3.branch_lengths == np.array([0.1, 0.4, 0.2, 0.1, 0.])))
        self.assertTrue(np.all(t1.get_branch_lengths(0) == np.array([0.4, 0.3])))
        self.assertTrue(np.all(t1.get_branch_lengths(1) == np.array([0.5, 0.2, 0.1])))
        self.assertTrue(np.all(t1.get_branch_lengths(2) == np.array([0.])))
        self.assertTrue(np.all(t2.get_branch_lengths(0) == np.array([0.1, 0.1])))
        self.assertTrue(np.all(t2.get_branch_lengths(1) == np.array([0.4, 0.3])))
        self.assertTrue(np.all(t2.get_branch_lengths(2) == np.array([0.5, 0.2, 0.1])))
        self.assertTrue(np.all(t2.get_branch_lengths(3) == np.array([0.])))
        self.assertTrue(np.all(t3.get_branch_lengths(0) == np.array([0.1, 0.4, 0.2, 0.1])))
        self.assertTrue(np.all(t3.get_branch_lengths(1) == np.array([0.])))


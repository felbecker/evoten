import unittest
import numpy as np
from tensortree.tree_handler import TreeHandler


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


class TestBackendTF(unittest.TestCase):

    def test_branch_lengths(self):
        from tensortree.backend_tf import make_branch_lengths
        kernel = np.array([[-3., -1., 0.], [1., 2., 3.]])
        branch_lengths = make_branch_lengths(kernel)
        self.assertTrue(np.all(branch_lengths > 0.))

    def test_rate_matrix(self):
        from tensortree.backend_tf import make_rate_matrix
        # 3 jukes cantor models
        exchangeabilities = [[[0., mue, mue], [mue, 0., mue], [mue, mue, 0.]] for mue in [1., 2., 5.]]
        equilibrium = np.ones((3,3)) / 3.
        rate_matrix = make_rate_matrix(exchangeabilities, equilibrium)
        #since the matrix is normalized, all mue should yield the same result
        ref = np.array([[[-1., .5, .5], [.5, -1, .5], [.5, .5, -1]]]*3)
        np.testing.assert_allclose(rate_matrix, ref)

    def test_transition_probs(self):
        from tensortree.backend_tf import make_rate_matrix
        from tensortree.backend_tf import make_transition_probs
        # 3 jukes cantor models
        exchangeabilities = [[[0., mue, mue], [mue, 0., mue], [mue, mue, 0.]] for mue in [1., 2., 5.]]
        equilibrium = np.ones((3,3)) / 3.
        rate_matrix = make_rate_matrix(exchangeabilities, equilibrium)
        P = make_transition_probs(rate_matrix, np.ones((1, 1)))
        # test if matrix is probabilistic
        np.testing.assert_almost_equal(np.sum(P, -1), 1.) 
        # unit time, so expect 1 mutation per site
        mut_prob = np.sum(P * (1-np.eye(3)), -1)
        number_of_expected_mutations = - 2./3 * np.log( 1 - 3./2 * mut_prob)
        np.testing.assert_almost_equal(number_of_expected_mutations, 1.) 


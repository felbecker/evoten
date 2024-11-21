import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import numpy as np
from tensortree.tree_handler import TreeHandler
from tensortree import model


class TestTree(unittest.TestCase):

    def test_indices(self):
        # tensortree sorts nodes descending by depth and
        # leaf-first among nodes with the same depth
        ind = TreeHandler.read("test/data/simple.tree").get_indices(['A', 'B', 'C', 'D'])
        self.assertEqual(set((ind[0], ind[1])), set((2,3)))
        self.assertEqual(set((ind[2], ind[3])), set((0,1)))
        ind2 = TreeHandler.read("test/data/simple2.tree").get_indices(['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(set((ind2[0], ind2[1])), set((4,5)))
        self.assertEqual(ind2[2], 2)
        self.assertEqual(set((ind2[3], ind2[4])), set((0,1)))
        ind3 = TreeHandler.read("test/data/simple3.tree").get_indices(['A', 'B', 'C', 'D', 'E', 'F'])
        self.assertEqual(set((ind3[0], ind3[1])), set((0,1)))
        self.assertEqual(ind3[2], 4)
        self.assertEqual(ind3[3], 5)
        self.assertEqual(set((ind3[4], ind3[5])), set((2,3)))
        ind4 = TreeHandler.read("test/data/star.tree").get_indices(['A', 'B', 'C', 'D'])
        self.assertEqual(set(ind4), 
                         set(range(4)))

    def test_layer_sizes(self):
        layer_sizes1 = TreeHandler.read("test/data/simple.tree").layer_sizes
        layer_sizes2 = TreeHandler.read("test/data/simple2.tree").layer_sizes
        layer_sizes3 = TreeHandler.read("test/data/simple3.tree").layer_sizes
        layer_sizes4 = TreeHandler.read("test/data/star.tree").layer_sizes
        np.testing.assert_equal(layer_sizes1, np.array([1, 3, 2]))
        np.testing.assert_equal(layer_sizes2, np.array([1, 3, 2, 2]))
        np.testing.assert_equal(layer_sizes3, np.array([1, 2, 4, 4]))
        np.testing.assert_equal(layer_sizes4, np.array([1, 4]))

    def test_layer_leaves(self):
        layer_leaves1 = TreeHandler.read("test/data/simple.tree").layer_leaves
        layer_leaves2 = TreeHandler.read("test/data/simple2.tree").layer_leaves
        layer_leaves3 = TreeHandler.read("test/data/simple3.tree").layer_leaves
        layer_leaves4 = TreeHandler.read("test/data/star.tree").layer_leaves
        np.testing.assert_equal(layer_leaves1, np.array([0, 2, 2]))
        np.testing.assert_equal(layer_leaves2, np.array([0, 2, 1, 2]))
        np.testing.assert_equal(layer_leaves3, np.array([0, 0, 2, 4]))
        np.testing.assert_equal(layer_leaves4, np.array([0, 4]))

    def test_child_counts(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.child_counts, np.array([0,0,0,0,2,3]))
        np.testing.assert_equal(t2.child_counts, np.array([0,0,0,2,0,0,2,3]))
        np.testing.assert_equal(t3.child_counts, np.array([0,0,0,0,0,0,2,2,2,2,2]))
        np.testing.assert_equal(t4.child_counts, np.array([0,0,0,0,4]))
        np.testing.assert_equal(t1.get_child_counts_by_depth(0), np.array([3]))
        np.testing.assert_equal(t1.get_child_counts_by_depth(1), np.array([0,0,2]))
        np.testing.assert_equal(t1.get_child_counts_by_depth(2), np.array([0,0]))
        np.testing.assert_equal(t2.get_child_counts_by_depth(0), np.array([3]))
        np.testing.assert_equal(t2.get_child_counts_by_depth(1), np.array([0,0,2]))
        np.testing.assert_equal(t2.get_child_counts_by_depth(2), np.array([0,2]))
        np.testing.assert_equal(t2.get_child_counts_by_depth(3), np.array([0,0]))
        np.testing.assert_equal(t3.get_child_counts_by_depth(0), np.array([2]))
        np.testing.assert_equal(t3.get_child_counts_by_depth(1), np.array([2,2]))
        np.testing.assert_equal(t3.get_child_counts_by_depth(2), np.array([0,0,2,2]))
        np.testing.assert_equal(t3.get_child_counts_by_depth(3), np.array([0,0,0,0]))
        np.testing.assert_equal(t4.get_child_counts_by_depth(0), np.array([4]))
        np.testing.assert_equal(t4.get_child_counts_by_depth(1), np.array([0,0,0,0]))

    # tests if values from external tensors are correctly reads
    def test_get_values_by_depth(self):
        # nodes are sorted by depth 
        t = TreeHandler.read("test/data/simple.tree")
        branch_lens = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
        np.testing.assert_equal(t.get_values_by_depth(branch_lens, 1, no_root=True), np.array([0.3, 0.7, 0.2]))
        np.testing.assert_equal(t.get_values_by_depth(branch_lens, 2, no_root=True), np.array([0.1, 0.5]))
        t2 = TreeHandler.read("test/data/simple2.tree")
        branch_lens2 = np.array([0.4, 0.6, 0.1, 0.5, 0.3, 0.7, 0.2])
        np.testing.assert_equal(t2.get_values_by_depth(branch_lens2, 1, no_root=True), np.array([0.3, 0.7, 0.2]))
        np.testing.assert_equal(t2.get_values_by_depth(branch_lens2, 2, no_root=True), np.array([0.1, 0.5]))
        np.testing.assert_equal(t2.get_values_by_depth(branch_lens2, 3, no_root=True), np.array([0.4, 0.6]))
        t3 = TreeHandler.read("test/data/star.tree")
        branch_lens3 = np.array([0.4, 0.6, 0.1, 0.5])
        np.testing.assert_equal(t3.get_values_by_depth(branch_lens3, 1, no_root=True), branch_lens3)


    # tests if the branch lengths from the tree files are read correctly
    def test_branch_lengths(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.branch_lengths[:,0], np.array([0.3,0.4,0.1,0.2,0.5]))
        np.testing.assert_equal(t2.branch_lengths[:,0], np.array([0.1,0.1,0.3,0.4,0.1,0.2,0.5]))
        np.testing.assert_equal(t3.branch_lengths[:,0], np.array([0.1,0.2,0.6,0.7,0.4,0.9,0.3,0.8,0.5,1.]))
        np.testing.assert_equal(t4.branch_lengths[:,0], np.array([0.1,0.2,0.4,0.1]))
        np.testing.assert_equal(t1.get_branch_lengths_by_depth(1)[:,0], np.array([0.1, 0.2, 0.5]))
        np.testing.assert_equal(t1.get_branch_lengths_by_depth(2)[:,0], np.array([0.3, 0.4]))
        np.testing.assert_equal(t2.get_branch_lengths_by_depth(1)[:,0], np.array([0.1, 0.2, 0.5]))
        np.testing.assert_equal(t2.get_branch_lengths_by_depth(2)[:,0], np.array([0.3, 0.4]))
        np.testing.assert_equal(t2.get_branch_lengths_by_depth(3)[:,0], np.array([0.1, 0.1]))
        np.testing.assert_equal(t4.get_branch_lengths_by_depth(1)[:,0], np.array([0.1, 0.2, 0.4, 0.1]))


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


class TestModel(unittest.TestCase):

    def test_anc_probs(self):
        import tensorflow as tf

        t = TreeHandler.read("test/data/star.tree")
        leaves = np.array([[0], [1], [2], [3]])
        leaves = np.eye(4)[leaves]
        leaves = leaves[:,:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']
        rate_matrix = np.array([[[-1., 1./3, 1./3, 1./3], 
                                 [1./3, -1, 1./3, 1./3], 
                                 [1./3, 1./3, -1, 1./3], 
                                 [1./3, 1./3, 1./3, -1]]])
        X = model.compute_ancestral_probabilities(leaves, leaf_names, t, rate_matrix, 
                                                    leaves_are_probabilities=True,
                                                    return_probabilities=True)

        def p(z, mue=4./3):
            t = np.array([0.1, 0.2, 0.4, 0.1])
            p = 1./4 - 1./4 * (np.exp(-mue*t))
            p[z] = 1./4 + 3./4 * np.exp(-mue*t[z])
            return np.prod(p)

        ref = np.array([p(i) for i in range(4)])
        np.testing.assert_almost_equal(X[0,0,0], ref)

        L = model.loglik(leaves, leaf_names, t, rate_matrix, 
                        equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]),
                        leaves_are_probabilities=True)
        np.testing.assert_almost_equal(L, np.log(np.sum(ref)/4))

    
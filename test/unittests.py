import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest
import numpy as np
from tensortree.tree_handler import TreeHandler
from tensortree import model


class TestTree(unittest.TestCase):

    def test_indices(self):
        # tensortree sorts nodes by height and left to right within a layer
        ind = TreeHandler.read("test/data/simple.tree").get_indices(['A', 'B', 'C', 'D'])
        np.testing.assert_equal(ind, np.array([0,1,2,3]))
        ind2 = TreeHandler.read("test/data/simple2.tree").get_indices(['A', 'B', 'C', 'D', 'E'])
        np.testing.assert_equal(ind2, np.array([0,1,2,3,4]))
        ind3 = TreeHandler.read("test/data/simple3.tree").get_indices(['A', 'B', 'C', 'D', 'E', 'F'])
        np.testing.assert_equal(ind3, np.array([0,1,2,3,4,5]))
        ind4 = TreeHandler.read("test/data/star.tree").get_indices(['A', 'B', 'C', 'D'])
        np.testing.assert_equal(ind4, np.array([0,1,2,3]))

    def test_layer_sizes(self):

        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")

        np.testing.assert_equal(t1.layer_sizes, np.array([4,1,1]))
        np.testing.assert_equal(t2.layer_sizes, np.array([5,1,1,1]))
        np.testing.assert_equal(t3.layer_sizes, np.array([6,2,2,1]))
        np.testing.assert_equal(t4.layer_sizes, np.array([4,1]))

    def test_child_counts(self):

        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")

        np.testing.assert_equal(t1.leaf_counts, np.array([0,0,0,0,2,2]))
        np.testing.assert_equal(t1.internal_counts, np.array([0,0,0,0,0,1]))
        np.testing.assert_equal(t2.leaf_counts, np.array([0,0,0,0,0,2,1,2]))
        np.testing.assert_equal(t2.internal_counts, np.array([0,0,0,0,0,0,1,1]))
        np.testing.assert_equal(t3.leaf_counts, np.array([0,0,0,0,0,0,2,2,1,1,0]))
        np.testing.assert_equal(t3.internal_counts, np.array([0,0,0,0,0,0,0,0,1,1,2]))
        np.testing.assert_equal(t4.leaf_counts, np.array([0,0,0,0,4]))
        np.testing.assert_equal(t4.internal_counts, np.array([0,0,0,0,0]))

        np.testing.assert_equal(t1.get_leaf_counts_by_height(0), np.array([0,0,0,0]))
        np.testing.assert_equal(t1.get_internal_counts_by_height(0), np.array([0,0,0,0]))
        np.testing.assert_equal(t1.get_leaf_counts_by_height(1), np.array([2]))
        np.testing.assert_equal(t1.get_internal_counts_by_height(1), np.array([0]))
        np.testing.assert_equal(t1.get_leaf_counts_by_height(2), np.array([2]))
        np.testing.assert_equal(t1.get_internal_counts_by_height(2), np.array([1]))

        np.testing.assert_equal(t2.get_leaf_counts_by_height(0), np.array([0,0,0,0,0]))
        np.testing.assert_equal(t2.get_internal_counts_by_height(0), np.array([0,0,0,0,0]))
        np.testing.assert_equal(t2.get_leaf_counts_by_height(1), np.array([2]))
        np.testing.assert_equal(t2.get_internal_counts_by_height(1), np.array([0]))
        np.testing.assert_equal(t2.get_leaf_counts_by_height(2), np.array([1]))
        np.testing.assert_equal(t2.get_internal_counts_by_height(2), np.array([1]))
        np.testing.assert_equal(t2.get_leaf_counts_by_height(3), np.array([2]))
        np.testing.assert_equal(t2.get_internal_counts_by_height(3), np.array([1]))

        np.testing.assert_equal(t3.get_leaf_counts_by_height(0), np.array([0,0,0,0,0,0]))
        np.testing.assert_equal(t3.get_internal_counts_by_height(0), np.array([0,0,0,0,0,0]))
        np.testing.assert_equal(t3.get_leaf_counts_by_height(1), np.array([2,2]))
        np.testing.assert_equal(t3.get_internal_counts_by_height(1), np.array([0,0]))
        np.testing.assert_equal(t3.get_leaf_counts_by_height(2), np.array([1,1]))
        np.testing.assert_equal(t3.get_internal_counts_by_height(2), np.array([1,1]))
        np.testing.assert_equal(t3.get_leaf_counts_by_height(3), np.array([0]))
        np.testing.assert_equal(t3.get_internal_counts_by_height(3), np.array([2]))

        np.testing.assert_equal(t4.get_leaf_counts_by_height(0), np.array([0,0,0,0]))
        np.testing.assert_equal(t4.get_internal_counts_by_height(0), np.array([0,0,0,0]))
        np.testing.assert_equal(t4.get_leaf_counts_by_height(1), np.array([4]))
        np.testing.assert_equal(t4.get_internal_counts_by_height(1), np.array([0]))

    # tests if values from external tensors are correctly reads
    def test_get_values_by_height(self): 
        # nodes are sorted by depth 
        t = TreeHandler.read("test/data/simple.tree")
        branch_lens = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
        np.testing.assert_equal(t.get_values_by_height(branch_lens, 0), np.array([0.1, 0.5, 0.3, 0.7]))
        np.testing.assert_equal(t.get_values_by_height(branch_lens, 1), np.array([0.2]))
        t2 = TreeHandler.read("test/data/simple2.tree")
        branch_lens2 = np.array([0.4, 0.6, 0.1, 0.5, 0.3, 0.7, 0.2])
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 0), np.array([0.4, 0.6, 0.1, 0.5, 0.3]))
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 1), np.array([0.7]))
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 2), np.array([0.2]))
        t3 = TreeHandler.read("test/data/star.tree")
        branch_lens3 = np.array([0.4, 0.6, 0.1, 0.5])
        np.testing.assert_equal(t3.get_values_by_height(branch_lens3, 0), branch_lens3)


    # tests if the branch lengths from the tree files are read correctly
    def test_branch_lengths(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.branch_lengths[:,0], np.array([.1, .2, .3, .4, .5]))
        np.testing.assert_equal(t2.branch_lengths[:,0], np.array([.1, .2, .3, .1, .1, .4, .5]))
        np.testing.assert_equal(t3.branch_lengths[:,0], np.array([.1, .2, .4, .9, .6, .7, .3, .8, .5, 1.]))
        np.testing.assert_equal(t4.branch_lengths[:,0], np.array([.1, .2, .4, .1]))
        np.testing.assert_equal(t1.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .3, .4]))
        np.testing.assert_equal(t1.get_branch_lengths_by_height(1)[:,0], np.array([.5]))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .3, .1, .1]))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(1)[:,0], np.array([.4]))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(2)[:,0], np.array([.5]))
        np.testing.assert_equal(t4.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .4, .1]))


    def test_parent_indices(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.parent_indices, np.array([5, 5, 4, 4, 5]))
        np.testing.assert_equal(t2.parent_indices, np.array([7, 7,  6, 5, 5, 6, 7]))
        np.testing.assert_equal(t3.parent_indices, np.array([6, 6, 8, 9, 7, 7, 8, 9, 10, 10]))
        np.testing.assert_equal(t4.parent_indices, np.array([4, 4, 4, 4]))



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
    
    # computes the correct likelihood of a star-shaped tree when
    # a Jukes-Cantor model is used and
    # character z is at the root
    def get_ref_star(self, z, obs_at_leaves, t = [0.1, 0.2, 0.4, 0.1], mue=4./3):
        p = []
        for i,x in enumerate(obs_at_leaves):
            if x == z:
                p.append(1./4 + 3./4 * np.exp(-mue*t[i]))
            else:
                p.append(1./4 - 1./4 * np.exp(-mue*t[i]))
        return np.prod(p)
    

    def make_jukes_cantor_rate_matrix(self, mue=4./3):
        d = -3./4*mue
        x = mue/4
        rate_matrix = np.array([[[d, x, x, x], 
                                 [x, d, x, x], 
                                 [x, x, d, x], 
                                 [x, x, x, d]]])
        return rate_matrix


    def get_star_inputs_refs(self):
        t = TreeHandler.read("test/data/star.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
        #compute reference values
        refs = np.array([[self.get_ref_star(i, leaves[:,j]) for i in range(4)] for j in range(3)])
        # one-hot encode the leaves
        leaves = np.eye(4)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']
        rate_matrix = self.make_jukes_cantor_rate_matrix()
        return leaves, leaf_names, t, rate_matrix, refs


    def test_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()

        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(leaves, leaf_names, t, rate_matrix, 
                                                    leaves_are_probabilities=True,
                                                    return_probabilities=True)
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_likelihood_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()
        L = model.loglik(leaves, leaf_names, t, rate_matrix, 
                        equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]),
                        leaves_are_probabilities=True)
        self.assertEqual(L.shape, (1,3))
        np.testing.assert_almost_equal(L[0], np.log(np.sum(refs, -1)/4))


    def test_anc_probs_star_unordered_leaves(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()
        permuation = [2,1,0,3]
        leaves = leaves[permuation]
        leaf_names = [leaf_names[i] for i in permuation]
        X = model.compute_ancestral_probabilities(leaves, leaf_names, t, rate_matrix, 
                                                    leaves_are_probabilities=True,
                                                    return_probabilities=True)
        np.testing.assert_almost_equal(X[-1,0], refs)
    

    # computes the correct likelihood of simple3.tree when 
    # a Jukes-Cantor model with mue=4/3 is used 
    # returns a vector of likelihoods for each character at the root K
    #           K
    #        /    \
    #       I      J
    #      / \    / \
    #     G   C  D   H
    #    / \        / \
    #    A B       E  F
    def get_ref_simple3(self, obs_at_leaves):
        # A B C D E F given
        mue=4./3

        G = [self.get_ref_star(i, obs_at_leaves[:2], [0.1, 0.2]) for i in range(4)]
        H = [self.get_ref_star(i, obs_at_leaves[-2:], [0.6, 0.7]) for i in range(4)]
        
        def get_transitions(t):
            Pii = np.eye(4) * (1./4 + 3./4 * np.exp(-mue*t))
            Pij = (1-np.eye(4)) * (1./4 - 1./4 * np.exp(-mue*t))
            return Pii + Pij
        
        I1 = np.dot(get_transitions(0.3), G) #need matrix transpose if not symmetric
        I2 = np.array([self.get_ref_star(i, obs_at_leaves[2:3], [0.4]) for i in range(4)])
        I = I1 * I2

        J = np.dot(get_transitions(0.8), H)
        J *= np.array([self.get_ref_star(i, obs_at_leaves[3:4], [0.9]) for i in range(4)])

        K = np.dot(get_transitions(0.5), I)
        K *= np.dot(get_transitions(1.), J)

        return K
    

    def get_simple3_inputs_refs(self):
        t = TreeHandler.read("test/data/simple3.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,1,3,3,1], [1,1,0,2,2], [2,1,0,3,0], 
                           [3,1,2,2,3], [0,1,2,3,0], [0,1,1,2,0]])
        #compute reference values
        refs = np.array([self.get_ref_simple3(leaves[:,j]) for j in range(5)])
        # one-hot encode the leaves
        leaves = np.eye(4)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D', 'E', 'F']
        rate_matrix = np.array([[[-1., 1./3, 1./3, 1./3], 
                                 [1./3, -1, 1./3, 1./3], 
                                 [1./3, 1./3, -1, 1./3], 
                                 [1./3, 1./3, 1./3, -1]]])
        return leaves, leaf_names, t, rate_matrix, refs


    def test_anc_probs_simple3(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_simple3_inputs_refs()
        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(leaves, leaf_names, t, rate_matrix, 
                                                    leaves_are_probabilities=True,
                                                    return_probabilities=True)
        self.assertEqual(X.shape, (5,1,5,4))  
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_multi_model_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()

        # test model dimension with Jukes Cantor for other choices of mue
        mues = [1., 2.]
        rate_matrices = [self.make_jukes_cantor_rate_matrix(mue=mue) for mue in mues]
        refs2 = np.array([[[self.get_ref_star(i, np.argmax(leaves[:,0,j], -1), mue=mue) 
                                for i in range(4)] for j in range(leaves.shape[2])] for mue in mues])
        refs_full = np.concatenate([refs[np.newaxis], refs2], axis=0)
        branch_lengths = t.branch_lengths

        # 1. no broadcasting for rates and leaves
        rate_matrices_full = np.concatenate([rate_matrix] + rate_matrices, axis=0)
        branch_lengths_full = np.concatenate([branch_lengths]*3, axis=1)
        t.set_branch_lengths(branch_lengths_full)
        leaves_full = np.concatenate([leaves]*3, axis=1)

        X_no_broadcast = model.compute_ancestral_probabilities(leaves_full, leaf_names, t, rate_matrices_full, 
                                                               leaves_are_probabilities=True,
                                                               return_probabilities=True)
        np.testing.assert_almost_equal(X_no_broadcast[-1], refs_full)

        # 2. broadcasting for leaves
        X_broadcast_leaves = model.compute_ancestral_probabilities(leaves, leaf_names, t, rate_matrices_full, 
                                                                    leaves_are_probabilities=True,
                                                                    return_probabilities=True)
        np.testing.assert_almost_equal(X_broadcast_leaves[-1], refs_full)

        # 3. broadcasting for rates
        X_broadcast_rates = model.compute_ancestral_probabilities(leaves_full, leaf_names, t, rate_matrix, 
                                                                 leaves_are_probabilities=True,
                                                                 return_probabilities=True)
        np.testing.assert_almost_equal(X_broadcast_rates[-1], np.stack([refs]*3))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #run unit tests on CPU for speed and to avoid interference 
import unittest
import numpy as np
import tensorflow as tf
import torch
from tensortree.tree_handler import TreeHandler
from tensortree import model, util, tree_handler, substitution_models


class TestTree(unittest.TestCase):
    
    def setUp(self):
        util.set_backend("tensorflow")  

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
        branch_lens = np.array([0.1, 0.5, 0.3, 0.7, 0.2], dtype=util.default_dtype)
        np.testing.assert_equal(t.get_values_by_height(branch_lens, 0), np.array([0.1, 0.5, 0.3, 0.7], dtype=util.default_dtype))
        np.testing.assert_equal(t.get_values_by_height(branch_lens, 1), np.array([0.2], dtype=util.default_dtype))
        t2 = TreeHandler.read("test/data/simple2.tree")
        branch_lens2 = np.array([0.4, 0.6, 0.1, 0.5, 0.3, 0.7, 0.2], dtype=util.default_dtype)
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 0), np.array([0.4, 0.6, 0.1, 0.5, 0.3], dtype=util.default_dtype))
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 1), np.array([0.7], dtype=util.default_dtype))
        np.testing.assert_equal(t2.get_values_by_height(branch_lens2, 2), np.array([0.2], dtype=util.default_dtype))
        t3 = TreeHandler.read("test/data/star.tree")
        branch_lens3 = np.array([0.4, 0.6, 0.1, 0.5], dtype=util.default_dtype)
        np.testing.assert_equal(t3.get_values_by_height(branch_lens3, 0), branch_lens3)


    # tests if the branch lengths from the tree files are read correctly
    def test_branch_lengths(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.branch_lengths[:,0], np.array([.1, .2, .3, .4, .5], dtype=util.default_dtype))
        np.testing.assert_equal(t2.branch_lengths[:,0], np.array([.1, .2, .3, .1, .1, .4, .5], dtype=util.default_dtype))
        np.testing.assert_equal(t3.branch_lengths[:,0], np.array([.1, .2, .4, .9, .6, .7, .3, .8, .5, 1.], dtype=util.default_dtype))
        np.testing.assert_equal(t4.branch_lengths[:,0], np.array([.1, .2, .4, .1], dtype=util.default_dtype))
        np.testing.assert_equal(t1.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .3, .4], dtype=util.default_dtype))
        np.testing.assert_equal(t1.get_branch_lengths_by_height(1)[:,0], np.array([.5], dtype=util.default_dtype))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .3, .1, .1], dtype=util.default_dtype))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(1)[:,0], np.array([.4], dtype=util.default_dtype))
        np.testing.assert_equal(t2.get_branch_lengths_by_height(2)[:,0], np.array([.5], dtype=util.default_dtype))
        np.testing.assert_equal(t4.get_branch_lengths_by_height(0)[:,0], np.array([.1, .2, .4, .1], dtype=util.default_dtype))


    def test_parent_indices(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(t1.parent_indices, np.array([5, 5, 4, 4, 5]))
        np.testing.assert_equal(t2.parent_indices, np.array([7, 7,  6, 5, 5, 6, 7]))
        np.testing.assert_equal(t3.parent_indices, np.array([6, 6, 8, 9, 7, 7, 8, 9, 10, 10]))
        np.testing.assert_equal(t4.parent_indices, np.array([4, 4, 4, 4]))


    def test_newick_strings(self):
        tree_file = "test/data/simple.tree"
        t = TreeHandler.read(tree_file)
        self.assertEqual(t.to_newick().strip(), 
                         "(A:0.10000,B:0.20000,(C:0.30000,D:0.40000):0.50000):0.00000;")
        
    
    def test_split_node(self):
        t = TreeHandler.read("test/data/simple.tree")
        t.split("C", n=2, branch_length=0.1)
        t.update()
        np.testing.assert_equal(t.layer_sizes, np.array([5,1,1,1]))


    def test_collapse_node(self):
        t = TreeHandler.read("test/data/simple.tree")
        t.collapse("C")
        t.update()
        np.testing.assert_equal(t.layer_sizes, np.array([3,1,1]))
        np.testing.assert_equal(t.branch_lengths[:,0], np.array([.1, .2, .4, .5], dtype=util.default_dtype))


    def test_prune(self):
        for tree_file in ["test/data/simple.tree", 
                          "test/data/simple2.tree", 
                          "test/data/simple3.tree",
                          "test/data/star.tree"]:
            t = TreeHandler.read(tree_file)
            height_before = t.height
            layer_1_indices = t.get_parent_indices_by_height(0)
            names = [t.node_names[i] for i in layer_1_indices]
            t.prune()
            t.update()
            self.assertEqual(t.height, height_before-1)
            # after pruning, the indices of the previous height layer 1 (now 0), 
            # should be in the same order
            for i, name in zip(layer_1_indices, names):
                self.assertEqual(t.get_index(name), i-len(names))


class TestBackend():

    def _test_branch_lengths(self, backend, decode=False):
        kernel = np.array([[-3., -1., 0.], [1., 2., 3.]])
        branch_lengths = backend.make_branch_lengths(kernel)
        if decode:
            branch_lengths = branch_lengths.numpy()
        self.assertTrue(np.all(branch_lengths > 0.))


    def _test_exchangeability_matrix(self):
        kernel = np.array([[[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]]])
        R = backend.make_symmetric_pos_semidefinite(kernel)
        _check_symmetry_and_zero_diagonal(self, R[0])


    def _test_rate_matrix(self, backend, decode=False):
        # 3 jukes cantor models
        exchangeabilities, equilibrium = substitution_models.jukes_cantor(mue = [1., 2., 5.], d=3)
        rate_matrix = backend.make_rate_matrix(exchangeabilities, equilibrium)
        #since the matrix is normalized, all mue should yield the same result
        if decode:
            rate_matrix = rate_matrix.numpy()
        ref = np.array([[[-1., .5, .5], [.5, -1, .5], [.5, .5, -1]]]*3)
        np.testing.assert_allclose(rate_matrix, ref)


    def _test_LG_rate_matrix(self, backend):
        R, p = substitution_models.LG()
        Q = backend.make_rate_matrix(R[None], p[None])
        for i in range(20):
            for j in range(20):
                np.testing.assert_almost_equal(Q[0,i,j] * p[i], 
                                               Q[0,j,i] * p[j])


    def _test_transition_probs(self, backend, decode=False):
        # 3 jukes cantor models
        exchangeabilities = [[[0., mue, mue], [mue, 0., mue], [mue, mue, 0.]] for mue in [1., 2., 5.]]
        equilibrium = np.ones((3,3)) / 3.
        rate_matrix = backend.make_rate_matrix(exchangeabilities, equilibrium)
        P = backend.make_transition_probs(rate_matrix, np.ones((1, 1)))
        if decode:
            P = P.numpy()
        # test if matrix is probabilistic
        np.testing.assert_almost_equal(np.sum(P, -1), 1.) 
        # unit time, so expect 1 mutation per site
        mut_prob = np.sum(P * (1-np.eye(3)), -1)
        number_of_expected_mutations = - 2./3 * np.log( 1 - 3./2 * mut_prob)
        np.testing.assert_almost_equal(number_of_expected_mutations, 1.) 

    def test_root(self):
        t = TreeHandler.read("test/data/simple3.tree")
        t.change_root("H")
        ref = "((((A:0.10000,B:0.20000):0.30000,C:0.40000):1.50000,D:0.90000):0.80000,E:0.60000,F:0.70000):0.00000;"
        self.assertEqual(t.to_newick().strip(), ref)



class TestBackendTF(unittest.TestCase, TestBackend):

    def test_branch_lengths(self):
        from tensortree.backend_tf import BackendTF
        self._test_branch_lengths(BackendTF())

    def test_rate_matrix(self):
        from tensortree.backend_tf import BackendTF
        self._test_rate_matrix(BackendTF())

    def test_LG_rate_matrix(self):
        from tensortree.backend_tf import BackendTF
        self._test_LG_rate_matrix(BackendTF())
    
    def test_transition_probs(self):
        from tensortree.backend_tf import BackendTF
        self._test_transition_probs(BackendTF())
    

class TestBackendPytorch(unittest.TestCase, TestBackend):

    def test_branch_lengths(self):
        from tensortree.backend_pytorch import BackendTorch
        self._test_branch_lengths(BackendTorch(), decode=True)

    def test_rate_matrix(self):
        from tensortree.backend_pytorch import BackendTorch
        self._test_rate_matrix(BackendTorch(), decode=True)

    def test_LG_rate_matrix(self):
        from tensortree.backend_pytorch import BackendTorch
        self._test_LG_rate_matrix(BackendTorch())

    def test_transition_probs(self):
        from tensortree.backend_pytorch import BackendTorch
        self._test_transition_probs(BackendTorch(), decode=True)



class TestModelTF(unittest.TestCase):

    def setUp(self):
        util.set_backend("tensorflow")
    
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
        return np.prod(p).astype(util.default_dtype)


    def get_star_inputs_refs(self):
        t = TreeHandler.read("test/data/star.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
        #compute reference values
        refs = np.array([[self.get_ref_star(i, leaves[:,j]) 
                          for i in range(4)] 
                            for j in range(3)], dtype=util.default_dtype)
        # one-hot encode the leaves
        leaves = np.eye(4, dtype=util.default_dtype)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']
        R, pi = substitution_models.jukes_cantor(4./3)
        rate_matrix = model.backend.make_rate_matrix(R, pi)
        return leaves, leaf_names, t, rate_matrix, refs


    def test_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()

        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(leaves, t, rate_matrix, t.branch_lengths, leaf_names,
                                                    leaves_are_probabilities=True,
                                                    return_probabilities=True)
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_likelihood_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()
        L = model.loglik(leaves, t, rate_matrix, t.branch_lengths, 
                        equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]).astype(util.default_dtype),
                        leaf_names=leaf_names,
                        leaves_are_probabilities=True)
        self.assertEqual(L.shape, (1,3))
        np.testing.assert_almost_equal(L[0], np.log(np.sum(refs, -1)/4))


    def test_anc_probs_star_unordered_leaves(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()
        permuation = [2,1,0,3]
        leaves = leaves[permuation]
        leaf_names = [leaf_names[i] for i in permuation]
        X = model.compute_ancestral_probabilities(leaves, t, rate_matrix, t.branch_lengths, leaf_names,
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
        
        I1 = np.dot(get_transitions(0.3), G) 
        I2 = np.array([self.get_ref_star(i, obs_at_leaves[2:3], [0.4]) for i in range(4)])
        I = I1 * I2

        J = np.dot(get_transitions(0.8), H)
        J *= np.array([self.get_ref_star(i, obs_at_leaves[3:4], [0.9]) for i in range(4)])

        K = np.dot(get_transitions(0.5), I)
        K *= np.dot(get_transitions(1.), J)

        return K.astype(util.default_dtype)
    

    def get_simple3_inputs_refs(self):
        t = TreeHandler.read("test/data/simple3.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,1,3,3,1], [1,1,0,2,2], [2,1,0,3,0], 
                           [3,1,2,2,3], [0,1,2,3,0], [0,1,1,2,0]])
        #compute reference values
        refs = np.array([self.get_ref_simple3(leaves[:,j]) for j in range(5)])
        # one-hot encode the leaves
        leaves = np.eye(4, dtype=util.default_dtype)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D', 'E', 'F']
        rate_matrix = np.array([[[-1., 1./3, 1./3, 1./3], 
                                 [1./3, -1, 1./3, 1./3], 
                                 [1./3, 1./3, -1, 1./3], 
                                 [1./3, 1./3, 1./3, -1]]], dtype=util.default_dtype)
        return leaves, leaf_names, t, rate_matrix, refs


    def test_anc_probs_simple3(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_simple3_inputs_refs()
        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(leaves, t, rate_matrix, t.branch_lengths, leaf_names,
                                                    leaves_are_probabilities=True, return_probabilities=True)
        self.assertEqual(X.shape, (5,1,5,4))  
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_anc_probs_just_root(self):
        tree = TreeHandler() # root only
        leaves_ind = np.array([[0,1,3,3,1]])
        leaves = np.eye(4, dtype=util.default_dtype)[leaves_ind]
        leaves = leaves[:,np.newaxis]
        _,_,_,rate_matrix,_ = self.get_simple3_inputs_refs()

        # this should throw an assertion error as there are no internal nodes
        catched = False
        try:
            X = model.compute_ancestral_probabilities(leaves, tree, rate_matrix, tree.branch_lengths,
                                                        leaves_are_probabilities=True, return_probabilities=True)
        except AssertionError:
            catched = True
        self.assertTrue(catched)
        
        # this should work
        X = model.loglik(leaves, tree, rate_matrix, tree.branch_lengths, 
                        equilibrium_logits=np.log([[1./10, 2./10, 3./10, 4./10]]))
        refs = np.array([np.log(float(i+1)/10) for i in leaves_ind[0]])
        np.testing.assert_almost_equal(X[0], refs, decimal=5)


    def test_multi_model_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, refs = self.get_star_inputs_refs()

        # test model dimension with Jukes Cantor for other choices of mue
        mues = [1., 2.]
        R, pi = substitution_models.jukes_cantor(mues) 
        rate_matrices = model.backend.make_rate_matrix(R, pi, normalized=False)
        refs2 = np.array([[[self.get_ref_star(i, np.argmax(leaves[:,0,j], -1), mue=mue) 
                                for i in range(4)] for j in range(leaves.shape[2])] for mue in mues])
        refs_full = np.concatenate([refs[np.newaxis], refs2], axis=0)
        branch_lengths = t.branch_lengths

        # 1. no broadcasting for rates and leaves
        rate_matrices_full = np.concatenate([rate_matrix, rate_matrices], axis=0)
        branch_lengths_full = np.concatenate([branch_lengths]*3, axis=1)
        t.set_branch_lengths(branch_lengths_full)
        leaves_full = np.concatenate([leaves]*3, axis=1)

        X_no_broadcast = model.compute_ancestral_probabilities(leaves_full, 
                                                               t, 
                                                               rate_matrices_full, 
                                                               t.branch_lengths, 
                                                               leaf_names,
                                                               leaves_are_probabilities=True, 
                                                               return_probabilities=True)
        np.testing.assert_almost_equal(X_no_broadcast[-1], refs_full)

        # 2. broadcasting for leaves
        X_broadcast_leaves = model.compute_ancestral_probabilities(leaves, 
                                                                   t, 
                                                                   rate_matrices_full, 
                                                                   t.branch_lengths, 
                                                                   leaf_names, 
                                                                    leaves_are_probabilities=True, 
                                                                    return_probabilities=True)
        np.testing.assert_almost_equal(X_broadcast_leaves[-1], refs_full)

        # 3. broadcasting for rates
        X_broadcast_rates = model.compute_ancestral_probabilities(leaves_full, 
                                                                  t, 
                                                                  rate_matrix,  
                                                                  t.branch_lengths, 
                                                                  leaf_names,
                                                                 leaves_are_probabilities=True, 
                                                                 return_probabilities=True)
        np.testing.assert_almost_equal(X_broadcast_rates[-1], np.stack([refs]*3))


    def test_marginals_star(self):
        leaves, leaf_names, t, rate_matrix, ref = self.get_star_inputs_refs()
        marginals = model.compute_ancestral_marginals(leaves, 
                                                      t, 
                                                      rate_matrix, 
                                                      t.branch_lengths,
                                                      equilibrium_logits=np.log([[1./4]*4]),
                                                      leaf_names=leaf_names,
                                                      leaves_are_probabilities=True, 
                                                      return_probabilities=True)
        np.testing.assert_almost_equal(np.sum(marginals.numpy(), -1), 1., decimal=6)
        np.testing.assert_almost_equal(marginals[0,0], ref / np.sum(ref, axis=-1, keepdims=True), decimal=6)
        

    def test_marginals_simple3(self):
        leaves, leaf_names, t, rate_matrix, _ = self.get_simple3_inputs_refs()
        marginals = model.compute_ancestral_marginals(leaves, 
                                                        t, 
                                                        rate_matrix, 
                                                        t.branch_lengths,
                                                        equilibrium_logits=np.log([[1./4]*4]),
                                                        leaf_names=leaf_names,
                                                        leaves_are_probabilities=True, 
                                                        return_probabilities=True)
        
        np.testing.assert_almost_equal(np.sum(marginals.numpy(), -1), 1., decimal=6)

        # root is already correct, nice
        np.testing.assert_almost_equal(marginals[0,0,0], [0.62048769, 0.28582751, 0.0721368 , 0.02154799], decimal=5)
        np.testing.assert_almost_equal(marginals[1,0,0], [0.81955362, 0.0523807 , 0.05419097, 0.07387471], decimal=5)
        np.testing.assert_almost_equal(marginals[2,0,0], [0.35317405, 0.17935929, 0.39939567, 0.06807099], decimal=5)
        np.testing.assert_almost_equal(marginals[3,0,0], [0.36416282, 0.13638493, 0.15399601, 0.34545625], decimal=5)
        np.testing.assert_almost_equal(marginals[4,0,0], [0.32532723, 0.19558108, 0.30170421, 0.17738748], decimal=6)

        # test a rotation to a new root, it must not affect the results
        # note that after rotating, the original root "K" disappears from the tree since it has only one child
        t.change_root("H")
        marginals_H = model.compute_ancestral_marginals(leaves, 
                                                        t, 
                                                        rate_matrix, 
                                                        t.branch_lengths,
                                                        equilibrium_logits=np.log([[1./4]*4]),
                                                        leaf_names=leaf_names,
                                                        leaves_are_probabilities=True, 
                                                        return_probabilities=True)
        np.testing.assert_almost_equal(marginals_H[0,0,0], [0.62048769, 0.28582751, 0.0721368 , 0.02154799], decimal=5)
        np.testing.assert_almost_equal(marginals_H[1,0,0], [0.35317405, 0.17935929, 0.39939567, 0.06807099], decimal=5)
        np.testing.assert_almost_equal(marginals_H[2,0,0], [0.36416282, 0.13638493, 0.15399601, 0.34545625], decimal=5)
        np.testing.assert_almost_equal(marginals_H[3,0,0], [0.81955362, 0.0523807 , 0.05419097, 0.07387471], decimal=5)
    



class TestModelPytorch(TestModelTF):
    
    def setUp(self):
        util.set_backend("pytorch")


class TestGradientTF(unittest.TestCase):

    def setUp(self):
        util.set_backend("tensorflow")


    def get_star_inputs(self):
        t = TreeHandler.read("test/data/star.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
        # one-hot encode the leaves
        leaves = np.eye(4, dtype=util.default_dtype)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']
        R, pi = substitution_models.jukes_cantor(4./3)
        rate_matrix = model.backend.make_rate_matrix(R, pi)
        return leaves, leaf_names, t, rate_matrix
    

    def test_gradient_star(self):
        leaves, leaf_names, t, rate_matrix = self.get_star_inputs()
        # we want to differentiate the likelihood 
        # w.r.t. to leaves, branch lengths or rate matrix
        # create a variable for the branch lengths and initialize it with the
        # branch lengths parsed from the tree file
        B = tf.Variable(t.branch_lengths)
        # make the tree use the variable when computing the lilkehood
        t.set_branch_lengths(B) 

        X = tf.Variable(leaves)
        Q = tf.Variable(rate_matrix)

        # compute the likelihood and test if it can be differentiated
        # w.r.t. to the leaves, branch lengths and rate matrix
        with tf.GradientTape(persistent=True) as tape:
            L = model.loglik(X, t, Q, t.branch_lengths,
                            equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]),
                            leaf_names=leaf_names,
                            leaves_are_probabilities=True)
            
        # currently, we only test if the gradient can be computed without errors
        dL_dB = tape.gradient(L, B)
        dL_dQ = tape.gradient(L, Q)
        dL_dX = tape.gradient(L, X)

        self.assertTrue(not np.any(np.isnan(dL_dB.numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dQ.numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dX.numpy())))
        

class TestGradientPytorch(TestGradientTF):
    
    def setUp(self):
        util.set_backend("pytorch")  

    
    def test_gradient_star(self):
        leaves, leaf_names, t, rate_matrix = self.get_star_inputs()
        
        B = torch.nn.Parameter(torch.tensor(t.branch_lengths))
        # make the tree use the variable when computing the lilkehood
        t.set_branch_lengths(B) 

        X = torch.nn.Parameter(torch.tensor(leaves))
        Q = torch.nn.Parameter(rate_matrix.clone().detach())

        # compute the likelihood and test if it can be differentiated
        # w.r.t. to the leaves, branch lengths and rate matrix
        L = model.loglik(X, t, Q, t.branch_lengths,
                        equilibrium_logits=torch.log(torch.tensor([[1./4, 1./4, 1./4, 1./4]])),
                        leaf_names=leaf_names,
                        leaves_are_probabilities=True)
        
        # sum up, in pytorch grad can be implicitly created only for scalar outputs
        L = L.sum()
        
        # currently, we only test if the gradient can be computed without errors
        L.backward()
        dL_dB = B.grad
        dL_dQ = Q.grad
        dL_dX = X.grad

        self.assertTrue(not np.any(np.isnan(dL_dB.detach().numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dQ.detach().numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dX.detach().numpy())))



class TestSubstitutionModels(unittest.TestCase):

    def test_LG(self):
        R, pi = substitution_models.LG()
        self.assertEqual(R.shape, (20, 20))
        self.assertEqual(pi.shape, (20,))

        exchange_A_R = 0.425093
        exchange_D_N = 5.076149
        exchange_V_Y = 0.249313
        equi_A = 0.079066
        equi_D = 0.053052

        np.testing.assert_almost_equal(R[0,1], exchange_A_R)
        np.testing.assert_almost_equal(R[1,0], exchange_A_R)
        np.testing.assert_almost_equal(R[3,2], exchange_D_N)
        np.testing.assert_almost_equal(R[2,3], exchange_D_N)
        np.testing.assert_almost_equal(R[18,19], exchange_V_Y)
        np.testing.assert_almost_equal(R[19,18], exchange_V_Y)
        np.testing.assert_almost_equal(pi[0], equi_A)
        np.testing.assert_almost_equal(pi[3], equi_D)

        # test if different amino acids orders are correctly handled
        alternative_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        R, pi = substitution_models.LG(alphabet=alternative_alphabet)
        a = alternative_alphabet.index("A")
        r = alternative_alphabet.index("R")
        d = alternative_alphabet.index("D")
        n = alternative_alphabet.index("N")
        v = alternative_alphabet.index("V")
        y = alternative_alphabet.index("Y")

        np.testing.assert_almost_equal(R[a,r], exchange_A_R)
        np.testing.assert_almost_equal(R[r,a], exchange_A_R)
        np.testing.assert_almost_equal(R[d,n], exchange_D_N)
        np.testing.assert_almost_equal(R[n,d], exchange_D_N)
        np.testing.assert_almost_equal(R[v,y], exchange_V_Y)
        np.testing.assert_almost_equal(R[y,v], exchange_V_Y)
        np.testing.assert_almost_equal(pi[a], equi_A)
        np.testing.assert_almost_equal(pi[d], equi_D)

        _check_symmetry_and_zero_diagonal(self, R)
        _check_if_sums_to_one(self, pi)





# utility functions
def _check_symmetry_and_zero_diagonal(test_case : unittest.TestCase, matrix):
    np.testing.assert_almost_equal(matrix, np.transpose(matrix))
    np.testing.assert_almost_equal(np.diag(matrix), 0.)

def _check_if_sums_to_one(test_case : unittest.TestCase, matrix):
    np.testing.assert_almost_equal(np.sum(matrix, -1), 1., decimal=6)
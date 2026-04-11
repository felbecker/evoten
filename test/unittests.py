import os

#run unit tests on CPU for speed and to avoid interference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import unittest

import numpy as np
import tensorflow as tf
import torch
from Bio import SeqIO

from evoten import model, set_backend, substitution_models, util
from evoten.tree_handler import TreeHandler
from evoten.backend import backend


class TestTree(unittest.TestCase):

    def setUp(self):
        set_backend("tensorflow")

    def test_indices(self):
        # evotensorts nodes by height and left to right within a layer
        ind = TreeHandler.read("test/data/simple.tree").get_indices(
            ['A', 'B', 'C', 'D']
        )
        np.testing.assert_equal(ind, np.array([0,1,2,3]))
        ind2 = TreeHandler.read("test/data/simple2.tree").get_indices(
            ['A', 'B', 'C', 'D', 'E']
        )
        np.testing.assert_equal(ind2, np.array([0,1,2,3,4]))
        ind3 = TreeHandler.read("test/data/simple3.tree").get_indices(
            ['A', 'B', 'C', 'D', 'E', 'F']
        )
        np.testing.assert_equal(ind3, np.array([0,1,2,3,4,5]))
        ind4 = TreeHandler.read("test/data/star.tree").get_indices(
            ['A', 'B', 'C', 'D']
        )
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

        np.testing.assert_equal(
            t1.leaf_counts, np.array([0,0,0,0,2,2])
        )
        np.testing.assert_equal(
            t1.internal_counts, np.array([0,0,0,0,0,1])
        )
        np.testing.assert_equal(
            t2.leaf_counts, np.array([0,0,0,0,0,2,1,2])
        )
        np.testing.assert_equal(
            t2.internal_counts, np.array([0,0,0,0,0,0,1,1])
        )
        np.testing.assert_equal(
            t3.leaf_counts, np.array([0,0,0,0,0,0,2,2,1,1,0])
        )
        np.testing.assert_equal(
            t3.internal_counts, np.array([0,0,0,0,0,0,0,0,1,1,2])
        )
        np.testing.assert_equal(
            t4.leaf_counts, np.array([0,0,0,0,4])
        )
        np.testing.assert_equal(
            t4.internal_counts, np.array([0,0,0,0,0])
        )

        np.testing.assert_equal(
            t1.get_leaf_counts_by_height(0), np.array([0,0,0,0])
        )
        np.testing.assert_equal(
            t1.get_internal_counts_by_height(0), np.array([0,0,0,0])
        )
        np.testing.assert_equal(
            t1.get_leaf_counts_by_height(1), np.array([2])
        )
        np.testing.assert_equal(
            t1.get_internal_counts_by_height(1), np.array([0])
        )
        np.testing.assert_equal(
            t1.get_leaf_counts_by_height(2), np.array([2])
        )
        np.testing.assert_equal(
            t1.get_internal_counts_by_height(2), np.array([1])
        )

        np.testing.assert_equal(
            t2.get_leaf_counts_by_height(0), np.array([0,0,0,0,0])
        )
        np.testing.assert_equal(
            t2.get_internal_counts_by_height(0), np.array([0,0,0,0,0])
        )
        np.testing.assert_equal(
            t2.get_leaf_counts_by_height(1), np.array([2])
        )
        np.testing.assert_equal(
            t2.get_internal_counts_by_height(1), np.array([0]))
        np.testing.assert_equal(
            t2.get_leaf_counts_by_height(2), np.array([1]))
        np.testing.assert_equal(
            t2.get_internal_counts_by_height(2), np.array([1]))
        np.testing.assert_equal(
            t2.get_leaf_counts_by_height(3), np.array([2]))
        np.testing.assert_equal(
            t2.get_internal_counts_by_height(3), np.array([1]))

        np.testing.assert_equal(
            t3.get_leaf_counts_by_height(0), np.array([0,0,0,0,0,0])
        )
        np.testing.assert_equal(
            t3.get_internal_counts_by_height(0), np.array([0,0,0,0,0,0])
        )
        np.testing.assert_equal(
            t3.get_leaf_counts_by_height(1), np.array([2,2])
        )
        np.testing.assert_equal(
            t3.get_internal_counts_by_height(1), np.array([0,0])
        )
        np.testing.assert_equal(
            t3.get_leaf_counts_by_height(2), np.array([1,1])
        )
        np.testing.assert_equal(
            t3.get_internal_counts_by_height(2), np.array([1,1])
        )
        np.testing.assert_equal(
            t3.get_leaf_counts_by_height(3), np.array([0])
        )
        np.testing.assert_equal(
            t3.get_internal_counts_by_height(3), np.array([2])
        )

        np.testing.assert_equal(
            t4.get_leaf_counts_by_height(0), np.array([0,0,0,0])
        )
        np.testing.assert_equal(
            t4.get_internal_counts_by_height(0), np.array([0,0,0,0])
        )
        np.testing.assert_equal(
            t4.get_leaf_counts_by_height(1), np.array([4])
        )
        np.testing.assert_equal(
            t4.get_internal_counts_by_height(1), np.array([0])
        )

    # tests if values from external tensors are correctly reads
    def test_get_values_by_height(self):
        # nodes are sorted by depth
        t = TreeHandler.read("test/data/simple.tree")
        branch_lens = np.array(
            [0.1, 0.5, 0.3, 0.7, 0.2], dtype=util.default_dtype
        )
        np.testing.assert_equal(
            t.get_values_by_height(branch_lens, 0),
            np.array([0.1, 0.5, 0.3, 0.7], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t.get_values_by_height(branch_lens, 1),
            np.array([0.2], dtype=util.default_dtype)
        )
        t2 = TreeHandler.read("test/data/simple2.tree")
        branch_lens2 = np.array(
            [0.4, 0.6, 0.1, 0.5, 0.3, 0.7, 0.2], dtype=util.default_dtype
        )
        np.testing.assert_equal(
            t2.get_values_by_height(branch_lens2, 0),
            np.array([0.4, 0.6, 0.1, 0.5, 0.3], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.get_values_by_height(branch_lens2, 1),
            np.array([0.7], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.get_values_by_height(branch_lens2, 2),
            np.array([0.2], dtype=util.default_dtype)
        )
        t3 = TreeHandler.read("test/data/star.tree")
        branch_lens3 = np.array(
            [0.4, 0.6, 0.1, 0.5], dtype=util.default_dtype
        )
        np.testing.assert_equal(
            t3.get_values_by_height(branch_lens3, 0), branch_lens3
        )


    # tests if the branch lengths from the tree files are read correctly
    def test_branch_lengths(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(
            t1.branch_lengths[:,0],
            np.array([.1, .2, .3, .4, .5], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.branch_lengths[:,0],
            np.array([.1, .2, .3, .1, .1, .4, .5], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t3.branch_lengths[:,0],
            np.array(
                [.1, .2, .4, .9, .6, .7, .3, .8, .5, 1.],
                dtype=util.default_dtype
            )
        )
        np.testing.assert_equal(
            t4.branch_lengths[:,0],
            np.array([.1, .2, .4, .1], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t1.get_branch_lengths_by_height(0)[:,0],
            np.array([.1, .2, .3, .4], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t1.get_branch_lengths_by_height(1)[:,0],
            np.array([.5], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.get_branch_lengths_by_height(0)[:,0],
            np.array([.1, .2, .3, .1, .1], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.get_branch_lengths_by_height(1)[:,0],
            np.array([.4], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t2.get_branch_lengths_by_height(2)[:,0],
            np.array([.5], dtype=util.default_dtype)
        )
        np.testing.assert_equal(
            t4.get_branch_lengths_by_height(0)[:,0],
            np.array([.1, .2, .4, .1], dtype=util.default_dtype)
        )


    def test_parent_indices(self):
        t1 = TreeHandler.read("test/data/simple.tree")
        t2 = TreeHandler.read("test/data/simple2.tree")
        t3 = TreeHandler.read("test/data/simple3.tree")
        t4 = TreeHandler.read("test/data/star.tree")
        np.testing.assert_equal(
            t1.parent_indices, np.array([5, 5, 4, 4, 5])
        )
        np.testing.assert_equal(
            t2.parent_indices, np.array([7, 7,  6, 5, 5, 6, 7])
        )
        np.testing.assert_equal(
            t3.parent_indices, np.array([6, 6, 8, 9, 7, 7, 8, 9, 10, 10])
        )
        np.testing.assert_equal(
            t4.parent_indices, np.array([4, 4, 4, 4])
        )


    def test_newick_strings(self):
        tree_file = "test/data/simple.tree"
        t = TreeHandler.read(tree_file)
        self.assertEqual(
            t.to_newick().strip(),
            "(A:0.10000,B:0.20000,(C:0.30000,D:0.40000):0.50000):0.00000;"
        )


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
        np.testing.assert_equal(
            t.branch_lengths[:,0],
            np.array([.1, .2, .4, .5], dtype=util.default_dtype)
        )


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


    def test_stratify_branchlen_compact_kernel(self):
        """Compact (stratified) kernel gathering must match full-kernel slicing."""
        # simple3.tree has 10 branches with distinct lengths, giving enough
        # variety for stratification to produce a non-trivial mapping.
        t = TreeHandler.read("test/data/simple3.tree")
        num_branches = t.num_nodes - 1  # 10

        M, d = 3, 4
        rng = np.random.default_rng(0)

        K = 4  # fewer strata than branches → stratification is active
        t.stratify_branchlen(K)
        self.assertTrue(t.is_stratified)
        self.assertEqual(t.num_strata, K)

        # Build compact kernel: one row per stratum with arbitrary distinct values.
        compact_kernel = rng.random(
            (K, M, d, d), dtype=np.float64
        ).astype(util.default_dtype)

        # Derive the equivalent full kernel: each branch gets its stratum's row.
        # get_values_by_height on the full kernel (plain-slice path) must then
        # return the same result as the compact gather path.
        full_kernel = compact_kernel[t.branch_to_stratum]  # (num_branches, M, d, d)

        # For each height layer, compact gathering must reproduce full slicing.
        for h in range(t.height + 1):
            t.is_stratified = False
            full_slice = t.get_values_by_height(
                full_kernel, h, indexes_branches=True
            )
            t.is_stratified = True
            compact_slice = t.get_values_by_height(
                compact_kernel, h, indexes_branches=True
            )
            full_np   = np.array(full_slice)
            compact_np = np.array(compact_slice)
            self.assertEqual(full_np.shape, compact_np.shape,
                             msg=f"shape mismatch at height {h}")
            np.testing.assert_array_equal(
                compact_np, full_np,
                err_msg=f"compact kernel mismatch at height {h}"
            )

    def test_stratify_branchlen_noop_when_k_ge_branches(self):
        """stratify_branchlen must set is_stratified=False when K >= num_branches."""
        t = TreeHandler.read("test/data/simple3.tree")
        num_branches = t.num_nodes - 1
        t.stratify_branchlen(num_branches)      # exactly equal → no reduction
        self.assertFalse(t.is_stratified)
        t.stratify_branchlen(num_branches + 5)  # larger → no reduction
        self.assertFalse(t.is_stratified)

    def test_stratify_branchlen_invalid_kernel_raises(self):
        """get_values_by_height must raise ValueError for a kernel whose first
        dimension is neither num_strata nor num_branches."""
        t = TreeHandler.read("test/data/simple3.tree")
        t.stratify_branchlen(4)
        self.assertTrue(t.is_stratified)
        bad_kernel = np.zeros((7,), dtype=util.default_dtype)  # neither 4 nor 10
        with self.assertRaises(ValueError):
            t.get_values_by_height(bad_kernel, 0, indexes_branches=True)


class TestBackend():

    def _test_branch_lengths(self, backend, decode=False):
        kernel = np.array([[-3., -1., 0.], [1., 2., 3.]])
        branch_lengths = backend.make_branch_lengths(kernel)
        if decode:
            branch_lengths = branch_lengths.numpy()
        self.assertTrue(np.all(branch_lengths > 0.))


    def _test_exchangeability_matrix(self, backend):
        kernel = np.array([[[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]]])
        R = backend.make_symmetric_pos_semidefinite(kernel)
        _check_symmetry_and_zero_diagonal(self, R[0])


    def _test_rate_matrix(self, backend, decode=False):
        # 3 jukes cantor models
        exchangeabilities, equilibrium = substitution_models.jukes_cantor(
            mue = [1., 2., 5.], d=3
        )
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
        exchangeabilities = [
            [[0., mue, mue],
             [mue, 0., mue],
             [mue, mue, 0.]]
             for mue in [1., 2., 5.]
        ]
        equilibrium = np.ones((3,3)) / 3.
        rate_matrix = backend.make_rate_matrix(exchangeabilities, equilibrium)
        P = backend.make_transition_probs(rate_matrix, np.ones((1, 1)), equilibrium)
        if decode:
            P = P.numpy()
        # test if matrix is probabilistic
        np.testing.assert_almost_equal(np.sum(P, -1), 1.)
        # unit time, so expect 1 mutation per site
        mut_prob = np.sum(P * (1-np.eye(3)), -1)
        number_of_expected_mutations = - 2./3 * np.log( 1 - 3./2 * mut_prob)
        np.testing.assert_almost_equal(number_of_expected_mutations, 1.)

    def _test_traverse_branch(self, backend, decode=False):
        # Simple 3x3 transition probability matrix (stochastic)
        P = np.array([[[
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6]
        ]]], dtype=util.default_dtype)
        # Probabilities for 2 positions (shape: 1, 1, 2, 3)
        # at the upper node of a branch
        x = np.array([[[
            [0.1, 0.3, 0.2],
            [0.1, 0.8, 0.1]
        ]]], dtype=util.default_dtype)
        y = x @ np.transpose(P, (0,1,3,2))
        result = backend.traverse_branch(
            x, P, logarithmic = False,
        )
        result_log = backend.traverse_branch(
            np.log(x), P, logarithmic = True,
        )
        if decode:
            result = result.numpy()
            result_log = result_log.numpy()
        np.testing.assert_allclose(result, y, atol=1e-6)
        np.testing.assert_allclose(result_log, np.log(y), atol=1e-6)

    def _test_traverse_branch_transposed(self, backend, decode=False):
        # Simple 3x3 transition probability matrix (stochastic)
        P = np.array([[[
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6]
        ]]], dtype=util.default_dtype)
        # Probabilities for 2 positions (shape: 1, 1, 2, 3)
        # at the upper node of a branch
        x = np.array([[[
            [0.1, 0.3, 0.2],
            [0.1, 0.8, 0.1]
        ]]], dtype=util.default_dtype)
        y = x @ P
        result = backend.traverse_branch(
            x, P, logarithmic = False, transposed = True,
        )
        result_log = backend.traverse_branch(
            np.log(x), P, logarithmic = True, transposed = True,
        )
        if decode:
            result = result.numpy()
            result_log = result_log.numpy()
        np.testing.assert_allclose(result, y, atol=1e-6)
        np.testing.assert_allclose(result_log, np.log(y), atol=1e-6)

    def test_root(self):
        t = TreeHandler.read("test/data/simple3.tree")
        t.change_root("H")
        ref = "((((A:0.10000,B:0.20000):0.30000,C:0.40000):1.50000,D:0.90000)"\
            ":0.80000,E:0.60000,F:0.70000):0.00000;"
        self.assertEqual(t.to_newick().strip(), ref)



class TestBackendTF(unittest.TestCase, TestBackend):

    def test_branch_lengths(self):
        from evoten.backend_tf import BackendTF
        self._test_branch_lengths(BackendTF())

    def test_rate_matrix(self):
        from evoten.backend_tf import BackendTF
        self._test_rate_matrix(BackendTF())

    def test_LG_rate_matrix(self):
        from evoten.backend_tf import BackendTF
        self._test_LG_rate_matrix(BackendTF())

    def test_transition_probs(self):
        from evoten.backend_tf import BackendTF
        self._test_transition_probs(BackendTF())

    def test_traverse_branch(self):
        from evoten.backend_tf import BackendTF
        self._test_traverse_branch(BackendTF())

    def test_traverse_branch_transposed(self):
        from evoten.backend_tf import BackendTF
        self._test_traverse_branch_transposed(BackendTF())

    def test_transition_probs_ntr_vs_gtr(self):
        """GTR eigendecomp and NTR expm (Padé) must agree on GTR matrices."""
        import time
        import evoten
        evoten.set_backend("tensorflow")
        b = evoten.backend

        rng = np.random.default_rng(0)
        d, n_t = 20, 32

        # Random symmetric exchangeability matrix (1, d, d)
        R_raw = rng.exponential(1.0, size=(d, d)).astype(np.float64)
        R_raw = (R_raw + R_raw.T) / 2.
        np.fill_diagonal(R_raw, 0.)
        R = R_raw[np.newaxis]               # (1, d, d)

        # Random stationary distribution (1, d)
        pi_raw = rng.exponential(1.0, size=(1, d)).astype(np.float64)
        pi = pi_raw / pi_raw.sum(axis=-1, keepdims=True)  # (1, d)

        Q = b.make_rate_matrix(R, pi)       # (1, d, d)

        # 32 log-spaced time values covering 0.01 – 10 substitutions/site
        t = np.logspace(-2, 1, n_t).astype(np.float64)   # (32,)
        # broadcasting: Q (1,d,d) × t (32,) → P (32,d,d) for both methods

        # warm-up (TF traces on first call)
        _ = b.make_transition_probs(Q, t[:1], pi)
        _ = b.make_transition_probs_ntr(Q, t[:1])

        REPS = 5
        t0 = time.perf_counter()
        for _ in range(REPS):
            P_gtr = b.make_transition_probs(Q, t, pi).numpy()
        t1 = time.perf_counter()
        for _ in range(REPS):
            P_ntr = b.make_transition_probs_ntr(Q, t).numpy()
        t2 = time.perf_counter()

        print(f"\n[NTR test] GTR eigendecomp : {(t1-t0)/REPS*1e3:.1f} ms/call")
        print(f"[NTR test] NTR expm (Padé) : {(t2-t1)/REPS*1e3:.1f} ms/call")

        np.testing.assert_allclose(P_gtr, P_ntr, atol=1e-4,
            err_msg="GTR and NTR transition matrices diverge beyond tolerance")


class TestBackendPytorch(unittest.TestCase, TestBackend):

    def test_branch_lengths(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_branch_lengths(BackendTorch(), decode=True)

    def test_rate_matrix(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_rate_matrix(BackendTorch(), decode=True)

    def test_LG_rate_matrix(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_LG_rate_matrix(BackendTorch())

    def test_transition_probs(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_transition_probs(BackendTorch(), decode=True)

    def test_traverse_branch(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_traverse_branch(BackendTorch(), decode=True)

    def test_traverse_branch_transposed(self):
        from evoten.backend_pytorch import BackendTorch
        self._test_traverse_branch_transposed(BackendTorch(), decode=True)



class TestModelTF(unittest.TestCase):

    def setUp(self):
        set_backend("tensorflow")

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
        rate_matrix = backend.make_rate_matrix(R, pi)
        return leaves, leaf_names, t, rate_matrix, pi, refs


    def test_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, pi, refs = self.get_star_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = pi[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(
            leaves,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_likelihood_star(self):
        leaves, leaf_names, t, rate_matrix, pi, refs = self.get_star_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = pi[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        L = model.loglik(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]).astype(util.default_dtype),
            leaf_names=leaf_names,
            leaves_are_probabilities=True
        )
        self.assertEqual(L.shape, (1,3))
        np.testing.assert_allclose(
            L[0], np.log(np.sum(refs, -1)/4), atol=1e-6, rtol=1e-6
        )


    def test_likelihood_simple4(self):
        t = TreeHandler.read("test/data/simple4.tree")
        seqs = []

        with open("test/data/seq-gen.out") as f:
            for record in SeqIO.parse(f, "fasta"):
                seqs.append(str(record.seq))

        leaves = util.encode_one_hot(seqs, alphabet="ACGT")
        leaves = leaves[:, np.newaxis]

        R, pi = substitution_models.jukes_cantor(d = 4)
        Q = backend.make_rate_matrix(R, pi)
        B = np.ones_like(t.branch_lengths)

        transition_probs = backend.make_transition_probs(Q, B, pi)
        L = model.loglik(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]).astype(util.default_dtype),
        )

        # TODO: likelihood comparison
        #refs=...
        #self.assertEqual(L.shape, (1,3))
        #np.testing.assert_almost_equal(L[0], np.log(np.sum(refs, -1)/4))


    def test_anc_probs_star_unordered_leaves(self):
        leaves, leaf_names, t, rate_matrix, pi, refs = self.get_star_inputs_refs()
        permuation = [2,1,0,3]
        leaves = leaves[permuation]
        leaf_names = [leaf_names[i] for i in permuation]
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = pi[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        X = model.compute_ancestral_probabilities(
            leaves,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
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

        G = [
            self.get_ref_star(i, obs_at_leaves[:2], [0.1, 0.2])
            for i in range(4)
        ]
        H = [
            self.get_ref_star(i, obs_at_leaves[-2:], [0.6, 0.7])
            for i in range(4)
        ]

        def get_transitions(t):
            Pii = np.eye(4) * (1./4 + 3./4 * np.exp(-mue*t))
            Pij = (1-np.eye(4)) * (1./4 - 1./4 * np.exp(-mue*t))
            return Pii + Pij

        I1 = np.dot(get_transitions(0.3), G)
        I2 = np.array([
            self.get_ref_star(i, obs_at_leaves[2:3], [0.4]) for i in range(4)
        ])
        I = I1 * I2

        J = np.dot(get_transitions(0.8), H)
        J *= np.array([
            self.get_ref_star(i, obs_at_leaves[3:4], [0.9]) for i in range(4)
        ])

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
        equilibrium = np.array([[1./4, 1./4, 1./4, 1./4]], dtype=util.default_dtype)
        return leaves, leaf_names, t, rate_matrix, equilibrium, refs


    def test_anc_probs_simple3(self):
        leaves, leaf_names, t, rate_matrix, equilibrium, refs = self.get_simple3_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = equilibrium[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        # test if the ancestral probabilities are computed correctly
        X = model.compute_ancestral_probabilities(
            leaves,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True, return_probabilities=True
        )
        self.assertEqual(X.shape, (5,1,5,4))
        np.testing.assert_almost_equal(X[-1,0], refs)


    def test_anc_probs_just_root(self):
        tree = TreeHandler() # root only
        leaves_ind = np.array([[0,1,3,3,1]])
        leaves = np.eye(4, dtype=util.default_dtype)[leaves_ind]
        leaves = leaves[:,np.newaxis]
        _,_,_,rate_matrix,equilibrium,_ = self.get_simple3_inputs_refs()

        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        tree.branch_lengths = tree.branch_lengths[:, np.newaxis]
        equilibrium = equilibrium[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, tree.branch_lengths, equilibrium
        )

        # this should throw an assertion error as there are no internal nodes
        catched = False
        try:
            X = model.compute_ancestral_probabilities(
                leaves,
                tree,
                transition_probs,
                leaves_are_probabilities=True, return_probabilities=True
            )
        except AssertionError:
            catched = True
        self.assertTrue(catched)

        # this should work
        X = model.loglik(
            leaves,
            tree,
            transition_probs,
            equilibrium_logits=np.log([[1./10, 2./10, 3./10, 4./10]])
        )
        refs = np.array([np.log(float(i+1)/10) for i in leaves_ind[0]])
        np.testing.assert_almost_equal(X[0], refs, decimal=5)


    def test_multi_model_anc_probs_star(self):
        leaves, leaf_names, t, rate_matrix, pi, refs = self.get_star_inputs_refs()

        # test model dimension with Jukes Cantor for other choices of mue
        mues = [1., 2.]
        R, pi_multi = substitution_models.jukes_cantor(mues)
        rate_matrices = backend.make_rate_matrix(R, pi_multi, normalized=False)
        refs2 = np.array([[[
            self.get_ref_star(i, np.argmax(leaves[:,0,j], -1), mue=mue)
            for i in range(4)]
            for j in range(leaves.shape[2])]
            for mue in mues
        ])
        refs_full = np.concatenate([refs[np.newaxis], refs2], axis=0)
        branch_lengths = t.branch_lengths

        # 1. both transition_probs and leaves have a model dimension

        rate_matrices_full = np.concatenate([rate_matrix, rate_matrices], axis=0)
        equilibrium_full = np.concatenate([pi, pi_multi], axis=0)
        # broadcast in the node and the length dimension
        rate_matrices_full = rate_matrices_full[np.newaxis, :, np.newaxis]
        equilibrium_full = equilibrium_full[np.newaxis, :, np.newaxis]
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        equilibrium = pi[np.newaxis, :, np.newaxis]

        # broad cast in the length dimension
        branch_lengths_full = np.concatenate([branch_lengths]*3, axis=1)
        branch_lengths_full = branch_lengths_full[..., np.newaxis]

        t.set_branch_lengths(branch_lengths_full, update_phylo_tree=False)
        leaves_full = np.concatenate([leaves]*3, axis=1)

        # produce transition probs of shape (4, 3, 1, 4, 4)
        transition_probs = backend.make_transition_probs(
            rate_matrices_full, t.branch_lengths, equilibrium_full
        )
        X_no_broadcast = model.compute_ancestral_probabilities(
            leaves_full,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(X_no_broadcast[-1], refs_full)

        # 2. leaves do not have a model dimension and are broadcasted
        X_broadcast_leaves = model.compute_ancestral_probabilities(
            leaves,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(X_broadcast_leaves[-1], refs_full)

        # 3. rate_matrices do not have a model dimension and are broadcasted
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        X_broadcast_rates = model.compute_ancestral_probabilities(
            leaves_full,
            t,
            transition_probs,
            leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(
            X_broadcast_rates[-1], np.stack([refs]*3)
        )


    def test_marginals_star(self):
        leaves, leaf_names, t, rate_matrix, pi, ref = self.get_star_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = pi[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        marginals = model.compute_ancestral_marginals(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]),
            leaf_names=leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_allclose(
            np.sum(marginals.numpy(), -1), 1., atol=3e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            marginals[0,0],
            ref / np.sum(ref, axis=-1, keepdims=True),
            atol=3e-6, rtol=1e-6
        )


    def test_marginals_simple3(self):
        leaves, leaf_names, t, rate_matrix, equilibrium, _ = self.get_simple3_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = equilibrium[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, branch_lengths, equilibrium
        )
        marginals = model.compute_ancestral_marginals(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]),
            leaf_names=leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )

        np.testing.assert_almost_equal(
            np.sum(marginals.numpy(), -1), 1., decimal=6
        )

        # see "doc/Message passing on a tree.pdf" for the expected marginals
        np.testing.assert_almost_equal(
            marginals[0,0,0],
            [0.62048769, 0.28582751, 0.0721368 , 0.02154799],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals[1,0,0],
            [0.81955362, 0.0523807 , 0.05419097, 0.07387471],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals[2,0,0],
            [0.35317405, 0.17935929, 0.39939567, 0.06807099],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals[3,0,0],
            [0.36416282, 0.13638493, 0.15399601, 0.34545625],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals[4,0,0],
            [0.32532723, 0.19558108, 0.30170421, 0.17738748],
            decimal=6
        )

        # test a rotation to a new root, it must not affect the marginals
        # note that after rotating, the original root "K" disappears,
        # as it was bifurcating
        t.change_root("H")
        # recompute the transition matrices, as the node order has changed
        branch_lengths = t.branch_lengths[:, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, branch_lengths, equilibrium
        )
        marginals_H = model.compute_ancestral_marginals(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]),
            leaf_names=leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(
            marginals_H[0,0,0],
            [0.62048769, 0.28582751, 0.0721368 , 0.02154799],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals_H[1,0,0],
            [0.35317405, 0.17935929, 0.39939567, 0.06807099],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals_H[2,0,0],
            [0.36416282, 0.13638493, 0.15399601, 0.34545625],
            decimal=5
        )
        np.testing.assert_almost_equal(
            marginals_H[3,0,0],
            [0.81955362, 0.0523807 , 0.05419097, 0.07387471],
            decimal=5
        )


    def test_leaf_out_marginals_simple3(self):
        leaves, leaf_names, t, rate_matrix, equilibrium, _ = self.get_simple3_inputs_refs()
        rate_matrix = rate_matrix[np.newaxis, :, np.newaxis]
        t.branch_lengths = t.branch_lengths[:, np.newaxis]
        equilibrium = equilibrium[np.newaxis, :, np.newaxis]
        transition_probs = backend.make_transition_probs(
            rate_matrix, t.branch_lengths, equilibrium
        )
        leaf_out_marginals = model.compute_leaf_out_marginals(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log([[1./4]*4]),
            leaf_names=leaf_names,
            leaves_are_probabilities=True,
            return_probabilities=True
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[0,0,0],
            [0.07784672, 0.65521162, 0.18869241, 0.07824925],
            decimal=5
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[1,0,0],
            [0.7055906 , 0.07967477, 0.13374548, 0.08098915],
            decimal=5
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[2,0,0],
            [0.4170595 , 0.26266192, 0.1564732 , 0.16380537],
            decimal=5
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[3,0,0],
            [0.31507678, 0.2272743 , 0.23406292, 0.223586],
            decimal=5
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[4,0,0],
            [0.36942383, 0.2008257 , 0.20300842, 0.22674205],
            decimal=5
        )
        np.testing.assert_almost_equal(
            leaf_out_marginals[5,0,0],
            [0.37125943, 0.20210377, 0.20384607, 0.22279073],
            decimal=5
        )


    def test_propagate_simple(self):
        tree = TreeHandler.read("test/data/star.tree")
        rate_matrix = np.array(
            [[-1., 1./3, 1./3, 1./3],
            [1./3, -1, 1./3, 1./3],
            [1./3, 1./3, -1, 1./3],
            [1./3, 1./3, 1./3, -1]],
            dtype=util.default_dtype
        )
        # broadcasting to shape (1, 1, 1, 4, 4)
        # i.e. the same rate matrix is used for all branches (dim 0)
        # we use 1 model (dim 1)
        # we broadcast in the length dimension (dim 2)
        rate_matrix = rate_matrix[np.newaxis, np.newaxis, np.newaxis]

        # we have lengths for all branches for, again, 1 model
        # again, we broadcast in the length dimension
        tree.branch_lengths = tree.branch_lengths[:, np.newaxis]

        transition_probs = backend.make_transition_probs(
            rate_matrix, tree.branch_lengths, [1./4, 1./4, 1./4, 1./4]
        )

        root = np.eye(4, dtype=util.default_dtype)[np.newaxis, np.newaxis]

        dist = model.propagate(root, tree, transition_probs)
        mue = 4./3
        t = tree.branch_lengths[:,0]
        for i in range(4):
            pii = 1./4 + 3./4 * np.exp(-mue*t[i])
            pij = 1./4 - 1./4 * np.exp(-mue*t[i])
            ref = np.eye(4) * pii + (1-np.eye(4)) * pij
            np.testing.assert_allclose(dist[i,0], ref, atol=1e-6)


class TestGapPruningEquivTF(unittest.TestCase):
    """ Tests that the log-likelihood of a tree with all-gap leaves is zero.
    This can e.g. fail if P is not symmetric but not transposed right.
    """
    def setUp(self):
        set_backend("tensorflow")

    def test_gap_pruning_loglik_is_zero(self):
        t = TreeHandler.read("test/data/simple.tree")

        # Tree of height 4 with four leaves, one model and one alignment column.
        # All-gap leaves are represented as [1, 1, 1, 1], which should be
        # equivalent to pruning away each leaf/edge and therefore yield
        # log-likelihood 0 for every site.
        leaves = np.ones((4, 1, 1, 4), dtype=util.default_dtype)

        # Kimura 80 with strong transition bias and non-uniform pi
        kappa = np.array([10.0], dtype=util.default_dtype)
        pi = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=util.default_dtype)
        R = np.array(
            [[
                [0.0, 1.0, kappa[0], 1.0],
                [1.0, 0.0, 1.0, kappa[0]],
                [kappa[0], 1.0, 0.0, 1.0],
                [1.0, kappa[0], 1.0, 0.0],
            ]],
            dtype=util.default_dtype,
        )

        Q = backend.make_rate_matrix(R, pi)
        transition_probs = backend.make_transition_probs(
            Q,
            t.branch_lengths[:, np.newaxis],
            pi,
        )

        L = model.loglik(
            leaves,
            t,
            transition_probs,
            equilibrium_logits=np.log(pi).astype(util.default_dtype),
            leaf_names=['A', 'B', 'C', 'D'],
            leaves_are_probabilities=True,
        )

        self.assertEqual(L.shape, (1, 1))
        np.testing.assert_allclose(
            L,
            np.zeros((1, 1), dtype=util.default_dtype),
            atol=1e-6,
            rtol=1e-6,
        )



class TestModelPytorch(TestModelTF):

    def setUp(self):
        set_backend("pytorch")


class TestGradientTF(unittest.TestCase):

    def setUp(self):
        set_backend("tensorflow")


    def get_star_inputs(self):
        t = TreeHandler.read("test/data/star.tree")
        # leaves will have shape (num_leaves, L, models, d)
        leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
        # one-hot encode the leaves
        leaves = np.eye(4, dtype=util.default_dtype)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']
        R, pi = substitution_models.jukes_cantor(4./3)
        rate_matrix = backend.make_rate_matrix(R, pi)
        return leaves, leaf_names, t, rate_matrix, pi


    def test_gradient_star(self):
        leaves, leaf_names, t, rate_matrix, pi = self.get_star_inputs()
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
            transition_probs = backend.make_transition_probs(
                Q[tf.newaxis, :, tf.newaxis], t.branch_lengths[:, tf.newaxis], pi[tf.newaxis, :, tf.newaxis]
            )
            L = model.loglik(
                X,
                t,
                transition_probs,
                equilibrium_logits=np.log([[1./4, 1./4, 1./4, 1./4]]),
                leaf_names=leaf_names,
                leaves_are_probabilities=True
            )

        # currently, we only test if the gradient can be computed without errors
        dL_dB = tape.gradient(L, B)
        dL_dQ = tape.gradient(L, Q)
        dL_dX = tape.gradient(L, X)

        self.assertTrue(not np.any(np.isnan(dL_dB.numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dQ.numpy())))
        self.assertTrue(not np.any(np.isnan(dL_dX.numpy())))


class TestGradientPytorch(TestGradientTF):

    def setUp(self):
        set_backend("pytorch")


    def test_gradient_star(self):
        leaves, leaf_names, t, rate_matrix, pi = self.get_star_inputs()

        B = torch.nn.Parameter(torch.tensor(t.branch_lengths))
        # make the tree use the variable when computing the lilkehood
        t.set_branch_lengths(B)

        X = torch.nn.Parameter(torch.tensor(leaves))
        Q = torch.nn.Parameter(rate_matrix.clone().detach())

        # compute the likelihood and test if it can be differentiated
        # w.r.t. to the leaves, branch lengths and rate matrix
        transition_probs = backend.make_transition_probs(
            Q[np.newaxis, :, np.newaxis], t.branch_lengths[:, np.newaxis], pi[np.newaxis, :, np.newaxis]
        )
        L = model.loglik(
            X,
            t,
            transition_probs,
            equilibrium_logits=torch.log(torch.tensor([[1./4, 1./4, 1./4, 1./4]])),
            leaf_names=leaf_names,
            leaves_are_probabilities=True
        )

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

class EvotenModel(tf.keras.Model):
    """Optimizes the parameters (but not the topology) of a tree."""

    def __init__(
        self,
        tree_handler : TreeHandler,
        exchangeability_matrix : np.ndarray,
        branch_lengths : np.ndarray,
        equilibrium_frequencies : np.ndarray,
        train_rate_matrix : bool = False,
        train_branch_lengths : bool = True,
        train_equilibrium_frequencies : bool = False,
        leaf_names : list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.branch_lengths = branch_lengths.astype(np.float32)
        self.branch_lengths_init = backend.inverse_softplus(
            self.branch_lengths
        ).numpy()
        self.exchangeability_matrix = exchangeability_matrix.astype(np.float32)
        self.exchangeability_matrix_init = (
            backend.inverse_softplus(
                self.exchangeability_matrix
            ).numpy()
        )
        self.equilibrium_frequencies = equilibrium_frequencies.astype(np.float32)
        self.equilibrium_frequencies_init = tf.math.log(
                self.equilibrium_frequencies
        ).numpy()
        self.tree_handler = tree_handler
        self.train_rate_matrix = train_rate_matrix
        self.train_branch_lengths = train_branch_lengths
        self.train_equilibrium_frequencies = train_equilibrium_frequencies
        self.leaf_names = leaf_names

    def build(self, input_shape=None):
        # [num_leaves, num_models, seq_length]
        self.branch_lengths_kernel = self.add_weight(
            shape=self.branch_lengths.shape,
            initializer=tf.constant_initializer(self.branch_lengths_init),
            trainable=self.train_branch_lengths,
            name="branch_lengths_kernel",
        )
        # [num_leaves, num_models, seq_length, d, d]
        self.exchangeability_matrix_kernel = self.add_weight(
            shape=self.exchangeability_matrix.shape,
            initializer=tf.constant_initializer(self.exchangeability_matrix_init),
            trainable=self.train_rate_matrix,
            name="exchangeability_matrix_kernel",
        )
        # [num_models, d]
        self.equilibrium_frequencies_kernel = self.add_weight(
            shape=self.equilibrium_frequencies.shape,
            initializer=tf.constant_initializer(self.equilibrium_frequencies_init),
            trainable=self.train_equilibrium_frequencies,
            name="equilibrium_frequencies_kernel",
        )
        self.built = True

    def make_branch_lengths(self):
        return backend.make_branch_lengths(
            self.branch_lengths_kernel
        )

    def make_equilibrium_frequencies(self):
        return backend.make_equilibrium(
            self.equilibrium_frequencies_kernel
        )

    def make_exchangeability_matrix(self):
        return backend.make_symmetric_pos_semidefinite(
            self.exchangeability_matrix_kernel
        )

    def make_rate_matrix(self):
        return backend.make_rate_matrix(
            self.make_exchangeability_matrix(),
            self.make_equilibrium_frequencies(),
        )

    def loglik(self, inputs):
        # inputs: [num_leaves, num_models, seq_length, num_features]
        # outputs: [num_models, seq_length]
        transition_probs = backend.make_transition_probs(
            self.make_rate_matrix(),
            self.make_branch_lengths(),
            self.make_equilibrium_frequencies()
        )
        return model.loglik(
            inputs,
            self.tree_handler,
            transition_probs,
            self.equilibrium_frequencies_kernel,
            leaf_names=self.leaf_names
        )

    def call(self, inputs):
        # inputs: [num_leaves, num_models, seq_length, num_features]
        # outputs: [num_models, seq_length]
        return self.loglik(inputs)

    def compute_loss(self, x, y, y_pred, sample_weight):
        # average over models and sum over length
        y_pred = tf.reduce_mean(y_pred, axis=0)
        y_pred = tf.reduce_sum(y_pred)
        return -y_pred

class TestKerasModel(unittest.TestCase):

    def setUp(self):
        set_backend("tensorflow")

    def test_jit_compile_model(self):
        # Load a tree
        t = TreeHandler.read("test/data/star.tree")
        t.set_branch_lengths(np.ones((t.num_nodes-1, 1)))

        # Set up leaves of shape (num_leaves, L, models, d)
        leaves = np.array([[0,2,3], [1,1,0], [2,1,0], [3,1,2]])
        leaves = np.eye(4, dtype=util.default_dtype)[leaves]
        leaves = leaves[:,np.newaxis]
        leaf_names = ['A', 'B', 'C', 'D']

        # Initialize the model
        R, pi = substitution_models.jukes_cantor(4./3)
        model = EvotenModel(
            tree_handler=t,
            exchangeability_matrix=R[np.newaxis, :, :],
            branch_lengths=t.branch_lengths[:, np.newaxis],
            equilibrium_frequencies=pi[np.newaxis, :],
            train_rate_matrix=True,
            train_branch_lengths=True,
            train_equilibrium_frequencies=True,
            leaf_names=leaf_names
        )
        model.build()
        model.compile(jit_compile=True)

        # Test calling the model
        _loglik = model(leaves)

        # Test gradient computation
        model.fit(
            x=leaves,
            batch_size=leaves.shape[0],
            epochs=1,
            steps_per_epoch=1
        )


class TestUtil(unittest.TestCase):

    def test_parse_rate_model(self):
        R, pi, s = util.parse_rate_model("evoten/data/LG.model")
        R2, pi2, s2 = util.parse_rate_model("test/data/LG_with_factor.model")
        np.testing.assert_allclose(R, R2)
        np.testing.assert_allclose(pi, pi2)
        np.testing.assert_allclose(s, 1.0)
        np.testing.assert_allclose(s2, 0.7)
        np.testing.assert_allclose(
            R,
            np.array([
                [0.000000,0.425093,0.276818,0.395144,2.489084,0.969894,
                 1.038545,2.066040,0.358858,0.149830,0.395337,0.536518,
                 1.124035,0.253701,1.177651,4.727182,2.139501,0.180717,
                 0.218959,2.547870],
                [0.425093,0.000000,0.751878,0.123954,0.534551,2.807908,
                 0.363970,0.390192,2.426601,0.126991,0.301848,6.326067,
                 0.484133,0.052722,0.332533,0.858151,0.578987,0.593607,
                 0.314440,0.170887],
                [0.276818,0.751878,0.000000,5.076149,0.528768,1.695752,
                 0.541712,1.437645,4.509238,0.191503,0.068427,2.145078,
                 0.371004,0.089525,0.161787,4.008358,2.000679,0.045376,
                 0.612025,0.083688],
                [0.395144,0.123954,5.076149,0.000000,0.062556,0.523386,
                 5.243870,0.844926,0.927114,0.010690,0.015076,0.282959,
                 0.025548,0.017416,0.394456,1.240275,0.425860,0.029890,
                 0.135107,0.037967],
                [2.489084,0.534551,0.528768,0.062556,0.000000,0.084808,
                 0.003499,0.569265,0.640543,0.320627,0.594007,0.013266,
                 0.893680,1.105251,0.075382,2.784478,1.143480,0.670128,
                 1.165532,1.959291],
                [0.969894,2.807908,1.695752,0.523386,0.084808,0.000000,
                 4.128591,0.267959,4.813505,0.072854,0.582457,3.234294,
                 1.672569,0.035855,0.624294,1.223828,1.080136,0.236199,
                 0.257336,0.210332],
                [1.038545,0.363970,0.541712,5.243870,0.003499,4.128591,
                 0.000000,0.348847,0.423881,0.044265,0.069673,1.807177,
                 0.173735,0.018811,0.419409,0.611973,0.604545,0.077852,
                 0.120037,0.245034],
                [2.066040,0.390192,1.437645,0.844926,0.569265,0.267959,
                 0.348847,0.000000,0.311484,0.008705,0.044261,0.296636,
                 0.139538,0.089586,0.196961,1.739990,0.129836,0.268491,
                 0.054679,0.076701],
                [0.358858,2.426601,4.509238,0.927114,0.640543,4.813505,
                 0.423881,0.311484,0.000000,0.108882,0.366317,0.697264,
                 0.442472,0.682139,0.508851,0.990012,0.584262,0.597054,
                 5.306834,0.119013],
                [0.149830,0.126991,0.191503,0.010690,0.320627,0.072854,
                 0.044265,0.008705,0.108882,0.000000,4.145067,0.159069,
                 4.273607,1.112727,0.078281,0.064105,1.033739,0.111660,
                 0.232523,10.649107],
                [0.395337,0.301848,0.068427,0.015076,0.594007,0.582457,
                 0.069673,0.044261,0.366317,4.145067,0.000000,0.137500,
                 6.312358,2.592692,0.249060,0.182287,0.302936,0.619632,
                 0.299648,1.702745],
                [0.536518,6.326067,2.145078,0.282959,0.013266,3.234294,
                 1.807177,0.296636,0.697264,0.159069,0.137500,0.000000,
                 0.656604,0.023918,0.390322,0.748683,1.136863,0.049906,
                 0.131932,0.185202],
                [1.124035,0.484133,0.371004,0.025548,0.893680,1.672569,
                 0.173735,0.139538,0.442472,4.273607,6.312358,0.656604,
                 0.000000,1.798853,0.099849,0.346960,2.020366,0.696175,
                 0.481306,1.898718],
                [0.253701,0.052722,0.089525,0.017416,1.105251,0.035855,
                 0.018811,0.089586,0.682139,1.112727,2.592692,0.023918,
                 1.798853,0.000000,0.094464,0.361819,0.165001,2.457121,
                 7.803902,0.654683],
                [1.177651,0.332533,0.161787,0.394456,0.075382,0.624294,
                 0.419409,0.196961,0.508851,0.078281,0.249060,0.390322,
                 0.099849,0.094464,0.000000,1.338132,0.571468,0.095131,
                 0.089613,0.296501],
                [4.727182,0.858151,4.008358,1.240275,2.784478,1.223828,
                 0.611973,1.739990,0.990012,0.064105,0.182287,0.748683,
                 0.346960,0.361819,1.338132,0.000000,6.472279,0.248862,
                 0.400547,0.098369],
                [2.139501,0.578987,2.000679,0.425860,1.143480,1.080136,
                 0.604545,0.129836,0.584262,1.033739,0.302936,1.136863,
                 2.020366,0.165001,0.571468,6.472279,0.000000,0.140825,
                 0.245841,2.188158],
                [0.180717,0.593607,0.045376,0.029890,0.670128,0.236199,
                 0.077852,0.268491,0.597054,0.111660,0.619632,0.049906,
                 0.696175,2.457121,0.095131,0.248862,0.140825,0.000000,
                 3.151815,0.189510],
                [0.218959,0.314440,0.612025,0.135107,1.165532,0.257336,
                 0.120037,0.054679,5.306834,0.232523,0.299648,0.131932,
                 0.481306,7.803902,0.089613,0.400547,0.245841,3.151815,
                 0.000000,0.249313],
                [2.547870,0.170887,0.083688,0.037967,1.959291,0.210332,
                 0.245034,0.076701,0.119013,10.649107,1.702745,0.185202,
                 1.898718,0.654683,0.296501,0.098369,2.188158,0.189510,
                 0.249313,0.000000]
            ])
        )

    def test_write_rate_model(self):
        import tempfile
        R, pi, s = util.parse_rate_model("test/data/LG_with_factor.model")
        with tempfile.NamedTemporaryFile(suffix=".model", delete=True) as tmp:
            util.write_rate_model(tmp.name, R, pi, s)
            tmp.flush()
            R2, pi2, s2 = util.parse_rate_model(tmp.name)
        np.testing.assert_allclose(R, R2)
        np.testing.assert_allclose(pi, pi2)
        np.testing.assert_allclose(s, s2)

    def test_permute_rate_model(self):
        R, pi, s = util.parse_rate_model("evoten/data/LG.model")
        alphabet = "ARNDCQEGHILKMFPSTWYV"
        # Permute the alphabet, e.g., reverse order
        new_alphabet = alphabet[::-1]
        R_perm, pi_perm = util.permute_rate_model(
            R, pi, alphabet, new_alphabet
        )
        # Permuting back should recover the original
        R_recover, pi_recover = util.permute_rate_model(
            R_perm, pi_perm, new_alphabet, alphabet
        )
        np.testing.assert_allclose(R, R_recover)
        np.testing.assert_allclose(pi, pi_recover)


class TestTupleAlignment(unittest.TestCase):

    def test_k2_with_gaps(self):
        # Hand-verifiable: k=2, 3 rows with gaps
        # Row 0 non-gaps: [0,1,2,3] → tuples {(0,1),(1,2),(2,3)}
        # Row 1 non-gaps: [0,2,3]   → tuples {(0,2),(2,3)}
        # Row 2 non-gaps: [0,1,2]   → tuples {(0,1),(1,2)}
        # Count>=2: (0,1), (1,2), (2,3)
        S = ['ACGT', 'A-GT', 'ACG-']
        result = util.tuple_alignment(S, k=2)
        self.assertEqual(result, ['ACCGGT', '----GT', 'ACCG--'])

    def test_gapless_column_count(self):
        # Gap-less MSA: L columns, output must have L-k+1 columns
        S = ['ACGT', 'TTGA']  # L=4, k=3 → 2 output columns
        result = util.tuple_alignment(S, k=3)
        self.assertEqual(result, ['ACGCGT', 'TTGTGA'])
        self.assertTrue(all(len(r) == (4 - 3 + 1) * 3 for r in result))

    def test_clamsa_example(self):
        # MSA from the clamsa tuple_alignment docstring, using k=3, no frame
        S = ['ac--ttgatgtcgataa',
             'ac--ctaa---cancag',
             'acg-ttga-gtcgacaa',
             'acgtttgat-tcgac-a',
             'acg-ttgatgttga-aa']
        result = util.tuple_alignment(S)
        expected = [
            '---act---ctt---ttgtgagatatgtgtgtctcgcgagatatataa',
            '---acc---cct---ctataa---------------canancncacag',
            'acg---cgt---gttttgtga---------gtctcgcgagacacacaa',
            'acg------------ttgtgagat---------tcgcgagac------',
            'acg---cgt---gttttgtgagatatgtgtgttttgtga---------',
        ]
        self.assertEqual(result, expected)
        # Structural: 16 output columns, each entry all-gap or all-non-gap
        self.assertTrue(all(len(r) == 16 * 3 for r in result))
        for row in result:
            for i in range(0, len(row), 3):
                entry = row[i:i+3]
                self.assertTrue(entry == '---' or '-' not in entry)


class TestEncodeTupleAlignment(unittest.TestCase):

    def test_shape(self):
        # k=2: 4**2=16 classes; 3 rows, 3 output columns
        ta = ['ACCGGT', '----GT', 'ACCG--']
        arr = util.encode_tuple_alignment(ta, k=2)
        self.assertEqual(arr.shape, (3, 3, 16))

    def test_one_hot_values(self):
        # 'ac' -> 0*4+1=1, 'cg' -> 1*4+2=6, 'gt' -> 2*4+3=11
        ta = ['ACCGGT', '----GT', 'ACCG--']
        arr = util.encode_tuple_alignment(ta, k=2)
        # row 0: [1,0,...], [0,0,...,1,0,...] at idx 6, [...,1,0,...] at idx 11
        np.testing.assert_array_equal(arr[0, 0], np.eye(16)[1])   # 'ac'
        np.testing.assert_array_equal(arr[0, 1], np.eye(16)[6])   # 'cg'
        np.testing.assert_array_equal(arr[0, 2], np.eye(16)[11])  # 'gt'

    def test_gap_entries_are_ones(self):
        # Gap entries are all-ones (neutral for Felsenstein's pruning)
        ta = ['ACCGGT', '----GT', 'ACCG--']
        arr = util.encode_tuple_alignment(ta, k=2)
        # row 1, columns 0 and 1 are gaps
        np.testing.assert_array_equal(arr[1, 0], np.ones(16))
        np.testing.assert_array_equal(arr[1, 1], np.ones(16))
        # row 2, column 2 is a gap
        np.testing.assert_array_equal(arr[2, 2], np.ones(16))

    def test_codon_index(self):
        # 'aaa'->0, 'cgt'->1*16+2*4+3=27
        ta = ['aaacgt', 'aaacgt']
        arr = util.encode_tuple_alignment(ta, k=3)
        self.assertEqual(arr.shape, (2, 2, 64))
        np.testing.assert_array_equal(arr[0, 0], np.eye(64)[0])   # 'aaa'->0
        np.testing.assert_array_equal(arr[0, 1], np.eye(64)[27])  # 'cgt'->27
        np.testing.assert_array_equal(arr[1, 0], np.eye(64)[0])
        np.testing.assert_array_equal(arr[1, 1], np.eye(64)[27])

    def test_ambiguous_base_is_ones(self):
        # 'n' is not in {a,c,g,t} → all-ones (neutral)
        ta = ['acgnnn', 'acgnnn']
        arr = util.encode_tuple_alignment(ta, k=3)
        np.testing.assert_array_equal(arr[0, 1], np.ones(64))


class TestTupleArray(unittest.TestCase):

    def test_matches_two_step_gappy(self):
        # clamsa example: gappy MSA, k=3
        S = ['ac--ttgatgtcgataa',
             'ac--ctaa---cancag',
             'acg-ttga-gtcgacaa',
             'acgtttgat-tcgac-a',
             'acg-ttgatgttga-aa']
        expected = util.encode_tuple_alignment(util.tuple_alignment(S, k=3), k=3)
        result, _ = util.tuple_array(S, k=3)
        np.testing.assert_array_equal(result, expected)

    def test_matches_two_step_gapless(self):
        # gap-less MSA: all L-k+1 columns present
        S = ['ACGTACGT', 'TTGACCGA', 'GCATTTCA']
        expected = util.encode_tuple_alignment(util.tuple_alignment(S, k=3), k=3)
        result, _ = util.tuple_array(S, k=3)
        np.testing.assert_array_equal(result, expected)

    def test_matches_two_step_k2(self):
        S = ['ACGT', 'A-GT', 'ACG-']
        expected = util.encode_tuple_alignment(util.tuple_alignment(S, k=2), k=2)
        result, _ = util.tuple_array(S, k=2)
        np.testing.assert_array_equal(result, expected)


# utility functions
def _check_symmetry_and_zero_diagonal(test_case : unittest.TestCase, matrix):
    np.testing.assert_almost_equal(matrix, np.transpose(matrix))
    np.testing.assert_almost_equal(np.diag(matrix), 0.)

def _check_if_sums_to_one(test_case : unittest.TestCase, matrix):
    np.testing.assert_almost_equal(np.sum(matrix, -1), 1., decimal=6)
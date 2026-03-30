import os
from collections import Counter

import numpy as np

default_dtype = np.float32
supported_backends = ["tensorflow", "pytorch"]
    
  

class Backend():
    def __init__(self):
        # the wrapped backend that contains the actual implementation
        # the global backend will always be the same object to avoid importing 
        # issues but the wrapped backend can change at will
        self.wrapped_backend : Backend = None
    
    
    def make_rate_matrix(
            self, 
            exchangeabilities, 
            equilibrium, 
            epsilon=1e-16, 
            normalized=True
        ):
        """Constructs a stack of normalized rate matrices, i.e. 1 time unit = 1 
        expected mutation per site. Exchangeabilities and equilibrium tensors
        can have arbitrary leading dimensions for which mutual broadcasting is
        supported. The leading dimensions should match otherwise.

        Args:
            exchangeabilities: Symmetric, positive-semidefinite exchangeability 
                matrices with zero diagonal. Shape: (..., d, d) 
            equilibrium: A vector of relative frequencies. Shape: (..., d) 
            epsilon: Small value to avoid division by zero.
            normalized: If True, the rate matrices are normalized.

        Returns:
            Normalized rate matrices. Output shape: (..., d, d) 
        """
        return self.wrapped_backend.make_rate_matrix(
            exchangeabilities, equilibrium, epsilon, normalized
        )
    
    
    def make_transition_probs(self, rate_matrix, distances, equilibrium):
        """Constructs a probability matrix of mutating or copying one an input
        to another over a given amount of evolutionary time. The rate matrix and
        distance tensor can have arbitrary leading dimensions for which 
        broadcasting is supported. The leading dimensions should match otherwise.
        
        Args:
            rate_matrix: Rate matrix of shape (..., d, d).
            distances: Evolutionary times of shape (...) 
        
        Returns:
            Stack of probability matrices of shape (..., d, d)
        """
        return self.wrapped_backend.make_transition_probs(
            rate_matrix, distances, equilibrium
        )
    

    def make_branch_lengths(self, kernel):
        """
        Converts a kernel of parameters to positive branch lengths.

        Args:
            kernel: Tensor of any shape.
        
        Returns:
            Tensor of the same shape with positive values.	
        """
        return self.wrapped_backend.make_branch_lengths(kernel)
    

    def inverse_softplus(self, branch_lengths):
        """
        Computes the inverse of the softplus function. Can be used to initialize
        kernels for branch lengths or exchangeabilities.

        Args:
            branch_lengths: Tensor of any shape with positive values.
        
        Returns:
            Tensor of the same shape with values in (-inf, inf).
        """
        return self.wrapped_backend.inverse_softplus(branch_lengths)
    

    def make_symmetric_pos_semidefinite(self, kernel):
        """
        Constructs a stack of symmetric, positive-semidefinite matrices with 
        zero diagonal from a parameter kernel.
        Note: Uses an overparameterized d x d kernel for speed of computation.

        Args:
            kernel: Tensor of shape (..., d, d).

        Returns:
            Symmetric, positive-semidefinite matrices of shape (..., d, d).
        """
        return self.wrapped_backend.make_symmetric_pos_semidefinite(kernel)
    

    def make_equilibrium(self, kernel):
        """ Constructs a stack of equilibrium distributions from a parameter kernel.

        Args:
            kernel: Tensor of shape (..., d).

        Returns:
            Equilibrium distributions of shape (..., d).
        """ 
        return self.wrapped_backend.make_equilibrium(kernel)
    

    def traverse_branch(
            self, X, transition_probs, transposed=False, logarithmic=True
        ):
        """
        Computes P(X | X') for a branch {X, X'} given the transition matrix.
        The tensors X and transition_probs can have arbitrary leading 
        dimensions for which mutual broadcasting is supported. The leading 
        dimensions should match otherwise. 

        Args:
            X: tensor with logits of shape (..., d). 
            transition_probs: The transition matrices along each branch. 
                Tensor of shape (..., d, d). 
            transposed: If True, computes P(X' | X) instead.
            
        Returns:
            logits of shape (..., d)
        """
        return self.wrapped_backend.traverse_branch(
            X, transition_probs, transposed, logarithmic
        )
    

    def aggregate_children_log_probs(self, X, parent_map, num_ancestral):
        r"""
        Aggregates the partial log-likelihoods of child nodes.

        Args:
            X: tensor of shape (n, ...)
            parent_map: A list-like object of shape (n) that contains the index 
            of parent nodes.
            num_ancestral: Total number of ancestral nodes.

        Returns:
            tensor of shape (num_ancestral, ...).
        """
        return self.wrapped_backend.aggregate_children_log_probs(
            X, parent_map, num_ancestral
        )
    
    
    def loglik_from_root_logits(self, root_logits, equilibrium_logits):
        """
        Computes log likelihoods given root logits and equilibrium distributions.
        The tensors root_logits and equilibrium_logits can have arbitrary leading
        dimensions for which mutual broadcasting is supported. The leading 
        dimensions should match otherwise.

        Args:
            root_logits: Logits at the root node of shape (..., d)
            equilibrium_logits: Equilibrium distribution logits of shape (..., d)

        Returns:
            Log-likelihoods of shape (k, L)
        """
        return self.wrapped_backend.loglik_from_root_logits(
            root_logits, equilibrium_logits
        )
    

    def marginals_from_beliefs(self, beliefs, same_loglik=True):
        """
        Computes marginal distributions log P(u) from beliefs log P(u, data).

        Args:
            beliefs: Logits of shape (..., d)
            same_loglik: If True, the likelihoods are assumed to be identical 
                for all inputs.

        Returns:
            Marginal distributions of shape (..., d)
        """
        return self.wrapped_backend.marginals_from_beliefs(beliefs, same_loglik)
    

    def logits_from_probs(self, probs, log_zero_val=-1e3):
        """ Computes element-wise logarithm with output_i=log_zero_val where 
        x_i=0.
        """
        return self.wrapped_backend.logits_from_probs(probs, log_zero_val)
    

    def probs_from_logits(self, logits):
        """ 
        Computes element-wise exp
        """
        return self.wrapped_backend.probs_from_logits(logits)


    def gather(self, tensor, indices, axis=0):
        """ 
        Gathers values from a tensor along an axis.
        """
        return self.wrapped_backend.gather(tensor, indices, axis)


    def concat(self, tensors, axis=0):
        """ 
        Concatenates tensors along an axis.
        """
        return self.wrapped_backend.concat(tensors, axis)


    def expand(self, X, axis):
        """
        Adds a dimension of size 1 at the given axis.
        """
        return self.wrapped_backend.expand(X, axis)


    def make_zeros(self, leaves, models, num_nodes):
        """
        Initializes the ancestral logits tensor with zeros.
        """
        return self.wrapped_backend.make_zeros(leaves, models, num_nodes)
    


def set_backend(backend_name = "tensorflow"):
    """ Loads one of the following backends: ["tensorflow", "pytorch"]
        Must be called before any other function in the library.
    """
    _validate_backend(backend_name)
    if backend_name == "tensorflow":
        from evoten.backend_tf import BackendTF 
        backend.wrapped_backend = BackendTF()
    elif backend_name == "pytorch":
        from evoten.backend_pytorch import BackendTorch 
        backend.wrapped_backend = BackendTorch()
    


def _validate_backend(backend):
    if backend not in supported_backends:
        raise ValueError(f"Backend must be one of {supported_backends}")



# null object that raises errors upon usage that request to call set_backend
class NullBackend(Backend):
    def __getattr__(self, name):
        raise ValueError("No backend loaded. Please call "\
                         "evoten.set_backend(<name>) first." 
                         + f"Supported backends are: {supported_backends}.")


# initial case: load dummy and wait for user to call set_backend
backend : Backend = Backend()
backend.wrapped_backend = NullBackend()



################################################################################
# Input/output utility
################################################################################

def encode_one_hot(sequences, alphabet):
    """ One-hot encodes a list of strings over the given alphabet.
    """
    ids = np.array([[alphabet.index(c) for c in seq] for seq in sequences])
    return np.eye(len(alphabet), dtype=default_dtype)[ids]


def tuple_alignment(sequences, k=3, gap_symbols='-'):
    """
    Construct a tuple alignment from a multiple sequence alignment (MSA).

    For each row, k-mers of k consecutive non-gap characters are identified by
    their column-index tuple (c_0, ..., c_{k-1}). Output columns are the
    equivalence classes — unique column-index tuples — that appear in at least
    two rows, sorted lexicographically. Each output entry is either the k
    characters at those columns or k gap characters.

    For a gap-less MSA with L columns the output has L-k+1 columns.

    Args:
        sequences (List[str]): MSA rows, all the same length.
        k (int): Tuple length; k=3 gives codon-level alignment.
        gap_symbols (str): Characters treated as gaps. The first character is
            used to fill missing entries in the output.

    Returns:
        List[str]: One string per row; length = (number of output columns) * k.

    Example:
        S = ['ACGT', 'A-GT', 'ACG-']
        tuple_alignment(S, k=2)
        # => ['ACCGGT', '----GT', 'ACCG--']
        # Column-index tuples with count>=2: (0,1), (1,2), (2,3)
        # Row 1 lacks (0,1) and (1,2); row 2 lacks (2,3).
    """
    missing = gap_symbols[0] * k

    # Step 1-2: non-gap positions and their k-tuple sets per row
    seq_tuples = []
    for s in sequences:
        nongap = [i for i, c in enumerate(s) if c not in gap_symbols]
        tuples_in_seq = {tuple(nongap[j:j+k]) for j in range(len(nongap) - k + 1)}
        seq_tuples.append(tuples_in_seq)

    # Step 3-4: keep tuples present in >=2 rows
    counter = Counter()
    for ts in seq_tuples:
        counter.update(ts)
    valid_tuples = sorted(t for t, cnt in counter.items() if cnt >= 2)

    # Step 5: build output
    result = []
    for i, s in enumerate(sequences):
        row = []
        for t in valid_tuples:
            if t in seq_tuples[i]:
                row.append(''.join(s[pos] for pos in t))
            else:
                row.append(missing)
        result.append(''.join(row))
    return result


_BASE_TO_INT = {'a': 0, 'c': 1, 'g': 2, 't': 3,
                'A': 0, 'C': 1, 'G': 2, 'T': 3}

_INT_TO_BASE = ['a', 'c', 'g', 't']


def tuple_labels(k):
    """Return the list of 4**k k-mer strings in base-4 index order.

    Index i corresponds to the k-mer whose base-4 digits (most-significant
    first) give bases _INT_TO_BASE[digit].  E.g. for k=2: index 0 → 'aa',
    index 1 → 'ac', index 4 → 'ca', index 15 → 'tt'.

    Args:
        k (int): Tuple length.

    Returns:
        List[str]: 4**k strings of length k.
    """
    if k == 1:
        return list(_INT_TO_BASE)
    labels = []
    for i in range(4 ** k):
        chars = []
        val = i
        for _ in range(k):
            chars.append(_INT_TO_BASE[val % 4])
            val //= 4
        labels.append(''.join(reversed(chars)))
    return labels


def print_tuple_rate_matrix(Q, k, label=None, file=None):
    """Pretty-print a rate matrix with k-mer row/column labels.

    Args:
        Q: 2-D array-like of shape (4**k, 4**k).
        k (int): Tuple size used to produce Q.
        label (str): Optional header line printed before the matrix.
        file: File object for output (default: sys.stdout).
    """
    import sys
    out = file or sys.stdout
    labs = tuple_labels(k)
    col_w = max(8, k + 5)
    if label:
        print(label, file=out)
    header = ' ' * (k + 2) + '  '.join(f'{l:>{col_w}}' for l in labs)
    print(header, file=out)
    for i, row_lab in enumerate(labs):
        row_str = '  '.join(f'{Q[i, j]:{col_w}.4f}' for j in range(len(labs)))
        print(f'{row_lab}  {row_str}', file=out)


def print_tuple_stationary(pi, k, label=None, file=None):
    """Pretty-print a stationary distribution with k-mer labels.

    Args:
        pi: 1-D array-like of shape (4**k,).
        k (int): Tuple size.
        label (str): Optional header line printed before the values.
        file: File object for output (default: sys.stdout).
    """
    import sys
    out = file or sys.stdout
    labs = tuple_labels(k)
    if label:
        print(label, file=out)
    for lab, val in zip(labs, pi):
        print(f'  {lab}: {val:.6f}', file=out)


def encode_tuple_alignment(ta, k=3, gap_symbols='-', gap_separate_state=0):
    """
    One-hot encode a tuple alignment produced by tuple_alignment().

    Each k-character entry is converted to a base-4 integer index
    (a=0, c=1, g=2, t=3), so 'aaa'->0 and 'cgt'->1*16+2*4+3=27.
    Gap entries and entries containing characters outside {a,c,g,t,A,C,G,T}
    are encoded as all-ones (unknown/missing) unless gap_separate_state >= 1,
    in which case gap entries are one-hot at index 4**k.

    Args:
        ta (List[str]): Output of tuple_alignment(); R strings each of
            length L*k.
        k (int): Tuple length used to produce ta (default 3).
        gap_symbols (str): Characters treated as gaps (default '-').
        gap_separate_state (int): Number of extra gap states appended to the
            alphabet (default 0).  If >= 1, gap entries are one-hot at index
            4**k instead of all-ones.

    Returns:
        np.ndarray: Shape [R, L, 4**k + gap_separate_state], dtype float32.
            Valid k-mer entries are one-hot. Ambiguous entries are all-ones.
            Gap entries are all-ones when gap_separate_state=0, or one-hot at
            index 4**k when gap_separate_state >= 1.
    """
    alphabet_size = 4 ** k + gap_separate_state
    R = len(ta)
    L = len(ta[0]) // k if R > 0 else 0
    result = np.ones((R, L, alphabet_size), dtype=default_dtype)
    for r, row in enumerate(ta):
        for j in range(L):
            entry = row[j*k:(j+1)*k]
            if entry[0] in gap_symbols:
                if gap_separate_state >= 1:
                    result[r, j, :] = 0.0
                    result[r, j, 4 ** k] = 1.0
                continue           # else: leave as all-ones
            try:
                idx = 0
                for c in entry:
                    idx = idx * 4 + _BASE_TO_INT[c]
                result[r, j, :] = 0.0
                result[r, j, idx] = 1.0
            except KeyError:
                pass  # ambiguous base (e.g. 'n') → leave as all-ones
    return result


def tuple_array(sequences, k=3, gap_symbols='-', gap_separate_state=0):
    """
    Directly compute the one-hot encoded tuple alignment array.

    Combines tuple_alignment() and encode_tuple_alignment() in a single pass,
    avoiding intermediate string allocation. Equivalent to:
        encode_tuple_alignment(tuple_alignment(sequences, k, gap_symbols),
                               k, gap_symbols, gap_separate_state)

    For each row, non-gap positions are found and k-tuples of consecutive
    positions are mapped to base-4 indices. Valid tuples (present in >=2 rows)
    are written as one-hot vectors into the output array using batch numpy
    indexing.

    With gap_separate_state=0 (default): absent/gap entries are all-ones
    (neutral for Felsenstein's pruning algorithm).
    With gap_separate_state>=1: absent entries are one-hot at index 4**k
    (explicit gap state); ambiguous entries remain all-ones.

    Args:
        sequences (List[str]): MSA rows, all the same length.
        k (int): Tuple length; k=3 for codons.
        gap_symbols (str): Characters treated as gaps (default '-').
        gap_separate_state (int): Extra gap states appended to the alphabet
            (default 0).  If >= 1, absent tuple positions are encoded as
            one-hot at index 4**k rather than all-ones.

    Returns:
        result : np.ndarray, shape [R, L, 4**k + gap_separate_state], float32.
        first_positions : np.ndarray, shape [L], dtype int64.
            For each output column j, the 0-based alignment column index of
            the first character of tuple j.
    """
    # Step 1: per row, map column-index tuple -> base-4 index (-1 if ambiguous)
    seq_dicts = []
    for s in sequences:
        nongap = [i for i, c in enumerate(s) if c not in gap_symbols]
        d = {}
        for j in range(len(nongap) - k + 1):
            pos = nongap[j:j+k]
            t = tuple(pos)
            idx = 0
            valid = True
            for p in pos:
                b = _BASE_TO_INT.get(s[p], -1)
                if b < 0:
                    valid = False
                    break
                idx = idx * 4 + b
            d[t] = idx if valid else -1
        seq_dicts.append(d)

    # Step 2: keep tuples present in >=2 rows, sorted lexicographically
    counter = Counter()
    for d in seq_dicts:
        counter.update(d.keys())
    valid_tuples = sorted(t for t, cnt in counter.items() if cnt >= 2)

    R = len(sequences)
    L = len(valid_tuples)
    first_positions = np.array([t[0] for t in valid_tuples], dtype=np.int64)
    alphabet_size = 4 ** k + gap_separate_state

    # Step 3: initialise result and fill one-hot entries per row
    if gap_separate_state >= 1:
        result = np.zeros((R, L, alphabet_size), dtype=default_dtype)
        result[:, :, 4 ** k] = 1.0   # default: gap state
    else:
        result = np.ones((R, L, alphabet_size), dtype=default_dtype)  # neutral

    tuple_to_col = {t: j for j, t in enumerate(valid_tuples)}
    for r, d in enumerate(seq_dicts):
        js_valid, idxs, js_ambig = [], [], []
        for t, base_idx in d.items():
            j = tuple_to_col.get(t)
            if j is None:
                continue
            if base_idx >= 0:
                js_valid.append(j)
                idxs.append(base_idx)
            elif gap_separate_state >= 1:
                js_ambig.append(j)    # present but ambiguous → all-ones
        if js_valid:
            js_v   = np.array(js_valid, dtype=np.intp)
            idxs_a = np.array(idxs,    dtype=np.intp)
            result[r, js_v, :] = 0.0
            result[r, js_v, idxs_a] = 1.0
        if js_ambig:
            result[r, np.array(js_ambig, dtype=np.intp), :] = 1.0

    return result, first_positions
from collections.abc import Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator
from collections import Counter

import numpy as np

default_dtype: type[np.floating] = np.float32


@contextmanager
def data_path(filename: str) -> Iterator[Path]:
    """Yields a real filesystem path to a data file."""
    resource = resources.files("evoten.data") / filename
    with resources.as_file(resource) as path:
        yield path

def encode_one_hot(sequences : Sequence[str], alphabet : str) -> np.ndarray:
    """ One-hot encodes a list of strings over the given alphabet.
    """
    ids = np.array([[alphabet.index(c) for c in seq] for seq in sequences])
    return np.eye(len(alphabet), dtype=default_dtype)[ids]

def parse_rate_model(path : Path | str) -> tuple[np.ndarray, np.ndarray, float]:
    """ Parses a rate model from a file. The first row is expected to contain
    the equilibrium frequencies, and the remaining rows are expected to contain
    the exchangeabilities (lower triangular matrix without diagonal).
    After the matrix, an optional scaling factor can be provided in a separate
    line.

    The file can contain comments starting with '#'.

    Args:
        path: Path to the rate model file.

    Returns:
        A tuple ``(exchangeabilities, equilibrium frequencies, scaling factor)``
        with shapes ``(n, n)``, ``(n,)`` and a scalar respectively.
        When the scaling factor is not provided, it defaults to 1.
    """
    with open(path) as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split('#')[0].strip() for line in lines]
    lines = [line for line in lines if line]

    assert len(lines) > 0, "File is empty."

    # Parse
    equilibrium = np.array(list(map(float, lines[0].split())))
    n = len(equilibrium)

    np.testing.assert_allclose(
        np.sum(equilibrium), 1.,
        err_msg="Equilibrium frequencies must sum to 1.",
        atol=1e-5
    )

    assert len(lines) in [n, n+1], "File contains an invalid number of lines."

    # Parse the exchangeabilities
    exchangeabilities = np.zeros((n, n), dtype=default_dtype)
    for row in range(1, n):
        split = lines[row].split()
        assert len(split) == row,\
            "No lower triangular matrix without diagonal. "\
            "Expected {} values, got {}.".format(row, len(split))
        for col, value in enumerate(split):
            exchangeabilities[row, col] = float(value)
            exchangeabilities[col, row] = exchangeabilities[row, col]

    if len(lines) == n + 1:
        scaling_factor = float(lines[-1])
    else:
        scaling_factor = 1.

    return exchangeabilities, equilibrium, scaling_factor

def permute_rate_model(
    exchangeabilities : np.ndarray,
    equilibrium : np.ndarray,
    alphabet : str,
    new_alphabet : str
) -> tuple[np.ndarray, np.ndarray]:
    """ Permutes a rate model to match a new alphabet. The new alphabet must be
    a permutation of the old alphabet.

    Args:
        exchangeabilities: Exchangeability matrix of shape (n, n).
        equilibrium: Equilibrium frequencies of shape (n,).
        alphabet: Original alphabet.
        new_alphabet: New alphabet.

    Returns:
        A tuple ``(new_exchangeabilities, new_equilibrium)`` with the same
        shapes as the input exchangeabilities and equilibrium, but permuted
        to match the new alphabet.
    """
    perm = [alphabet.index(aa) for aa in new_alphabet if aa in alphabet]
    equilibrium = equilibrium[perm]
    exchangeabilities = exchangeabilities[perm, :]
    exchangeabilities = exchangeabilities[:, perm]
    return exchangeabilities, equilibrium

def write_rate_model(
    path : Path | str,
    exchangeabilities : np.ndarray,
    equilibrium : np.ndarray,
    scaling_factor : float = 1.0,
) -> None:
    """ Writes a rate model to a file. The first row contains the equilibrium
    frequencies, and the remaining rows contain the exchangeabilities (lower
    triangular matrix without diagonal). An optional scaling factor can be
    provided in a separate line after the matrix.

    Args:
        exchangeabilities: Exchangeability matrix of shape (n, n).
        equilibrium: Equilibrium frequencies of shape (n,).
        scaling_factor: Optional scaling factor to write to the file.
    """
    n = len(equilibrium)
    assert exchangeabilities.shape == (n, n),\
        "Exchangeabilities must be a square matrix of shape (n, n)."
    with open(path, 'w') as f:
        f.write(' '.join(map(str, equilibrium)) + '\n')
        for row in range(1, n):
            f.write(' '.join(map(str, exchangeabilities[row, :row])) + '\n')
        if scaling_factor != 1.0:
            f.write(str(scaling_factor) + '\n')



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

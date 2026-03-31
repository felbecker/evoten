from collections.abc import Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator

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

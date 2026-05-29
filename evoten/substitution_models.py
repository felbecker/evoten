from collections.abc import Sequence

import numpy as np

from evoten import util


def jukes_cantor(
    mue: float | Sequence[float] = 4./3,
    d: int = 4,
    dtype: type[np.floating] = util.default_dtype
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies for the
    Jukes-Cantor model.

    Args:
        mue: Scalar, list or 1D array.

    Returns:
        symmetric k x d x d  tensor of exchangeabilities and k x d matrix of
            equilibrium frequencies.
        k is the length of mue or 1 if mue is a scalar.
    """
    mue_array = np.atleast_1d(mue).astype(dtype)
    I = (1-np.eye(d, dtype=dtype))
    I = np.stack([I]*mue_array.size)
    exchangeabilities = I * mue_array[:, np.newaxis, np.newaxis]
    equilibrium = np.ones((mue_array.size, d), dtype=dtype) / d
    return exchangeabilities, equilibrium

def LG(
    alphabet:str = "ARNDCQEGHILKMFPSTWYV",
    dtype: type[np.floating] = util.default_dtype
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies for the LG
        model.
        Si Quang Le, Olivier Gascuel
        An Improved General Amino Acid Replacement Matrix, 2008
        Use for amino acids.

    Args:
        alphabet: A string with the amino acids in the desired order.
        dtype: The desired dtype for the returned arrays.

    Returns:
        symmetric d x d  tensor of exchangeabilities and d vector of
        equilibrium frequencies.
    """
    with util.data_path("LG.model") as path:
        R, p, s = util.parse_rate_model(path)
    # TODO: s is omitted for now, but can be used in the future
    R, pi = util.permute_rate_model(R, p, "ARNDCQEGHILKMFPSTWYV", alphabet)
    R = R.astype(dtype)
    pi = pi.astype(dtype)
    return R, pi

def foldseek_3Di(
    alphabet:str = "ACDEFGHIKLMNPQRSTVWY",
    dtype: type[np.floating] = util.default_dtype
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies for a model
    derived from Foldseek's 3Di substitution matrix.

    Based on https://github.com/steineggerlab/foldseek/blob/master/data/mat3di.out
    See `docs/misc/3Di_Q_from_P.ipynb` for construction details.

    Args:
        alphabet: A string with the amino acids in the desired order.
        dtype: The desired dtype for the returned arrays.

    Returns:
        symmetric d x d  tensor of exchangeabilities and d vector of
        equilibrium frequencies.
    """
    with util.data_path("foldseek_3Di.model") as path:
        R, p, s = util.parse_rate_model(path)
    # TODO: s is omitted for now, but can be used in the future
    R, pi = util.permute_rate_model(R, p, "ACDEFGHIKLMNPQRSTVWY", alphabet)
    R = R.astype(dtype)
    pi = pi.astype(dtype)
    return R, pi

def AF_3Di(
    alphabet:str = "ACDEFGHIKLMNPQRSTVWY",
    dtype: type[np.floating] = util.default_dtype
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies for a for 3Di
    characters derived from AlphaFold structures.

    Hochberg, Georg, 2025,
    "A general substitution matrix for structural phylogenetics.",
    https://doi.org/10.17617/3.1MJJBH, Edmond, V3

    Args:
        alphabet: A string with the amino acids in the desired order.
        dtype: The desired dtype for the returned arrays.

    Returns:
        symmetric d x d  tensor of exchangeabilities and d vector of
        equilibrium frequencies.
    """
    with util.data_path("AF_3Di.model") as path:
        R, p, s = util.parse_rate_model(path)
    # TODO: s is omitted for now, but can be used in the future
    R, pi = util.permute_rate_model(R, p, "ARNDCQEGHILKMFPSTWYV", alphabet)
    R = R.astype(dtype)
    pi = pi.astype(dtype)
    return R, pi

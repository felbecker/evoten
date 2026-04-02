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

    Returns:
        symmetric d x d  tensor of exchangeabilities and d matrix of
        equilibrium frequencies.
    """
    with util.data_path("LG.model") as path:
        R, p, s = util.parse_rate_model(path)
    # TODO: s is omitted for now, but can be used in the future
    R, pi = util.permute_rate_model(R, p, "ARNDCQEGHILKMFPSTWYV", alphabet)
    R = R.astype(dtype)
    pi = pi.astype(dtype)
    return R, pi

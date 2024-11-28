import numpy as np


""" Returns the exchangeabilities and equilibrium frequencies for the Jukes-Cantor model.
Args:
    mue: Scalar, list or 1D array.
Returns:
    symmetric k x d x d  tensor of exchangeabilities and k x d matrix of equilibrium frequencies.
    k is the length of mue or 1 if mue is a scalar.
"""
def jukes_cantor(mue=4./3, d=4, dtype=np.float64):
    if isinstance(mue, list):
        mue = np.array(mue, dtype=dtype)
    if np.isscalar(mue):
        mue = np.array([mue], dtype=dtype)
    assert(mue.ndim == 1)
    I = (1-np.eye(d, dtype=dtype))
    I = np.stack([I]*mue.size)
    exchangeabilities = I * mue[:, np.newaxis, np.newaxis]
    equilibrium = np.ones((mue.size, d), dtype=dtype) / d
    return exchangeabilities, equilibrium
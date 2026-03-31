import numpy as np

default_dtype = np.float32


################################################################################
# Input/output utility
################################################################################

def encode_one_hot(sequences, alphabet):
    """ One-hot encodes a list of strings over the given alphabet.
    """
    ids = np.array([[alphabet.index(c) for c in seq] for seq in sequences])
    return np.eye(len(alphabet), dtype=default_dtype)[ids]

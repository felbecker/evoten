import itertools
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


# ---------------------------------------------------------------------------
# Codon and triplet models
# ---------------------------------------------------------------------------
# The 64-state models use the k-mer order of util.tuple_labels(3): the state of
# the triplet b1 b2 b3 (a=0, c=1, g=2, t=3) is 16*b1 + 4*b2 + b3.

# NCBI translation table 1 in NCBI codon order (bases t, c, a, g with the first
# codon position outermost); '*' marks a stop codon.
_NCBI_STANDARD_CODE = \
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"


def genetic_code(name: str = "standard") -> np.ndarray:
    """ Returns the amino acid encoded by each of the 64 codons.

    Args:
        name: Name of the genetic code. Only "standard" (NCBI translation
            table 1) is supported.

    Returns:
        Array of shape (64,) with one-letter amino acid codes in the order of
        util.tuple_labels(3); stop codons are '*'.
    """
    if name != "standard":
        raise ValueError(
            f"Unknown genetic code {name!r}, supported: 'standard'."
        )
    ncbi_order = ("".join(c) for c in itertools.product("tcag", repeat=3))
    table = dict(zip(ncbi_order, _NCBI_STANDARD_CODE))
    return np.array([table[codon] for codon in util.tuple_labels(3)])


def nucleotide_gtr(
    rates: Sequence[float] | None = None,
    kappa: float | None = None,
    freqs: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    dtype: type[np.floating] = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies of a
    nucleotide model (order a, c, g, t).

    Args:
        rates: The six exchangeabilities (ac, ag, at, cg, ct, gt) of a GTR
            model.
        kappa: Transition/transversion ratio of an HKY model, i.e. the
            exchangeability of a<->g and c<->t while all others are 1.
            If neither rates nor kappa is given, all exchangeabilities are 1.
        freqs: Equilibrium frequencies, normalized to sum to 1.

    Returns:
        Symmetric 4 x 4 exchangeabilities with zero diagonal and the 4
        equilibrium frequencies.
    """
    if rates is not None and kappa is not None:
        raise ValueError("Specify either rates or kappa, not both.")
    if rates is not None:
        values = np.asarray(rates, dtype=np.float64)
        if values.shape != (6,):
            raise ValueError(
                f"rates must have 6 entries, got shape {values.shape}."
            )
    elif kappa is not None:
        values = np.array([1.0, kappa, 1.0, 1.0, kappa, 1.0])
    else:
        values = np.ones(6)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Exchangeabilities must be positive.")
    R = np.zeros((4, 4))
    R[np.triu_indices(4, k=1)] = values
    R = R + R.T
    pi = _positive_frequencies(freqs, "freqs", (4,))
    return R.astype(dtype), pi.astype(dtype)


def codon_frequencies(
    nuc_freqs: np.ndarray | Sequence | None = None,
    stop_freq: float = 1e-6,
    code: str = "standard",
    dtype: type[np.floating] = np.float64
) -> np.ndarray:
    """ Returns equilibrium frequencies of the 64 codons.

    The sense codons get the total frequency 1 - stop_freq, distributed
    uniformly (nuc_freqs None) or proportionally to the product of nucleotide
    frequencies (F1x4 for shape (4,), F3x4 for shape (3, 4) with one row per
    codon position). The stop codons share stop_freq equally. A positive
    stop_freq keeps all frequencies positive, as required by reversible
    transition matrices (expm_gtr_torch.expm_gtr) and softmax parameterizations.

    Args:
        nuc_freqs: None or nucleotide frequencies (a, c, g, t) of shape (4,)
            or (3, 4).
        stop_freq: Total frequency of the stop codons, in (0, 1).
        code: Genetic code, see genetic_code.

    Returns:
        Array of shape (64,) in the order of util.tuple_labels(3).
    """
    if not 0 < stop_freq < 1:
        raise ValueError(f"stop_freq must be in (0, 1), got {stop_freq}.")
    stop = genetic_code(code) == "*"
    if nuc_freqs is None:
        freqs = np.ones(64)
    else:
        nf = np.asarray(nuc_freqs, dtype=np.float64)
        if nf.shape == (4,):
            nf = np.stack([nf] * 3)
        nf = _positive_frequencies(nf, "nuc_freqs", (3, 4))
        freqs = np.einsum("a,b,c->abc", nf[0], nf[1], nf[2]).reshape(64)
    freqs = np.where(stop, 0.0, freqs)
    freqs = freqs / freqs.sum() * (1.0 - stop_freq)
    freqs[stop] = stop_freq / stop.sum()
    return freqs.astype(dtype)


def gy94(
    kappa: float = 4.0,
    omega: float = 1.0,
    codon_freqs: np.ndarray | Sequence[float] | None = None,
    *,
    min_exchangeability: float = 1e-3,
    code: str = "standard",
    dtype: type[np.floating] = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies of a codon
    model in the style of Goldman and Yang (1994) on all 64 triplets.

    Two codons that differ at one position have the exchangeability
    kappa^[transition] * omega^[different amino acids]; stop codons count as
    an amino acid of their own. With Q_ij = R_ij * pi_j (see
    Backend.make_rate_matrix) this is the GY94 rate matrix. Unlike in GY94,
    every off-diagonal exchangeability is at least a small floor, which
    applies in particular to codons that differ at 2 or 3 positions. The
    exchangeabilities are thus strictly positive, e.g. representable as the
    softplus of a kernel.

    Args:
        kappa: Transition/transversion ratio.
        omega: Nonsynonymous/synonymous rate ratio (dN/dS).
        codon_freqs: Equilibrium frequencies of shape (64,), default
            codon_frequencies() (uniform sense codons, rare stop codons).
        min_exchangeability: Floor relative to the mean exchangeability of
            the codon pairs that differ at one position; 0 disables it.
        code: Genetic code, see genetic_code.

    Returns:
        Symmetric 64 x 64 exchangeabilities with zero diagonal (not
        normalized) and 64 equilibrium frequencies, in the order of
        util.tuple_labels(3).
    """
    if kappa <= 0 or omega <= 0:
        raise ValueError(
            f"kappa and omega must be positive, got {kappa} and {omega}."
        )
    aa = genetic_code(code)
    single, transition = _triplet_single_changes()
    R = np.where(single, 1.0, 0.0)
    R = R * np.where(transition, kappa, 1.0)
    R = R * np.where(aa[:, None] != aa[None, :], omega, 1.0)
    R = _floor_exchangeabilities(R, single, min_exchangeability)
    if codon_freqs is None:
        pi = codon_frequencies(code=code)
    else:
        pi = _positive_frequencies(codon_freqs, "codon_freqs", (64,))
    return R.astype(dtype), pi.astype(dtype)


def triplet_independent(
    nuc_models,
    *,
    min_exchangeability: float = 1e-3,
    dtype: type[np.floating] = np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the exchangeabilities and equilibrium frequencies of a 64-state
    triplet model in which the three nucleotides evolve independently, e.g. a
    model of non-coding sequence on the state space of a codon model.

    Position p evolves under the nucleotide rate matrix Q_p of (R_p, pi_p),
    normalized to one expected substitution per time unit. The triplet rate
    matrix is the Kronecker sum Q = Q_1 x I x I + I x Q_2 x I + I x I x Q_3
    (three expected substitutions per time unit) with the equilibrium
    pi_1 x pi_2 x pi_3. Q is reversible and R_ij = Q_ij / pi_j. As in gy94,
    the off-diagonal exchangeabilities are raised to a floor.

    Args:
        nuc_models: One nucleotide model (R, pi) with shapes (4, 4) and (4,)
            for all positions or a sequence of three, e.g. from nucleotide_gtr.
        min_exchangeability: Floor relative to the mean exchangeability of
            the triplet pairs that differ at one position; 0 disables it.

    Returns:
        Symmetric 64 x 64 exchangeabilities with zero diagonal and 64
        equilibrium frequencies, in the order of util.tuple_labels(3).
    """
    if len(nuc_models) == 2 and np.shape(nuc_models[0]) == (4, 4):
        nuc_models = [nuc_models] * 3
    if len(nuc_models) != 3:
        raise ValueError("nuc_models must be one (R, pi) pair or three pairs.")
    Qs, pis = [], []
    for R_nuc, pi_nuc in nuc_models:
        R_nuc = np.asarray(R_nuc, dtype=np.float64)
        pi_nuc = _positive_frequencies(pi_nuc, "pi", (4,))
        if R_nuc.shape != (4, 4) or not np.allclose(R_nuc, R_nuc.T):
            raise ValueError(
                "Nucleotide exchangeabilities must be symmetric 4 x 4."
            )
        Q_nuc = R_nuc * (1.0 - np.eye(4)) * pi_nuc[None, :]
        Q_nuc -= np.diag(Q_nuc.sum(axis=1))
        Qs.append(Q_nuc / -np.sum(pi_nuc * np.diag(Q_nuc)))
        pis.append(pi_nuc)
    I = np.eye(4)
    Q = (np.kron(np.kron(Qs[0], I), I) + np.kron(np.kron(I, Qs[1]), I)
         + np.kron(np.kron(I, I), Qs[2]))
    pi = np.kron(np.kron(pis[0], pis[1]), pis[2])
    R = Q / pi[None, :]
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 0.0)
    single, _ = _triplet_single_changes()
    R = _floor_exchangeabilities(R, single, min_exchangeability)
    return R.astype(dtype), pi.astype(dtype)


def mean_rate(R: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """ Returns the expected substitution rate sum_i pi_i sum_j R_ij pi_j of
    the unnormalized rate matrix Q_ij = R_ij * pi_j (R with zero diagonal),
    i.e. the normalizing factor of Backend.make_rate_matrix.

    Args:
        R: Exchangeabilities of shape (..., d, d).
        pi: Equilibrium frequencies of shape (..., d).

    Returns:
        Array of shape (...).
    """
    R = np.asarray(R, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    return np.einsum("...i,...ij,...j->...", pi, R, pi)


def _positive_frequencies(freqs, name, shape):
    freqs = np.asarray(freqs, dtype=np.float64)
    if freqs.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {freqs.shape}.")
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0):
        raise ValueError(f"{name} must be positive.")
    return freqs / freqs.sum(axis=-1, keepdims=True)


def _triplet_single_changes():
    """ Boolean (64, 64) masks of the triplet pairs that differ at exactly one
    position and of the pairs whose single difference is a transition. """
    idx = np.arange(64)
    digits = np.stack([idx // 16, (idx // 4) % 4, idx % 4], axis=1)
    differ = digits[:, None, :] != digits[None, :, :]
    single = differ.sum(axis=-1) == 1
    # a=0 <-> g=2 and c=1 <-> t=3: different digits of equal parity
    same_parity = (digits[:, None, :] - digits[None, :, :]) % 2 == 0
    transition = single & np.any(differ & same_parity, axis=-1)
    return single, transition


def _floor_exchangeabilities(R, single, min_exchangeability):
    if min_exchangeability < 0:
        raise ValueError("min_exchangeability must be non-negative.")
    floor = min_exchangeability * R[single].mean()
    R = np.maximum(R, floor)
    np.fill_diagonal(R, 0.0)
    return R

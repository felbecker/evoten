"""Tests for the codon and triplet substitution models (pytest).

    pytest test/test_codon_models.py
"""
import itertools
from collections import Counter

import numpy as np
import pytest
import scipy.linalg

from evoten import substitution_models as sm
from evoten import util

LABELS = util.tuple_labels(3)

# Codon counts per amino acid of the standard genetic code.
STANDARD_DEGENERACY = dict(A=4, R=6, N=2, D=2, C=2, Q=2, E=2, G=4, H=2, I=3,
                           L=6, K=2, M=1, F=2, P=4, S=6, T=4, W=1, Y=2, V=4)


def _hamming(i, j):
    return sum(a != b for a, b in zip(LABELS[i], LABELS[j]))


def test_genetic_code_standard():
    code = sm.genetic_code()
    assert code.shape == (64,)
    counts = Counter(code.tolist())
    assert counts.pop("*") == 3
    assert counts == STANDARD_DEGENERACY
    assert [LABELS[i] for i in np.flatnonzero(code == "*")] == ["taa", "tag", "tga"]
    for codon, aa in [("atg", "M"), ("tgg", "W"), ("ttt", "F"), ("aag", "K"),
                      ("gct", "A"), ("tgc", "C"), ("cgg", "R"), ("agt", "S")]:
        assert code[LABELS.index(codon)] == aa
    with pytest.raises(ValueError):
        sm.genetic_code("vertebrate_mitochondrial")


def test_nucleotide_gtr():
    R, pi = sm.nucleotide_gtr(kappa=4.0, freqs=(1, 2, 3, 4))
    np.testing.assert_allclose(pi, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(R, R.T)
    assert R[0, 2] == R[1, 3] == 4.0 and R[0, 1] == R[0, 3] == R[1, 2] == R[2, 3] == 1.0
    R, _ = sm.nucleotide_gtr(rates=(1, 2, 3, 4, 5, 6))
    assert [R[0, 1], R[0, 2], R[0, 3], R[1, 2], R[1, 3], R[2, 3]] == [1, 2, 3, 4, 5, 6]
    assert np.all(np.diag(R) == 0)
    with pytest.raises(ValueError):
        sm.nucleotide_gtr(rates=(1,) * 6, kappa=2.0)


def test_codon_frequencies():
    stop = sm.genetic_code() == "*"
    f = sm.codon_frequencies()
    assert f.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(f[stop], 1e-6 / 3)
    np.testing.assert_allclose(f[~stop], (1 - 1e-6) / 61)
    nf = np.array([[0.1, 0.2, 0.3, 0.4], [0.25] * 4, [0.4, 0.3, 0.2, 0.1]])
    f3x4 = sm.codon_frequencies(nf, stop_freq=1e-4)
    base = {"a": 0, "c": 1, "g": 2, "t": 3}
    raw = np.array([np.prod([nf[p, base[c]] for p, c in enumerate(codon)]) for codon in LABELS])
    raw[stop] = 0.0
    expected = raw / raw.sum() * (1 - 1e-4)
    expected[stop] = 1e-4 / 3
    np.testing.assert_allclose(f3x4, expected)
    f1x4 = sm.codon_frequencies([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(f1x4, sm.codon_frequencies([[0.1, 0.2, 0.3, 0.4]] * 3))
    with pytest.raises(ValueError):
        sm.codon_frequencies(stop_freq=0.0)


def test_gy94_structure():
    kappa, omega, floor = 4.0, 0.2, 1e-3
    R, pi = sm.gy94(kappa=kappa, omega=omega, min_exchangeability=floor)
    code = sm.genetic_code()
    assert R.shape == (64, 64) and pi.shape == (64,)
    np.testing.assert_array_equal(R, R.T)
    off_diagonal = ~np.eye(64, dtype=bool)
    assert np.all(np.diag(R) == 0) and np.all(R[off_diagonal] > 0)
    np.testing.assert_allclose(pi, sm.codon_frequencies())

    single = np.zeros((64, 64), dtype=bool)
    expected = np.zeros((64, 64))
    for i, j in itertools.product(range(64), repeat=2):
        if _hamming(i, j) == 1:
            p = next(p for p in range(3) if LABELS[i][p] != LABELS[j][p])
            pair = {LABELS[i][p], LABELS[j][p]}
            value = kappa if pair in ({"a", "g"}, {"c", "t"}) else 1.0
            if code[i] != code[j]:
                value *= omega
            single[i, j] = True
            expected[i, j] = value
    f = floor * expected[single].mean()
    np.testing.assert_allclose(R[single], np.maximum(expected[single], f))
    np.testing.assert_allclose(R[off_diagonal & ~single], f)

    ctt, ctc, cta = LABELS.index("ctt"), LABELS.index("ctc"), LABELS.index("cta")
    assert R[ctt, ctc] == pytest.approx(kappa)          # synonymous transition (Leu)
    assert R[ctt, cta] == pytest.approx(1.0)            # synonymous transversion (Leu)
    aaa, aca = LABELS.index("aaa"), LABELS.index("aca")
    assert R[aaa, aca] == pytest.approx(omega)          # nonsynonymous transversion (Lys -> Thr)


def test_gy94_rate_matrix_is_reversible_with_mean_rate_one():
    torch = pytest.importorskip("torch")
    from evoten.backend_pytorch import BackendTorch
    pi = sm.codon_frequencies([[0.1, 0.2, 0.3, 0.4]] * 3)
    R, pi = sm.gy94(kappa=4.0, omega=0.1, codon_freqs=pi)
    Q = BackendTorch().make_rate_matrix(torch.tensor(R), torch.tensor(pi)).numpy()
    flux = pi[:, None] * Q
    np.testing.assert_allclose(flux, flux.T, atol=1e-15)
    np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-12)
    assert -np.sum(pi * np.diag(Q)) == pytest.approx(1.0)
    assert sm.mean_rate(R, pi) == pytest.approx(np.sum(pi[:, None] * R * pi[None, :]))


def test_triplet_independent_is_kronecker_sum():
    models = [sm.nucleotide_gtr(rates=(1.0, 2.0, 0.5, 0.7, 3.0, 1.2), freqs=(0.3, 0.2, 0.2, 0.3)),
              sm.nucleotide_gtr(kappa=4.0, freqs=(0.1, 0.4, 0.4, 0.1)),
              sm.nucleotide_gtr()]
    R, pi = sm.triplet_independent(models, min_exchangeability=0.0)

    def rate_matrix(R_nuc, pi_nuc):
        Q = R_nuc * pi_nuc[None, :]
        Q -= np.diag(Q.sum(axis=1))
        return Q / -np.sum(pi_nuc * np.diag(Q))

    Qs = [rate_matrix(*m) for m in models]
    Q = R * pi[None, :]
    Q -= np.diag(Q.sum(axis=1))
    # the first base is the most significant digit of the state index
    aca = LABELS.index("aca")
    assert Q[aca, LABELS.index("gca")] == pytest.approx(Qs[0][0, 2])
    assert Q[aca, LABELS.index("ata")] == pytest.approx(Qs[1][1, 3])
    assert Q[aca, LABELS.index("acg")] == pytest.approx(Qs[2][0, 2])
    I = np.eye(4)
    kron_sum = (np.kron(np.kron(Qs[0], I), I) + np.kron(np.kron(I, Qs[1]), I)
                + np.kron(np.kron(I, I), Qs[2]))
    np.testing.assert_allclose(Q, kron_sum, atol=1e-12)
    np.testing.assert_allclose(pi, np.kron(np.kron(models[0][1], models[1][1]), models[2][1]))
    assert sm.mean_rate(R, pi) == pytest.approx(3.0)
    t = 0.37
    P = np.kron(np.kron(scipy.linalg.expm(Qs[0] * t), scipy.linalg.expm(Qs[1] * t)),
                scipy.linalg.expm(Qs[2] * t))
    np.testing.assert_allclose(scipy.linalg.expm(Q * t), P, atol=1e-10)

    # one model for all positions, with the floor on multi-nucleotide changes
    R0, _ = sm.triplet_independent(models[1], min_exchangeability=0.0)
    R1, pi1 = sm.triplet_independent(models[1], min_exchangeability=1e-3)
    single = np.array([[_hamming(i, j) == 1 for j in range(64)] for i in range(64)])
    multi = ~single & ~np.eye(64, dtype=bool)
    assert np.all(R0[multi] == 0)
    np.testing.assert_allclose(R1[multi], 1e-3 * R0[single].mean())
    np.testing.assert_allclose(R1[single], R0[single])
    np.testing.assert_allclose(pi1, np.kron(np.kron(models[1][1], models[1][1]), models[1][1]))


def test_float32_kernels_reproduce_the_rate_matrix():
    """The models are representable by softplus/softmax kernels in float32."""
    torch = pytest.importorskip("torch")
    from evoten.backend_pytorch import BackendTorch
    backend = BackendTorch()
    for R, pi in [sm.gy94(kappa=4.0, omega=0.03, codon_freqs=sm.codon_frequencies([0.3, 0.2, 0.2, 0.3])),
                  sm.triplet_independent(sm.nucleotide_gtr(kappa=4.0, freqs=(0.3, 0.2, 0.2, 0.3)))]:
        scale = R[~np.eye(64, dtype=bool)].mean()
        R_kernel = backend.inverse_softplus(torch.tensor(R / scale)).float()
        pi_kernel = torch.tensor(np.log(pi) - np.log(pi).mean()).float()
        Q32 = backend.make_rate_matrix(backend.make_symmetric_pos_semidefinite(R_kernel),
                                       backend.make_equilibrium(pi_kernel))
        Q64 = backend.make_rate_matrix(torch.tensor(R), torch.tensor(pi))
        np.testing.assert_allclose(Q32.double().numpy(), Q64.numpy(), rtol=2e-4, atol=1e-9)

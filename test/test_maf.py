"""Tests for evoten.maf (pytest; the round trip through taffy needs taffy).

    pytest test/test_maf.py
"""
import io

import numpy as np
import pytest

from evoten import maf, util

LABELS = util.tuple_labels(3)


def _parse_maf(text):
    """Blocks of a MAF text as lists of (name, start, size, strand, src_size, text) rows."""
    lines = text.split("\n")
    assert lines[0].startswith("##maf")
    blocks = []
    for line in lines[1:]:
        if line.startswith("a"):
            blocks.append([])
        elif line.startswith("s "):
            _, name, start, size, strand, src_size, seq = line.split()
            blocks[-1].append((name, int(start), int(size), strand, int(src_size), seq))
    return blocks


def test_separator_layout_coordinates_and_mask():
    rng = np.random.default_rng(0)
    names = ["A", "B", "C"]
    states = rng.integers(0, 64, size=(3, 25))
    mask = np.zeros(states.shape, dtype=bool)
    mask[2, 4] = True
    buf = io.StringIO()
    info = maf.write_maf(buf, states, names, mask=mask, block_units=10, reference="B",
                         comments=["seed=0"])
    assert buf.getvalue().split("\n")[:3] == ["##maf version=1 scoring=none", "# seed=0", ""]
    blocks = _parse_maf(buf.getvalue())
    assert [block[0][2] for block in blocks] == [40, 40, 20]
    assert info["n_blocks"] == 3 and info["src_size"] == 100 and info["cols_per_unit"] == 4
    assert info["block_offsets"].tolist() == [0, 10, 20, 25]
    assert info["row_names"] == ["B.sim", "A.sim", "C.sim"]
    for b, block in enumerate(blocks):
        assert [row[0] for row in block] == ["B.sim", "A.sim", "C.sim"]
        for name, start, size, strand, src_size, seq in block:
            assert (start, strand, src_size, size) == (40 * b, "+", 100, len(seq))
            r = names.index(name.split(".")[0])
            for u in range(size // 4):
                unit = seq[4 * u:4 * u + 4]
                expected = "NNN" if mask[r, 10 * b + u] else LABELS[states[r, 10 * b + u]].upper()
                assert unit == expected + "N"


def test_contiguous_layout_merges_a_short_last_block():
    buf = io.StringIO()
    info = maf.write_maf(buf, np.zeros((2, 21), dtype=int), ["x", "y"], layout="contiguous",
                         block_units=10, contig=None)
    blocks = _parse_maf(buf.getvalue())
    assert [block[0][2] for block in blocks] == [30, 33]          # 1-codon tail (3 < 10 columns) merged
    assert blocks[1][0][:2] == ("x", 30) and blocks[1][1][4] == 63
    assert info["block_offsets"].tolist() == [0, 10, 21]
    with pytest.raises(ValueError):                               # 9 columns in total
        maf.write_maf(io.StringIO(), np.zeros((2, 3), dtype=int), ["x", "y"], layout="contiguous")
    with pytest.raises(ValueError):                               # blocks of 2 codons = 8 columns
        maf.write_maf(io.StringIO(), np.zeros((2, 30), dtype=int), ["x", "y"], block_units=2)
    with pytest.raises(ValueError):                               # duplicate row names
        maf.write_maf(io.StringIO(), np.zeros((2, 30), dtype=int), ["x", "x"])


def test_streaming_writes_are_identical_to_one_write(tmp_path):
    rng = np.random.default_rng(1)
    states = rng.integers(0, 64, size=(4, 1003))
    names = ["a", "b", "c", "d"]
    whole = io.StringIO()
    maf.write_maf(whole, states, names, block_units=100)
    parts = io.StringIO()
    with maf.MafWriter(parts, names, 1003, block_units=100) as writer:
        for lo in range(0, 1003, 77):
            writer.write(states[:, lo:lo + 77])
    assert parts.getvalue() == whole.getvalue()
    path = tmp_path / "x.maf"
    maf.write_maf(str(path), states, names, block_units=100)
    assert path.read_text() == whole.getvalue()
    writer = maf.MafWriter(io.StringIO(), names, 1003)
    writer.write(states[:, :10])
    with pytest.raises(ValueError):                               # fewer units than announced
        writer.close()


@pytest.mark.parametrize("gap_separate_state", [0, 1])
def test_triplet_encoding_scores_only_the_codons(gap_separate_state):
    rng = np.random.default_rng(2)
    n = 12
    states = rng.integers(0, 64, size=(3, n))
    buf = io.StringIO()
    maf.write_maf(buf, states, ["a", "b", "c"], block_units=n)
    (block,) = _parse_maf(buf.getvalue())
    tt, first = util.tuple_array([row[5] for row in block], k=3, gap_separate_state=gap_separate_state)
    assert tt.shape == (3, 4 * n - 2, 64 + gap_separate_state)
    np.testing.assert_array_equal(first, np.arange(4 * n - 2))
    in_frame = first % 4 == 0
    np.testing.assert_array_equal(tt[:, in_frame].argmax(-1), states)
    assert np.all(tt[:, in_frame].sum(-1) == 1)
    assert np.all(tt[:, ~in_frame] == 1)


def test_taffy_reads_the_written_maf(tmp_path):
    taffy_lib = pytest.importorskip("taffy.lib")
    states = np.random.default_rng(3).integers(0, 64, size=(3, 30))
    path = tmp_path / "sim.maf"
    maf.write_maf(str(path), states, ["x", "GCF_000001.2", "hg38"], block_units=10, reference="hg38")
    expected = _parse_maf(path.read_text())
    read = []
    for block in taffy_lib.AlignmentReader(str(path)):
        read.append([(row.sequence_name(), row.start(), row.length(), "+" if row.strand() else "-",
                      row.sequence_length(), row.bases()) for row in block])
    assert read == expected
    assert [row[0] for row in read[0]] == ["hg38.sim", "x.sim", "GCF_000001.2.sim"]

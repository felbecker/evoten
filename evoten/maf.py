"""Writing simulated alignments of k-mer states (e.g. codons) as MAF files.

numpy only. The files can be read with taffy (``taffy.lib.AlignmentReader``)
or converted to TAF (``taffy view -i sim.maf -o sim.taf``). Every row is a
gap-free contig named ``<leaf>.<contig>`` and the reference row comes first in
every block.

With ``layout="separator"``, every k-mer is followed by one separator column
(``N`` by default). Encodings that use all overlapping k-mers of a row, such
as util.tuple_array, then find the separator in every k-mer that straddles two
simulated units; such k-mers are ambiguous (all-ones, i.e. missing data) and
only the simulated k-mers carry likelihood.
"""
import os
from collections.abc import Iterable, Sequence

import numpy as np

from evoten import util


def block_layout(
    n_units: int,
    block_units: int,
    cols_per_unit: int,
    min_block_columns: int = 10
) -> np.ndarray:
    """ Splits an alignment of n_units units into blocks of block_units
    units. A last block with fewer than min_block_columns columns is merged
    into the previous block.

    Args:
        n_units: Number of units (k-mers) of the alignment.
        block_units: Number of units per block.
        cols_per_unit: Number of alignment columns per unit.
        min_block_columns: Minimum number of columns of a block.

    Returns:
        Unit offsets of the blocks, shape (n_blocks + 1,), from 0 to n_units.
    """
    n_units, block_units = int(n_units), int(block_units)
    cols_per_unit = int(cols_per_unit)
    if n_units < 1 or block_units < 1 or cols_per_unit < 1:
        raise ValueError(
            "n_units, block_units and cols_per_unit must be positive."
        )
    if n_units * cols_per_unit < min_block_columns:
        raise ValueError(
            f"The alignment has {n_units * cols_per_unit} columns, fewer than "
            f"min_block_columns={min_block_columns}."
        )
    if n_units > block_units and block_units * cols_per_unit < min_block_columns:
        raise ValueError(
            f"Blocks of {block_units} units have fewer than "
            f"min_block_columns={min_block_columns} columns."
        )
    offsets = list(range(0, n_units, block_units)) + [n_units]
    if (len(offsets) > 2
            and (offsets[-1] - offsets[-2]) * cols_per_unit < min_block_columns):
        del offsets[-2]
    return np.asarray(offsets, dtype=np.int64)


class MafWriter:
    """ Streams alignments of k-mer states into a MAF file.

    Args:
        file: Path or writable text file handle.
        leaf_names: Names of the rows of the states passed to write.
        n_units: Total number of units (k-mers per row) that will be written.
        k: k-mer length; states are indices into util.tuple_labels(k).
        layout: "separator" writes every unit followed by the separator
            column, "contiguous" writes the units without separators.
        separator: Separator character.
        block_units: Number of units per MAF block, see block_layout.
        reference: Index or name of the leaf whose row comes first in every
            block (taffy and Agrippa treat the first row as the reference).
        contig: Rows are named f"{leaf}.{contig}", or just by the leaf name if
            contig is None.
        comments: Strings written as "# " lines after the header, e.g.
            "key=value".
        min_block_columns: See block_layout.
        mask_char: Character of masked units. The default "N" makes masked
            units missing data without the gaps that tuple encodings skip.
    """
    def __init__(
        self,
        file,
        leaf_names: Sequence[str],
        n_units: int,
        *,
        k: int = 3,
        layout: str = "separator",
        separator: str = "N",
        block_units: int = 250,
        reference: int | str = 0,
        contig: str | None = "sim",
        comments: Iterable[str] = (),
        min_block_columns: int = 10,
        mask_char: str = "N"
    ):
        if layout not in ("separator", "contiguous"):
            raise ValueError(
                f"layout must be 'separator' or 'contiguous', got {layout!r}."
            )
        for char, name in ((separator, "separator"), (mask_char, "mask_char")):
            if len(char) != 1 or char.isspace() or char == "-":
                raise ValueError(f"{name} must be one character other than "
                                 "whitespace and '-'.")
        self.leaf_names = [str(name) for name in leaf_names]
        num_rows = len(self.leaf_names)
        if num_rows < 1:
            raise ValueError("At least one row is required.")
        if isinstance(reference, str):
            if reference not in self.leaf_names:
                raise ValueError(f"Unknown reference leaf {reference!r}.")
            reference = self.leaf_names.index(reference)
        if not 0 <= int(reference) < num_rows:
            raise ValueError(f"reference index {reference} is out of range.")
        self._order = [int(reference)] + [
            i for i in range(num_rows) if i != int(reference)
        ]
        names = [n if contig is None else f"{n}.{contig}"
                 for n in self.leaf_names]
        if (len(set(names)) != num_rows
                or any(not n or any(c.isspace() for c in n) for n in names)):
            raise ValueError("Row names must be unique, non-empty and "
                             "without whitespace.")
        self.row_names = [names[i] for i in self._order]
        comments = [str(c) for c in comments]
        if any("\n" in c for c in comments):
            raise ValueError("Comments must not contain newlines.")

        self.k = int(k)
        self.n_units = int(n_units)
        self.layout = layout
        self.cols_per_unit = self.k + 1 if layout == "separator" else self.k
        self.block_offsets = block_layout(
            self.n_units, block_units, self.cols_per_unit, min_block_columns
        )
        self.src_size = self.n_units * self.cols_per_unit
        labels = "".join(util.tuple_labels(self.k)).upper().encode("ascii")
        self._lut = np.frombuffer(labels, dtype=np.uint8).reshape(-1, self.k)
        self._separator = ord(separator)
        self._mask_char = ord(mask_char)
        self._pending = np.zeros((num_rows, 0), dtype=np.int64)
        self._pending_mask = np.zeros((num_rows, 0), dtype=bool)
        self._next_block = 0
        self._units_received = 0
        self._closed = False
        self._info = None

        if isinstance(file, (str, os.PathLike)):
            self._fh = open(file, "w")
            self._owns_file = True
        else:
            self._fh = file
            self._owns_file = False
        self._fh.write("##maf version=1 scoring=none\n")
        for comment in comments:
            self._fh.write(f"# {comment}\n")
        self._fh.write("\n")


    def write(self, states, mask=None) -> None:
        """ Appends units to the alignment and writes all completed blocks.

        Args:
            states: Integer array (num_rows, n) of k-mer indices < 4**k, rows
                in the order of leaf_names.
            mask: Optional boolean array (num_rows, n); True marks missing
                units, which are written as mask_char * k.
        """
        if self._closed:
            raise ValueError("The writer is closed.")
        states = np.asarray(states)
        num_rows = len(self.leaf_names)
        if states.ndim != 2 or states.shape[0] != num_rows:
            raise ValueError(f"states must have shape ({num_rows}, n), "
                             f"got {states.shape}.")
        if not np.issubdtype(states.dtype, np.integer):
            raise TypeError("states must be integers.")
        if states.size and (states.min() < 0
                            or states.max() >= self._lut.shape[0]):
            raise ValueError(f"states must be in [0, {self._lut.shape[0]}).")
        if mask is None:
            mask = np.zeros(states.shape, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != states.shape:
                raise ValueError("mask must have the shape of states.")
        if self._units_received + states.shape[1] > self.n_units:
            raise ValueError(f"More than n_units={self.n_units} units.")
        self._units_received += states.shape[1]
        self._pending = np.concatenate(
            [self._pending, states.astype(np.int64)], axis=1
        )
        self._pending_mask = np.concatenate([self._pending_mask, mask], axis=1)
        while self._next_block < len(self.block_offsets) - 1:
            lo = int(self.block_offsets[self._next_block])
            n = int(self.block_offsets[self._next_block + 1]) - lo
            if self._pending.shape[1] < n:
                break
            self._write_block(lo, self._pending[:, :n], self._pending_mask[:, :n])
            self._pending = self._pending[:, n:]
            self._pending_mask = self._pending_mask[:, n:]
            self._next_block += 1


    def close(self) -> dict:
        """ Closes the writer (and the file if the writer opened it).

        Returns:
            A dict with n_blocks, n_units, n_columns (= src_size),
            cols_per_unit, layout, block_offsets (unit offsets) and row_names
            (in file order).
        """
        if self._closed:
            return self._info
        self._closed = True
        try:
            if self._units_received != self.n_units:
                raise ValueError(f"Wrote {self._units_received} of "
                                 f"n_units={self.n_units} units.")
        finally:
            if self._owns_file:
                self._fh.close()
        self._info = dict(
            n_blocks=len(self.block_offsets) - 1,
            n_units=self.n_units,
            n_columns=self.src_size,
            src_size=self.src_size,
            cols_per_unit=self.cols_per_unit,
            layout=self.layout,
            block_offsets=self.block_offsets.copy(),
            row_names=list(self.row_names),
        )
        return self._info


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self._closed = True
            if self._owns_file:
                self._fh.close()
        return False


    def _write_block(self, unit_offset, states, mask):
        chars = self._lut[states]                               # (rows, n, k)
        chars[mask] = self._mask_char
        if self.layout == "separator":
            sep = np.full(chars.shape[:2] + (1,), self._separator, np.uint8)
            chars = np.concatenate([chars, sep], axis=2)
        text = chars.reshape(chars.shape[0], -1)
        start = unit_offset * self.cols_per_unit
        size = text.shape[1]
        lines = ["a\n"]
        for name, row in zip(self.row_names, self._order):
            seq = text[row].tobytes().decode("ascii")
            lines.append(f"s {name} {start} {size} + {self.src_size} {seq}\n")
        lines.append("\n")
        self._fh.write("".join(lines))


def write_maf(file, states, leaf_names, *, mask=None, **kwargs) -> dict:
    """ Writes an alignment of k-mer states (num_rows, n) as a MAF file.

    Args:
        file: Path or writable text file handle.
        states: Integer array (num_rows, n) of k-mer indices.
        leaf_names: Names of the rows of states.
        mask: Optional boolean array (num_rows, n) of missing units.
        kwargs: Passed to MafWriter.

    Returns:
        The dict returned by MafWriter.close.
    """
    states = np.asarray(states)
    writer = MafWriter(file, leaf_names, states.shape[1], **kwargs)
    with writer:
        writer.write(states, mask)
    return writer.close()

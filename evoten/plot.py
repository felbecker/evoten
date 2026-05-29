"""Utility for plotting substitution model rate matrices as heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evoten import util


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_rate_matrix_numpy(
    exchangeabilities: np.ndarray,
    equilibrium: np.ndarray,
) -> np.ndarray:
    """Pure-numpy GTR rate matrix fallback (no backend required)."""
    Q = exchangeabilities * equilibrium[np.newaxis, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mu = -np.dot(equilibrium, np.diag(Q))
    if mu > 0:
        Q /= mu
    return Q


def _compute_rate_matrix(
    exchangeabilities: np.ndarray,
    equilibrium: np.ndarray,
) -> np.ndarray:
    """Compute Q via the active evoten backend; fall back to numpy on failure."""
    try:
        from evoten.backend import backend
        Q = backend.make_rate_matrix(exchangeabilities, equilibrium)
        if hasattr(Q, "numpy"):
            Q = Q.numpy()
        return np.asarray(Q, dtype=float)
    except Exception:
        return _make_rate_matrix_numpy(
            np.asarray(exchangeabilities, dtype=float),
            np.asarray(equilibrium, dtype=float),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_rate_matrix(
    Q: np.ndarray,
    alphabet: str,
    output_path: str | Path,
    title: str = "Rate Matrix",
    cmap: str = "RdBu_r",
    mask_upper_triangle: bool = False,
    colorbar_label: str = "Rate",
) -> None:
    """Plot a matrix as a heatmap and save to a PDF.

    Args:
        Q: Square matrix of shape (D, D).
        alphabet: String of length D used as tick labels on both axes.
        output_path: Destination PDF path.
        title: Plot title.
        cmap: Matplotlib colormap name.
        mask_upper_triangle: If True, entries on and above the diagonal are
            masked. Useful for symmetric matrices such as exchangeabilities.
        colorbar_label: Label for the color scale legend (default: "Rate").
    """
    D = Q.shape[0]
    if Q.shape != (D, D):
        raise ValueError(f"Q must be square, got shape {Q.shape}.")
    if len(alphabet) != D:
        raise ValueError(
            f"len(alphabet) ({len(alphabet)}) must equal matrix dimension ({D})."
        )

    data = Q.copy().astype(float)
    if mask_upper_triangle:
        upper_idx = np.triu_indices(D, k=0)
        data[upper_idx] = np.nan

    fig_size = max(5.0, D * 0.38)
    fig, ax = plt.subplots(figsize=(fig_size + 1.2, fig_size))

    im = ax.imshow(data, cmap=cmap, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    labels = list(alphabet)
    ticks = list(range(D))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_rate_matrix_diff(
    Q1: np.ndarray,
    Q2: np.ndarray,
    alphabet: str,
    output_path: str | Path,
    title: str = "Rate Matrix Difference (Q1 − Q2)",
    cmap: str = "RdBu_r",
) -> None:
    """Plot the element-wise difference Q1 − Q2 as a heatmap and save to a PDF.

    The color scale is symmetric around zero so that positive values (Q1 > Q2)
    and negative values (Q1 < Q2) are directly comparable.  The diagonal is
    always shown.

    Args:
        Q1: First rate matrix of shape (D, D).
        Q2: Second rate matrix of shape (D, D), must match Q1's shape.
        alphabet: String of length D used as tick labels on both axes.
        output_path: Destination PDF path.
        title: Plot title.
        cmap: Matplotlib colormap name (a diverging map is recommended).
    """
    if Q1.shape != Q2.shape:
        raise ValueError(
            f"Q1 and Q2 must have the same shape, got {Q1.shape} vs {Q2.shape}."
        )
    diff = Q1.astype(float) - Q2.astype(float)
    plot_rate_matrix(
        diff, alphabet, output_path,
        title=title, cmap=cmap,
        colorbar_label="Rate Difference",
    )


def plot_exchangeabilities(
    R: np.ndarray,
    alphabet: str,
    output_path: str | Path,
    title: str = "Exchangeabilities",
    cmap: str = "RdBu_r",
) -> None:
    """Plot an exchangeability matrix R as a lower-triangular heatmap.

    Because R is symmetric, only the lower triangle (i > j) carries unique
    information.  The upper triangle and diagonal are always masked.

    Args:
        R: Symmetric exchangeability matrix of shape (D, D).
        alphabet: String of length D used as tick labels on both axes.
        output_path: Destination PDF path.
        title: Plot title.
        cmap: Matplotlib colormap name.
    """
    plot_rate_matrix(
        R, alphabet, output_path,
        title=title, cmap=cmap,
        mask_upper_triangle=True,
        colorbar_label="Exchangeability",
    )


def plot_exchangeabilities_diff(
    R1: np.ndarray,
    R2: np.ndarray,
    alphabet: str,
    output_path: str | Path,
    title: str = "Exchangeability Difference (R1 − R2)",
    cmap: str = "RdBu_r",
) -> None:
    """Plot the lower-triangular difference R1 − R2 as a heatmap.

    The diagonal is always masked.

    Args:
        R1: First exchangeability matrix of shape (D, D).
        R2: Second exchangeability matrix of shape (D, D), must match R1's shape.
        alphabet: String of length D used as tick labels on both axes.
        output_path: Destination PDF path.
        title: Plot title.
        cmap: Matplotlib colormap name (a diverging map is recommended).
    """
    if R1.shape != R2.shape:
        raise ValueError(
            f"R1 and R2 must have the same shape, got {R1.shape} vs {R2.shape}."
        )
    diff = R1.astype(float) - R2.astype(float)
    plot_rate_matrix(
        diff, alphabet, output_path,
        title=title, cmap=cmap,
        mask_upper_triangle=True,
        colorbar_label="Exchangeability Difference",
    )


def plot_substitution_model(
    exchangeabilities: np.ndarray,
    equilibrium: np.ndarray,
    alphabet: str,
    output_path: str | Path,
    title: str = "Rate Matrix",
    cmap: str = "RdBu_r",
) -> None:
    """Compute a rate matrix from a substitution model and plot it as a heatmap.

    Uses the active evoten backend (``make_rate_matrix``); falls back to a
    pure-numpy implementation when no backend has been loaded.  The diagonal
    is always shown.

    Args:
        exchangeabilities: Symmetric exchangeability matrix of shape (D, D).
        equilibrium: Equilibrium frequencies of shape (D,), must sum to 1.
        alphabet: String of length D; tick labels on both axes.  Must already
            correspond to the column order of ``exchangeabilities``.
        output_path: Destination PDF path.
        title: Plot title.
        cmap: Matplotlib colormap name.
    """
    Q = _compute_rate_matrix(
        np.asarray(exchangeabilities),
        np.asarray(equilibrium),
    )
    plot_rate_matrix(Q, alphabet, output_path, title=title, cmap=cmap)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a substitution-model rate matrix as a PDF heatmap. "
            "Model file format: first line = equilibrium frequencies, "
            "remaining lines = lower-triangular exchangeabilities "
            "(see evoten.util.parse_rate_model)."
        )
    )
    parser.add_argument(
        "model_path", type=Path,
        help="Path to the rate model file.",
    )
    parser.add_argument(
        "output_path", type=Path,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--original_alphabet",
        type=str,
        default="ARNDCQEGHILKMFPSTWYV",
        help=(
            "Alphabet matching the column order in the model file "
            "(default: ARNDCQEGHILKMFPSTWYV)."
        ),
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="ARNDCQEGHILKMFPSTWYV",
        help=(
            "Alphabet matching the column order in the model file "
            "(default: ARNDCQEGHILKMFPSTWYV)."
        ),
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Plot title (default: model file name).",
    )
    parser.add_argument(
        "--cmap", type=str, default="RdBu_r",
        help="Matplotlib colormap name (default: RdBu_r).",
    )
    parser.add_argument(
        "--diff-model", type=Path, default=None, metavar="MODEL_PATH",
        help=(
            "Path to a second rate model file.  When given, the plot shows "
            "the difference (first model − second model) instead of the "
            "first model alone.  Both models must share the same alphabet."
        ),
    )
    parser.add_argument(
        "--diff-original-alphabet",
        type=str,
        default=None,
        metavar="ALPHABET",
        help=(
            "Alphabet matching the column order of the --diff-model file. "
            "Defaults to --original_alphabet when omitted."
        ),
    )
    parser.add_argument(
        "--plot-r",
        action="store_true",
        help=(
            "Plot the exchangeability matrix R instead of the rate matrix Q. "
            "Only the lower triangular part is shown (R is symmetric)."
        ),
    )
    args = parser.parse_args()

    title = args.title if args.title is not None else args.model_path.name

    R1_raw, pi1, _ = util.parse_rate_model(args.model_path)
    R1_raw, pi1 = util.permute_rate_model(R1_raw, pi1, args.original_alphabet, args.alphabet)

    if args.diff_model is not None:
        diff_orig_alphabet = (
            args.diff_original_alphabet
            if args.diff_original_alphabet is not None
            else args.original_alphabet
        )
        R2_raw, pi2, _ = util.parse_rate_model(args.diff_model)
        R2_raw, pi2 = util.permute_rate_model(R2_raw, pi2, diff_orig_alphabet, args.alphabet)
        if title == args.model_path.name:
            title = f"{args.model_path.name} − {args.diff_model.name}"
        if args.plot_r:
            plot_exchangeabilities_diff(
                R1_raw, R2_raw, args.alphabet, args.output_path,
                title=title, cmap=args.cmap,
            )
        else:
            Q1 = _compute_rate_matrix(R1_raw, pi1)
            Q2 = _compute_rate_matrix(R2_raw, pi2)
            plot_rate_matrix_diff(
                Q1, Q2, args.alphabet, args.output_path,
                title=title, cmap=args.cmap,
            )
    else:
        if args.plot_r:
            plot_exchangeabilities(
                R1_raw, args.alphabet, args.output_path,
                title=title, cmap=args.cmap,
            )
        else:
            Q1 = _compute_rate_matrix(R1_raw, pi1)
            plot_rate_matrix(
                Q1, args.alphabet, args.output_path,
                title=title, cmap=args.cmap,
            )
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()

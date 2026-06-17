# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Plot the per-query range-hit distribution (scheme A: zero/non-zero + tail hist).

Reads a JSON file produced by compute_range_gt.py (must contain
`per_query_counts`, an array of length n_queries), and writes a 2-panel PNG:

  [ left panel  ]  zero-hit vs nonzero-hit bar counts (with absolute + pct labels)
  [ right panel ]  log-x histogram of nonzero counts, with quantile markers

Recommended invocation (bypasses the surrounding DiskANN project's pyproject):

    uv run --script plot_per_query_hist.py --input <gt.json>

Output path defaults to <input_stem>_per_query_hist.png next to the input file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per_query_counts distribution (scheme A) from a range-GT JSON."
    )
    p.add_argument("--input", type=pathlib.Path, required=True,
                   help="Path to a range-GT JSON produced by compute_range_gt.py.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output PNG path; defaults to <input_stem>_per_query_hist.png "
                        "next to the input JSON.")
    p.add_argument("--bins", type=int, default=30,
                   help="Number of log-spaced bins for the right panel (default 30).")
    p.add_argument("--dpi", type=int, default=150,
                   help="PNG resolution (default 150).")
    return p.parse_args(argv)


def load_counts(json_path: pathlib.Path) -> tuple[np.ndarray, dict]:
    data = json.loads(json_path.read_text())
    if "per_query_counts" not in data:
        raise KeyError(
            f"{json_path} has no 'per_query_counts' field. "
            f"Regenerate it with the updated compute_range_gt.py."
        )
    counts = np.asarray(data["per_query_counts"], dtype=np.int64)
    if data.get("n_queries") is not None and len(counts) != int(data["n_queries"]):
        raise ValueError(
            f"per_query_counts length {len(counts)} != n_queries {data['n_queries']}"
        )
    return counts, data


def draw_left_panel(ax: plt.Axes, counts: np.ndarray) -> None:
    n = len(counts)
    n_zero = int((counts == 0).sum())
    n_nonzero = n - n_zero

    bars = ax.bar(
        ["zero hit", "nonzero hit"],
        [n_zero, n_nonzero],
        color=["#b0b0b0", "#4c78a8"],
        edgecolor="black", linewidth=0.5,
    )
    for bar, v in zip(bars, [n_zero, n_nonzero]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v}\n({100.0 * v / n:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylabel("Number of queries")
    ax.set_title(f"Any-hit summary  (n_queries = {n})")
    ax.set_ylim(0, max(n_zero, n_nonzero) * 1.18)
    ax.grid(axis="y", alpha=0.3)


def draw_right_panel(ax: plt.Axes, counts: np.ndarray, nbins: int) -> None:
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        ax.text(0.5, 0.5, "No non-zero queries", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Nonzero distribution (n = 0)")
        return

    lo = 1
    hi = int(nonzero.max()) + 1
    bins = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    ax.hist(nonzero, bins=bins, color="#4c78a8",
            edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("In-range hits per query  (log scale)")
    ax.set_ylabel("Number of queries (nonzero only)")
    ax.set_title(f"Nonzero distribution  (n = {len(nonzero)})")
    ax.grid(True, which="both", alpha=0.3)

    # quantile markers — computed on NONZERO so they describe the tail, not the full set
    markers = [
        ("min",    int(nonzero.min()),                  "#555555"),
        ("median", int(np.median(nonzero)),             "#e45756"),
        ("p95",    int(np.percentile(nonzero, 95)),     "#f58518"),
        ("max",    int(nonzero.max()),                  "#72b7b2"),
    ]
    ymax = ax.get_ylim()[1]
    for i, (label, value, color) in enumerate(markers):
        ax.axvline(value, color=color, linestyle="--", alpha=0.85, linewidth=1.2)
        ax.text(value, ymax * (0.96 - i * 0.08), f" {label}={value}",
                color=color, fontsize=9, va="top")


def build_suptitle(meta: dict) -> str:
    dataset = meta.get("dataset", "?")
    r2 = meta.get("radius_squared", float("nan"))
    l2 = float(np.sqrt(r2)) if r2 == r2 else float("nan")  # nan-safe sqrt
    k_ref = meta.get("k_reference", "?")
    pct = meta.get("radius_percentile", "?")
    total = meta.get("total_in_range", "?")
    return (
        f"{dataset}   |   radius (L2) = {l2:.4f}  (r² = {r2:.4f})   "
        f"|   k_ref = {k_ref},  pct = {pct}   |   total_in_range = {total}"
    )


def make_figure(counts: np.ndarray, meta: dict, nbins: int) -> plt.Figure:
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 2.2]},
    )
    draw_left_panel(ax_left, counts)
    draw_right_panel(ax_right, counts, nbins)
    fig.suptitle(build_suptitle(meta), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    counts, meta = load_counts(args.input)
    out_path = args.out or args.input.with_name(
        args.input.stem + "_per_query_hist.png"
    )
    fig = make_figure(counts, meta, args.bins)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}  ({counts.size} queries, "
          f"{int((counts == 0).sum())} zero, {int((counts > 0).sum())} nonzero)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

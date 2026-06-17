#!/usr/bin/env python3
"""Compute recall-QPS Pareto front from the alpha x ef grid and plot it.

Run with an ephemeral env (no repo pyproject needed):
    uv run --with matplotlib --with numpy python scripts/plot_pareto.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_CSV = Path("results/deep10m_alpha_ef_grid.csv")
OUT_PNG = Path("results/deep10m_recall_qps_pareto.png")
OUT_CSV = Path("results/deep10m_recall_qps_pareto_front.csv")


def load_grid(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "alpha": float(r["approx_alpha"]),
                    "ef": int(r["sqg_ef_search"]),
                    "recall": float(r["recall"]),
                    "qps": float(r["qps_median"]),
                }
            )
    return rows


def pareto_front(points):
    """Upper-right frontier: keep points not dominated on (recall, qps).

    p dominated if another q has q.recall >= p.recall and q.qps >= p.qps with
    at least one strict.
    """
    front = []
    for p in points:
        dominated = any(
            (q["recall"] >= p["recall"] and q["qps"] >= p["qps"])
            and (q["recall"] > p["recall"] or q["qps"] > p["qps"])
            for q in points
        )
        if not dominated:
            front.append(p)
    front.sort(key=lambda d: d["recall"])
    return front


def main() -> None:
    rows = load_grid(GRID_CSV)
    front = pareto_front(rows)

    # Save the frontier table.
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recall", "qps_median", "approx_alpha", "sqg_ef_search"])
        for p in front:
            w.writerow([p["recall"], p["qps"], p["alpha"], p["ef"]])

    # Plot: all grid points colored by ef, frontier overlaid.
    efs = sorted({r["ef"] for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, ef in enumerate(efs):
        pts = [r for r in rows if r["ef"] == ef]
        ax.scatter(
            [p["recall"] for p in pts],
            [p["qps"] for p in pts],
            s=28,
            color=cmap(i / max(1, len(efs) - 1)),
            label=f"ef={ef}",
            alpha=0.75,
            zorder=2,
        )
    ax.plot(
        [p["recall"] for p in front],
        [p["qps"] for p in front],
        "-o",
        color="crimson",
        lw=2,
        ms=6,
        label="Pareto front",
        zorder=3,
    )
    # Annotate frontier points with their alpha.
    for p in front:
        ax.annotate(
            f"α={p['alpha']:g}\nef={p['ef']}",
            (p["recall"], p["qps"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="crimson",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS (median of 3 rounds)")
    ax.set_title("deep10m range search: recall-QPS Pareto front (approx_alpha x sqg_ef_search)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"frontier points: {len(front)}")
    for p in front:
        print(f"  recall={p['recall']:.6f}  qps={p['qps']:.1f}  alpha={p['alpha']:g}  ef={p['ef']}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

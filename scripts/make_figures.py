#!/usr/bin/env python3
"""Generate publication figures for TurnDrift-Control.

Outputs to results/figures/. Requires matplotlib.

Figures:
  1. ICR by injection strategy (no defense) — bar chart
  2. Defense comparison — grouped bar chart (once monitors run)
  3. D2 AUROC top-5 layers — bar chart from A4 summary
  4. Fragment innocuousness — single vs multi-fragment comparison

Usage:
    python scripts/make_figures.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGY_LABELS = {
    "single_turn": "Single\nturn",
    "two_fragment": "Two\nfragment",
    "three_fragment": "Three\nfragment",
    "semantic_split": "Semantic\nsplit",
    "delayed_trigger": "Delayed\ntrigger",
}

STRATEGY_ORDER = ["single_turn", "two_fragment", "three_fragment",
                  "semantic_split", "delayed_trigger"]

COLORS = {
    "none": "#4878CF",
    "full_context_monitor": "#6ACC65",
    "per_turn_monitor": "#D65F5F",
    "D1_steering": "#B47CC7",
    "D2_steering": "#C4AD66",
    "full_context_monitor+best_steering": "#77BEDB",
}

DEFENSE_LABELS = {
    "none": "No defense",
    "full_context_monitor": "Full-context monitor",
    "per_turn_monitor": "Per-turn monitor",
    "D1_steering": "D1 steering",
    "D2_steering": "D2 steering",
    "full_context_monitor+best_steering": "Monitor + steering",
}


def load_egta(results_dir: Path, split: str) -> dict | None:
    p = results_dir / "evaluations" / f"egta_{split}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig_icr_by_strategy(egta: dict, fig_dir: Path):
    """Bar chart: ICR by injection strategy, no defense."""
    matrix = egta["payoff_matrix"]
    if "none" not in matrix:
        return

    none_cells = matrix["none"]
    strats = [s for s in STRATEGY_ORDER if s in none_cells]
    asrs = [none_cells[s]["asr"] for s in strats]
    ns = [none_cells[s]["n"] for s in strats]
    labels = [STRATEGY_LABELS.get(s, s) for s in strats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(strats)), asrs, color="#4878CF", edgecolor="white", width=0.6)

    for bar, asr, n in zip(bars, asrs, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{asr:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Injection Compliance Rate (%)")
    ax.set_title(f"ICR by Attack Strategy — No Defense ({egta['split']} split)")
    ax.set_ylim(0, max(asrs) * 1.35 if asrs else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = fig_dir / f"icr_by_strategy_{egta['split']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_defense_comparison(egta: dict, fig_dir: Path):
    """Grouped bar chart: ASR by strategy for each defense."""
    matrix = egta["payoff_matrix"]
    defenders = [d for d in egta["available_defenders"] if d in matrix]

    if len(defenders) < 2:
        print("  Skipping defense comparison (need ≥2 defenders)")
        return

    strats = [s for s in STRATEGY_ORDER if all(s in matrix[d] for d in defenders)]
    n_strats = len(strats)
    n_defs = len(defenders)
    width = 0.8 / n_defs
    x = np.arange(n_strats)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, d in enumerate(defenders):
        asrs = [matrix[d][s]["asr"] for s in strats]
        offset = (i - n_defs/2 + 0.5) * width
        ax.bar(x + offset, asrs, width, label=DEFENSE_LABELS.get(d, d),
               color=COLORS.get(d, f"C{i}"), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strats], fontsize=9)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(f"Defense Comparison ({egta['split']} split)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = fig_dir / f"defense_comparison_{egta['split']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_d2_auroc(results_dir: Path, fig_dir: Path):
    """D2 AUROC by layer from A4 summary."""
    p = results_dir / "pilot" / "a4_summary.json"
    if not p.exists():
        print("  Skipping D2 AUROC (no a4_summary.json)")
        return

    with open(p) as f:
        a4 = json.load(f)

    top5 = a4["top5"]
    layers = [t["layer"] for t in top5]
    aurocs = [t["mean_auroc"] for t in top5]
    fold_aurocs = [t["fold_aurocs"] for t in top5]
    errors = [np.std(fa) for fa in fold_aurocs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(layers)), aurocs, yerr=errors,
                  color="#6ACC65", edgecolor="white", width=0.6,
                  capsize=4, error_kw={"linewidth": 1.2})

    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{auroc:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers], fontsize=9)
    ax.set_ylabel("Mean AUROC (3-fold CV)")
    ax.set_title("D2 Direction — AUROC by Layer (Top 5)")
    ax.set_ylim(0.85, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance")

    fig.tight_layout()
    out = fig_dir / "d2_auroc_by_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_fragment_innocuousness(egta: dict, fig_dir: Path):
    """Single vs multi-fragment ICR comparison."""
    matrix = egta["payoff_matrix"]
    if "none" not in matrix:
        return

    none_cells = matrix["none"]
    single = none_cells.get("single_turn", {}).get("asr", 0)
    multi_strats = ["two_fragment", "three_fragment", "semantic_split", "delayed_trigger"]
    multi_asrs = [none_cells.get(s, {}).get("asr", 0) for s in multi_strats]
    avg_multi = np.mean(multi_asrs)

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ["Single turn", "Avg multi-\nfragment"]
    values = [single, avg_multi]
    colors_bar = ["#D65F5F", "#4878CF"]
    bars = ax.bar(categories, values, color=colors_bar, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Injection Compliance Rate (%)")
    ax.set_title(f"Fragment Innocuousness ({egta['split']} split)")
    ax.set_ylim(0, max(values) * 1.4 if max(values) > 0 else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if single > 0:
        ratio = avg_multi / single
        ax.text(0.95, 0.95, f"Ratio: {ratio:.2f}x",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    fig.tight_layout()
    out = fig_dir / f"fragment_innocuousness_{egta['split']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate TurnDrift figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    egta = load_egta(results_dir, args.split)
    if not egta:
        print(f"No EGTA data for split={args.split}")
        return

    print(f"Generating figures (split={args.split})...\n")

    fig_icr_by_strategy(egta, fig_dir)
    fig_defense_comparison(egta, fig_dir)
    fig_d2_auroc(results_dir, fig_dir)
    fig_fragment_innocuousness(egta, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()

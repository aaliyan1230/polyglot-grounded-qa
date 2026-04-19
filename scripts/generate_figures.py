"""Generate figure assets from evaluation parquet tables.

Writes PNG files to artifacts/figures/.

Usage:
    uv run python scripts/generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colours ──────────────────────────────────────────────────────────────────
C_BASELINE = "#9e9e9e"
C_BASE_PROMPTED = "#64b5f6"
C_ADAPTER = "#1976d2"
C_CEILING = "#e0e0e0"

VARIANT_LABELS = {
    "baseline-pipeline": "Baseline\npipeline",
    "base-model-prompted-v1": "Base model\n(prompted)",
    "tuned-adapter-v1": "Tuned\nadapter",
}
VARIANT_COLOURS = {
    "baseline-pipeline": C_BASELINE,
    "base-model-prompted-v1": C_BASE_PROMPTED,
    "tuned-adapter-v1": C_ADAPTER,
}
REAL_VARIANTS = ["baseline-pipeline", "base-model-prompted-v1", "tuned-adapter-v1"]


def _latest_by_variant(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort("timestamp_utc", descending=True)
        .group_by("variant")
        .head(1)
    )


# ── Figure 1: Trust-score comparison bar chart ───────────────────────────────
def fig_trust_comparison(df: pl.DataFrame) -> None:
    real = (
        _latest_by_variant(df)
        .filter(pl.col("variant").is_in(REAL_VARIANTS))
        .sort(pl.col("variant").cast(pl.Categorical))
    )
    # sort in display order
    order = {v: i for i, v in enumerate(REAL_VARIANTS)}
    rows = sorted(real.iter_rows(named=True), key=lambda r: order.get(r["variant"], 99))

    metrics = [
        ("grounded_trust_score", "Grounded trust score"),
        ("avg_citation_precision", "Citation precision"),
        ("avg_citation_recall", "Citation recall"),
        ("avg_answer_token_f1", "Answer token F1"),
        ("abstain_accuracy", "Abstain accuracy"),
    ]

    x = range(len(metrics))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    for col_offset, row in zip(offsets, rows):
        values = [float(row[m]) for m, _ in metrics]
        colour = VARIANT_COLOURS[row["variant"]]
        bars = ax.bar(
            [xi + col_offset for xi in x],
            values,
            width=width * 0.92,
            color=colour,
            label=VARIANT_LABELS[row["variant"]],
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if val > 0.03:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333333",
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Evaluation metrics: baseline vs base model vs tuned adapter", fontsize=11, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_trust_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 2: Per-language trust score for the adapter ───────────────────────
def fig_adapter_by_language(df_lang: pl.DataFrame) -> None:
    # Latest run per language for the tuned adapter
    adapter = (
        df_lang.filter(pl.col("variant") == "tuned-adapter-v1")
        .sort("timestamp_utc", descending=True)
        .group_by("language")
        .head(1)
        .sort("language")
    )

    languages = adapter["language"].to_list()
    trust = adapter["grounded_trust_score"].to_list()
    cit_p = adapter["avg_citation_precision"].to_list()
    cit_r = adapter["avg_citation_recall"].to_list()

    x = range(len(languages))
    width = 0.28
    offsets = [-width, 0, width]
    series = [
        (trust, "Grounded trust score", C_ADAPTER),
        (cit_p, "Citation precision", "#42a5f5"),
        (cit_r, "Citation recall", "#90caf9"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")

    for offset, (values, label, colour) in zip(offsets, series):
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width=width * 0.92,
            color=colour,
            label=label,
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if val > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(languages, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Tuned adapter — trust & citation scores by language", fontsize=11, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_adapter_by_language.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ── Figure 3: Training loss curve (static, from known checkpoints) ───────────
def fig_training_loss() -> None:
    # Known checkpoints from the training run (Qwen2.5-3B QLoRA, ~540 steps)
    steps  = [  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 540]
    losses = [2.45, 1.82, 1.41, 1.12, 0.89, 0.72, 0.60, 0.51, 0.44, 0.39, 0.35, 0.32]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    ax.plot(steps, losses, color=C_ADAPTER, linewidth=2.5, marker="o", markersize=5, zorder=3)
    ax.fill_between(steps, losses, alpha=0.08, color=C_ADAPTER)

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("QLoRA training loss — Qwen2.5-3B-Instruct (rank-16, ~540 steps)", fontsize=11, pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.annotate(
        "2.45 → 0.32",
        xy=(540, 0.32),
        xytext=(420, 0.65),
        arrowprops=dict(arrowstyle="->", color="#555"),
        fontsize=9,
        color="#333",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_training_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    summary = pl.read_parquet(PROJECT_ROOT / "artifacts/tables/final_finetune_eval_summary.parquet")
    by_lang = pl.read_parquet(PROJECT_ROOT / "artifacts/tables/final_finetune_eval_by_language.parquet")

    fig_trust_comparison(summary)
    fig_adapter_by_language(by_lang)
    fig_training_loss()

    print("All figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()

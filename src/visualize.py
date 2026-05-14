"""
visualize.py
------------
Generates EDA and model evaluation charts using seaborn + matplotlib.

Charts produced:
  1. Correlation heatmap of differential features
  2. Distribution of key differentials by game outcome
  3. ROC curve
  4. Win rate by rest-day advantage bucket
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
VIZ_DIR   = os.path.join(os.path.dirname(__file__), "..", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

DATA_PATH  = os.path.join(DATA_DIR,  "model_ready.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "playoff_predictor.pkl")

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = {"Home Win": "#1565C0", "Away Win": "#C62828"}


# ── 1. Correlation Heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    diff_cols = [c for c in df.columns if c.startswith("DIFF_")]
    corr = df[diff_cols + ["H_WIN"]].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap="coolwarm",
        center=0, linewidths=0.4, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix\n(Differential Stats + Target)", fontsize=15, pad=12)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.show()


# ── 2. Feature Distributions by Outcome ──────────────────────────────────────

TOP_FEATURES = [
    "DIFF_PTS", "DIFF_FG_PCT", "DIFF_FG3_PCT",
    "DIFF_OFF_RTG_PROXY", "DIFF_PLUS_MINUS", "DIFF_REST_DAYS",
]

def plot_distributions(df: pd.DataFrame):
    df = df.copy()
    df["Outcome"] = df["H_WIN"].map({1: "Home Win", 0: "Away Win"})

    available = [f for f in TOP_FEATURES if f in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        sns.histplot(
            data=df, x=feat, hue="Outcome",
            palette=PALETTE, kde=True, alpha=0.5,
            ax=axes[i], bins=30,
        )
        label = feat.replace("DIFF_", "").replace("_", " ").title()
        axes[i].set_title(f"Home − Away: {label}", fontsize=11)
        axes[i].axvline(0, color="black", linestyle="--", linewidth=0.8)
        axes[i].set_xlabel("")

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Key Differential Stats by Game Outcome",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ── 3. ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(df: pd.DataFrame, pipeline):
    diff_cols = [c for c in df.columns if c.startswith("DIFF_")]
    X = df[diff_cols]
    y = df["H_WIN"]

    # Use last 2 seasons as test set (mirrors train.py)
    test_mask = df["SEASON"].isin(["2022-23", "2023-24"])
    X_test, y_test = X[test_mask], y[test_mask]

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1565C0")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Playoff Predictor", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.show()


# ── 4. Win Rate by Rest-Day Advantage ─────────────────────────────────────────

def plot_rest_day_win_rate(df: pd.DataFrame):
    df = df.copy()
    if "DIFF_REST_DAYS" not in df.columns:
        print("DIFF_REST_DAYS not found — skipping rest day plot.")
        return

    bins = [-8, -3, -1, 0, 1, 3, 8]
    labels = ["≤-3", "-1 to -2", "0", "+1", "+2", "≥+3"]
    df["RestBucket"] = pd.cut(df["DIFF_REST_DAYS"], bins=bins, labels=labels)
    win_rate = df.groupby("RestBucket", observed=True)["H_WIN"].agg(["mean", "count"]).reset_index()
    win_rate.columns = ["RestBucket", "HomeWinRate", "Count"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        win_rate["RestBucket"].astype(str),
        win_rate["HomeWinRate"],
        color=[("#C62828" if r < 0.5 else "#1565C0") for r in win_rate["HomeWinRate"]],
        edgecolor="white", linewidth=0.5,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9, label="50% baseline")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("Rest Day Advantage (Home − Away)", fontsize=12)
    ax.set_ylabel("Home Team Win Rate", fontsize=12)
    ax.set_title("Home Win Rate by Rest Day Advantage", fontsize=14)
    ax.legend()

    for bar, (_, row) in zip(bars, win_rate.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={int(row['Count'])}",
            ha="center", va="bottom", fontsize=9, color="gray",
        )

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "rest_day_win_rate.png")
    plt.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize_all():
    print("Loading data and model...")
    df = pd.read_csv(DATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    print("\n[1/4] Correlation heatmap...")
    plot_correlation_heatmap(df)

    print("[2/4] Feature distributions...")
    plot_distributions(df)

    print("[3/4] ROC curve...")
    plot_roc_curve(df, pipeline)

    print("[4/4] Rest day win rate...")
    plot_rest_day_win_rate(df)

    print(f"\nAll charts saved to {VIZ_DIR}/")


if __name__ == "__main__":
    visualize_all()

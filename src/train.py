"""
train.py
--------
Trains a Logistic Regression model on playoff game differential features.

Pipeline:
  1. Load model_ready.csv
  2. Temporal train/test split (no data leakage — earlier seasons train, recent seasons test)
  3. StandardScaler → LogisticRegression
  4. Evaluate: accuracy, ROC-AUC, classification report, confusion matrix
  5. Save trained model to models/playoff_predictor.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(DATA_DIR, "model_ready.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "playoff_predictor.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# Seasons used for training vs. held-out test
TEST_SEASONS = ["2022-23", "2023-24"]


# ── Load & Split ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["GAME_DATE"])
    feature_cols = [c for c in df.columns if c.startswith("DIFF_")]
    target_col = "H_WIN"

    X = df[feature_cols]
    y = df[target_col]
    seasons = df["SEASON"]

    return X, y, seasons, feature_cols


def temporal_split(X, y, seasons):
    """
    Temporal split: train on earlier seasons, test on most recent.
    This is CRITICAL for sports ML — never shuffle game data.
    """
    test_mask = seasons.isin(TEST_SEASONS)
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"Train: {train_mask.sum()} games  |  Test: {test_mask.sum()} games")
    print(f"Test seasons: {TEST_SEASONS}")
    return X_train, X_test, y_train, y_test


# ── Model ─────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
        )),
    ])


def cross_validate(pipeline, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle — temporal
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nCV ROC-AUC scores: {scores.round(3)}")
    print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test, feature_cols):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"Test Accuracy : {acc:.3f} ({acc*100:.1f}%)")
    print(f"Test ROC-AUC  : {auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))

    # Feature importance (logistic regression coefficients)
    coef = pipeline.named_steps["clf"].coef_[0]
    importance = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    print(f"\nTop 10 Most Predictive Features:")
    print(importance.head(10).to_string(index=False))

    return y_pred, y_prob, importance


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Away Win", "Home Win"],
        yticklabels=["Away Win", "Home Win"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Playoff Predictor", fontsize=14, pad=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("\nSaved confusion_matrix.png")
    plt.show()


def plot_feature_importance(importance: pd.DataFrame):
    top = importance.head(15).copy()
    top["color"] = top["coefficient"].apply(lambda x: "#2196F3" if x > 0 else "#F44336")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top["feature"][::-1], top["coefficient"][::-1], color=top["color"][::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Logistic Regression Coefficients\n(Positive = Favors Home Team)", fontsize=14)
    ax.set_xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    print("Saved feature_importance.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("Loading data...")
    X, y, seasons, feature_cols = load_data()

    X_train, X_test, y_train, y_test = temporal_split(X, y, seasons)

    pipeline = build_pipeline()

    print("\nRunning cross-validation...")
    cross_validate(pipeline, X_train, y_train)

    print("\nFitting final model on full training set...")
    pipeline.fit(X_train, y_train)

    y_pred, y_prob, importance = evaluate(pipeline, X_test, y_test, feature_cols)

    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(importance)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    return pipeline, importance


if __name__ == "__main__":
    train()

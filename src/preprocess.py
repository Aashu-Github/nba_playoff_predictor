"""
preprocess.py
-------------
Cleans and engineers features from raw_playoff_logs.csv.
Produces model_ready.csv with one row per game, home vs. away differentials.

Feature engineering:
  - Differential stats (home - away) for all key metrics
  - Rolling 5-game averages per team (within-season momentum)
  - Rest day advantage
  - Seed-implied features (derived from season context)
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw_playoff_logs.csv")
OUT_PATH = os.path.join(DATA_DIR, "model_ready.csv")

# Core box-score stats we pulled from nba_api LeagueGameLog
# (advanced stats like OffRtg/DefRtg need TeamDashboard — see note below)
BOX_STATS = [
    "PTS", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV",
    "PLUS_MINUS", "REST_DAYS",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["GAME_DATE"])
    print(f"Loaded raw data: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def add_rolling_averages(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each team, compute a rolling average of key stats
    over the previous `window` games (shift by 1 to avoid leakage).
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    roll_cols = ["PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "OREB", "DREB",
                 "AST", "TOV", "STL", "BLK", "PLUS_MINUS"]

    for col in roll_cols:
        if col in df.columns:
            df[f"ROLL_{col}"] = (
                df.groupby("TEAM_ID")[col]
                  .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
    return df


def compute_offensive_rating_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    True OffRtg requires possession data from advanced endpoints.
    Proxy: PTS / (FGA - OREB + TOV + 0.44*FTA) * 100
    This is a solid approximation used widely in analytics.
    """
    df = df.copy()
    possessions = df["FGA"] - df["OREB"] + df["TOV"] + 0.44 * df["FTA"]
    possessions = possessions.replace(0, np.nan)
    df["OFF_RTG_PROXY"] = (df["PTS"] / possessions * 100).round(2)

    # DefRtg proxy: we'll compute as opponent OffRtg after pairing
    return df


def compute_pace_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pace proxy = FGA + TOV + 0.44*FTA - OREB (per-game possessions estimate).
    """
    df = df.copy()
    df["PACE_PROXY"] = (df["FGA"] + df["TOV"] + 0.44 * df["FTA"] - df["OREB"]).round(2)
    return df


def compute_efg(df: pd.DataFrame) -> pd.DataFrame:
    """Effective FG% = (FGM + 0.5*FG3M) / FGA"""
    df = df.copy()
    df["EFG_PCT"] = ((df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)).round(4)
    return df


def compute_tov_rate(df: pd.DataFrame) -> pd.DataFrame:
    """TOV rate = TOV / (FGA + 0.44*FTA + TOV)"""
    df = df.copy()
    denom = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
    df["TOV_RATE"] = (df["TOV"] / denom.replace(0, np.nan)).round(4)
    return df


def build_game_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split into home/away teams and merge into one row per game.
    MATCHUP format: 'TOR vs. MIL' (home) or 'MIL @ TOR' (away).
    """
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(r" vs\. ").astype(int)

    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    # All stats get prefixed
    exclude = ["GAME_ID", "GAME_DATE", "SEASON"]
    home = home.rename(columns={c: f"H_{c}" for c in home.columns if c not in exclude})
    away = away.rename(columns={c: f"A_{c}" for c in away.columns if c not in exclude})

    paired = home.merge(away, on="GAME_ID", suffixes=("", "_dup"))
    paired = paired[[c for c in paired.columns if not c.endswith("_dup")]]

    return paired


def build_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create HOME - AWAY differential features.
    These are the actual predictive signals for logistic regression.
    """
    df = df.copy()

    diff_stats = [
        "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "OREB", "DREB", "AST", "TOV", "STL", "BLK",
        "PLUS_MINUS", "REST_DAYS",
        "OFF_RTG_PROXY", "PACE_PROXY", "EFG_PCT", "TOV_RATE",
        "ROLL_PTS", "ROLL_FG_PCT", "ROLL_FG3_PCT", "ROLL_AST",
        "ROLL_TOV", "ROLL_PLUS_MINUS",
    ]

    for stat in diff_stats:
        h_col = f"H_{stat}"
        a_col = f"A_{stat}"
        if h_col in df.columns and a_col in df.columns:
            df[f"DIFF_{stat}"] = df[h_col] - df[a_col]

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only differentials + metadata + target."""
    diff_cols = [c for c in df.columns if c.startswith("DIFF_")]
    meta_cols = ["GAME_ID", "GAME_DATE", "SEASON"]
    target = ["H_WIN"]

    available = [c for c in meta_cols + diff_cols + target if c in df.columns]
    return df[available].dropna()


# ── Main ──────────────────────────────────────────────────────────────────────

def preprocess():
    df = load_raw()

    # Feature engineering on per-team rows
    df = add_rolling_averages(df)
    df = compute_offensive_rating_proxy(df)
    df = compute_pace_proxy(df)
    df = compute_efg(df)
    df = compute_tov_rate(df)

    # Pair home/away
    paired = build_game_pairs(df)
    print(f"Game pairs: {len(paired):,}")

    # Build differentials
    paired = build_differentials(paired)

    # Compute DefRtg proxy as opponent's OffRtg
    if "H_OFF_RTG_PROXY" in paired.columns and "A_OFF_RTG_PROXY" in paired.columns:
        paired["DIFF_DEF_RTG_PROXY"] = paired["A_OFF_RTG_PROXY"] - paired["H_OFF_RTG_PROXY"]

    final = select_model_features(paired)
    print(f"Final dataset: {final.shape[0]} games, {final.shape[1]} columns")
    print(f"Home win rate: {final['H_WIN'].mean():.1%}")

    final.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    return final


if __name__ == "__main__":
    preprocess()

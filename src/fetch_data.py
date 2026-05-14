"""
fetch_data.py
-------------
Pulls playoff game logs and advanced team stats (last 10 seasons)
from the nba_api. Saves raw CSVs to data/.

Stats collected per team per game:
  - OffRtg, DefRtg, Pace, eFG%, TS%, OREB%, DREB%, TOV%, AST%
  - FG%, 3P%, FT%, Plus/Minus
  - Rest days between games
"""

import time
import os
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import (
    leaguegamelog,
    teamdashboardbyteamperformance,
    teamgamelogs,
)
from nba_api.stats.static import teams

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DELAY = 0.7          # seconds between API calls (respect rate limits)
SEASON_TYPE = "Playoffs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_playoff_game_logs(season: str) -> pd.DataFrame:
    """Return a DataFrame of every playoff game log for all teams in a season."""
    time.sleep(DELAY)
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=SEASON_TYPE,
        league_id_nullable="00",
    )
    df = log.get_data_frames()[0]
    df["SEASON"] = season
    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a REST_DAYS column = days since a team's previous game.
    First game of playoffs defaults to 3 (typical after regular season).
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["PREV_GAME"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME"]).dt.days.fillna(3).astype(int)
    return df


def label_winner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add WIN column: 1 if team won, 0 if lost.
    WL column from nba_api is 'W' or 'L'.
    """
    df = df.copy()
    df["WIN"] = (df["WL"] == "W").astype(int)
    return df


def build_game_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the per-team-per-game log into one row per game with
    home and away team stats side-by-side.

    Game ID appears twice (once per team). We split into home/away
    using the MATCHUP string: 'TEAM vs. OPP' = home, 'TEAM @ OPP' = away.
    """
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)

    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    home = home.rename(columns=lambda c: f"HOME_{c}" if c not in ["GAME_ID", "GAME_DATE", "SEASON"] else c)
    away = away.rename(columns=lambda c: f"AWAY_{c}" if c not in ["GAME_ID"] else c)

    merged = home.merge(away, on="GAME_ID", suffixes=("", "_away"))
    merged["HOME_WIN"] = merged["HOME_WIN"]  # target label

    return merged


# ── Main fetch routine ────────────────────────────────────────────────────────

def fetch_all_seasons():
    all_logs = []

    print(f"Fetching {SEASON_TYPE} game logs for {len(SEASONS)} seasons...\n")
    for season in tqdm(SEASONS, desc="Seasons"):
        try:
            df = get_playoff_game_logs(season)
            df = compute_rest_days(df)
            df = label_winner(df)
            all_logs.append(df)
            tqdm.write(f"  ✓ {season}: {len(df)} team-game rows")
        except Exception as e:
            tqdm.write(f"  ✗ {season}: {e}")
            time.sleep(2)

    combined = pd.concat(all_logs, ignore_index=True)

    raw_path = os.path.join(OUTPUT_DIR, "raw_playoff_logs.csv")
    combined.to_csv(raw_path, index=False)
    print(f"\nSaved raw logs → {raw_path}")
    print(f"Total rows: {len(combined):,}  |  Unique games: {combined['GAME_ID'].nunique():,}")

    return combined


if __name__ == "__main__":
    fetch_all_seasons()

"""Backfill nflverse data into data_cache/ as parquet.

Pulls multi-year history (player stats, rosters, depth charts, injuries)
plus the full schedule through the upcoming season. Re-run anytime to
refresh — everything here is regenerable, nothing here is a source of truth
on its own.
"""

import os

import nflreadpy as nfl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

HISTORY_START = 2018
CURRENT_SEASON = 2026
HISTORY_SEASONS = list(range(HISTORY_START, CURRENT_SEASON))  # completed seasons only
ALL_SEASONS = list(range(HISTORY_START, CURRENT_SEASON + 1))  # includes in-progress/upcoming season


def _save(df, name):
    path = os.path.join(CACHE_DIR, f"{name}.parquet")
    df.write_parquet(path)
    print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols -> {path}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Schedules ({HISTORY_START}-{CURRENT_SEASON}, includes historical Vegas lines + upcoming matchups)...")
    _save(nfl.load_schedules(seasons=ALL_SEASONS), "schedules")

    print(f"Weekly player stats ({HISTORY_START}-{CURRENT_SEASON - 1}, completed seasons)...")
    _save(nfl.load_player_stats(seasons=HISTORY_SEASONS), "player_stats_weekly")

    print(f"Rosters ({HISTORY_START}-{CURRENT_SEASON})...")
    _save(nfl.load_rosters(seasons=ALL_SEASONS), "rosters")

    print(f"Depth charts ({HISTORY_START}-{CURRENT_SEASON}, ESPN-sourced, used for O-line continuity)...")
    _save(nfl.load_depth_charts(seasons=ALL_SEASONS), "depth_charts")

    print(f"Injuries ({HISTORY_START}-{CURRENT_SEASON - 1}, current season has no injury reports until games start)...")
    _save(nfl.load_injuries(seasons=HISTORY_SEASONS), "injuries")

    print(f"Snap counts ({HISTORY_START}-{CURRENT_SEASON - 1}, completed seasons)...")
    _save(nfl.load_snap_counts(seasons=HISTORY_SEASONS), "snap_counts")

    print(f"Team stats ({HISTORY_START}-{CURRENT_SEASON - 1}, completed seasons -- team defense scoring)...")
    _save(nfl.load_team_stats(seasons=HISTORY_SEASONS), "team_stats")

    print("Team descriptive info (colors/names/ids, no season split)...")
    _save(nfl.load_teams(), "teams")

    print("Done.")


if __name__ == "__main__":
    main()

"""Shared helpers used by both fantasy_scan.py and betting_scan.py."""

import math
import os

import polars as pl

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
REF_DIR = os.path.join(BASE_DIR, "scripts", "reference")

OFFENSIVE_LINE_POS = ["LT", "LG", "C", "RG", "RT"]
DEFENSIVE_LINE_POS = ["LDE", "LDT", "NT", "RDT", "RDE"]


def json_safe(obj):
    """Recursively replace NaN/Infinity with None -- json.dump emits bare
    NaN/Infinity that browsers' JSON.parse rejects, and nflverse data has
    real NaNs (byes, unattempted stat categories, etc)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def load_cache(name):
    return pl.read_parquet(os.path.join(CACHE_DIR, f"{name}.parquet"))


def season_from_dt(dt_col):
    """NFL season year from a depth-chart snapshot timestamp (ISO string,
    e.g. "2026-03-14T07:32:09Z"): Jan/Feb belongs to the season that started
    the previous calendar year. Sliced as plain strings to sidestep polars'
    strict timezone-aware datetime parsing -- we only need year/month ints."""
    year = dt_col.str.slice(0, 4).cast(pl.Int32)
    month = dt_col.str.slice(5, 2).cast(pl.Int32)
    return pl.when(month <= 2).then(year - 1).otherwise(year)


def build_oline_continuity(depth_charts, prior_season, current_season, positions=None):
    """Starter continuity for a position group (offensive line by default;
    pass positions=DEFENSIVE_LINE_POS for the defensive-line equivalent used
    by DST rows)."""
    positions = positions or OFFENSIVE_LINE_POS
    dc = depth_charts.filter(pl.col("pos_abb").is_in(positions) & (pl.col("pos_rank") == 1)).with_columns(
        season_from_dt(pl.col("dt")).alias("season")
    )
    prior = dc.filter(pl.col("season") == prior_season)
    prior_latest = prior.group_by("team").agg(pl.col("dt").max().alias("dt"))
    prior_starters = prior.join(prior_latest, on=["team", "dt"], how="inner").select(["team", "pos_abb", "gsis_id"]).unique()

    current = dc.filter(pl.col("season") == current_season)
    current_latest = current.group_by("team").agg(pl.col("dt").max().alias("dt"))
    current_starters = current.join(current_latest, on=["team", "dt"], how="inner").select(["team", "pos_abb", "gsis_id"]).unique()

    result = {}
    for team in sorted(set(prior_starters["team"].to_list()) | set(current_starters["team"].to_list())):
        prior_ids = set(prior_starters.filter(pl.col("team") == team)["gsis_id"].drop_nulls().to_list())
        current_ids = set(current_starters.filter(pl.col("team") == team)["gsis_id"].drop_nulls().to_list())
        if not prior_ids or not current_ids:
            continue
        returning = prior_ids & current_ids
        result[team] = {
            "returning_starters": len(returning),
            "total_starters": len(current_ids),
            "continuity_pct": round(len(returning) / max(len(current_ids), 1), 3),
        }
    return result


def build_coach_continuity(schedules, prior_season, current_season):
    def primary_coach(season):
        home = schedules.filter(pl.col("season") == season).select(pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach"))
        away = schedules.filter(pl.col("season") == season).select(pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach"))
        combined = pl.concat([home, away]).drop_nulls()
        if combined.is_empty():
            return {}
        counts = combined.group_by(["team", "coach"]).len().sort("len", descending=True)
        top = counts.group_by("team").first()
        return dict(zip(top["team"].to_list(), top["coach"].to_list()))

    prior_coach = primary_coach(prior_season)
    current_coach = primary_coach(current_season)
    result = {}
    for team in sorted(set(prior_coach) | set(current_coach)):
        c_prior = prior_coach.get(team)
        c_current = current_coach.get(team)
        result[team] = {
            "coach_prior": c_prior,
            "coach_current": c_current,
            "same_coach": bool(c_prior and c_current and c_prior == c_current),
        }
    return result

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



# nflverse's rosters table alone uses a handful of team codes that its other
# tables (schedules, depth_charts, team_stats, the coordinator reference)
# don't -- normalize to the codes used everywhere else on the site so
# lookups keyed by those other tables (continuity, coordinators) still hit.
TEAM_CODE_ALIASES = {"AZ": "ARI", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def build_current_team_lookup(rosters, season):
    """Most recent known team per player (by gsis_id) for a season -- picks
    up offseason trades/signings that a prior-season stat aggregation can't
    reflect (e.g. a player's 2025 stats are tagged with their 2025 team even
    after they sign elsewhere for 2026)."""
    snap = rosters.filter((pl.col("season") == season) & pl.col("gsis_id").is_not_null())
    latest = snap.group_by("gsis_id").agg(pl.col("week").max().alias("week"))
    current = snap.join(latest, on=["gsis_id", "week"], how="inner").unique(subset=["gsis_id"]).select(["gsis_id", "team"])
    return {
        gsis_id: TEAM_CODE_ALIASES.get(team, team)
        for gsis_id, team in zip(current["gsis_id"].to_list(), current["team"].to_list())
    }


def primary_coach(schedules, season):
    """Each team's primary coach for a season, by majority vote across that
    team's home/away games (handles the rare mid-season coaching change --
    the coach who ran the most games "wins," same as an interim coach who
    only had 2 games would lose to a fired coach who had 15)."""
    home = schedules.filter(pl.col("season") == season).select(pl.col("home_team").alias("team"), pl.col("home_coach").alias("coach"))
    away = schedules.filter(pl.col("season") == season).select(pl.col("away_team").alias("team"), pl.col("away_coach").alias("coach"))
    combined = pl.concat([home, away]).drop_nulls()
    if combined.is_empty():
        return {}
    counts = combined.group_by(["team", "coach"]).len().sort("len", descending=True)
    top = counts.group_by("team").first()
    return dict(zip(top["team"].to_list(), top["coach"].to_list()))


def build_coach_continuity(schedules, prior_season, current_season):
    prior_coach = primary_coach(schedules, prior_season)
    current_coach = primary_coach(schedules, current_season)
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


def build_former_coach_matchups(schedules, prior_season, current_season):
    """For each team whose prior_season coach left and is now coaching a
    DIFFERENT team this season, maps team -> {coach_name, now_with}. A
    caller checks former_coach.get(team, {}).get("now_with") == opponent to
    know whether THIS specific matchup is a former-coach game -- shared
    between betting_scan.py (a team-margin adjustment) and fantasy_scan.py
    (a team-wide skill-position bump), since both need the identical
    "who used to coach whom" fact, not two independently-computed copies."""
    prior_coach = primary_coach(schedules, prior_season)
    current_coach = primary_coach(schedules, current_season)
    current_team_by_coach = {v: k for k, v in current_coach.items()}
    result = {}
    for team, old_coach in prior_coach.items():
        if not old_coach:
            continue
        new_team = current_team_by_coach.get(old_coach)
        if new_team and new_team != team:
            result[team] = {"coach_name": old_coach, "now_with": new_team}
    return result

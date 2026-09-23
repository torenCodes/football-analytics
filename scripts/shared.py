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


QB_OUT_STATUSES = {"Out", "Doubtful"}  # "Questionable" is a real coin-flip -- shown as a tag, not scored


def starting_qb_by_team(depth_charts, season):
    """Each team's current starting QB (gsis_id), from that team's most
    recent depth-chart snapshot this season -- same snapshot-latest pattern
    build_oline_continuity uses for O-line starters."""
    qb = depth_charts.filter((pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1)).with_columns(
        season_from_dt(pl.col("dt")).alias("dc_season")
    )
    qb = qb.filter(pl.col("dc_season") == season)
    if qb.is_empty():
        return {}
    latest = qb.group_by("team").agg(pl.col("dt").max().alias("dt"))
    current = qb.join(latest, on=["team", "dt"], how="inner").unique(subset=["team"])
    return dict(zip(current["team"].to_list(), current["gsis_id"].to_list()))


def build_qb_injury_flags(depth_charts, injuries, season, week, rosters=None):
    """Flags a team's starting QB as a real game-time question when the
    NFL's own official weekly injury report lists them Out or Doubtful --
    the standard, free signal sportsbooks and fantasy platforms already key
    off, available well before a player is ever formally placed on IR.
    Shared between betting_scan.py (a team-margin adjustment) and
    fantasy_scan.py (a WR/TE ripple penalty), since both need the identical
    "is this team's starter really playing" fact.

    Optional rosters fallback: the weekly injury report only covers players
    still tracked on it -- a starter formally moved to Reserve (nflverse's
    umbrella "RES" status, covering IR/PUP/NFI/Suspended) often drops off
    that report entirely, even though the depth chart may not yet show a
    new starter. When rosters is given, also flags a starter whose most
    recent roster status this season is "RES", even if they're absent from
    this week's injury report -- confirmed this bit us for Jaxson Dart
    (2026 wk3): a likely season-ending injury that hadn't yet produced a
    weekly report entry."""
    starters = starting_qb_by_team(depth_charts, season)
    if not starters:
        return {}

    status_by_id = {}
    if injuries is not None and not injuries.is_empty():
        wk = injuries.filter((pl.col("season") == season) & (pl.col("week") == week))
        if not wk.is_empty():
            status_by_id = dict(zip(wk["gsis_id"].to_list(), wk["report_status"].to_list()))

    reserve_ids = set()
    if rosters is not None:
        snap = rosters.filter((pl.col("season") == season) & pl.col("gsis_id").is_not_null())
        if not snap.is_empty():
            latest = snap.group_by("gsis_id").agg(pl.col("week").max().alias("week"))
            current = snap.join(latest, on=["gsis_id", "week"], how="inner")
            reserve_ids = set(current.filter(pl.col("status") == "RES")["gsis_id"].to_list())

    flags = {}
    for team, gsis_id in starters.items():
        status = status_by_id.get(gsis_id)
        if status in QB_OUT_STATUSES:
            flags[team] = {"gsis_id": gsis_id, "status": status}
        elif gsis_id in reserve_ids:
            flags[team] = {"gsis_id": gsis_id, "status": "Reserve"}
    return flags

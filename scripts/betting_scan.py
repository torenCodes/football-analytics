"""Individual Game Betting scan: 2025 spread backtest + 2026 game-by-game
picks.

Reads from data_cache/ (populated by ingest_historical.py) and writes
data/betting/best_calls_backtest.json + data/betting/game_board.json.

Core signal is a leave-one-out team efficiency model (offensive EPA/play vs.
opponent's defensive EPA/play allowed, each excluding the game being
predicted, so the backtest isn't grading itself on data it already knows the
outcome of). Situational/continuity/weather factors are layered on top as
explainable adjustments, not folded invisibly into one black-box number.

IMPORTANT -- spread_line sign convention (verified against nflverse's own
docs and cross-checked against real 2025 results before writing this):
a POSITIVE spread_line means the HOME team was favored by that many points.
`result` = home_score - away_score. Home covers when result > spread_line.
Getting this backwards would silently invert every pick in the backtest.

Run standalone, re-run anytime (weekly via GitHub Actions once the season
is live -- picks should tighten up as more of the current season's own
efficiency data accumulates) to refresh.
"""

import json
import os
from datetime import datetime, timezone

import polars as pl
import requests

from shared import (
    build_coach_continuity,
    build_oline_continuity,
    json_safe,
    load_cache,
    REF_DIR,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "betting")

RETRO_SEASON = 2025  # most recently completed season -> backtest + fallback efficiency baseline
UPCOMING_SEASON = 2026
PLAY_SCALE = 65  # rough plays-per-team-per-game, converts EPA/play diff to a point-scale margin
HOME_FIELD_ADJ = 1.5  # points, standard analytics convention (~1-3)
MIN_GAMES_FOR_CURRENT_SEASON = 3  # below this, fall back to prior-season efficiency for that team


def team_game_efficiency(team_stats):
    """Per (game_id, team) offensive EPA/play that game."""
    plays = (pl.col("attempts").fill_null(0) + pl.col("carries").fill_null(0)).alias("plays")
    epa_sum = (pl.col("passing_epa").fill_null(0.0) + pl.col("rushing_epa").fill_null(0.0)).alias("epa_sum")
    return (
        team_stats.select(["game_id", "season", "week", "team", "opponent_team", "attempts", "carries", "passing_epa", "rushing_epa"])
        .with_columns([plays, epa_sum])
        .with_columns((pl.col("epa_sum") / pl.when(pl.col("plays") > 0).then(pl.col("plays")).otherwise(1)).alias("off_epa_pp"))
        .select(["game_id", "season", "week", "team", "opponent_team", "plays", "epa_sum", "off_epa_pp"])
    )


def leave_one_out_efficiency(game_eff):
    """For each (game_id, team) row, this team's offensive EPA/play and this
    team's opponent-perspective defensive EPA/play allowed, both averaged
    across every OTHER game that team played that season (excludes the game
    being predicted, so a backtest doesn't already know its own answer)."""
    off_totals = game_eff.group_by("team").agg(
        pl.col("epa_sum").sum().alias("season_epa_sum"), pl.col("plays").sum().alias("season_plays")
    )
    off_excl = game_eff.join(off_totals, on="team").with_columns(
        ((pl.col("season_epa_sum") - pl.col("epa_sum")) / (pl.col("season_plays") - pl.col("plays")).clip(lower_bound=1)).alias(
            "off_epa_pp_excl"
        )
    ).select(["game_id", "team", "off_epa_pp_excl"])

    # game_eff row: team=offense that game, opponent_team=defense that allowed
    # it -- so the "defense allowed" table is keyed by opponent_team, not team.
    def_view = game_eff.rename({"opponent_team": "def_team"}).select(["game_id", "def_team", "epa_sum", "plays"])
    def_totals = def_view.group_by("def_team").agg(
        pl.col("epa_sum").sum().alias("season_epa_allowed_sum"), pl.col("plays").sum().alias("season_plays_allowed")
    )
    def_excl = def_view.join(def_totals, on="def_team").with_columns(
        (
            (pl.col("season_epa_allowed_sum") - pl.col("epa_sum"))
            / (pl.col("season_plays_allowed") - pl.col("plays")).clip(lower_bound=1)
        ).alias("def_epa_pp_allowed_excl")
    ).select(["game_id", "def_team", "def_epa_pp_allowed_excl"])

    return off_excl, def_excl


def season_efficiency_fallback(game_eff):
    """Full-season (non-excluded) team offensive/defensive EPA/play, used as
    the prior for a season with too few games played so far (e.g. every team
    entering Week 1)."""
    off = game_eff.group_by("team").agg((pl.col("epa_sum").sum() / pl.col("plays").sum().clip(lower_bound=1)).alias("off_epa_pp"))
    def_view = game_eff.rename({"opponent_team": "def_team"})
    deff = def_view.group_by("def_team").agg(
        (pl.col("epa_sum").sum() / pl.col("plays").sum().clip(lower_bound=1)).alias("def_epa_pp_allowed")
    ).rename({"def_team": "team"})
    return dict(zip(off["team"].to_list(), off["off_epa_pp"].to_list())), dict(zip(deff["team"].to_list(), deff["def_epa_pp_allowed"].to_list()))


def build_backtest(schedules, team_stats):
    games = schedules.filter((pl.col("season") == RETRO_SEASON) & (pl.col("game_type") == "REG")).drop_nulls(["result", "spread_line"])
    game_eff = team_game_efficiency(team_stats.filter(pl.col("season") == RETRO_SEASON))
    off_excl, def_excl = leave_one_out_efficiency(game_eff)

    home_off = off_excl.rename({"team": "home_team", "off_epa_pp_excl": "home_off_epa_pp"})
    away_off = off_excl.rename({"team": "away_team", "off_epa_pp_excl": "away_off_epa_pp"})
    home_def = def_excl.rename({"def_team": "home_team", "def_epa_pp_allowed_excl": "home_def_epa_pp_allowed"})
    away_def = def_excl.rename({"def_team": "away_team", "def_epa_pp_allowed_excl": "away_def_epa_pp_allowed"})

    g = games.join(home_off, on=["game_id", "home_team"], how="inner")
    g = g.join(away_off, on=["game_id", "away_team"], how="inner")
    g = g.join(home_def, on=["game_id", "home_team"], how="inner")
    g = g.join(away_def, on=["game_id", "away_team"], how="inner")

    g = g.with_columns(
        (
            ((pl.col("home_off_epa_pp") - pl.col("away_def_epa_pp_allowed")) - (pl.col("away_off_epa_pp") - pl.col("home_def_epa_pp_allowed")))
            * PLAY_SCALE
            + HOME_FIELD_ADJ
        ).alias("model_margin_home")
    )

    picks = []
    correct = 0
    graded = 0
    for r in g.iter_rows(named=True):
        our_pick = "home" if r["model_margin_home"] > r["spread_line"] else "away"
        home_covered = r["result"] > r["spread_line"]
        push = r["result"] == r["spread_line"]
        correct_pick = None
        if not push:
            correct_pick = (our_pick == "home" and home_covered) or (our_pick == "away" and not home_covered)
            graded += 1
            correct += int(correct_pick)
        picks.append(
            {
                "game_id": r["game_id"],
                "week": r["week"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "spread_line": r["spread_line"],
                "result": r["result"],
                "model_margin_home": round(r["model_margin_home"], 1),
                "our_pick": our_pick,
                "push": push,
                "correct": correct_pick,
            }
        )

    accuracy = round(correct / graded, 3) if graded else None
    return {
        "season": RETRO_SEASON,
        "method": "Leave-one-out team offensive/defensive EPA-per-play model vs. the closing spread_line, home-field adjusted. Each game's efficiency inputs exclude that game itself.",
        "games_graded": graded,
        "correct_picks": correct,
        "accuracy": accuracy,
        "picks": sorted(picks, key=lambda x: (x["week"], x["game_id"])),
    }


def last_meeting(schedules, home_team, away_team, before_season):
    prior = (
        schedules.filter(
            (
                ((pl.col("home_team") == home_team) & (pl.col("away_team") == away_team))
                | ((pl.col("home_team") == away_team) & (pl.col("away_team") == home_team))
            )
            & (pl.col("season") < before_season)
            & pl.col("result").is_not_null()
        )
        .sort(["season", "week"], descending=True)
        .head(1)
    )
    if prior.is_empty():
        return None
    row = prior.row(0, named=True)
    winner = row["home_team"] if row["result"] > 0 else (row["away_team"] if row["result"] < 0 else None)
    return {"season": row["season"], "winner": winner}


def fetch_weather(lat, lon, gameday):
    """NWS point forecast for a specific date. Free, no key, US-only, but
    NWS only publishes ~7 days out -- periods[0] is just "today", NOT the
    game date, so blindly taking it would mislabel today's weather as the
    forecast for a game weeks away. We match periods by calendar date
    instead, and return None (not a stale guess) when the game is further
    out than NWS forecasts, which is the normal case until game week."""
    try:
        game_date = datetime.strptime(gameday, "%Y-%m-%d").date()
        if (game_date - datetime.now(timezone.utc).date()).days > 7:
            return None
        points = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=10, headers={"User-Agent": "gridiron-edge (personal research project)"})
        if points.status_code != 200:
            return None
        forecast_url = points.json()["properties"]["forecast"]
        forecast = requests.get(forecast_url, timeout=10, headers={"User-Agent": "gridiron-edge (personal research project)"})
        if forecast.status_code != 200:
            return None
        periods = forecast.json()["properties"]["periods"]
        match = next((p for p in periods if p.get("startTime", "")[:10] == gameday and p.get("isDaytime")), None) or next(
            (p for p in periods if p.get("startTime", "")[:10] == gameday), None
        )
        if not match:
            return None
        return {
            "short_forecast": match.get("shortForecast"),
            "temperature_f": match.get("temperature"),
            "wind": match.get("windSpeed"),
        }
    except (requests.RequestException, ValueError):
        return None


def build_game_board(schedules, team_stats):
    with open(os.path.join(REF_DIR, "stadiums.json"), encoding="utf-8") as f:
        stadiums = json.load(f)

    upcoming = schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG") & pl.col("home_score").is_null())
    if upcoming.is_empty():
        return {"season": UPCOMING_SEASON, "note": "No upcoming games found in the cached schedule.", "weeks": {}}
    next_week = int(upcoming["week"].min())
    week_games = upcoming.filter(pl.col("week") == next_week)

    # A team's "real" home stadium name for international-game detection is
    # whichever name is most common across ITS OWN season -- majority vote,
    # not a match against our reference table's name. nflverse's own stadium
    # text isn't stable (Houston's 2026 rows say "Reliant Stadium", a decade-
    # old name, vs. "NRG Stadium" in every prior season) so comparing against
    # a fixed reference name produces false positives; comparing against the
    # team's own season correctly isolates the actual one-off outlier game.
    season_home_games = schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG")).drop_nulls(["stadium"])
    primary_stadium_name = {}
    if not season_home_games.is_empty():
        counts = season_home_games.group_by(["home_team", "stadium"]).len().sort("len", descending=True)
        top = counts.group_by("home_team").first()
        primary_stadium_name = dict(zip(top["home_team"].to_list(), top["stadium"].to_list()))

    current_season_stats = team_stats.filter(pl.col("season") == UPCOMING_SEASON)
    games_played_by_team = {}
    if not current_season_stats.is_empty():
        counts = current_season_stats.group_by("team").agg(pl.col("game_id").n_unique().alias("games"))
        games_played_by_team = dict(zip(counts["team"].to_list(), counts["games"].to_list()))

    fallback_stats = team_stats.filter(pl.col("season") == RETRO_SEASON)
    fallback_eff = team_game_efficiency(fallback_stats)
    fallback_off, fallback_def = season_efficiency_fallback(fallback_eff)

    current_off, current_def = {}, {}
    if not current_season_stats.is_empty():
        current_eff = team_game_efficiency(current_season_stats)
        current_off, current_def = season_efficiency_fallback(current_eff)

    print("Computing continuity for game board...")
    depth_charts = load_cache("depth_charts")
    oline_continuity = build_oline_continuity(depth_charts, RETRO_SEASON, UPCOMING_SEASON)
    coach_continuity = build_coach_continuity(schedules, RETRO_SEASON, UPCOMING_SEASON)

    def team_off(team):
        if games_played_by_team.get(team, 0) >= MIN_GAMES_FOR_CURRENT_SEASON:
            return current_off.get(team), "current_season"
        return fallback_off.get(team), "prior_season_fallback"

    def team_def(team):
        if games_played_by_team.get(team, 0) >= MIN_GAMES_FOR_CURRENT_SEASON:
            return current_def.get(team), "current_season"
        return fallback_def.get(team), "prior_season_fallback"

    picks = []
    for r in week_games.iter_rows(named=True):
        home, away = r["home_team"], r["away_team"]
        home_off_pp, home_off_src = team_off(home)
        away_off_pp, away_off_src = team_off(away)
        home_def_pp, _ = team_def(home)
        away_def_pp, _ = team_def(away)

        model_margin_home = None
        if None not in (home_off_pp, away_off_pp, home_def_pp, away_def_pp):
            model_margin_home = round(
                ((home_off_pp - away_def_pp) - (away_off_pp - home_def_pp)) * PLAY_SCALE + HOME_FIELD_ADJ, 1
            )

        rest_edge = None
        if r["home_rest"] is not None and r["away_rest"] is not None:
            rest_edge = r["home_rest"] - r["away_rest"]

        meeting = last_meeting(schedules, home, away, UPCOMING_SEASON)
        revenge_flag = None
        if meeting and r["div_game"]:
            revenge_flag = {"home": meeting["winner"] == away, "away": meeting["winner"] == home, "last_season": meeting["season"]}

        stadium = stadiums.get(home)
        primary_name = primary_stadium_name.get(home)
        international_site = bool(primary_name and r.get("stadium") and r["stadium"] != primary_name)
        weather = None  # must default every iteration -- otherwise a skipped fetch (e.g. this game's
        # condition below is False) would silently inherit the PREVIOUS game's weather value, since
        # Python loop variables aren't re-scoped per-iteration.
        if stadium and not international_site and stadium.get("nws_coverage") and r.get("gameday"):
            weather = fetch_weather(stadium["lat"], stadium["lon"], r["gameday"])

        weekday = r.get("weekday")
        gametime = r.get("gametime")
        primetime = bool(gametime and gametime >= "18:00")

        # "Juicy" ranking for the homepage/game-board ordering: primetime
        # billing and rivalry/revenge storylines surface first, tie-broken by
        # kickoff time. This is a display-ordering heuristic, not part of the
        # pick model itself.
        juice_score = (
            (4 if primetime else 0)
            + (2 if r["div_game"] else 0)
            + (2 if revenge_flag and (revenge_flag["home"] or revenge_flag["away"]) else 0)
        )

        picks.append(
            {
                "game_id": r["game_id"],
                "week": r["week"],
                "gameday": r["gameday"],
                "weekday": weekday,
                "gametime": gametime,
                "primetime": primetime,
                "juice_score": juice_score,
                "home_team": home,
                "away_team": away,
                "div_game": bool(r["div_game"]),
                "model_margin_home": model_margin_home,
                "efficiency_source": {"home": home_off_src, "away": away_off_src},
                "our_pick": ("home" if model_margin_home > 0 else "away") if model_margin_home is not None else None,
                "rest_days_edge_home": rest_edge,
                "oline_continuity": {"home": oline_continuity.get(home), "away": oline_continuity.get(away)},
                "coach_continuity": {"home": coach_continuity.get(home), "away": coach_continuity.get(away)},
                "revenge_game": revenge_flag,
                "roof": r["roof"],
                "international_site": international_site,
                "weather": weather,
            }
        )

    picks.sort(key=lambda g: (-g["juice_score"], g["gametime"] or "", g["game_id"]))

    return {"season": UPCOMING_SEASON, "week": next_week, "games": picks}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    schedules = load_cache("schedules")
    team_stats_25 = load_cache_team_stats(RETRO_SEASON)

    print(f"Building {RETRO_SEASON} spread backtest (leave-one-out EPA model)...")
    backtest = build_backtest(schedules, team_stats_25)
    print(f"  {backtest['games_graded']} games graded, accuracy={backtest['accuracy']}")

    print("Building upcoming-week game board...")
    team_stats_current = load_cache_team_stats(UPCOMING_SEASON, allow_empty=True)
    all_team_stats = pl.concat([team_stats_25, team_stats_current]) if team_stats_current is not None else team_stats_25
    game_board = build_game_board(schedules, all_team_stats)
    print(f"  {len(game_board.get('games', []))} games in week {game_board.get('week')}")

    meta = {"generated_at": datetime.now(timezone.utc).isoformat()}

    with open(os.path.join(OUT_DIR, "best_calls_backtest.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **backtest}), f, indent=2)
    with open(os.path.join(OUT_DIR, "game_board.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **game_board}), f, indent=2)

    print("Wrote best_calls_backtest.json and game_board.json")


def load_cache_team_stats(season, allow_empty=False):
    import nflreadpy as nfl

    try:
        return nfl.load_team_stats(seasons=[season])
    except Exception as e:
        if allow_empty:
            print(f"  no team_stats for {season} yet ({e}) -- expected before that season's games start")
            return None
        raise


if __name__ == "__main__":
    main()

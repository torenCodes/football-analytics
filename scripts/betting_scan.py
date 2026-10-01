"""Game Picks scan: 2025 walk-forward backtest + 2026 game-by-game picks
against the spread.

Reads completed seasons from data_cache/ (populated by ingest_historical.py)
plus the in-progress season live from nflverse, and writes
data/betting/{best_calls_backtest,game_board,accountability}.json.

Core signal: each team's offensive EPA/play and defensive EPA/play ALLOWED,
as deviations from league average. Each rating blends last season (shrunk
toward average) with this season to date, weighted by plays played, so a
team's rating moves off last year's tape gradually instead of all at once.
margin = MARGIN_SCALE * [(home_off - away_off) + (away_def - home_def)] + HFA.
A defense that allows MORE EPA is worse, so the opponent's def number is
ADDED -- an earlier version subtracted it, which made the model close to
noise (0.06 correlation with 2025 results). Constants are fitted by
scripts/calibrate_game_model.py. Situational factors (QB out, revenge,
former coach) are layered on top as explainable point nudges.

The backtest runs the exact same rating code walk-forward: each 2025 game is
predicted using only games before it, like the live board.

IMPORTANT -- spread_line sign convention (verified against nflverse's own
docs and cross-checked against real 2025 results before writing this):
a POSITIVE spread_line means the HOME team was favored by that many points.
`result` = home_score - away_score. Home covers when result > spread_line.
Getting this backwards would silently invert every pick in the backtest.
"""

import glob
import json
import os
import zlib
from datetime import datetime, timezone

import polars as pl
import requests

from shared import (
    build_coach_continuity,
    build_former_coach_matchups,
    build_oline_continuity,
    build_qb_injury_flags,
    json_safe,
    load_cache,
    REF_DIR,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "betting")
ARCHIVE_DIR = os.path.join(OUT_DIR, "archive")

RETRO_SEASON = 2025  # most recently completed season -> backtest + prior-season baseline
UPCOMING_SEASON = 2026
# Fitted by scripts/calibrate_game_model.py on 2026-09-23 (least squares on final margins,
# walk-forward over 2021-2024 REG games; best fit rho 0.6/0.2, K 520, scale 50.6, HFA 2.00 --
# these round values are within noise of it). 2025 holdout, never seen by the fit: margin
# correlation 0.37 (spread: 0.50), outright winners 62.6%, 47.6% against the spread. Kept
# fixed through 2026 so the published 2025 backtest stays a true holdout -- refit after the season.
MARGIN_SCALE = 50.0  # points per 1.0 EPA/play of combined off/def rating gap
HOME_FIELD_ADJ = 2.0  # points; 0 at neutral sites (schedules.location == "Neutral")
RHO_OFF = 0.6  # share of last season's offensive rating carried into this one (offense is stickier)
RHO_DEF = 0.2  # same for defense -- year-to-year defensive EPA barely persists
K_PLAYS = 480  # plays of this season's data that count as much as last season's (shrunk) rating
MIN_BACKTEST_CORR = 0.2  # sanity guard: below this the model is broken (e.g. a sign flip), fail the run
QB_OUT_PENALTY = 4.0  # points; backtested estimates of a backup QB's scoring impact commonly range
# ~3-7 points -- 4 is a defensible, modest v1 value, consistent with this model's philosophy of small
# explainable nudges rather than a fully-modeled per-player value system (that's a future project, not this one)
REVENGE_BONUS = 1.0  # points, for a team that lost the last meeting against this same opponent (only
# ever set for division games -- see last_meeting/revenge_flag). Below HOME_FIELD_ADJ, well under
# QB_OUT_PENALTY (4.0) -- a psychological nudge stays softer than a concrete injury impact. Not
# validated by the backtest (not reconstructable historically without leaking later news)
FORMER_COACH_BONUS = 1.0  # points, for a team facing the coach who coached THEM last season -- same
# magnitude as REVENGE_BONUS, same reasoning
INDOOR_ROOFS = ("dome", "closed")  # nflverse roof values where the forecast doesn't matter


def team_game_efficiency(team_stats):
    """Per (game_id, team) offensive EPA and plays, regular season only --
    playoff games exist only for the best teams and would skew their season
    numbers relative to everyone else's."""
    plays = (pl.col("attempts").fill_null(0) + pl.col("carries").fill_null(0)).alias("plays")
    epa_sum = (pl.col("passing_epa").fill_null(0.0) + pl.col("rushing_epa").fill_null(0.0)).alias("epa_sum")
    return (
        team_stats.filter(pl.col("season_type") == "REG")
        .with_columns([plays, epa_sum])
        .select(["game_id", "season", "week", "team", "opponent_team", "plays", "epa_sum"])
    )


def _side_totals(rows):
    """{team: (epa, plays)} on offense, the same keyed by the defense that
    allowed it (opponent_team), and the period's league EPA/play."""
    off = rows.group_by("team").agg(pl.col("epa_sum").sum(), pl.col("plays").sum())
    deff = rows.group_by("opponent_team").agg(pl.col("epa_sum").sum(), pl.col("plays").sum())
    total_plays = rows["plays"].sum()
    mean = rows["epa_sum"].sum() / total_plays if total_plays else 0.0
    return {t: (e, p) for t, e, p in off.iter_rows()}, {t: (e, p) for t, e, p in deff.iter_rows()}, mean


def rating_inputs_asof(eff, season, week):
    """What a live model would know going into `season` week `week`: that
    season's REG games strictly before `week`, plus the full prior season.
    Every number is a deviation from its own period's league average."""
    cur_off, cur_def, cur_mean = _side_totals(eff.filter((pl.col("season") == season) & (pl.col("week") < week)))
    pri_off, pri_def, pri_mean = _side_totals(eff.filter(pl.col("season") == season - 1))
    inputs = {}
    for team in set(cur_off) | set(pri_off):
        co_e, co_p = cur_off.get(team, (0.0, 0))
        cd_e, cd_p = cur_def.get(team, (0.0, 0))
        po_e, po_p = pri_off.get(team, (0.0, 0))
        pd_e, pd_p = pri_def.get(team, (0.0, 0))
        inputs[team] = {
            "cur_off": co_e / co_p - cur_mean if co_p else 0.0,
            "cur_off_plays": co_p,
            "cur_def": cd_e / cd_p - cur_mean if cd_p else 0.0,
            "cur_def_plays": cd_p,
            "prior_off": po_e / po_p - pri_mean if po_p else 0.0,
            "prior_def": pd_e / pd_p - pri_mean if pd_p else 0.0,
        }
    return inputs


def blend(inputs, rho_off=RHO_OFF, rho_def=RHO_DEF, k_plays=K_PLAYS):
    """Per-team off/def rating: this season's deviation weighted by plays
    played so far, the remainder from last season's deviation shrunk toward
    average by rho. w_off/w_def = share of the rating that is this season."""
    ratings = {}
    for team, x in inputs.items():
        w_off = x["cur_off_plays"] / (x["cur_off_plays"] + k_plays)
        w_def = x["cur_def_plays"] / (x["cur_def_plays"] + k_plays)
        ratings[team] = {
            "off": w_off * x["cur_off"] + (1 - w_off) * rho_off * x["prior_off"],
            "def": w_def * x["cur_def"] + (1 - w_def) * rho_def * x["prior_def"],
            "w_off": w_off,
            "w_def": w_def,
        }
    return ratings


def base_margin(ratings, home, away, neutral, scale=MARGIN_SCALE, hfa=HOME_FIELD_ADJ):
    """Home-relative margin before situational nudges. "def" is EPA/play a
    defense ALLOWS (higher = worse), so the opponent's def rating is added."""
    h, a = ratings.get(home), ratings.get(away)
    if h is None or a is None:
        return None
    return scale * ((h["off"] - a["off"]) + (a["def"] - h["def"])) + (0.0 if neutral else hfa)


def ats_pick(margin_home, spread_home):
    """The side of the spread the model favors: "home" if it has the home
    team beating the line, "away" if short of it, None (no lean) on an exact
    tie or with no line. Always called with the ROUNDED stored margin, so the
    pick shown on the site and the pick graded later can't disagree."""
    if margin_home is None or spread_home is None or margin_home == spread_home:
        return None
    return "home" if margin_home > spread_home else "away"


def _record(flags):
    graded = len(flags)
    correct = sum(flags)
    return {"graded": graded, "correct": correct, "accuracy": round(correct / graded, 3) if graded else None}


def _corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return round(sxy / (sxx * syy) ** 0.5, 3) if sxx and syy else None


def build_backtest(schedules, eff):
    """Walk-forward: each RETRO_SEASON REG game predicted by the same rating
    code the live board runs, using only games before it plus the prior
    season, then picked against the closing spread with the same ats_pick
    rule. Base model only -- the QB/revenge/former-coach nudges can't be
    reconstructed historically without leaking later news."""
    games = schedules.filter((pl.col("season") == RETRO_SEASON) & (pl.col("game_type") == "REG")).drop_nulls(["result", "spread_line"])
    picks = []
    for week in sorted(games["week"].unique().to_list()):
        ratings = blend(rating_inputs_asof(eff, RETRO_SEASON, week))
        for r in games.filter(pl.col("week") == week).iter_rows(named=True):
            margin = base_margin(ratings, r["home_team"], r["away_team"], r["location"] == "Neutral")
            if margin is None:
                continue
            margin = round(margin, 1)
            spread, result = r["spread_line"], r["result"]
            pick = ats_pick(margin, spread)
            push = result == spread
            takes_favorite = None if (pick is None or spread == 0) else ((pick == "home") == (spread > 0))
            picks.append(
                {
                    "game_id": r["game_id"],
                    "week": r["week"],
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "spread_line": spread,
                    "result": result,
                    "model_margin_home": margin,
                    "our_pick": pick,
                    "takes_favorite": takes_favorite,
                    "push": push,
                    "correct": None if (pick is None or push) else (pick == "home") == (result > spread),
                    "winner_correct": None if (margin == 0 or result == 0) else (margin > 0) == (result > 0),
                }
            )

    ats = [p["correct"] for p in picks if p["correct"] is not None]
    outright = [p["winner_correct"] for p in picks if p["winner_correct"] is not None]
    market_fav = [(p["spread_line"] > 0) == (p["result"] > 0) for p in picks if p["spread_line"] != 0 and p["result"] != 0]
    fav_side = [p["correct"] for p in picks if p["correct"] is not None and p["takes_favorite"] is True]
    dog_side = [p["correct"] for p in picks if p["correct"] is not None and p["takes_favorite"] is False]
    sided = [p for p in picks if p["takes_favorite"] is not None]
    margins, results, spreads = [p["model_margin_home"] for p in picks], [p["result"] for p in picks], [p["spread_line"] for p in picks]
    return {
        "season": RETRO_SEASON,
        "method": (
            f"Walk-forward: each {RETRO_SEASON} regular-season game predicted with only the games before it plus "
            f"{RETRO_SEASON - 1}, using the same rating code as the live board, then picked against the closing spread. "
            f"Base model only — QB-injury, revenge and former-coach adjustments aren't backtested. Constants were fitted on "
            f"2021-{RETRO_SEASON - 1}, so {RETRO_SEASON} is data the fit never saw."
        ),
        "games": len(picks),
        "ats": _record(ats),
        "outright": _record(outright),
        "market_favorite_outright": _record(market_fav),
        "no_lean": sum(1 for p in picks if p["our_pick"] is None),
        "pushes": sum(1 for p in picks if p["push"]),
        "correlation": _corr(margins, results),
        "market_correlation": _corr(spreads, results),
        "underdog_pick_share": round(sum(1 for p in sided if not p["takes_favorite"]) / len(sided), 3) if sided else None,
        "ats_by_side": {"favorite": _record(fav_side), "underdog": _record(dog_side)},
        "picks": picks,
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


def _team_line(spread_home, side):
    """A team's own spread in standard notation (favorite negative)."""
    line = -spread_home if side == "home" else spread_home
    return "PK" if line == 0 else f"{'+' if line > 0 else '−'}{abs(line):g}"


def unit_ranks(ratings):
    """1-32 per unit from the blended ratings: offense by most EPA/play
    gained, defense by fewest allowed."""
    by_off = sorted(ratings, key=lambda t: -ratings[t]["off"])
    by_def = sorted(ratings, key=lambda t: ratings[t]["def"])
    return {t: {"off": by_off.index(t) + 1, "def": by_def.index(t) + 1} for t in ratings}


def margin_breakdown(ratings, home, away, neutral, adjustments, margin_home):
    """The model's margin split into the pieces that add up to it: each
    offense against the defense it faces (points vs. an average matchup --
    base_margin regrouped), home field, then the situational nudges. Rows are
    home-relative (positive helps home), rounded to 0.1 with the rounding
    slack put on the biggest row, so the rows shown sum exactly to the
    margin shown."""
    h, a = ratings[home], ratings[away]
    rows = [
        {"kind": "offense", "team": home, "opp": away, "raw": MARGIN_SCALE * (h["off"] + a["def"])},
        {"kind": "offense", "team": away, "opp": home, "raw": -MARGIN_SCALE * (a["off"] + h["def"])},
    ]
    if not neutral:
        rows.append({"kind": "home_field", "team": home, "raw": HOME_FIELD_ADJ})
    rows += adjustments
    for r in rows:
        r["pts_home"] = round(r["raw"], 1)
    slack = round(margin_home - sum(r["pts_home"] for r in rows), 1)
    if slack:
        biggest = max(rows, key=lambda r: abs(r["raw"]))
        biggest["pts_home"] = round(biggest["pts_home"] + slack, 1)
    return [{k: v for k, v in r.items() if k != "raw"} for r in rows]


class Phrasing:
    """Rotates through each situation's phrasings across a week's board, so
    16 cards don't keep landing on the same sentence. Each key starts at an
    offset seeded by the season/week: a re-scan with the same situations in
    the same order phrases them the same way, but each week opens fresh."""

    def __init__(self, seed=""):
        self.seed = seed
        self.used = {}

    def pick(self, key, options):
        i = self.used.get(key, zlib.crc32(f"{self.seed}:{key}".encode()))
        self.used[key] = i + 1
        return options[i % len(options)]


def _half(x):
    """Points as prose, to the nearest half point: "3", "3.5"."""
    return f"{round(x * 2) / 2:g}"


def _qb_status(flag):
    return "on Reserve" if flag["status"] == "Reserve" else f"listed {flag['status']}"


def _weakness_story(own_rank, opp_rank):
    """True when the opposing unit sits further below average than this
    team's unit sits above it -- the story is then the other side's hole,
    not this side's strength (a #20 defense isn't why ATL's #25 offense stalls)."""
    return (opp_rank - 16.5) > (16.5 - own_rank)


def driver_sentence(phr, home, away, margin_home, rows, ranks, qb_injury):
    """What carries the model's projected winner: the breakdown row pulling
    hardest their way, told as a unit-vs-unit matchup with ranks.
    Returns (sentence, kind of row it describes)."""
    if abs(margin_home) < 1.0:
        return phr.pick("even", [
            "The model can barely separate these two.",
            "By the model's ratings, this is about as even as matchups get.",
            "The model sees close to a coin flip here.",
        ]), "even"
    w, l = (home, away) if margin_home > 0 else (away, home)
    sign = 1 if w == home else -1
    best = max(rows, key=lambda r: sign * r["pts_home"])
    pts = _half(sign * best["pts_home"])
    if best["kind"] == "offense" and best["team"] == w:
        ro, rd = ranks[w]["off"], ranks[l]["def"]
        if _weakness_story(ro, rd):
            return phr.pick("drv_def_hole", [
                f"The soft spot is {l}'s defense, #{rd} in the model's ratings, and {w}'s #{ro} offense gets to face it.",
                f"{l}'s defense ranks #{rd}, and the model expects {w}'s #{ro} offense to take advantage — worth about {pts} points by its math.",
                f"Look at {l}'s #{rd} defense: that's the hole the model sees {w}'s #{ro} offense getting through.",
            ]), "offense"
        return phr.pick("drv_off", [
            f"It starts with {w}'s offense, #{ro} in the model's ratings, against a {l} defense that ranks #{rd}.",
            f"The matchup doing the most work: {w}'s #{ro} offense vs. {l}'s #{rd} defense, worth about {pts} points by the model's math.",
            f"{w} brings the #{ro} offense in the model's ratings and draws a {l} defense ranked #{rd}.",
            f"Follow {w}'s #{ro} offense into {l}'s #{rd} defense — that's where the model finds its edge.",
        ]), "offense"
    if best["kind"] == "offense":
        rd, ro = ranks[w]["def"], ranks[l]["off"]
        if _weakness_story(rd, ro):
            against = "even against" if rd > 16 else "up against"
            return phr.pick("drv_off_hole", [
                f"{l}'s offense ranks #{ro} in the model's ratings, and it runs into a {w} defense ranked #{rd}.",
                f"{l}'s #{ro} offense is the weak link the model keys on, {against} {w}'s #{rd} defense.",
                f"The model doesn't trust {l}'s offense (#{ro}) to move the ball, {against} {w}'s #{rd} defense — worth about {pts} points by its math.",
            ]), "defense"
        return phr.pick("drv_def", [
            f"{w}'s #{rd} defense against {l}'s #{ro} offense is where the model sees this one tilting.",
            f"The model trusts {w}'s defense (#{rd}) to slow {l}'s #{ro} offense — worth about {pts} points by its math.",
            f"This projection is built on defense: {w}'s #{rd} unit against {l}'s #{ro} offense.",
        ]), "defense"
    if best["kind"] == "home_field":
        return phr.pick("drv_home", [
            f"On paper these two grade out close — home field does most of the work for {w}.",
            f"Take away home field and the model has this one nearly even; the {HOME_FIELD_ADJ:g}-point home edge tips it to {w}.",
        ]), "home_field"
    if best["kind"] == "qb_out" and qb_injury.get(l):
        status = _qb_status(qb_injury[l])
        return phr.pick("drv_qb", [
            f"{l}'s starting QB is {status}, and that {QB_OUT_PENALTY:g}-point adjustment is the biggest single piece of the model's number.",
            f"The headline is under center: {l}'s starting QB is {status}, which costs them {QB_OUT_PENALTY:g} points in the model.",
        ]), "qb_out"
    return f"The model's numbers tilt toward {w}.", best["kind"]


def counter_sentence(phr, home, away, margin_home, rows, ranks):
    """The other side's best answer, when it's worth a point and a half or
    more -- so the readout isn't one-sided."""
    if abs(margin_home) < 1.0:
        return None
    w, l = (home, away) if margin_home > 0 else (away, home)
    sign = 1 if w == home else -1
    best = max(rows, key=lambda r: -sign * r["pts_home"])
    pts = -sign * best["pts_home"]
    if pts < 1.5:
        return None
    if best["kind"] == "offense" and best["team"] == l:
        ro, rd = ranks[l]["off"], ranks[w]["def"]
        if _weakness_story(ro, rd):
            return phr.pick("ctr_def_hole", [
                f"The worry for {w}: a #{rd} defense that {l}'s #{ro} offense can attack.",
                f"{w}'s defense (#{rd}) gives {l}'s #{ro} offense a way back in, worth about {_half(pts)} points.",
            ])
        return phr.pick("ctr_off", [
            f"{l}'s best counter: its #{ro} offense against {w}'s #{rd} defense, worth about {_half(pts)} points back.",
            f"{l} isn't without answers — its #{ro} offense should find room against {w}'s #{rd} defense.",
        ])
    if best["kind"] == "offense":
        rd, ro = ranks[l]["def"], ranks[w]["off"]
        if _weakness_story(rd, ro):
            return phr.pick("ctr_off_hole", [
                f"The catch: {w}'s own offense ranks just #{ro}, and {l}'s #{rd} defense can keep this close.",
                f"{w}'s #{ro} offense is the question mark, facing {l}'s #{rd} defense.",
            ])
        return phr.pick("ctr_def", [
            f"{l}'s best counter is its #{rd} defense, which should make life hard on {w}'s #{ro} offense.",
            f"The catch: {l}'s defense ranks #{rd} and matches up well with {w}'s #{ro} offense.",
        ])
    if best["kind"] == "home_field":
        return f"{l} does get {HOME_FIELD_ADJ:g} points of home field back."
    return None


def pick_verdict(phr, home, away, model_margin_home, spread_home, our_pick):
    """The model's margin set against the spread, and the side of the spread
    that produces. Every branch reads off the same two numbers the pick is
    derived from, so the text can't contradict the pick."""
    if model_margin_home is None:
        return "Not enough efficiency data yet to model this matchup — check back closer to kickoff."
    model_winner = home if model_margin_home > 0 else away
    margin_abs = abs(model_margin_home)
    if spread_home is None:
        return (f"No line is posted yet; the model has {model_winner} by {margin_abs:.1f}." if margin_abs
                else "No line is posted yet, and the model has this one dead even.")
    if our_pick is None:
        if spread_home == 0:
            return "Vegas and the model both have this one even — no lean either way."
        fav = home if spread_home > 0 else away
        return f"The model lands exactly on the {fav} {abs(spread_home):g}-point spread — no lean either way."

    pick = home if our_pick == "home" else away
    pick_line = _team_line(spread_home, our_pick)
    if spread_home == 0:
        return phr.pick("pk", [
            f"Vegas can't separate these two; the model can — {pick} by {margin_abs:.1f}.",
            f"It's a pick'em in Vegas, but the model has {pick} by {margin_abs:.1f}.",
        ])
    fav, dog = (home, away) if spread_home > 0 else (away, home)
    line = f"{abs(spread_home):g}"
    m = model_margin_home if fav == home else -model_margin_home
    if pick == fav:
        gap = m - abs(spread_home)
        size = "a hair" if gap < 1 else ("a bit" if gap < 3 else "well")
        return phr.pick("fav", [
            f"Vegas has {fav} by {line}; the model goes {size} past that, to {m:.1f}. Lean {fav} {pick_line}.",
            f"{fav} is a {line}-point favorite, and the model thinks that's not enough — it has them by {m:.1f}.",
            f"The model is even higher on {fav} than Vegas is: {m:.1f} points to the market's {line}.",
        ])
    if m > 0:
        return phr.pick("dog", [
            f"The model likes {fav} too, just not by {line} — it has them by {m:.1f}, so the lean is {dog} {pick_line}.",
            f"Asking {fav} to win by more than {line} is a lot when the model has them by {m:.1f}. Lean {dog} {pick_line}.",
            f"Vegas and the model agree {fav} should win; the model just expects {dog} to keep it closer than {line}.",
            f"The model's {fav} by {m:.1f} lands inside the {line}-point spread, so {dog} {pick_line} gets the lean.",
        ])
    if m == 0:
        return f"The model calls it a dead heat; Vegas has {fav} by {line}. That's enough to lean {dog} {pick_line}."
    return phr.pick("upset", [
        f"Upset call: Vegas favors {fav} by {line}, but the model has {dog} winning outright by {-m:.1f}.",
        f"The model goes against the grain, projecting {dog} to win straight up by {-m:.1f} as a {line}-point underdog.",
        f"Vegas has {fav}; the model has {dog}, by {-m:.1f}. That flips the lean to {dog} {pick_line}.",
    ])


def build_storyline(
    home, away, model_margin_home, spread_home, our_pick, weights, rest_edge,
    oline_home, oline_away, coach_home, coach_away, revenge_flag,
    weather, international_site, neutral_site, roof, breakdown=None, ranks=None,
    qb_injury_home=None, qb_injury_away=None, former_coach_home=None, former_coach_away=None,
    phrasing=None,
):
    """A short prose readout of *why* the model landed where it did --
    the same factors already computed for this game, synthesized into
    sentences instead of left as a pile of tags. Template-based (not an
    LLM call -- this runs in a GitHub Actions scan, no API budget for
    that), but branches on enough of the real inputs per game that it
    reads as a genuine per-matchup readout rather than boilerplate, and
    each situation has a few phrasings (rotated across the board by
    Phrasing) so 16 games don't read like one sentence 16 times.
    Factors that aren't in the model's margin are described as context,
    never as pushing the pick.

    Returns (full, short): short is the matchup that drives the projection
    plus the verdict against the spread, surfaced on the collapsed game card
    so the "why" is visible before a click, not just after."""
    phr = phrasing or Phrasing()
    verdict = pick_verdict(phr, home, away, model_margin_home, spread_home, our_pick)
    if model_margin_home is None:
        return verdict, verdict
    qb_injury = {home: qb_injury_home, away: qb_injury_away}
    driver_kind = None
    short = [verdict]
    if breakdown and ranks:
        driver, driver_kind = driver_sentence(phr, home, away, model_margin_home, breakdown, ranks, qb_injury)
        short = [driver, verdict]
    sentences = list(short)
    if breakdown and ranks:
        counter = counter_sentence(phr, home, away, model_margin_home, breakdown, ranks)
        if counter:
            sentences.append(counter)

    pick_side = our_pick or ("home" if model_margin_home >= 0 else "away")
    pick_team, other_team = (home, away) if pick_side == "home" else (away, home)
    # The driver sentence already told the story of the projected loser's QB being out.
    model_loser = away if model_margin_home > 0 else home
    told = {model_loser} if driver_kind == "qb_out" else set()

    other_qb_out = qb_injury[other_team]
    pick_qb_out = qb_injury[pick_team]
    if other_qb_out and other_team not in told:
        sentences.append(
            f"{other_team}'s starting QB is {_qb_status(other_qb_out)} — the model already docks them "
            f"{QB_OUT_PENALTY:g} points for it."
        )
    if pick_qb_out and pick_team not in told and our_pick is not None:
        sentences.append(
            f"{pick_team}'s own starting QB is {_qb_status(pick_qb_out)}. The model docks them {QB_OUT_PENALTY:g} "
            "points for it and still thinks the line moved too far."
        )
    elif pick_qb_out and pick_team not in told:
        sentences.append(f"{pick_team}'s starting QB is {_qb_status(pick_qb_out)} — the model already docks them {QB_OUT_PENALTY:g} points for it.")

    if weights:
        share = sum(weights[s][p] for s in ("home", "away") for p in ("off", "def")) / 4
        pct = round(share * 100)
        if pct < 50:
            sentences.append(
                f"Early-season read: only about {pct}% of these ratings come from {UPCOMING_SEASON} games so far — "
                f"the rest is {UPCOMING_SEASON - 1}'s tape, shrunk toward average, so expect it to move."
            )
        else:
            sentences.append(f"About {pct}% of these ratings now come from {UPCOMING_SEASON} games.")

    pick_oline = oline_home if pick_side == "home" else oline_away
    if pick_oline and pick_oline.get("continuity_pct") is not None:
        pct = round(pick_oline["continuity_pct"] * 100)
        if pct >= 80:
            sentences.append(f"{pick_team}'s offensive line is largely intact from a year ago ({pct}% continuity), which tends to mean a faster start to the season.")
        elif pct <= 40:
            sentences.append(f"{pick_team} is breaking in a mostly new offensive line ({pct}% continuity) — a real wildcard this early in the year.")

    pick_coach = coach_home if pick_side == "home" else coach_away
    if pick_coach and pick_coach.get("same_coach") is False:
        sentences.append(f"{pick_team} is also playing under a new head coach this season, adding scheme uncertainty no efficiency stat fully captures yet.")

    if rest_edge is not None and rest_edge != 0:
        rested_team = home if rest_edge > 0 else away
        days = abs(rest_edge)
        sentences.append(f"{rested_team} comes in with {days} extra day{'s' if days != 1 else ''} of rest — context, not something the model scores.")

    if revenge_flag and (revenge_flag.get("home") or revenge_flag.get("away")):
        revenge_team = home if revenge_flag.get("home") else away
        sentences.append(f"{revenge_team} lost the last meeting between these division rivals — worth a {REVENGE_BONUS:g}-point nudge in the model.")

    if former_coach_home or former_coach_away:
        team, coach = (home, former_coach_home) if former_coach_home else (away, former_coach_away)
        sentences.append(
            f"{team} is also facing {coach['coach_name']}, who coached {team} last season — worth a {FORMER_COACH_BONUS:g}-point nudge in the model."
        )

    if neutral_site:
        where = "an international neutral site" if international_site else "a neutral site"
        sentences.append(f"This one's at {where}, so the model gives neither team home-field points.")
    elif international_site:
        sentences.append("This one's at an international site, so the usual home-field and weather assumptions carry more uncertainty than a normal week.")
    elif roof in INDOOR_ROOFS:
        sentences.append("Played indoors, so weather isn't a factor here.")
    elif weather:
        sentences.append(
            f"Forecast for kickoff: {weather['short_forecast']}, {weather['temperature_f']}°F, wind {weather['wind']} "
            "— something to watch if it turns into a run-heavy day."
        )

    return " ".join(sentences), " ".join(short)


def build_market(r, model_margin_home):
    """Real market odds straight from nflverse's schedules -- already
    populated ahead of kickoff, no paid odds API needed. spread_home follows
    the same sign convention as the rest of the site: positive means the
    HOME team is favored (verified against real results in betting_scan's
    module docstring notes)."""
    spread_home = r.get("spread_line")
    if spread_home is None:
        return None

    market_favorite = "home" if spread_home > 0 else ("away" if spread_home < 0 else None)
    # How much MORE (or less) the model favors the home team than Vegas
    # does, both expressed as the same home-relative margin so they're
    # directly comparable. Its sign is the pick: > 0 home, < 0 away, 0 no lean.
    edge_vs_market = round(model_margin_home - spread_home, 1) if model_margin_home is not None else None

    return {
        "spread_home": spread_home,
        "spread_home_odds": r.get("home_spread_odds"),
        "spread_away_odds": r.get("away_spread_odds"),
        "moneyline_home": r.get("home_moneyline"),
        "moneyline_away": r.get("away_moneyline"),
        "total": r.get("total_line"),
        "over_odds": r.get("over_odds"),
        "under_odds": r.get("under_odds"),
        "market_favorite": market_favorite,
        "edge_vs_market": edge_vs_market,
    }


def build_game_board(schedules, eff, injuries, rosters=None):
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

    ratings = blend(rating_inputs_asof(eff, UPCOMING_SEASON, next_week))
    ranks = unit_ranks(ratings)
    phrasing = Phrasing(seed=f"{UPCOMING_SEASON}-{next_week}")

    print("Computing continuity for game board...")
    depth_charts = load_cache("depth_charts")
    oline_continuity = build_oline_continuity(depth_charts, RETRO_SEASON, UPCOMING_SEASON)
    coach_continuity = build_coach_continuity(schedules, RETRO_SEASON, UPCOMING_SEASON)
    former_coach_matchups = build_former_coach_matchups(schedules, RETRO_SEASON, UPCOMING_SEASON)
    qb_injury_flags = build_qb_injury_flags(depth_charts, injuries, UPCOMING_SEASON, next_week, rosters=rosters)

    picks = []
    for r in week_games.iter_rows(named=True):
        home, away = r["home_team"], r["away_team"]
        neutral_site = r["location"] == "Neutral"
        model_margin_home = base_margin(ratings, home, away, neutral_site)

        meeting = last_meeting(schedules, home, away, UPCOMING_SEASON)
        revenge_flag = None
        if meeting and r["div_game"]:
            revenge_flag = {"home": meeting["winner"] == away, "away": meeting["winner"] == home, "last_season": meeting["season"]}

        former_coach_home = former_coach_matchups.get(home) if former_coach_matchups.get(home, {}).get("now_with") == away else None
        former_coach_away = former_coach_matchups.get(away) if former_coach_matchups.get(away, {}).get("now_with") == home else None

        qb_injury_home, qb_injury_away = qb_injury_flags.get(home), qb_injury_flags.get(away)
        adjustments = []  # the same nudges as breakdown rows (home-relative points)
        if qb_injury_home:
            adjustments.append({"kind": "qb_out", "team": home, "raw": -QB_OUT_PENALTY})
        if qb_injury_away:
            adjustments.append({"kind": "qb_out", "team": away, "raw": QB_OUT_PENALTY})
        if revenge_flag and revenge_flag.get("home"):
            adjustments.append({"kind": "revenge", "team": home, "raw": REVENGE_BONUS})
        if revenge_flag and revenge_flag.get("away"):
            adjustments.append({"kind": "revenge", "team": away, "raw": -REVENGE_BONUS})
        if former_coach_home:
            adjustments.append({"kind": "former_coach", "team": home, "coach": former_coach_home["coach_name"], "raw": FORMER_COACH_BONUS})
        if former_coach_away:
            adjustments.append({"kind": "former_coach", "team": away, "coach": former_coach_away["coach_name"], "raw": -FORMER_COACH_BONUS})
        breakdown = None
        if model_margin_home is not None:
            model_margin_home = round(model_margin_home + sum(a["raw"] for a in adjustments), 1)
            breakdown = margin_breakdown(ratings, home, away, neutral_site, adjustments, model_margin_home)

        rest_edge = None
        if r["home_rest"] is not None and r["away_rest"] is not None:
            rest_edge = r["home_rest"] - r["away_rest"]

        stadium = stadiums.get(home)
        primary_name = primary_stadium_name.get(home)
        international_site = bool(primary_name and r.get("stadium") and r["stadium"] != primary_name)
        weather = None  # must default every iteration -- otherwise a skipped fetch (e.g. this game's
        # condition below is False) would silently inherit the PREVIOUS game's weather value, since
        # Python loop variables aren't re-scoped per-iteration.
        if stadium and not international_site and stadium.get("nws_coverage") and r.get("gameday") and r["roof"] not in INDOOR_ROOFS:
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
            + (2 if former_coach_home or former_coach_away else 0)
        )

        spread_home = r.get("spread_line")
        our_pick = ats_pick(model_margin_home, spread_home)
        if model_margin_home is None:
            pick_status = "no_data"
        elif spread_home is None:
            pick_status = "no_line"
        else:
            pick_status = "pick" if our_pick else "no_lean"
        weights = None
        if home in ratings and away in ratings:
            weights = {
                side: {"off": round(ratings[t]["w_off"], 2), "def": round(ratings[t]["w_def"], 2)}
                for side, t in (("home", home), ("away", away))
            }
        oline_home, oline_away = oline_continuity.get(home), oline_continuity.get(away)
        coach_home, coach_away = coach_continuity.get(home), coach_continuity.get(away)

        game_ranks = {"home": ranks[home], "away": ranks[away]} if home in ranks and away in ranks else None
        storyline, storyline_short = build_storyline(
            home, away, model_margin_home, spread_home, our_pick, weights, rest_edge,
            oline_home, oline_away, coach_home, coach_away, revenge_flag,
            weather, international_site, neutral_site, r["roof"],
            breakdown=breakdown, ranks={home: ranks[home], away: ranks[away]} if game_ranks else None,
            qb_injury_home=qb_injury_home, qb_injury_away=qb_injury_away,
            former_coach_home=former_coach_home, former_coach_away=former_coach_away,
            phrasing=phrasing,
        )
        market = build_market(r, model_margin_home)

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
                "margin_breakdown": breakdown,
                "unit_ranks": game_ranks,
                "current_season_weight": weights,
                "our_pick": our_pick,
                "pick_status": pick_status,
                "neutral_site": neutral_site,
                "rest_days_edge_home": rest_edge,
                "oline_continuity": {"home": oline_home, "away": oline_away},
                "coach_continuity": {"home": coach_home, "away": coach_away},
                "qb_injury": {"home": qb_injury_home, "away": qb_injury_away},
                "revenge_game": revenge_flag,
                "former_coach": {"home": former_coach_home, "away": former_coach_away} if (former_coach_home or former_coach_away) else None,
                "roof": r["roof"],
                "international_site": international_site,
                "weather": weather,
                "storyline": storyline,
                "storyline_short": storyline_short,
                "market": market,
            }
        )

    picks.sort(key=lambda g: (-g["juice_score"], g["gametime"] or "", g["game_id"]))

    return {"season": UPCOMING_SEASON, "week": next_week, "games": picks}


def archive_current_week(game_board):
    """Freeze this week's picks before kickoff so they can be graded later
    against the actual final results without any risk of a later re-run
    silently rewriting a pick after the fact. Only ever writes/overwrites
    the archive file for a week where NONE of its games have started yet
    (by calendar day, not by whether a score has posted -- a manual
    workflow_dispatch re-run mid-week could otherwise land between two of
    the week's games and mix a completed one with an unplayed one into a
    single "frozen" week). Once any game in the week has kicked off, that
    week's archive is permanently left alone."""
    games = game_board.get("games") or []
    if not games:
        return
    season, week = game_board["season"], game_board["week"]
    today = datetime.now(timezone.utc).date().isoformat()
    if any((g.get("gameday") or "9999-99-99") <= today for g in games):
        print(f"  Week {week} has already started -- leaving its archive untouched.")
        return
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    path = os.path.join(ARCHIVE_DIR, f"{season}-w{week:02d}.json")
    payload = {"season": season, "week": week, "locked_at": datetime.now(timezone.utc).isoformat(), "pick_rule": "ats", "games": games}
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=2)
    print(f"  Locked picks for {season} week {week} ({len(games)} games) -> archive/{season}-w{week:02d}.json")


def grade_archived_weeks(schedules):
    """Grade every archived week's FROZEN numbers against final results,
    two ways: against the spread (the pick) and outright (did the model's
    projected winner win). The ATS pick is always re-derived from the frozen
    margin vs. the frozen spread with ats_pick -- one rule for every week,
    including weeks locked before picks were made against the spread. Only
    pre-kickoff numbers are used, so this is never hindsight. Idempotent; a
    week in progress carries partial grades until its games go final."""
    paths = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*-w*.json")))
    result_by_game = dict(zip(schedules["game_id"].to_list(), schedules["result"].to_list()))

    weeks, all_ats, all_outright = [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            archived = json.load(f)
        week_games, week_ats, week_outright = [], [], []
        for g in archived["games"]:
            result = result_by_game.get(g["game_id"])
            spread_home = (g.get("market") or {}).get("spread_home")
            margin = g.get("model_margin_home")
            pick = ats_pick(margin, spread_home)
            if archived.get("pick_rule") == "ats" and g.get("our_pick") != pick:
                raise ValueError(f"{g['game_id']}: archived pick {g.get('our_pick')} != derived {pick}")
            no_lean = margin is not None and spread_home is not None and pick is None
            push = correct = winner_correct = None
            if result is not None and spread_home is not None:
                push = result == spread_home
                if pick is not None and not push:
                    correct = (pick == "home") == (result > spread_home)
                    week_ats.append(correct)
            if result is not None and margin is not None and margin != 0 and result != 0:
                winner_correct = (margin > 0) == (result > 0)
                week_outright.append(winner_correct)
            week_games.append(
                {
                    "game_id": g["game_id"],
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "model_margin_home": margin,
                    "spread_home": spread_home,
                    "our_pick": pick,
                    "no_lean": no_lean,
                    "result": result,
                    "push": push,
                    "correct": correct,
                    "winner_correct": winner_correct,
                }
            )
        weeks.append(
            {
                "season": archived["season"],
                "week": archived["week"],
                "locked_at": archived.get("locked_at"),
                "ats": _record(week_ats),
                "outright": _record(week_outright),
                "games": week_games,
            }
        )
        all_ats += week_ats
        all_outright += week_outright

    return {
        "weeks_graded": len(weeks),
        "ats": _record(all_ats),
        "outright": _record(all_outright),
        "weeks": sorted(weeks, key=lambda w: (w["season"], w["week"])),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    schedules = load_cache("schedules")
    rosters = load_cache("rosters")
    # Completed seasons from the cache (ingest_historical.py runs right before this in CI, so
    # it's just as fresh) -- the same inputs calibrate_game_model.py fits on. Only the
    # in-progress season is pulled live.
    team_stats = load_cache("team_stats").filter(pl.col("season") < UPCOMING_SEASON)
    live_stats = load_live_team_stats(UPCOMING_SEASON, allow_empty=True)
    if live_stats is not None and not live_stats.is_empty():
        team_stats = pl.concat([team_stats, live_stats], how="diagonal_relaxed")
    eff = team_game_efficiency(team_stats)

    print(f"Building {RETRO_SEASON} walk-forward backtest...")
    backtest = build_backtest(schedules, eff)
    print(f"  {backtest['games']} games | corr {backtest['correlation']} | ATS {backtest['ats']} | outright {backtest['outright']}")
    if backtest["correlation"] is None or backtest["correlation"] < MIN_BACKTEST_CORR:
        # A failed run commits nothing, so the site keeps its last good data instead of
        # quietly publishing a broken model (the 2026 sign bug scored 0.06 here).
        raise RuntimeError(f"Backtest correlation {backtest['correlation']} < {MIN_BACKTEST_CORR} -- model looks broken, refusing to publish")

    print("Building upcoming-week game board...")
    injuries_hist = load_cache("injuries")
    injuries_current = load_current_season_injuries(UPCOMING_SEASON, allow_empty=True)
    all_injuries = pl.concat([injuries_hist, injuries_current], how="diagonal_relaxed") if injuries_current is not None else injuries_hist
    game_board = build_game_board(schedules, eff, all_injuries, rosters=rosters)
    print(f"  {len(game_board.get('games', []))} games in week {game_board.get('week')}")

    print("Archiving this week's picks (skipped if the week has already started)...")
    archive_current_week(game_board)

    print("Grading archived weeks against final results...")
    accountability = grade_archived_weeks(schedules)
    print(f"  {accountability['weeks_graded']} weeks | ATS {accountability['ats']} | outright {accountability['outright']}")

    meta = {"generated_at": datetime.now(timezone.utc).isoformat()}

    with open(os.path.join(OUT_DIR, "best_calls_backtest.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **backtest}), f, indent=2)
    with open(os.path.join(OUT_DIR, "game_board.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **game_board}), f, indent=2)
    with open(os.path.join(OUT_DIR, "accountability.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **accountability}), f, indent=2)

    print("Wrote best_calls_backtest.json, game_board.json, and accountability.json")


def load_live_team_stats(season, allow_empty=False):
    import nflreadpy as nfl

    try:
        return nfl.load_team_stats(seasons=[season])
    except Exception as e:
        if allow_empty:
            print(f"  no team_stats for {season} yet ({e}) -- expected before that season's games start")
            return None
        raise


def load_current_season_injuries(season, allow_empty=False):
    import nflreadpy as nfl

    try:
        return nfl.load_injuries(seasons=[season])
    except Exception as e:
        if allow_empty:
            print(f"  no injuries for {season} yet ({e}) -- expected before that season's games start")
            return None
        raise


if __name__ == "__main__":
    main()

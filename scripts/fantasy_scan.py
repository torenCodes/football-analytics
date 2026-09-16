"""Fantasy dashboard scan: "Perfect Team" retrospective, 2026 Draft Board,
and 2026 Dream Team (realistic snake-draft simulation).

Reads from data_cache/ (populated by ingest_historical.py) and writes
data/fantasy/perfect_team.json, draft_board.json, and dream_team.json. All
three score every player under BOTH PPR and Standard rules -- the frontend
toggles between the two client-side from one fetch.

Covers QB/RB/WR/TE (individual players), K (individual players, distance-
tiered scoring), and DST (team defense/special teams, scored from team-level
defensive stats + points allowed -- there's no individual "player" for a
team defense, so it's identified as "<TEAM>_DST").

Run standalone, re-run anytime (weekly via GitHub Actions once the season
is live) to refresh.
"""

import json
import os
from datetime import datetime, timezone

import polars as pl

from shared import (
    DEFENSIVE_LINE_POS,
    REF_DIR,
    TEAM_CODE_ALIASES,
    build_coach_continuity,
    build_current_team_lookup,
    build_former_coach_matchups,
    build_oline_continuity,
    json_safe,
    load_cache,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "fantasy")
ARCHIVE_DIR = os.path.join(OUT_DIR, "archive")

RETRO_SEASON = 2025  # most recently completed season -> "Perfect Team" + draft baseline
UPCOMING_SEASON = 2026
ROSTER_POS = ["QB", "RB", "WR", "TE", "K", "DST"]
SKILL_POS = ["QB", "RB", "WR", "TE"]  # positions where O-line continuity is a meaningful adjustment
RECENCY_WEIGHT = 0.5  # blend weight given to last-6-games PPG vs full-season PPG in the draft baseline
CONTINUITY_ADJ_WEIGHT = 1.5  # points added/removed per 1.0 of (continuity_pct - 0.6)
SAME_HC_BONUS, NEW_HC_PENALTY = 0.2, -0.5
SAME_COORDINATOR_BONUS, NEW_COORDINATOR_PENALTY = 0.2, -0.5  # same weight as HC -- the OC/DC is who actually calls the scheme a player has to relearn
MIN_GAMES_FOR_DRAFT_BOARD = 4  # a 1-2 game PPG sample isn't a signal -- exclude rather than rank on noise
MIN_GAMES_FOR_WEEKLY_RANKINGS = 1  # deliberately looser than the Draft Board's 4 -- this needs to be
# useful starting Week 2, not Week 5; games_this_season is shown alongside the score so a thin sample
# is visibly a thin sample rather than silently blended away
MATCHUP_POS = SKILL_POS + ["K"]  # positions the defense-vs-position matchup signal is computed for --
# DST is excluded (see build_dst_weekly's opponent_team comment): "who does this DST face" needs an
# inverted offense-strength model, out of scope for v1
MATCHUP_ADJ_WEIGHT = 0.5  # half the raw points-allowed-vs-league-average differential -- a modest
# nudge, same philosophy as CONTINUITY_ADJ_WEIGHT below
MIN_GAMES_FOR_DEFENSE_SAMPLE = 3  # below this, fall back to the defense's full-prior-season number --
# reuses betting_scan.py's MIN_GAMES_FOR_CURRENT_SEASON threshold for consistency, though a 3-game
# points-allowed-to-position total is a smaller, higher-variance sample than the play-level EPA data
# that threshold was originally calibrated against -- an acceptable, explicitly-flagged judgment call
QUESTIONABLE_PENALTY, DOUBTFUL_PENALTY, OUT_PENALTY = -2.0, -10.0, -20.0  # soft nudges, never a hard
# exclusion -- an "Out" player still shows up, clearly flagged, sunk to the bottom of their position
# group (a typical 8-20 ppg skill player's score goes negative), same "explainable, not a black box"
# philosophy as betting_scan.py's QB_OUT_PENALTY
MIN_GAMES_FOR_WAIVER_WIRE = 1  # a single monster game is often exactly what should trigger a waiver
# pickup in the first place -- requiring 2+ games would leave this section completely empty for the
# first two weeks of the season, missing the exact window a hot pickup is most available. games is
# still shown alongside ppg (same "visible sample size, not a hidden blend" convention as Weekly
# Rankings) so a 1-game outlier reads as exactly that, not a false guarantee.
# Rough draftable depth per position in a standard 12-team league (2 QB/5 RB/6 WR/2 TE/1 K/1 DST per
# roster, some slack for FLEX/bench) -- a player ranked deeper than this on the frozen preseason board,
# or missing from it entirely (rookies, anyone with too little 2025 usage to be ranked at all), is a
# reasonable proxy for "probably sitting on waivers in most leagues." Real per-league ownership data
# doesn't exist for free, so this is a proxy, not a fact -- same "model, not gospel" honesty as everywhere
# else on the site.
WAIVER_DRAFTABLE_DEPTH = {"QB": 24, "RB": 60, "WR": 72, "TE": 24, "K": 14, "DST": 14}

FORMER_COACH_TEAM_BONUS = 0.5  # points, for a SKILL_POS player whose team faces the coach who ran
# them last season -- above the Draft Board's flat SAME_HC_BONUS (0.2), below the injury penalties: a
# genuinely matchup-specific motivation signal, still clearly softer than "this player might not play"
FORMER_TEAM_BONUS = 1.0  # points, for a player individually facing a team they used to play for --
# a bigger, more personal storyline than facing a former coach, same magnitude as betting_scan.py's
# REVENGE_BONUS/FORMER_COACH_BONUS

UNIFIED_COLS = ["player_id", "player_display_name", "position", "team", "week", "season", "season_type", "fpts_ppr", "fpts_standard", "opponent_team"]


def build_skill_weekly(raw_stats):
    """QB/RB/WR/TE fpts_ppr/fpts_standard, in the unified schema shared with K/DST."""
    zero = pl.lit(0.0)
    df = raw_stats.filter(pl.col("position").is_in(SKILL_POS))

    def col(name):
        return pl.col(name).fill_null(0.0) if name in df.columns else zero

    base = (
        col("passing_yards") * 0.04
        + col("passing_tds") * 4
        - col("passing_interceptions") * 2
        + col("rushing_yards") * 0.1
        + col("rushing_tds") * 6
        + col("receiving_yards") * 0.1
        + col("receiving_tds") * 6
        - col("rushing_fumbles_lost") * 2
        - col("receiving_fumbles_lost") * 2
        + (col("passing_2pt_conversions") + col("rushing_2pt_conversions") + col("receiving_2pt_conversions")) * 2
    )
    df = df.with_columns((base + col("receptions") * 1.0).alias("fpts_ppr"), base.alias("fpts_standard"))
    return df.select(UNIFIED_COLS)


def build_kicker_weekly(raw_stats):
    """Distance-tiered kicker scoring: 3pts (<40yd), 4pts (40-49), 5pts (50+),
    1pt/PAT, -1/missed FG. A standard, commonly-used convention -- not the
    only one platforms use, but a defensible, documented v1."""
    zero = pl.lit(0.0)
    k = raw_stats.filter(pl.col("position") == "K")

    def col(name):
        return pl.col(name).fill_null(0.0) if name in k.columns else zero

    fg_low = col("fg_made_0_19") + col("fg_made_20_29") + col("fg_made_30_39")
    fg_mid = col("fg_made_40_49")
    fg_high = col("fg_made_50_59") + col("fg_made_60_")
    pts = fg_low * 3 + fg_mid * 4 + fg_high * 5 + col("pat_made") * 1 - col("fg_missed") * 1

    k = k.with_columns(pts.alias("fpts_ppr"), pts.alias("fpts_standard"))
    return k.select(UNIFIED_COLS)


# (max points allowed inclusive, fantasy points) -- standard D/ST points-allowed
# scale used by most default platform scoring. Anything above the last tier
# (35+) scores -4.
PA_TIERS = [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1)]


def _points_allowed_score(points_allowed_col):
    expr = pl.lit(-4)
    for max_pa, score in reversed(PA_TIERS):
        expr = pl.when(points_allowed_col <= max_pa).then(score).otherwise(expr)
    return expr


def build_dst_weekly(team_stats, schedules):
    """Team defense/special teams scoring: 1/sack, 2/INT, 2/fumble recovery,
    2/safety, 6/defensive or return TD, plus the points-allowed tiers above.
    There's no individual player_id for a team defense -- identified as
    "<TEAM>_DST" so it can flow through the same pipeline as every other
    position."""
    ts = team_stats.select(
        [
            "season", "week", "season_type", "team", "game_id",
            "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "def_tds", "special_teams_tds",
        ]
    )
    sched = schedules.select(["game_id", "home_team", "away_team", "home_score", "away_score"])
    ts = ts.join(sched, on="game_id", how="inner")
    ts = ts.with_columns(
        pl.when(pl.col("team") == pl.col("home_team")).then(pl.col("away_score")).otherwise(pl.col("home_score")).alias("points_allowed")
    )

    def col(name):
        return pl.col(name).fill_null(0.0)

    def_td = col("def_tds") + col("special_teams_tds")
    pts = (
        col("def_sacks") * 1
        + col("def_interceptions") * 2
        + col("fumble_recovery_opp") * 2
        + col("def_safeties") * 2
        + def_td * 6
        + _points_allowed_score(pl.col("points_allowed"))
    )
    ts = ts.with_columns(
        pts.alias("fpts_ppr"),
        pts.alias("fpts_standard"),
        (pl.col("team") + "_DST").alias("player_id"),
        (pl.col("team") + " D/ST").alias("player_display_name"),
        pl.lit("DST").alias("position"),
        # DST rows are excluded from the weekly-rankings matchup-difficulty
        # computation (that needs an inverted offense-strength model, not
        # built for v1) -- null here rather than deriving it, since points_allowed
        # above already captures DST's own matchup in a DST-appropriate way.
        pl.lit(None, dtype=pl.String).alias("opponent_team"),
    )
    return ts.select(UNIFIED_COLS)


def _normalize_name(s):
    return "".join(ch for ch in s.lower() if ch.isalnum()) if s else s


def load_coordinator_continuity():
    """The one hand-maintained reference table on the site -- nflverse only
    tracks head coaches, not coordinators. See the file's own _comment for
    sourcing and staleness caveats."""
    with open(os.path.join(REF_DIR, "coordinators_2026.json"), encoding="utf-8") as f:
        return json.load(f)


def load_market_consensus(raw_stats, teams):
    """FantasyPros consensus expert rankings (redraft, positional) via
    nflreadpy/ffverse. QB/RB/WR/TE join to gsis_id through the ff_playerids
    crosswalk; kickers aren't in that crosswalk at all (checked -- zero rows),
    so they're matched by normalized full name instead; DST has no player
    identity at all, so it's matched by team full name -> team_abbr. Treated
    as one PPR-convention market baseline (FantasyPros' default) applied to
    both scoring formats -- a fully format-split public consensus isn't
    available for free, and this is honestly labeled as such rather than
    overclaiming precision."""
    import nflreadpy as nfl

    rankings = nfl.load_ff_rankings(type="draft")
    positional = rankings.filter(
        pl.col("page_type").is_in(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te", "redraft-k", "redraft-dst"])
    )
    if positional.is_empty():
        return {}
    latest_date = positional["scrape_date"].max()
    positional = positional.filter(pl.col("scrape_date") == latest_date)

    result = {}

    skill = positional.filter(pl.col("page_type").is_in(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"]))
    ids = nfl.load_ff_playerids().select(["fantasypros_id", "gsis_id"]).drop_nulls()
    joined = skill.join(ids, left_on="id", right_on="fantasypros_id", how="inner")
    for row in joined.select(["gsis_id", "ecr", "pos"]).iter_rows(named=True):
        result[row["gsis_id"]] = {"market_rank": row["ecr"], "market_position": row["pos"]}

    k_rank = positional.filter(pl.col("page_type") == "redraft-k")
    k_name_to_rank = {_normalize_name(r["player"]): r["ecr"] for r in k_rank.iter_rows(named=True)}
    k_players = raw_stats.filter(pl.col("position") == "K").select(["player_id", "player_display_name"]).unique()
    for r in k_players.iter_rows(named=True):
        rank = k_name_to_rank.get(_normalize_name(r["player_display_name"]))
        if rank is not None:
            result[r["player_id"]] = {"market_rank": rank, "market_position": "K"}

    dst_rank = positional.filter(pl.col("page_type") == "redraft-dst")
    team_name_to_abbr = dict(zip(teams["team_name"].to_list(), teams["team_abbr"].to_list()))
    for r in dst_rank.iter_rows(named=True):
        abbr = team_name_to_abbr.get(r["player"])
        if abbr:
            result[f"{abbr}_DST"] = {"market_rank": r["ecr"], "market_position": "DST"}

    return result


def optimal_lineup(players):
    """players: list of dicts with player_id/name/position/points. Picks the
    highest-scoring 1QB/2RB/2WR/1TE/1FLEX/1K/1DST -- greedy-by-position is
    optimal here since there's no shared cap across slots. Also fills a
    6-man bench from the best remaining skill-position players (mirrors the
    Dream Team's bench: best-available depth regardless of position, not
    backup K/DST -- nobody rosters a second kicker)."""
    by_pos = {p: sorted((x for x in players if x["position"] == p), key=lambda x: -x["points"]) for p in ROSTER_POS}
    lineup = {}
    used_ids = set()

    def take(pos, n):
        picks = by_pos[pos][:n]
        used_ids.update(p["player_id"] for p in picks)
        return picks

    lineup["QB"] = take("QB", 1)
    lineup["RB"] = take("RB", 2)
    lineup["WR"] = take("WR", 2)
    lineup["TE"] = take("TE", 1)
    flex_pool = sorted(
        (x for pos in ("RB", "WR", "TE") for x in by_pos[pos] if x["player_id"] not in used_ids),
        key=lambda x: -x["points"],
    )
    lineup["FLEX"] = flex_pool[:1]
    used_ids.update(p["player_id"] for p in lineup["FLEX"])  # previously missed -- let the FLEX pick double as a bench pick too
    lineup["K"] = take("K", 1)
    lineup["DST"] = take("DST", 1)
    total = sum(p["points"] for slot in lineup.values() for p in slot)

    # Best-remaining-points, not pure position-blind: without a cap, QBs
    # (who score more raw points per game than bench-caliber RB/WR/TE) fill
    # every bench slot -- mathematically the "best remaining scorers," but
    # not what any real fantasy bench looks like. One reserve QB and one
    # reserve TE is realistic; RB/WR (where real rosters carry the most
    # depth) fill the rest.
    bench_position_cap = {"QB": 1, "TE": 1}
    bench_pool = sorted(
        (x for pos in SKILL_POS for x in by_pos[pos] if x["player_id"] not in used_ids),
        key=lambda x: -x["points"],
    )
    bench = []
    bench_pos_count = {}
    for p in bench_pool:
        cap = bench_position_cap.get(p["position"])
        if cap is not None and bench_pos_count.get(p["position"], 0) >= cap:
            continue
        bench.append(p)
        bench_pos_count[p["position"]] = bench_pos_count.get(p["position"], 0) + 1
        if len(bench) == 6:
            break
    return lineup, total, bench


def compute_perfect_team(stats):
    reg = stats.filter((pl.col("season") == RETRO_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(ROSTER_POS))
    agg = reg.group_by(["player_id", "player_display_name", "position", "team"]).agg(
        pl.col("fpts_ppr").sum().alias("total_ppr"),
        pl.col("fpts_standard").sum().alias("total_standard"),
        pl.col("week").n_unique().alias("games"),
    )

    out = {}
    for fmt, points_col in (("ppr", "total_ppr"), ("standard", "total_standard")):
        rows = agg.select(["player_id", "player_display_name", "position", "team", points_col, "games"]).sort(points_col, descending=True)
        leaders = {}
        for pos in ROSTER_POS:
            leaders[pos] = [
                {
                    "player_id": r["player_id"],
                    "name": r["player_display_name"],
                    "team": r["team"],
                    "points": round(r[points_col], 1),
                    "games": r["games"],
                    "ppg": round(r[points_col] / max(r["games"], 1), 1),
                }
                for r in rows.filter(pl.col("position") == pos).head(10).iter_rows(named=True)
            ]

        players = [
            {"player_id": r["player_id"], "name": r["player_display_name"], "team": r["team"], "position": r["position"], "points": r[points_col]}
            for r in rows.iter_rows(named=True)
        ]
        lineup, total, bench = optimal_lineup(players)
        out[fmt] = {
            "season": RETRO_SEASON,
            "leaders_by_position": leaders,
            "perfect_lineup": {
                slot: [{"player_id": p["player_id"], "name": p["name"], "team": p["team"], "points": round(p["points"], 1)} for p in picks]
                for slot, picks in lineup.items()
            },
            "perfect_lineup_total_points": round(total, 1),
            "bench": [
                {"player_id": p["player_id"], "name": p["name"], "team": p["team"], "position": p["position"], "points": round(p["points"], 1)}
                for p in bench
            ],
        }
    return out


def compute_team_of_week(stats, week):
    """The actual highest-scoring lineup for ONE already-played 2026 week --
    same shape and greedy-optimal logic as compute_perfect_team, just scoped
    to a single week instead of summed across a whole season. Returns None
    if that week has no stat rows yet (not played, or stats not synced)."""
    wk = stats.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("week") == week) & (pl.col("season_type") == "REG") & pl.col("position").is_in(ROSTER_POS))
    if wk.is_empty():
        return None
    agg = wk.group_by(["player_id", "player_display_name", "position", "team"]).agg(
        pl.col("fpts_ppr").sum().alias("total_ppr"),
        pl.col("fpts_standard").sum().alias("total_standard"),
    )

    out = {}
    for fmt, points_col in (("ppr", "total_ppr"), ("standard", "total_standard")):
        rows = agg.select(["player_id", "player_display_name", "position", "team", points_col])
        players = [
            {"player_id": r["player_id"], "name": r["player_display_name"], "team": r["team"], "position": r["position"], "points": r[points_col]}
            for r in rows.iter_rows(named=True)
        ]
        lineup, total, bench = optimal_lineup(players)
        out[fmt] = {
            "season": UPCOMING_SEASON,
            "week": week,
            "perfect_lineup": {
                slot: [{"player_id": p["player_id"], "name": p["name"], "team": p["team"], "points": round(p["points"], 1)} for p in picks]
                for slot, picks in lineup.items()
            },
            "perfect_lineup_total_points": round(total, 1),
            "bench": [
                {"player_id": p["player_id"], "name": p["name"], "team": p["team"], "position": p["position"], "points": round(p["points"], 1)}
                for p in bench
            ],
        }
    return out


def archive_team_of_week(stats, schedules):
    """Recompute + overwrite every completed 2026 week's Team of the Week
    unconditionally (unlike the betting picks archive, this is a
    retrospective stat aggregation, not a locked-in-advance prediction, so
    there's no hindsight-bias risk in refreshing it -- it should self-correct
    if nflverse issues a late stat correction)."""
    completed_weeks = (
        schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG") & pl.col("result").is_not_null())
        ["week"].unique().sort().to_list()
    )
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    written = []
    for week in completed_weeks:
        team_of_week = compute_team_of_week(stats, week)
        if team_of_week is None:
            continue
        path = os.path.join(ARCHIVE_DIR, f"{UPCOMING_SEASON}-w{week:02d}-team-of-week.json")
        with open(path, "w") as f:
            json.dump(json_safe(team_of_week), f, indent=2)
        written.append(week)
    return written


def _season_started(schedules):
    """Whether any 2026 regular-season game has reached its kickoff date
    yet, by calendar day rather than by whether a score has posted -- same
    guard convention as betting_scan.py's archive_current_week."""
    today = datetime.now(timezone.utc).date().isoformat()
    reg = schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG"))
    if reg.is_empty():
        return False
    return bool((reg["gameday"] <= today).any())


def freeze_preseason_board_if_needed(draft_board, dream_team, schedules):
    """One-time snapshot of the preseason Draft Board/Dream Team, taken the
    first run after Week 1 kicks off. Existence-checked so each file is
    only ever written once per season -- draft_board.json/dream_team.json
    keep regenerating weekly after this (their own computation is
    untouched), but this frozen copy is what the pivoted frontend reads as
    "what we said before the season started"."""
    if not _season_started(schedules):
        return False
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    board_path = os.path.join(ARCHIVE_DIR, f"{UPCOMING_SEASON}-preseason-draft-board.json")
    team_path = os.path.join(ARCHIVE_DIR, f"{UPCOMING_SEASON}-preseason-dream-team.json")
    wrote_any = False
    if not os.path.exists(board_path):
        with open(board_path, "w") as f:
            json.dump(json_safe(draft_board), f, indent=2)
        wrote_any = True
    if not os.path.exists(team_path):
        with open(team_path, "w") as f:
            json.dump(json_safe(dream_team), f, indent=2)
        wrote_any = True
    if wrote_any:
        print("  Froze preseason Draft Board + Dream Team snapshot at the Week 1 boundary.")
    return wrote_any


def compute_draft_board_accountability(stats, schedules):
    """Preseason Draft Board rank vs. actual 2026 season-to-date PPG rank,
    per position. Mirrors the site's existing market_gap convention
    (market_rank - our_rank, fantasy_scan.py's compute_draft_board) but
    pointed at "us-then vs. reality-now" instead of "us vs. the market."
    Negative rank_error = we underrated them (a steal); positive = we
    overrated them (a bust). Guarded by MIN_GAMES_FOR_DRAFT_BOARD, the same
    noisy-small-sample guard the Draft Board itself already uses."""
    board_path = os.path.join(ARCHIVE_DIR, f"{UPCOMING_SEASON}-preseason-draft-board.json")
    if not os.path.exists(board_path):
        return {"available": False, "note": "Accountability grading unlocks once the 2026 preseason Draft Board has been frozen at Week 1 kickoff."}
    with open(board_path, encoding="utf-8") as f:
        preseason_board = json.load(f)

    reg = stats.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(ROSTER_POS))
    if reg.is_empty():
        return {"available": False, "note": "Accountability grading unlocks once 2026 games have been played."}
    agg = reg.group_by(["player_id", "player_display_name", "position"]).agg(
        pl.col("fpts_ppr").sum().alias("total_ppr"),
        pl.col("fpts_standard").sum().alias("total_standard"),
        pl.col("week").n_unique().alias("games"),
    )

    out = {}
    for fmt, points_col in (("ppr", "total_ppr"), ("standard", "total_standard")):
        preseason_rank_by_id = {}
        for pos, rows in preseason_board[fmt]["rankings_by_position"].items():
            for row in rows:
                preseason_rank_by_id[row["player_id"]] = row["our_rank"]

        by_pos = {}
        for pos in ROSTER_POS:
            pos_rows = (
                agg.filter((pl.col("position") == pos) & (pl.col("games") >= MIN_GAMES_FOR_DRAFT_BOARD))
                .with_columns((pl.col(points_col) / pl.col("games")).alias("ppg"))
                .sort("ppg", descending=True)
            )
            graded = []
            for actual_rank, r in enumerate(pos_rows.iter_rows(named=True), start=1):
                preseason_rank = preseason_rank_by_id.get(r["player_id"])
                if preseason_rank is None:
                    continue
                graded.append(
                    {
                        "player_id": r["player_id"],
                        "name": r["player_display_name"],
                        "preseason_rank": preseason_rank,
                        "actual_rank": actual_rank,
                        "rank_error": actual_rank - preseason_rank,
                        "games": r["games"],
                        "ppg": round(r["ppg"], 1),
                    }
                )
            graded.sort(key=lambda g: g["rank_error"])
            n = len(graded)
            by_pos[pos] = {
                "players_graded": n,
                "mean_abs_rank_error": round(sum(abs(g["rank_error"]) for g in graded) / n, 1) if n else None,
                "hit_rate_within_5": round(sum(1 for g in graded if abs(g["rank_error"]) <= 5) / n, 3) if n else None,
                "biggest_steals": graded[:5],
                "biggest_busts": graded[-5:][::-1],
            }
        out[fmt] = by_pos
    return {"available": True, "season": UPCOMING_SEASON, "by_format": out}


def build_waiver_wire_explain(row):
    """Short prose readout, same spirit as build_draft_board_explain and
    build_weekly_ranking_explain -- surfaced as this row's tooltip."""
    if row["preseason_rank"] is None:
        expectation = "wasn't on our preseason board at all"
    else:
        expectation = f"we had them {row['preseason_rank']}th at the position entering the season"
    game_word = "game" if row["games"] == 1 else "games"
    return f"Likely available on waivers in most leagues -- {expectation}, and they're averaging {row['ppg']:.1f} ppg over {row['games']} {game_word} since."


def compute_waiver_wire(stats, schedules):
    """Likely-undrafted players (probably sitting on waivers in most
    leagues) who are outperforming that expectation -- the inverse of
    compute_draft_board_accountability's "steals" list, which explicitly
    SKIPS any player with no preseason rank (`if preseason_rank is None:
    continue`) since it's grading rank accuracy, not surfacing waiver
    adds. This is the mirror image: it only wants exactly the players
    that filter throws away, plus anyone ranked deeper than a realistic
    draftable cutoff (WAIVER_DRAFTABLE_DEPTH)."""
    board_path = os.path.join(ARCHIVE_DIR, f"{UPCOMING_SEASON}-preseason-draft-board.json")
    if not os.path.exists(board_path):
        return {"available": False, "note": "Waiver wire tracking unlocks once the 2026 preseason Draft Board has been frozen at Week 1 kickoff."}
    with open(board_path, encoding="utf-8") as f:
        preseason_board = json.load(f)

    reg = stats.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(ROSTER_POS))
    if reg.is_empty():
        return {"available": False, "note": "Waiver wire tracking unlocks once 2026 games have been played."}
    agg = reg.group_by(["player_id", "player_display_name", "position", "team"]).agg(
        pl.col("fpts_ppr").sum().alias("total_ppr"),
        pl.col("fpts_standard").sum().alias("total_standard"),
        pl.col("week").n_unique().alias("games"),
    )

    out = {}
    for fmt, points_col in (("ppr", "total_ppr"), ("standard", "total_standard")):
        preseason_rank_by_id = {}
        for pos, rows in preseason_board[fmt]["rankings_by_position"].items():
            for row in rows:
                preseason_rank_by_id[row["player_id"]] = row["our_rank"]

        by_pos = {}
        for pos in ROSTER_POS:
            depth = WAIVER_DRAFTABLE_DEPTH.get(pos, 24)
            pos_rows = (
                agg.filter((pl.col("position") == pos) & (pl.col("games") >= MIN_GAMES_FOR_WAIVER_WIRE))
                .with_columns((pl.col(points_col) / pl.col("games")).alias("ppg"))
                .sort("ppg", descending=True)
            )
            candidates = []
            for r in pos_rows.iter_rows(named=True):
                preseason_rank = preseason_rank_by_id.get(r["player_id"])
                if preseason_rank is not None and preseason_rank <= depth:
                    continue  # ranked within realistic draftable depth -- not a waiver-wire story
                candidates.append(
                    {
                        "player_id": r["player_id"],
                        "name": r["player_display_name"],
                        "team": r["team"],
                        "preseason_rank": preseason_rank,
                        "games": r["games"],
                        "ppg": round(r["ppg"], 1),
                    }
                )
            for c in candidates:
                c["explain"] = build_waiver_wire_explain(c)
            by_pos[pos] = candidates[:15]
        out[fmt] = by_pos
    return {"available": True, "season": UPCOMING_SEASON, "by_position": out}


def compute_season_state(schedules):
    """Single source of truth for the frontend: has the season started, and
    what's the latest week with completed games? Lets fantasy.html branch
    between the preseason layout and the in-season pivot without
    re-deriving this from multiple JSON files client-side."""
    started = _season_started(schedules)
    current_week = None
    if started:
        reg = schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG"))
        played = reg.filter(pl.col("result").is_not_null())
        current_week = int(played["week"].max()) if not played.is_empty() else int(reg["week"].min())
    return {"season": UPCOMING_SEASON, "season_started": started, "current_week": current_week}


def build_draft_board_explain(row):
    """A short prose readout of *why* this player's Edge Score/rank landed
    where it did -- same spirit as betting_scan.py's build_storyline, but
    surfaced as this row's tooltip rather than a visible table column,
    since the Draft Board's table-layout:fixed columns have no room left
    for prose without reopening past column-alignment bugs."""
    name = row["name"]
    line_label = "D-line" if row["line_continuity_type"] == "dline" else "O-line"
    role = row["coordinator_role"] or "OC"
    sentences = []

    pct = row["line_continuity_pct"]
    if pct is not None:
        pct_r = round(pct * 100)
        if pct_r >= 80:
            sentences.append(f"{line_label} continuity is elite ({pct_r}%), a real tailwind here.")
        elif pct_r <= 40:
            sentences.append(f"{line_label} continuity is shaky ({pct_r}%), a drag on the Edge Score this early.")

    if row["same_head_coach"] is False:
        sentences.append("Playing under a new head coach this season adds scheme uncertainty.")

    if row["same_coordinator"] is False and row["coordinator_name"]:
        sentences.append(f"New {role} ({row['coordinator_name']}) means a new scheme to learn too.")
    elif row["same_coordinator"] is True and row["coordinator_name"]:
        sentences.append(f"Same {role} ({row['coordinator_name']}) returning, so scheme continuity holds.")

    gap = row["market_gap"]
    if gap is not None and gap >= 5:
        sentences.append(f"We have {name} {round(gap)} spots higher than market consensus -- a value the market hasn't priced in yet.")
    elif gap is not None and gap <= -5:
        sentences.append(f"The market ranks {name} {round(abs(gap))} spots higher than we do -- our model is more cautious here.")

    if not sentences and row["coordinator_name"]:
        sentences.append(f"{role}: {row['coordinator_name']}.")

    return " ".join(sentences)


def compute_draft_board(stats, oline_continuity, dline_continuity, coach_continuity, coordinator_continuity, market, current_team):
    reg = stats.filter((pl.col("season") == RETRO_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(ROSTER_POS))

    season_agg = reg.group_by(["player_id", "player_display_name", "position", "team"]).agg(
        pl.col("fpts_ppr").sum().alias("season_ppr"),
        pl.col("fpts_standard").sum().alias("season_standard"),
        pl.col("week").n_unique().alias("games"),
    )

    last6 = (
        reg.sort("week")
        .group_by("player_id")
        .tail(6)
        .group_by("player_id")
        .agg(
            pl.col("fpts_ppr").sum().alias("last6_ppr_total"),
            pl.col("fpts_standard").sum().alias("last6_standard_total"),
            pl.col("week").n_unique().alias("last6_games"),
        )
    )

    merged = season_agg.join(last6, on="player_id", how="left")

    board = {}
    for fmt, season_col, last6_col in (("ppr", "season_ppr", "last6_ppr_total"), ("standard", "season_standard", "last6_standard_total")):
        rows = []
        for r in merged.iter_rows(named=True):
            games = max(r["games"], 1)
            season_ppg = r[season_col] / games
            last6_games = r["last6_games"] or 0
            last6_ppg = (r[last6_col] / last6_games) if last6_games else season_ppg
            baseline_ppg = RECENCY_WEIGHT * last6_ppg + (1 - RECENCY_WEIGHT) * season_ppg

            position = r["position"]
            # This board ranks players for the 2026 draft, so team should
            # reflect where they'll actually play in 2026 -- not wherever
            # they racked up their 2025 stats (r["team"]), which goes stale
            # the moment a player is traded or signs elsewhere. DST rows
            # aren't real gsis_ids so they always miss this lookup and fall
            # back to their (accurate) team-as-player_id team.
            team = current_team.get(r["player_id"], r["team"])
            is_skill = position in SKILL_POS
            is_dst = position == "DST"
            # Skill positions care about O-line continuity (pass protection/
            # run blocking in front of them); a DST's own fantasy output
            # tracks its D-line continuity instead (pass rush/run defense
            # personnel) -- same signal, mirrored to the other side of the
            # ball, not the same stat relabeled.
            line = oline_continuity.get(team) if is_skill else (dline_continuity.get(team) if is_dst else None)
            coach = coach_continuity.get(team)
            continuity_pct = line["continuity_pct"] if line else None
            same_coach = coach["same_coach"] if coach else None

            # Same split as O-line/D-line: skill positions track the offensive
            # coordinator (who calls their plays), DST tracks the defensive
            # coordinator. Display-only for now -- not folded into edge_score,
            # since this is hand-maintained data rather than pipeline-derived
            # like everything else feeding the ranking.
            coord = coordinator_continuity.get(team, {})
            if is_skill:
                same_coordinator, coordinator_name, coordinator_role = coord.get("same_oc"), coord.get("oc"), "OC"
            elif is_dst:
                same_coordinator, coordinator_name, coordinator_role = coord.get("same_dc"), coord.get("dc"), "DC"
            else:
                same_coordinator, coordinator_name, coordinator_role = None, None, None

            adjustment = 0.0
            if continuity_pct is not None:
                adjustment += CONTINUITY_ADJ_WEIGHT * (continuity_pct - 0.6)
            if same_coach is False:
                adjustment += NEW_HC_PENALTY
            elif same_coach is True:
                adjustment += SAME_HC_BONUS
            if same_coordinator is False:
                adjustment += NEW_COORDINATOR_PENALTY
            elif same_coordinator is True:
                adjustment += SAME_COORDINATOR_BONUS

            edge_score = round(baseline_ppg + adjustment, 2)
            market_info = market.get(r["player_id"], {})

            rows.append(
                {
                    "player_id": r["player_id"],
                    "name": r["player_display_name"],
                    "position": position,
                    "team": team,
                    "baseline_ppg": round(baseline_ppg, 2),
                    "season_ppg": round(season_ppg, 2),
                    "last6_ppg": round(last6_ppg, 2),
                    "games_played_2025": r["games"],
                    "line_continuity_pct": continuity_pct,
                    "line_continuity_type": "oline" if is_skill else ("dline" if is_dst else None),
                    "same_head_coach": same_coach,
                    "same_coordinator": same_coordinator,
                    "coordinator_name": coordinator_name,
                    "coordinator_role": coordinator_role,
                    "edge_score": edge_score,
                    "market_rank": market_info.get("market_rank"),
                }
            )

        by_pos = {}
        for pos in ROSTER_POS:
            pos_rows = sorted(
                (x for x in rows if x["position"] == pos and x["games_played_2025"] >= MIN_GAMES_FOR_DRAFT_BOARD),
                key=lambda x: -x["edge_score"],
            )
            for i, x in enumerate(pos_rows, start=1):
                x["our_rank"] = i
                x["market_gap"] = round(x["market_rank"] - i, 1) if x["market_rank"] is not None else None
                x["explain"] = build_draft_board_explain(x)
            by_pos[pos] = pos_rows[:50]
        board[fmt] = {"season_baseline": RETRO_SEASON, "draft_season": UPCOMING_SEASON, "rankings_by_position": by_pos}
    return board


# ---- 2026 Dream Team: realistic snake-draft simulation ----
# Precomputes every (league size x draft slot) combination the frontend's
# selectors offer, so picking a new league size/slot is an instant lookup
# into already-computed data -- same pattern as the PPR/Standard toggle --
# rather than re-running the simulation live (and duplicating this logic
# in JS).

LEAGUE_SIZES = [10, 12]
STARTERS = [("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)]
FLEX_ELIGIBLE = ["RB", "WR", "TE"]
BENCH_SIZE = 6
SKILL_PICKS_NEEDED = sum(n for _, n in STARTERS) + 1 + BENCH_SIZE  # 6 fixed starters + 1 FLEX + 6 bench = 13


def snake_pick_numbers(num_teams, draft_slot, num_rounds):
    """Overall pick number for a given draft slot, each round of a snake draft."""
    picks = []
    for rnd in range(1, num_rounds + 1):
        picks.append((rnd - 1) * num_teams + draft_slot if rnd % 2 == 1 else rnd * num_teams - draft_slot + 1)
    return picks


def build_big_board(ranked, league_teams):
    """VOR = edge score minus the edge score of the last starter-worthy
    player at that position league-wide -- replacement level scales with
    league size (e.g. RB replacement is RB20 in a 10-team league, RB24 in
    a 12-team league, since it's "2 starters x every team")."""
    big_board = []
    for pos, starters_per_team in STARTERS:
        players = ranked.get(pos, [])
        if not players:
            continue
        replacement_rank = starters_per_team * league_teams
        idx = min(replacement_rank - 1, len(players) - 1)
        replacement_score = players[idx]["edge_score"]
        for p in players:
            big_board.append({**p, "vor": round(p["edge_score"] - replacement_score, 2)})
    big_board.sort(key=lambda x: -x["vor"])
    for i, p in enumerate(big_board, start=1):
        p["overall_vor_rank"] = i
    return big_board


def simulate_draft(big_board, ranked, league_teams, draft_slot):
    """Pick-by-pick simulation, not just "take whoever's at our exact VOR
    rank": every pick from 1 to our last skill pick has to be resolved in
    order, because a player only reaches OUR slot if no one before us (in
    a chalk best-player-available draft) took him first. Other teams' picks
    are pure best-VOR-available (we have no visibility into their needs).
    OUR picks prefer the best-available player at a position we still need
    for our required starters -- a real drafter reaches for need over
    letting a starting slot go unfilled -- falling back to pure BPA once
    starters are set (bench picks). K/DST are conventionally punted to the
    final two picks since real drafts essentially never take them earlier
    regardless of raw VOR (replacement-level kickers/defenses sit on
    waivers all season, so there's no real cost to waiting)."""
    pick_numbers = snake_pick_numbers(league_teams, draft_slot, num_rounds=SKILL_PICKS_NEEDED + 2)
    skill_pick_numbers = pick_numbers[:SKILL_PICKS_NEEDED]
    our_picks = set(skill_pick_numbers)
    last_skill_pick = skill_pick_numbers[-1]

    taken = set()
    starters = {pos: [] for pos, _ in STARTERS}
    starters["FLEX"] = []
    bench = []
    required = dict(STARTERS)
    required["FLEX"] = 1
    need = dict(required)
    reach_notes = []
    base_positions = [pos for pos, _ in STARTERS]

    def best_available(position_filter=None):
        for p in big_board:
            if p["player_id"] in taken:
                continue
            if position_filter and p["position"] not in position_filter:
                continue
            return p
        return None

    for overall_pick in range(1, last_skill_pick + 1):
        if overall_pick in our_picks:
            # Fill the fixed positional slots (QB/RB/WR/TE) before FLEX, and
            # FLEX before bench -- a real drafter locks in required starters
            # first, then takes the best remaining RB/WR/TE for the flex
            # spot, same priority order as everything else here.
            base_needed = [pos for pos in base_positions if need.get(pos, 0) > 0]
            if base_needed:
                needed_positions = base_needed
            elif need.get("FLEX", 0) > 0:
                needed_positions = FLEX_ELIGIBLE
            else:
                needed_positions = None
            pick = best_available(needed_positions) if needed_positions else best_available()
            if pick is None:
                continue
            taken.add(pick["player_id"])
            pick = {**pick, "drafted_at_pick": overall_pick}
            if need.get(pick["position"], 0) > 0:
                starters[pick["position"]].append(pick)
                need[pick["position"]] -= 1
            elif need.get("FLEX", 0) > 0 and pick["position"] in FLEX_ELIGIBLE:
                starters["FLEX"].append(pick)
                need["FLEX"] -= 1
            else:
                bench.append(pick)
        else:
            pick = best_available()
            if pick is not None:
                taken.add(pick["player_id"])

    for pos, remaining in need.items():
        if remaining > 0:
            filled = required[pos] - remaining
            reach_notes.append(
                f"Only {filled} of {required[pos]} {pos} slot(s) filled by pick {last_skill_pick} -- "
                "every remaining option at the position was already off the board."
            )

    bench.sort(key=lambda x: -x["vor"])

    k_list, dst_list = ranked.get("K", []), ranked.get("DST", [])
    top_k, top_dst = (k_list[0] if k_list else None), (dst_list[0] if dst_list else None)

    def simplify(p):
        return {
            "player_id": p["player_id"],
            "name": p["name"],
            "team": p["team"],
            "position": p["position"],
            "edge_score": p["edge_score"],
            "vor": p.get("vor"),
            "overall_vor_rank": p.get("overall_vor_rank"),
            "drafted_at_pick": p.get("drafted_at_pick"),
        }

    replacement_ranks = {pos: n * league_teams for pos, n in STARTERS}
    methodology = (
        f"Simulates a {league_teams}-team snake draft from pick {draft_slot}. Every pick -- ours and the "
        f"other {league_teams - 1} teams' -- is modeled as best-player-available by Value Over Replacement "
        "(VOR): edge score minus the edge score of the last starter-worthy player at that position "
        f"league-wide (QB{replacement_ranks['QB']}/RB{replacement_ranks['RB']}/WR{replacement_ranks['WR']}/"
        f"TE{replacement_ranks['TE']} for a {league_teams}-team, 1QB/2RB/2WR/1TE/1FLEX-starter league; FLEX "
        "draws from whichever RB/WR/TE has the best remaining VOR once the fixed slots are filled). That "
        f"determines our exact pick numbers: {', '.join(str(p) for p in skill_pick_numbers)} for skill "
        f"positions, with K and DST punted to the final two picks ({pick_numbers[-2]}, {pick_numbers[-1]}) "
        "since real drafts essentially never take them earlier regardless of raw VOR."
    )

    return {
        "pick_numbers": pick_numbers,
        "methodology": methodology,
        "roster": {
            "QB": [simplify(p) for p in starters["QB"]],
            "RB": [simplify(p) for p in starters["RB"]],
            "WR": [simplify(p) for p in starters["WR"]],
            "TE": [simplify(p) for p in starters["TE"]],
            "FLEX": [simplify(p) for p in starters["FLEX"]],
            "K": [simplify(top_k)] if top_k else [],
            "DST": [simplify(top_dst)] if top_dst else [],
            "BENCH": [simplify(p) for p in bench],
        },
        "notes": reach_notes,
    }


def compute_dream_team(draft_board):
    scenarios = {}
    for league_teams in LEAGUE_SIZES:
        scenarios[str(league_teams)] = {str(slot): {} for slot in range(1, league_teams + 1)}
        for fmt in ("ppr", "standard"):
            ranked = draft_board[fmt]["rankings_by_position"]
            big_board = build_big_board(ranked, league_teams)
            for draft_slot in range(1, league_teams + 1):
                scenarios[str(league_teams)][str(draft_slot)][fmt] = simulate_draft(big_board, ranked, league_teams, draft_slot)

    return {
        "league_sizes": LEAGUE_SIZES,
        "roster_requirements": {**{pos: n for pos, n in STARTERS}, "FLEX": 1, "K": 1, "DST": 1, "BENCH": BENCH_SIZE},
        "scenarios": scenarios,
    }


# ---- Weekly start/sit rankings: the in-season successor to the Draft Board,
# recomputed every run against the upcoming matchup instead of frozen once.


def _defense_vs_position(stats, season):
    """Points allowed per game, by defense and offensive position, for ONE
    season -- excludes DST rows (see build_dst_weekly's opponent_team
    comment) and is never blended across seasons, which would dilute the
    signal with years-old scoring environments. Returns (by_team_pos,
    league_avg): by_team_pos maps (team, position) -> {ppr_ppg,
    standard_ppg, games}, league_avg maps position -> {ppr_ppg,
    standard_ppg} -- the SAME-season league baseline a defense's own number
    should be diffed against, so a thin current-season sample is never
    compared to a different season's scoring environment."""
    df = stats.filter(
        (pl.col("season") == season)
        & (pl.col("season_type") == "REG")
        & pl.col("position").is_in(MATCHUP_POS)
        & pl.col("opponent_team").is_not_null()
    )
    if df.is_empty():
        return {}, {}
    agg = df.group_by(["opponent_team", "position"]).agg(
        pl.col("fpts_ppr").sum().alias("ppr_total"),
        pl.col("fpts_standard").sum().alias("standard_total"),
        pl.col("week").n_unique().alias("games"),
    )
    by_team_pos = {}
    for r in agg.iter_rows(named=True):
        games = max(r["games"], 1)
        by_team_pos[(r["opponent_team"], r["position"])] = {
            "ppr_ppg": r["ppr_total"] / games,
            "standard_ppg": r["standard_total"] / games,
            "games": r["games"],
        }
    league = agg.group_by("position").agg(
        (pl.col("ppr_total").sum() / pl.col("games").sum()).alias("ppr_ppg"),
        (pl.col("standard_total").sum() / pl.col("games").sum()).alias("standard_ppg"),
    )
    league_avg = {r["position"]: {"ppr_ppg": r["ppr_ppg"], "standard_ppg": r["standard_ppg"]} for r in league.iter_rows(named=True)}
    return by_team_pos, league_avg


def build_player_former_teams(rosters):
    """gsis_id -> set of teams this player was ACTIVE on in a prior season,
    excluding their current team. Filtered to status=="ACT" -- unfiltered,
    over half of all (player, team) pairs in the roster history are
    practice-squad/cut-before-a-snap stints, not real former-team stories.
    Applies the same TEAM_CODE_ALIASES normalization build_current_team_lookup
    already uses, so a pre-2020 Raiders stint (tagged "OAK") still matches
    against today's "LV" -- otherwise it silently never would."""
    active = rosters.filter((pl.col("status") == "ACT") & pl.col("gsis_id").is_not_null()).with_columns(
        pl.col("team").replace(TEAM_CODE_ALIASES).alias("team")
    )
    by_player = active.group_by("gsis_id").agg(pl.col("team").unique().alias("teams"))
    return dict(zip(by_player["gsis_id"].to_list(), (set(t) for t in by_player["teams"].to_list())))


def build_weekly_ranking_explain(row):
    """Short prose readout of why this player's Start Score landed where it
    did -- same explainable-nudge spirit as build_draft_board_explain and
    betting_scan.py's build_storyline, surfaced as this row's tooltip."""
    sentences = [f"Averaging {row['recent_ppg']:.1f} ppg over {row['games_this_season']} game(s) this season."]
    if row["matchup_adj"] is not None:
        if row["matchup_adj"] >= 1.5:
            sentences.append(f"Facing {row['opponent']}, a favorable matchup for {row['position']}s this week.")
        elif row["matchup_adj"] <= -1.5:
            sentences.append(f"Facing {row['opponent']}, a tough matchup for {row['position']}s this week.")
    if row["injury_status"]:
        sentences.append(f"Listed {row['injury_status']} on the official injury report.")
    if row.get("former_coach_matchup"):
        sentences.append(f"{row['team']} also faces the coach who ran them last season -- extra motivation baked into the score.")
    if row.get("former_team_matchup"):
        sentences.append(f"{row['name']} is facing a former team this week, another motivation nudge factored in.")
    return " ".join(sentences)


def compute_weekly_rankings(stats, schedules, injuries, current_team, rosters):
    """Forward-looking start/sit rankings for the upcoming week -- the
    in-season successor to the Draft Board, using the exact same
    recency-weighted-production-plus-explainable-adjustments philosophy,
    just recomputed weekly against the actual next matchup instead of
    frozen once at the start of the season."""
    upcoming = schedules.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("game_type") == "REG") & pl.col("home_score").is_null())
    if upcoming.is_empty():
        return {"available": False, "note": "No upcoming games found in the cached schedule."}
    next_week = int(upcoming["week"].min())
    week_games = upcoming.filter(pl.col("week") == next_week)

    opponent_by_team = {}
    for r in week_games.iter_rows(named=True):
        opponent_by_team[r["home_team"]] = r["away_team"]
        opponent_by_team[r["away_team"]] = r["home_team"]

    # Soft, data-driven gate rather than _season_started(): a mid-week
    # workflow_dispatch re-run (e.g. Thursday night after only one Week-1
    # game) would already have _season_started()==True by calendar day,
    # but there's still nothing completed to build a recent-form baseline
    # from -- report unavailable rather than compute on an empty pool.
    completed = stats.filter(
        (pl.col("season") == UPCOMING_SEASON) & (pl.col("season_type") == "REG") & (pl.col("week") < next_week) & pl.col("position").is_in(ROSTER_POS)
    )
    if completed.is_empty():
        return {"available": False, "note": "Weekly rankings unlock once at least one 2026 game has been played."}

    season_agg = completed.group_by(["player_id", "player_display_name", "position"]).agg(
        pl.col("fpts_ppr").sum().alias("season_ppr"),
        pl.col("fpts_standard").sum().alias("season_standard"),
        pl.col("week").n_unique().alias("games"),
    )
    last6 = (
        completed.sort("week")
        .group_by("player_id")
        .tail(6)
        .group_by("player_id")
        .agg(
            pl.col("fpts_ppr").sum().alias("last6_ppr_total"),
            pl.col("fpts_standard").sum().alias("last6_standard_total"),
            pl.col("week").n_unique().alias("last6_games"),
        )
    )
    merged = season_agg.join(last6, on="player_id", how="left")

    current_by_team_pos, league_avg_current = _defense_vs_position(stats, UPCOMING_SEASON)
    fallback_by_team_pos, league_avg_fallback = _defense_vs_position(stats, RETRO_SEASON)
    games_played_by_defense = {}
    for (team, _pos), row in current_by_team_pos.items():
        games_played_by_defense[team] = max(games_played_by_defense.get(team, 0), row["games"])

    def defense_allowed(team, position, fmt):
        ppg_key = f"{fmt}_ppg"
        if games_played_by_defense.get(team, 0) >= MIN_GAMES_FOR_DEFENSE_SAMPLE and (team, position) in current_by_team_pos:
            return current_by_team_pos[(team, position)][ppg_key], league_avg_current.get(position, {}).get(ppg_key), "current_season"
        if (team, position) in fallback_by_team_pos:
            return fallback_by_team_pos[(team, position)][ppg_key], league_avg_fallback.get(position, {}).get(ppg_key), "prior_season_fallback"
        return None, None, None

    injury_by_player = {}
    if injuries is not None and not injuries.is_empty():
        wk_injuries = injuries.filter((pl.col("season") == UPCOMING_SEASON) & (pl.col("week") == next_week))
        if not wk_injuries.is_empty():
            injury_by_player = dict(zip(wk_injuries["gsis_id"].to_list(), wk_injuries["report_status"].to_list()))
    injury_penalty_by_status = {"Questionable": QUESTIONABLE_PENALTY, "Doubtful": DOUBTFUL_PENALTY, "Out": OUT_PENALTY}

    former_coach_matchups = build_former_coach_matchups(schedules, RETRO_SEASON, UPCOMING_SEASON)
    player_former_teams = build_player_former_teams(rosters)

    board = {}
    for fmt, season_col, last6_col in (("ppr", "season_ppr", "last6_ppr_total"), ("standard", "season_standard", "last6_standard_total")):
        rows = []
        for r in merged.iter_rows(named=True):
            position = r["position"]
            # DST player_ids ("<TEAM>_DST") aren't real gsis_ids so they
            # always miss current_team -- recover the team directly, same
            # reasoning as compute_draft_board's team-resolution comment.
            team = r["player_id"].removesuffix("_DST") if position == "DST" else current_team.get(r["player_id"])
            if team is None or team not in opponent_by_team:
                continue  # unresolved current team, or this team has a bye this week

            opponent = opponent_by_team[team]
            games = max(r["games"], 1)
            season_ppg = r[season_col] / games
            last6_games = r["last6_games"] or 0
            last6_ppg = (r[last6_col] / last6_games) if last6_games else season_ppg
            recent_ppg = RECENCY_WEIGHT * last6_ppg + (1 - RECENCY_WEIGHT) * season_ppg

            matchup_adj, matchup_source = None, None
            if position in MATCHUP_POS:
                allowed_ppg, league_ppg, source = defense_allowed(opponent, position, fmt)
                if allowed_ppg is not None and league_ppg is not None:
                    matchup_adj = round(MATCHUP_ADJ_WEIGHT * (allowed_ppg - league_ppg), 2)
                    matchup_source = source

            injury_status = injury_by_player.get(r["player_id"])
            injury_penalty = injury_penalty_by_status.get(injury_status, 0.0)

            former_coach_matchup = position in SKILL_POS and former_coach_matchups.get(team, {}).get("now_with") == opponent
            former_team_matchup = opponent in player_former_teams.get(r["player_id"], set())
            motivation_bonus = (FORMER_COACH_TEAM_BONUS if former_coach_matchup else 0.0) + (FORMER_TEAM_BONUS if former_team_matchup else 0.0)

            start_score = round(recent_ppg + (matchup_adj or 0.0) + injury_penalty + motivation_bonus, 2)

            row = {
                "player_id": r["player_id"],
                "name": r["player_display_name"],
                "position": position,
                "team": team,
                "opponent": opponent,
                "start_score": start_score,
                "recent_ppg": round(recent_ppg, 2),
                "games_this_season": r["games"],
                "matchup_adj": matchup_adj,
                "matchup_source": matchup_source,
                "injury_status": injury_status,
                "former_coach_matchup": former_coach_matchup,
                "former_team_matchup": former_team_matchup,
            }
            row["explain"] = build_weekly_ranking_explain(row)
            rows.append(row)

        by_pos = {}
        for pos in ROSTER_POS:
            pos_rows = sorted(
                (x for x in rows if x["position"] == pos and x["games_this_season"] >= MIN_GAMES_FOR_WEEKLY_RANKINGS),
                key=lambda x: -x["start_score"],
            )
            for i, x in enumerate(pos_rows, start=1):
                x["rank"] = i
            by_pos[pos] = pos_rows[:50]
        board[fmt] = {"week": next_week, "rankings_by_position": by_pos}

    return {"available": True, "season": UPCOMING_SEASON, **board}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_stats = load_cache("player_stats_weekly")
    team_stats = load_cache("team_stats")
    depth_charts = load_cache("depth_charts")
    schedules = load_cache("schedules")
    teams = load_cache("teams")
    rosters = load_cache("rosters")

    print(f"Loading live {UPCOMING_SEASON} player/team stats (in-season, empty until Week 1 kicks off)...")
    current_player_stats = load_current_season_player_stats(UPCOMING_SEASON, allow_empty=True)
    current_team_stats = load_current_season_team_stats(UPCOMING_SEASON, allow_empty=True)
    if current_player_stats is not None and not current_player_stats.is_empty():
        raw_stats = pl.concat([raw_stats, current_player_stats], how="diagonal_relaxed")
        print(f"  {current_player_stats.shape[0]} {UPCOMING_SEASON} player-weeks added")
    if current_team_stats is not None and not current_team_stats.is_empty():
        team_stats = pl.concat([team_stats, current_team_stats], how="diagonal_relaxed")
        print(f"  {current_team_stats.shape[0]} {UPCOMING_SEASON} team-weeks added")

    print("Building unified weekly fantasy points (QB/RB/WR/TE + K + DST)...")
    stats = pl.concat([build_skill_weekly(raw_stats), build_kicker_weekly(raw_stats), build_dst_weekly(team_stats, schedules)])
    print(f"  {stats.shape[0]} player-weeks across {stats['position'].n_unique()} positions")

    print("Computing O-line continuity...")
    oline_continuity = build_oline_continuity(depth_charts, RETRO_SEASON, UPCOMING_SEASON)
    print(f"  {len(oline_continuity)} teams")

    print("Computing D-line continuity (for DST rows)...")
    dline_continuity = build_oline_continuity(depth_charts, RETRO_SEASON, UPCOMING_SEASON, positions=DEFENSIVE_LINE_POS)
    print(f"  {len(dline_continuity)} teams")

    print("Computing head-coach continuity...")
    coach_continuity = build_coach_continuity(schedules, RETRO_SEASON, UPCOMING_SEASON)
    print(f"  {len(coach_continuity)} teams")

    print("Loading coordinator continuity (hand-maintained reference)...")
    coordinator_continuity = load_coordinator_continuity()
    print(f"  {sum(1 for k in coordinator_continuity if not k.startswith('_'))} teams")

    print("Loading market consensus (FantasyPros ECR via ffverse, incl. K/DST)...")
    try:
        market = load_market_consensus(raw_stats, teams)
        print(f"  {len(market)} players/teams matched")
    except Exception as e:
        print(f"  WARNING: market consensus unavailable ({e}), continuing without it")
        market = {}

    print("Computing current (2026) team per player, for traded/signed players...")
    current_team = build_current_team_lookup(rosters, UPCOMING_SEASON)
    print(f"  {len(current_team)} players")

    print("Computing Perfect Team retrospective...")
    perfect_team = compute_perfect_team(stats)

    print("Computing Draft Board...")
    draft_board = compute_draft_board(stats, oline_continuity, dline_continuity, coach_continuity, coordinator_continuity, market, current_team)

    print("Computing 2026 Dream Team (snake-draft simulation)...")
    dream_team = compute_dream_team(draft_board)

    print("Freezing preseason Draft Board + Dream Team snapshot, if Week 1 has started and not yet frozen...")
    freeze_preseason_board_if_needed(draft_board, dream_team, schedules)

    print("Computing Draft Board accountability (preseason rank vs. actual performance)...")
    accountability = compute_draft_board_accountability(stats, schedules)
    print(f"  available={accountability['available']}")

    print("Archiving Team of the Week for every completed 2026 week...")
    weeks_archived = archive_team_of_week(stats, schedules)
    print(f"  {len(weeks_archived)} week(s): {weeks_archived}")

    print("Computing season state (for the frontend's preseason/in-season pivot)...")
    season_state = compute_season_state(schedules)
    print(f"  {season_state}")

    print("Loading live 2026 injury reports...")
    injuries = load_current_season_injuries(UPCOMING_SEASON, allow_empty=True)

    print("Computing weekly start/sit rankings...")
    weekly_rankings = compute_weekly_rankings(stats, schedules, injuries, current_team, rosters)
    print(f"  available={weekly_rankings['available']}")

    print("Computing waiver wire watch (likely-undrafted players outperforming expectations)...")
    waiver_wire = compute_waiver_wire(stats, schedules)
    print(f"  available={waiver_wire['available']}")

    meta = {"generated_at": datetime.now(timezone.utc).isoformat()}

    with open(os.path.join(OUT_DIR, "perfect_team.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **perfect_team}), f, indent=2)
    with open(os.path.join(OUT_DIR, "draft_board.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **draft_board}), f, indent=2)
    with open(os.path.join(OUT_DIR, "dream_team.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **dream_team}), f, indent=2)
    with open(os.path.join(OUT_DIR, "accountability.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **accountability}), f, indent=2)
    with open(os.path.join(OUT_DIR, "season_state.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **season_state}), f, indent=2)
    with open(os.path.join(OUT_DIR, "weekly_rankings.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **weekly_rankings}), f, indent=2)
    with open(os.path.join(OUT_DIR, "waiver_wire.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **waiver_wire}), f, indent=2)

    print("Wrote perfect_team.json, draft_board.json, dream_team.json, accountability.json, season_state.json, weekly_rankings.json, and waiver_wire.json")


def load_current_season_player_stats(season, allow_empty=False):
    import nflreadpy as nfl

    try:
        return nfl.load_player_stats(seasons=[season])
    except Exception as e:
        if allow_empty:
            print(f"  no player_stats for {season} yet ({e}) -- expected before that season's games start")
            return None
        raise


def load_current_season_team_stats(season, allow_empty=False):
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

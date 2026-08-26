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
    build_coach_continuity,
    build_oline_continuity,
    json_safe,
    load_cache,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "fantasy")

RETRO_SEASON = 2025  # most recently completed season -> "Perfect Team" + draft baseline
UPCOMING_SEASON = 2026
ROSTER_POS = ["QB", "RB", "WR", "TE", "K", "DST"]
SKILL_POS = ["QB", "RB", "WR", "TE"]  # positions where O-line continuity is a meaningful adjustment
RECENCY_WEIGHT = 0.5  # blend weight given to last-6-games PPG vs full-season PPG in the draft baseline
CONTINUITY_ADJ_WEIGHT = 1.5  # points added/removed per 1.0 of (continuity_pct - 0.6)
SAME_HC_BONUS, NEW_HC_PENALTY = 0.2, -0.5
SAME_COORDINATOR_BONUS, NEW_COORDINATOR_PENALTY = 0.2, -0.5  # same weight as HC -- the OC/DC is who actually calls the scheme a player has to relearn
MIN_GAMES_FOR_DRAFT_BOARD = 4  # a 1-2 game PPG sample isn't a signal -- exclude rather than rank on noise

UNIFIED_COLS = ["player_id", "player_display_name", "position", "team", "week", "season", "season_type", "fpts_ppr", "fpts_standard"]


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


def compute_draft_board(stats, oline_continuity, dline_continuity, coach_continuity, coordinator_continuity, market):
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
            team = r["team"]
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
BENCH_SIZE = 6
SKILL_PICKS_NEEDED = sum(n for _, n in STARTERS) + BENCH_SIZE  # 6 starters + 6 bench = 12


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
    bench = []
    required = dict(STARTERS)
    need = dict(required)
    reach_notes = []

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
            needed_positions = [pos for pos, remaining in need.items() if remaining > 0]
            pick = best_available(needed_positions) if needed_positions else best_available()
            if pick is None:
                continue
            taken.add(pick["player_id"])
            pick = {**pick, "drafted_at_pick": overall_pick}
            if need.get(pick["position"], 0) > 0:
                starters[pick["position"]].append(pick)
                need[pick["position"]] -= 1
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
        f"TE{replacement_ranks['TE']} for a {league_teams}-team, 1QB/2RB/2WR/1TE-starter league). That "
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
        "roster_requirements": {**{pos: n for pos, n in STARTERS}, "K": 1, "DST": 1, "BENCH": BENCH_SIZE},
        "scenarios": scenarios,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw_stats = load_cache("player_stats_weekly")
    team_stats = load_cache("team_stats")
    depth_charts = load_cache("depth_charts")
    schedules = load_cache("schedules")
    teams = load_cache("teams")

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

    print("Computing Perfect Team retrospective...")
    perfect_team = compute_perfect_team(stats)

    print("Computing Draft Board...")
    draft_board = compute_draft_board(stats, oline_continuity, dline_continuity, coach_continuity, coordinator_continuity, market)

    print("Computing 2026 Dream Team (snake-draft simulation)...")
    dream_team = compute_dream_team(draft_board)

    meta = {"generated_at": datetime.now(timezone.utc).isoformat()}

    with open(os.path.join(OUT_DIR, "perfect_team.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **perfect_team}), f, indent=2)
    with open(os.path.join(OUT_DIR, "draft_board.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **draft_board}), f, indent=2)
    with open(os.path.join(OUT_DIR, "dream_team.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **dream_team}), f, indent=2)

    print("Wrote perfect_team.json, draft_board.json, and dream_team.json")


if __name__ == "__main__":
    main()

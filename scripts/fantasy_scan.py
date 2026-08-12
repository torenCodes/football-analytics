"""Fantasy dashboard scan: "Perfect Team" retrospective + 2026 Draft Board.

Reads from data_cache/ (populated by ingest_historical.py) and writes
data/fantasy/perfect_team.json + data/fantasy/draft_board.json. Both
outputs score every player under BOTH PPR and Standard rules -- the
frontend toggles between the two client-side from one fetch.

Run standalone, re-run anytime (weekly via GitHub Actions once the season
is live) to refresh.
"""

import json
import os
from datetime import datetime, timezone

import polars as pl

from shared import (
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
SKILL_POS = ["QB", "RB", "WR", "TE"]
RECENCY_WEIGHT = 0.5  # blend weight given to last-6-games PPG vs full-season PPG in the draft baseline
CONTINUITY_ADJ_WEIGHT = 1.5  # points added/removed per 1.0 of (continuity_pct - 0.6)
MIN_GAMES_FOR_DRAFT_BOARD = 4  # a 1-2 game PPG sample isn't a signal -- exclude rather than rank on noise


def add_fantasy_points(df):
    """Adds fpts_ppr and fpts_standard columns to a weekly stat frame."""
    zero = pl.lit(0.0)

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
    return df.with_columns(
        (base + col("receptions") * 1.0).alias("fpts_ppr"),
        base.alias("fpts_standard"),
    )


def load_market_consensus():
    """FantasyPros consensus expert rankings (redraft, positional) via
    nflreadpy/ffverse, joined to gsis_id through the ff_playerids crosswalk.
    Treated as one PPR-convention market baseline (FantasyPros' default)
    applied to both scoring formats -- a fully format-split public consensus
    isn't available for free, and this is honestly labeled as such rather
    than overclaiming precision."""
    import nflreadpy as nfl

    rankings = nfl.load_ff_rankings(type="draft")
    positional = rankings.filter(pl.col("page_type").is_in(["redraft-qb", "redraft-rb", "redraft-wr", "redraft-te"]))
    if positional.is_empty():
        return {}
    latest_date = positional["scrape_date"].max()
    positional = positional.filter(pl.col("scrape_date") == latest_date)

    ids = nfl.load_ff_playerids().select(["fantasypros_id", "gsis_id"]).drop_nulls()
    joined = positional.join(ids, left_on="id", right_on="fantasypros_id", how="inner")

    result = {}
    for row in joined.select(["gsis_id", "ecr", "pos"]).iter_rows(named=True):
        result[row["gsis_id"]] = {"market_rank": row["ecr"], "market_position": row["pos"]}
    return result


def optimal_lineup(players):
    """players: list of dicts with player_id/name/position/points. Picks the
    highest-scoring 1QB/2RB/2WR/1TE/1FLEX -- greedy-by-position is optimal
    here since there's no shared cap across slots."""
    by_pos = {p: sorted((x for x in players if x["position"] == p), key=lambda x: -x["points"]) for p in SKILL_POS}
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
    total = sum(p["points"] for slot in lineup.values() for p in slot)
    return lineup, total


def compute_perfect_team(stats):
    reg = stats.filter((pl.col("season") == RETRO_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POS))
    agg = reg.group_by(["player_id", "player_display_name", "position", "team"]).agg(
        pl.col("fpts_ppr").sum().alias("total_ppr"),
        pl.col("fpts_standard").sum().alias("total_standard"),
        pl.col("week").n_unique().alias("games"),
    )

    out = {}
    for fmt, points_col in (("ppr", "total_ppr"), ("standard", "total_standard")):
        rows = agg.select(["player_id", "player_display_name", "position", "team", points_col, "games"]).sort(points_col, descending=True)
        leaders = {}
        for pos in SKILL_POS:
            pos_rows = rows.filter(pl.col("position") == pos).head(10)
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
        lineup, total = optimal_lineup(players)
        out[fmt] = {
            "season": RETRO_SEASON,
            "leaders_by_position": leaders,
            "perfect_lineup": {
                slot: [{"player_id": p["player_id"], "name": p["name"], "team": p["team"], "points": round(p["points"], 1)} for p in picks]
                for slot, picks in lineup.items()
            },
            "perfect_lineup_total_points": round(total, 1),
        }
    return out


def compute_draft_board(stats, oline_continuity, coach_continuity, market):
    reg = stats.filter((pl.col("season") == RETRO_SEASON) & (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POS))

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

            team = r["team"]
            oline = oline_continuity.get(team)
            coach = coach_continuity.get(team)
            continuity_pct = oline["continuity_pct"] if oline else None
            same_coach = coach["same_coach"] if coach else None

            adjustment = 0.0
            if continuity_pct is not None:
                adjustment += CONTINUITY_ADJ_WEIGHT * (continuity_pct - 0.6)
            if same_coach is False:
                adjustment -= 0.5
            elif same_coach is True:
                adjustment += 0.2

            edge_score = round(baseline_ppg + adjustment, 2)
            market_info = market.get(r["player_id"], {})

            rows.append(
                {
                    "player_id": r["player_id"],
                    "name": r["player_display_name"],
                    "position": r["position"],
                    "team": team,
                    "baseline_ppg": round(baseline_ppg, 2),
                    "season_ppg": round(season_ppg, 2),
                    "last6_ppg": round(last6_ppg, 2),
                    "games_played_2025": r["games"],
                    "oline_continuity_pct": continuity_pct,
                    "same_head_coach": same_coach,
                    "edge_score": edge_score,
                    "market_rank": market_info.get("market_rank"),
                }
            )

        by_pos = {}
        for pos in SKILL_POS:
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = add_fantasy_points(load_cache("player_stats_weekly"))
    depth_charts = load_cache("depth_charts")
    schedules = load_cache("schedules")

    print("Computing O-line continuity...")
    oline_continuity = build_oline_continuity(depth_charts, RETRO_SEASON, UPCOMING_SEASON)
    print(f"  {len(oline_continuity)} teams")

    print("Computing head-coach continuity...")
    coach_continuity = build_coach_continuity(schedules, RETRO_SEASON, UPCOMING_SEASON)
    print(f"  {len(coach_continuity)} teams")

    print("Loading market consensus (FantasyPros ECR via ffverse)...")
    try:
        market = load_market_consensus()
        print(f"  {len(market)} players matched")
    except Exception as e:
        print(f"  WARNING: market consensus unavailable ({e}), continuing without it")
        market = {}

    print("Computing Perfect Team retrospective...")
    perfect_team = compute_perfect_team(stats)

    print("Computing Draft Board...")
    draft_board = compute_draft_board(stats, oline_continuity, coach_continuity, market)

    meta = {"generated_at": datetime.now(timezone.utc).isoformat()}

    with open(os.path.join(OUT_DIR, "perfect_team.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **perfect_team}), f, indent=2)
    with open(os.path.join(OUT_DIR, "draft_board.json"), "w") as f:
        json.dump(json_safe({"meta": meta, **draft_board}), f, indent=2)

    print("Wrote perfect_team.json and draft_board.json")


if __name__ == "__main__":
    main()

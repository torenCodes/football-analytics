"""Streaming Lab: is there a better way to pick a defense off the waiver wire?

Every week since 2021, among defenses rostered in at most half of leagues, take
the one whose opponent has the lowest Vegas implied total and the one FantasyPros
ranks highest, and compare what each actually scored.

Writes data/lab/streaming_lab.json, which fantasy_scan.py quotes on the page.
Local-only (not run in CI): python scripts/streaming_lab.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import fantasy_lab as lab
import fantasy_scan as fs
from shared import CACHE_DIR, load_cache

HISTORY = os.path.join(CACHE_DIR, "ff_dst_history.parquet")
MAX_ROSTERED = 50.0
rng = np.random.default_rng(31)


def history():
    if not os.path.exists(HISTORY):
        import nflreadpy as nfl
        (nfl.load_ff_rankings(type="all").filter(pl.col("page_type") == "weekly-dst")
         .select("player", "team", "ecr", "player_owned_avg", "scrape_date").write_parquet(HISTORY))
    return pl.read_parquet(HISTORY)


def main():
    schedules = load_cache("schedules")
    lines = lab.game_lines(schedules)
    stats = lab.load_stats().filter(pl.col("position") == "DST")
    h = history()
    rows = []
    for season in lab.SEASONS:
        weeks = lab.week_for_scrape(schedules, season)
        dates = sorted(d for d in h["scrape_date"].unique().to_list() if d.startswith(str(season)))
        for week, last in weeks:
            prev_last = next((l for w, l in weeks if w == week - 1), "0000")
            snapshot = [d for d in dates if prev_last <= d <= last]
            now = stats.filter((pl.col("season") == season) & (pl.col("week") == week))
            if not snapshot or now.is_empty():
                continue
            actual = dict(zip(now["team"].to_list(), now["fpts_ppr"].to_list()))
            pool = []
            for r in h.filter(pl.col("scrape_date") == max(snapshot)).iter_rows(named=True):
                team = lab.FP_TEAM.get(r["team"], r["team"])
                team = lab.TEAM_CODE_ALIASES.get(team, team)
                if team not in actual or (season, week, team) not in lines or r["player_owned_avg"] is None or r["player_owned_avg"] > MAX_ROSTERED:
                    continue
                opp = lines[(season, week, team)][0]
                implied = lines[(season, week, opp)][1]
                if implied is not None and r["ecr"] is not None:
                    pool.append((team, implied, r["ecr"], actual[team]))
            if len(pool) < 3:
                continue
            ours = min(pool, key=lambda x: x[1])
            fp = min(pool, key=lambda x: x[2])
            rows.append({"season": season, "week": week, "pool": len(pool), "ours": ours[3], "fp": fp[3], "avg": float(np.mean([x[3] for x in pool])), "same": ours[0] == fp[0]})
    t = pl.DataFrame(rows)
    diff = (t["ours"] - t["fp"]).to_numpy()
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]
    gain = (t["ours"] - t["avg"]).to_numpy()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "max_rostered": MAX_ROSTERED, "weeks": t.height,
        "ours": round(float(t["ours"].mean()), 2), "fantasypros": round(float(t["fp"].mean()), 2), "pool_average": round(float(t["avg"].mean()), 2),
        "ours_minus_fp": round(float(diff.mean()), 2), "lo": round(float(np.percentile(boots, 2.5)), 2), "hi": round(float(np.percentile(boots, 97.5)), 2),
        "same_pick": round(float(t["same"].mean()), 3), "ours_beat_pool_rate": round(float((gain > 0).mean()), 3),
        "by_season": [{"season": int(s), "weeks": g.height, "ours": round(float(g["ours"].mean()), 2), "fantasypros": round(float(g["fp"].mean()), 2), "pool_average": round(float(g["avg"].mean()), 2)} for (s,), g in t.group_by(["season"], maintain_order=True)],
    }
    with open(os.path.join(fs.BASE_DIR, "data", "lab", "streaming_lab.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

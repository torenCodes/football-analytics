"""One-time backfill: rebuild each 2026 week's frozen rankings from git history.

Weekly Rankings weren't archived before Week 9, but every published version is
in git. This replays them oldest to newest with the same rule the live archive
uses -- a team's rows stay open until its kickoff date -- so a week's archive
holds what the site actually showed before each game. FantasyPros' Friday
snapshot for that week is attached, so all three orderings can be graded.

Existing archives are skipped, so a week the live pipeline froze stays as it is.

Local-only: python scripts/backfill_weekly_rankings.py
"""

import json
import os
import subprocess
from datetime import datetime

import polars as pl

import fantasy_lab as lab
import fantasy_scan as fs
from shared import load_cache

REL = "data/fantasy/weekly_rankings.json"
LAST_V1_WEEK = 8


def git(*args):
    return subprocess.run(["git", "-C", fs.BASE_DIR, *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def main():
    schedules = load_cache("schedules").filter((pl.col("season") == fs.UPCOMING_SEASON) & (pl.col("game_type") == "REG"))
    stats = lab.load_stats()
    mapper = lab.fp_to_player_id(stats)
    history = lab.load_history()
    weeks = lab.week_for_scrape(load_cache("schedules"), fs.UPCOMING_SEASON)
    dates = sorted(d for d in history.filter(pl.col("page_type") == "weekly-wr")["scrape_date"].unique().to_list() if d.startswith(str(fs.UPCOMING_SEASON)))

    frozen = {}
    commits = [line.split() for line in git("log", "--reverse", "--format=%H %cI", "--", REL).strip().split("\n")]
    for commit, when in commits:
        day = datetime.fromisoformat(when).astimezone(tz=None).date().isoformat() if False else when[:10]
        try:
            version = json.loads(git("show", f"{commit}:{REL}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        if not version.get("available"):
            continue
        week = version["ppr"]["week"]
        if week > LAST_V1_WEEK:
            continue
        games = schedules.filter(pl.col("week") == week)
        started = set()
        for r in games.iter_rows(named=True):
            if r["gameday"] <= day:
                started.update((r["home_team"], r["away_team"]))
        book = frozen.setdefault(week, {fmt: {pos: [] for pos in fs.ROSTER_POS} for fmt in ("ppr", "standard")})
        for fmt in ("ppr", "standard"):
            for pos in fs.ROSTER_POS:
                book[fmt][pos] = fs.merge_frozen_rows(book[fmt][pos], version[fmt]["rankings_by_position"].get(pos, []), started)

    for week, book in sorted(frozen.items()):
        path = fs.weekly_archive_path(fs.UPCOMING_SEASON, week)
        if os.path.exists(path):
            print(f"week {week}: archive already exists -- skipped")
            continue
        prev_last = next((l for w, l in weeks if w == week - 1), "0000")
        last = next(l for w, l in weeks if w == week)
        snapshot_date = max((d for d in dates if prev_last <= d <= last), default=None)
        for pos in fs.ROSTER_POS:
            market = lab.snapshot(history, f"weekly-{pos.lower()}", snapshot_date, mapper) if snapshot_date else {}
            for fmt in ("ppr", "standard"):
                for row in book[fmt][pos]:
                    row["market_rank"] = market.get(row["player_id"])
        with open(path, "w") as f:
            json.dump(fs.json_safe({
                "season": fs.UPCOMING_SEASON, "week": week, "start_score_version": "v1", "backfilled_from_git": True,
                "consensus_snapshot": snapshot_date, "rankings": book,
            }), f, indent=1)
        print(f"week {week}: {sum(len(v) for v in book['ppr'].values())} PPR rows, consensus {snapshot_date}")


if __name__ == "__main__":
    main()

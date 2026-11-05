"""One-time repair: put each week's Thursday game back into its frozen archive.

Through Week 8 of 2026, archive_current_week replaced the whole week's file on
every run, and the board it was handed only lists unplayed games. So a Friday
run, after Thursday's game was final, rewrote the week without it: those games
were never graded.

Git still has them. This takes each missing game from the last version of that
week's archive committed before the game's kickoff -- pre-kickoff numbers only,
so nothing here is hindsight -- and adds it back. Idempotent.

Local-only: python scripts/restore_thursday_games.py
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

import polars as pl

from betting_scan import ARCHIVE_DIR, BASE_DIR
from shared import load_cache


def git(*args):
    return subprocess.run(["git", "-C", BASE_DIR, *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def kickoff_utc(gameday, gametime):
    """nflverse times are US Eastern; EDT ends Nov 1 in 2026, and every game
    repaired here was played before that."""
    return datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=timezone.utc) + timedelta(hours=4)


def main():
    schedules = load_cache("schedules").filter((pl.col("season") == 2026) & (pl.col("game_type") == "REG"))
    for name in sorted(os.listdir(ARCHIVE_DIR)):
        if not name.startswith("2026-w"):
            continue
        path = os.path.join(ARCHIVE_DIR, name)
        with open(path, encoding="utf-8") as f:
            archive = json.load(f)
        have = {g["game_id"] for g in archive["games"]}
        scheduled = schedules.filter(pl.col("week") == archive["week"])
        missing = scheduled.filter(~pl.col("game_id").is_in(list(have)))
        if missing.is_empty():
            continue
        rel = f"data/betting/archive/{name}"
        history = [line.split() for line in git("log", "--format=%H %cI", "--", rel).strip().split("\n")]
        for r in missing.iter_rows(named=True):
            kickoff = kickoff_utc(r["gameday"], r["gametime"])
            found = None
            for commit, when in history:  # newest first
                if datetime.fromisoformat(when) >= kickoff:
                    continue
                version = json.loads(git("show", f"{commit}:{rel}"))
                game = next((g for g in version["games"] if g["game_id"] == r["game_id"]), None)
                if game is not None:
                    found = {**game, "locked_at": version["locked_at"], "restored_from": commit[:8]}
                    break
            if found is None:
                print(f"week {archive['week']}: {r['game_id']} has no pre-kickoff version -- left out")
                continue
            archive["games"].append(found)
            print(f"week {archive['week']}: restored {r['game_id']} from {found['restored_from']} (locked {found['locked_at'][:16]}Z, kickoff {kickoff:%Y-%m-%d %H:%M}Z)")
        with open(path, "w") as f:
            json.dump(archive, f, indent=2)


if __name__ == "__main__":
    main()

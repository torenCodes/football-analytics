"""Wind Watch: over/under spots where the forecast says wind.

The rule, set before any live game was graded: an open-air game with 10+ mph of
forecast wind over its first four hours leans under the total. scripts/wind_lab.py
has the evidence, which is promising but not proven, so this runs as a live test.

Every outdoor game's forecast and total stay open until that game's kickoff, then
freeze for good, and only frozen numbers get graded. Runs from the betting scan and
from its own game-day workflow, so a spot reflects the forecast close to kickoff.

Writes data/betting/wind_watch.json and data/betting/archive/wind/<season>-w<week>.json.
"""

import glob
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import requests

from shared import REF_DIR, json_safe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "betting", "wind_watch.json")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data", "betting", "archive", "wind")
LAB_PATH = os.path.join(BASE_DIR, "data", "lab", "wind_lab.json")
SEASON = 2026
FIRST_WEEK = 9
FIRST_GAMEDAY = "2026-11-08"  # the live test starts with Week 9's Sunday games, not that Thursday's
RULE_MPH = 10
ROOFED = ("dome", "closed", "open")  # "open" is a retractable roof: teams close it when it's windy
ET = ZoneInfo("America/New_York")


def kickoff_utc(gameday, gametime):
    return datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=ET).astimezone(timezone.utc)


def roofed_teams(schedules):
    """Teams whose home stadium has a roof: nflverse leaves roof blank on upcoming
    games, so this reads each team's recent home games."""
    played = schedules.filter(pl.col("game_type") == "REG").drop_nulls("roof")
    return set(played.filter(pl.col("roof").is_in(ROOFED))["home_team"].to_list())


def primary_stadiums(schedules):
    counts = schedules.filter((pl.col("season") == SEASON) & (pl.col("game_type") == "REG")).drop_nulls("stadium").group_by(["home_team", "stadium"]).len().sort("len", descending=True)
    top = counts.group_by("home_team", maintain_order=True).first()
    return dict(zip(top["home_team"].to_list(), top["stadium"].to_list()))


def fetch_forecast(lat, lon, kick):
    """Mean forecast wind over the first four hours from kickoff, and the peak gust."""
    start = kick.date()
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": (start + timedelta(days=1)).isoformat(),
            "hourly": "wind_speed_10m,wind_gusts_10m", "wind_speed_unit": "mph", "timezone": "GMT"}, timeout=20, headers={"User-Agent": "football-forecast"})
        if r.status_code != 200:
            return None
        hourly = r.json()["hourly"]
        i = hourly["time"].index(kick.strftime("%Y-%m-%dT%H:00"))
    except (requests.RequestException, ValueError, KeyError):
        return None
    wind = [x for x in hourly["wind_speed_10m"][i:i + 4] if x is not None]
    gust = [x for x in hourly["wind_gusts_10m"][i:i + 4] if x is not None]
    if not wind:
        return None
    mean = round(sum(wind) / len(wind), 1)
    return mean, max(round(max(gust), 1), mean) if gust else None


def archive_path(week):
    return os.path.join(ARCHIVE_DIR, f"{SEASON}-w{week:02d}.json")


def update_week(schedules, week, now):
    """Refresh every outdoor game that hasn't kicked off; kicked-off games keep what they froze with."""
    with open(os.path.join(REF_DIR, "stadiums.json"), encoding="utf-8") as f:
        stadiums = json.load(f)
    roofed, primary = roofed_teams(schedules), primary_stadiums(schedules)
    path = archive_path(week)
    existing = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = {g["game_id"]: g for g in json.load(f)["games"]}
    games = dict(existing)
    upcoming = schedules.filter((pl.col("season") == SEASON) & (pl.col("week") == week) & (pl.col("game_type") == "REG"))
    for r in upcoming.iter_rows(named=True):
        home = r["home_team"]
        international = r["location"] == "Neutral" or (primary.get(home) and r["stadium"] and r["stadium"] != primary[home])
        st = stadiums.get(home)
        if home in roofed or international or not st or not r["gameday"] or not r["gametime"] or r["gameday"] < FIRST_GAMEDAY:
            continue
        kick = kickoff_utc(r["gameday"], r["gametime"])
        if kick <= now or (kick - now) > timedelta(days=9):
            continue
        fc = fetch_forecast(st["lat"], st["lon"], kick)
        if fc is None:
            continue  # keep the last good forecast rather than publish a gap
        games[r["game_id"]] = {
            "game_id": r["game_id"], "week": week, "gameday": r["gameday"], "gametime": r["gametime"], "weekday": r["weekday"], "kickoff_utc": kick.isoformat(),
            "away_team": r["away_team"], "home_team": home, "forecast_mph": fc[0], "gust_mph": fc[1], "total": r["total_line"],
            "spot": fc[0] >= RULE_MPH and r["total_line"] is not None, "locked_at": now.isoformat(),
        }
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    payload = {"season": SEASON, "week": week, "rule_mph": RULE_MPH, "games": sorted(games.values(), key=lambda g: g["kickoff_utc"])}
    with open(path, "w") as f:
        json.dump(json_safe(payload), f, indent=1)
    return payload["games"]


def grade(schedules):
    """Frozen numbers against the final score. Only a spot whose game has been played counts."""
    finals = {r["game_id"]: r["total"] for r in schedules.filter(pl.col("season") == SEASON).iter_rows(named=True)}
    past, under, over = [], [], []
    push = 0
    for path in sorted(glob.glob(os.path.join(ARCHIVE_DIR, f"{SEASON}-w*.json"))):
        with open(path, encoding="utf-8") as f:
            archived = json.load(f)
        for g in archived["games"]:
            final = finals.get(g["game_id"])
            if not g.get("spot") or final is None:
                continue
            result = "under" if final < g["total"] else "over" if final > g["total"] else "push"
            past.append({k: g[k] for k in ("game_id", "week", "away_team", "home_team", "forecast_mph", "total")} | {"final": final, "result": result})
            under += [1] if result == "under" else []
            over += [1] if result == "over" else []
            push += result == "push"
    return past, {"under": len(under), "over": len(over), "push": push}


def main():
    import nflreadpy as nfl

    schedules = nfl.load_schedules(seasons=[SEASON - 1, SEASON])
    now = datetime.now(timezone.utc)
    unplayed = schedules.filter((pl.col("season") == SEASON) & (pl.col("game_type") == "REG") & pl.col("result").is_null())
    week = int(unplayed["week"].min()) if not unplayed.is_empty() else None
    live = week is not None and week >= FIRST_WEEK
    rows = update_week(schedules, week, now) if live else []
    past, record = grade(schedules)
    played = {g["game_id"] for g in past}
    backtest = None
    if os.path.exists(LAB_PATH):
        with open(LAB_PATH, encoding="utf-8") as f:
            backtest = json.load(f)
    result = {
        "meta": {"generated_at": now.isoformat()},
        "season": SEASON, "week": week, "first_week": FIRST_WEEK, "live": live, "rule_mph": RULE_MPH,
        "spots": [g for g in rows if g["spot"] and g["game_id"] not in played],
        "outdoor_games": len(rows),
        "record": record, "past": past[::-1],
        "backtest": {"forecast": backtest["forecast"], "recorded": backtest["recorded"], "break_even": backtest["break_even"]} if backtest else None,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(json_safe(result), f, indent=1)
    print(f"Wind Watch: week {week}, live={live}, {len(rows)} outdoor games, {len(result['spots'])} spots, record {record}")


if __name__ == "__main__":
    main()

"""Wind Lab: do over/under totals under-react to wind?

Two looks, each graded on the closing total:
  recorded -- wind recorded at the game, 2006-2026, split into two eras.
  forecast -- archived short-range model forecasts for each stadium, 2022-2026
              (Open-Meteo's Historical Forecast API), the mean over the first
              four hours from kickoff. This is the version a reader could have
              acted on, and it's why the live rule reads a forecast.

Writes data/lab/wind_lab.json, which wind_watch.py quotes on the page.
Local-only (not run in CI): python scripts/wind_lab.py
"""

import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import requests

from shared import CACHE_DIR, REF_DIR, load_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIND_CACHE = os.path.join(CACHE_DIR, "wind")
ET = ZoneInfo("America/New_York")
RULE_MPH = 10
INDOOR = ["dome", "closed"]
rng = np.random.default_rng(9)


def summary(under):
    p = float(under.mean())
    se = float(np.sqrt(p * (1 - p) / len(under)))
    return {"rate": round(p, 3), "n": int(len(under)), "lo": round(p - 1.96 * se, 3), "hi": round(p + 1.96 * se, 3)}


def slope(x, y):
    b = np.polyfit(x, y, 1)[0]
    boots = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    return {"pts_per_mph": round(float(b), 3), "lo": round(float(np.percentile(boots, 2.5)), 3), "hi": round(float(np.percentile(boots, 97.5)), 3)}


def recorded():
    s = pl.read_parquet(os.path.join(CACHE_DIR, "schedules_2006_2026.parquet")).filter(
        (pl.col("game_type") == "REG") & pl.col("total").is_not_null() & pl.col("total_line").is_not_null() & pl.col("wind").is_not_null() & ~pl.col("roof").is_in(INDOOR))
    wind, season = s["wind"].cast(pl.Float64).to_numpy(), s["season"].to_numpy()
    tot, line = s["total"].cast(pl.Float64).to_numpy(), s["total_line"].cast(pl.Float64).to_numpy()
    out = {}
    for name, m in (("2006-2018", season <= 2018), ("2019-2026", season >= 2019)):
        pick = m & (wind >= RULE_MPH) & (tot != line)
        out[name] = {**summary((tot < line)[pick]), **slope(wind[m], (tot - line)[m])}
    return out


def hourly(team, season, stadiums):
    os.makedirs(WIND_CACHE, exist_ok=True)
    path = os.path.join(WIND_CACHE, f"{team}_{season}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    st = stadiums.get(team)
    if not st:
        return None
    today = datetime.now(timezone.utc).date().isoformat()
    r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params={
        "latitude": st["lat"], "longitude": st["lon"], "start_date": f"{season}-09-01", "end_date": min(f"{season + 1}-01-15", today),
        "hourly": "wind_speed_10m,wind_gusts_10m", "wind_speed_unit": "mph", "timezone": "GMT"}, timeout=60, headers={"User-Agent": "football-forecast-research"})
    if r.status_code != 200:
        return None
    data = r.json()["hourly"]
    with open(path, "w") as f:
        json.dump(data, f)
    time.sleep(0.25)
    return data


def forecast():
    with open(os.path.join(REF_DIR, "stadiums.json"), encoding="utf-8") as f:
        stadiums = json.load(f)
    s = load_cache("schedules").filter(
        (pl.col("game_type") == "REG") & (pl.col("season") >= 2022) & pl.col("result").is_not_null() & pl.col("total_line").is_not_null()
        & (pl.col("location") != "Neutral") & ~pl.col("roof").is_in(INDOOR))
    rows = []
    for r in s.iter_rows(named=True):
        d = hourly(r["home_team"], r["season"], stadiums)
        if not d:
            continue
        kick = datetime.fromisoformat(f"{r['gameday']}T{r['gametime']}").replace(tzinfo=ET).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00")
        if kick not in d["time"]:
            continue
        i = d["time"].index(kick)
        w = [x for x in d["wind_speed_10m"][i:i + 4] if x is not None]
        if w:
            rows.append((r["season"], float(np.mean(w)), float(r["total"]), float(r["total_line"])))
    a = np.array(rows)
    season, fc, tot, line = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    under, ok = tot < line, tot != line
    buckets = []
    for lo, hi in ((0, 6), (6, 10), (10, 15), (15, 99)):
        buckets.append({"lo": lo, "hi": hi, **summary(under[ok & (fc >= lo) & (fc < hi)])})
    by_season = {str(y): summary(under[ok & (fc >= RULE_MPH) & (season == y)]) for y in np.unique(season).astype(int) if (ok & (fc >= RULE_MPH) & (season == y)).sum()}
    return {"games": len(a), "rule": summary(under[ok & (fc >= RULE_MPH)]), "buckets": buckets, "by_season": by_season, **slope(fc, tot - line)}


def main():
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "rule_mph": RULE_MPH, "break_even": 0.524, "recorded": recorded(), "forecast": forecast()}
    with open(os.path.join(BASE_DIR, "data", "lab", "wind_lab.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=0)[:1800])


if __name__ == "__main__":
    main()

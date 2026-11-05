"""Team Lab: what's a fair answer to "is this team for real?"

For each team at several points in each 2019-2025 season, using only games
played so far, this measures what happened the rest of that season:

  forecast  -- which numbers best predict a team's point margin from here on?
               Weights are fit leave-one-season-out, then on every season for
               the live site.
  hold rate -- how often does a unit ranked this high stay in the top 10? This
               is the base rate each team card quotes.
  splits    -- do the stories fans tell (built on pressure, built on takeaways,
               built on the run) change those odds?
  verdicts  -- each possible verdict, and how those teams actually finished.

Writes scripts/reference/team_lab.json, which team_profile.py reads.

Local-only (not run in CI): python scripts/team_lab.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

from shared import CACHE_DIR, REF_DIR, TEAM_CODE_ALIASES, load_cache
from team_profile import team_game_tables, verdict_for

CACHE_OFF = os.path.join(CACHE_DIR, "pbp_team_games.parquet")
CACHE_DEF = os.path.join(CACHE_DIR, "pbp_team_games_def.parquet")
SEASONS = list(range(2019, 2026))
CUTS = (4, 6, 8, 10, 12, 14)  # weeks completed when the question gets asked
FEATURES = ["pd", "off", "dfn", "off_prior", "def_prior"]
PERIODS = {"early": (0, 6), "mid": (7, 10), "late": (11, 99)}


def load_tables():
    if not (os.path.exists(CACHE_OFF) and os.path.exists(CACHE_DEF)):
        import nflreadpy as nfl
        off, dfn = team_game_tables(nfl.load_pbp(seasons=list(range(2018, 2026))))
        off.write_parquet(CACHE_OFF)
        dfn.write_parquet(CACHE_DEF)
    fix = lambda t: t.with_columns(pl.col("team").replace(TEAM_CODE_ALIASES))
    return fix(pl.read_parquet(CACHE_OFF)), fix(pl.read_parquet(CACHE_DEF))


def totals(t, season, lo, hi):
    return {r["team"]: r for r in t.filter((pl.col("season") == season) & (pl.col("week") >= lo) & (pl.col("week") <= hi)).group_by("team").agg(pl.exclude("game_id", "season", "week", "team").sum()).iter_rows(named=True)}


def ranks(values, best_high):
    order = sorted(values, key=lambda k: -values[k] if best_high else values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def period_of(cut):
    return next(name for name, (lo, hi) in PERIODS.items() if lo <= cut <= hi)


def build_rows(off, dfn):
    games = load_cache("schedules").filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null())
    margin = pl.concat([
        games.select("season", "week", pl.col("home_team").alias("team"), pl.col("result").cast(pl.Float64).alias("pd")),
        games.select("season", "week", pl.col("away_team").alias("team"), (-pl.col("result")).cast(pl.Float64).alias("pd")),
    ]).with_columns(pl.col("team").replace(TEAM_CODE_ALIASES))
    rows = []
    for season in SEASONS:
        prior_o, prior_d = totals(off, season - 1, 1, 18), totals(dfn, season - 1, 1, 18)
        m = margin.filter(pl.col("season") == season)
        for cut in CUTS:
            o, d = totals(off, season, 1, cut), totals(dfn, season, 1, cut)
            o2, d2 = totals(off, season, cut + 1, 18), totals(dfn, season, cut + 1, 18)
            teams = sorted(set(o) & set(o2))
            if len(teams) < 30:
                continue
            share = lambda t, n, den: {x: t[x][n] / max(t[x][den], 1) for x in teams}
            sets = {
                "off": share(o, "epa", "plays"), "dfn": share(d, "epa", "plays"), "off_later": share(o2, "epa", "plays"), "dfn_later": share(d2, "epa", "plays"),
                "hit": share(d, "qb_hits", "dropbacks"), "sack": share(d, "sacks", "dropbacks"), "takeaway": share(d, "turnovers", "plays"), "def_no_to": share(d, "epa_no_to", "plays_no_to"),
                "off_pass": share(o, "pass_epa", "pass_plays"), "off_rush": share(o, "rush_epa", "rush_plays"),
            }
            so_far = {x: m.filter((pl.col("team") == x) & (pl.col("week") <= cut))["pd"] for x in teams}
            later = {x: m.filter((pl.col("team") == x) & (pl.col("week") > cut))["pd"] for x in teams}
            pd_now = {x: float(so_far[x].mean()) for x in teams}
            pd_next = {x: float(later[x].mean()) for x in teams}
            rk = {
                "off": ranks(sets["off"], True), "dfn": ranks(sets["dfn"], False), "off_later": ranks(sets["off_later"], True), "dfn_later": ranks(sets["dfn_later"], False),
                "hit": ranks(sets["hit"], True), "takeaway": ranks(sets["takeaway"], True), "def_no_to": ranks(sets["def_no_to"], False),
                "off_pass": ranks(sets["off_pass"], True), "off_rush": ranks(sets["off_rush"], True), "pd": ranks(pd_now, True), "pd_next": ranks(pd_next, True),
            }
            for x in teams:
                rows.append({
                    "season": season, "cut": cut, "period": period_of(cut), "team": x, "pd": pd_now[x], "pd_next": pd_next[x],
                    "off": sets["off"][x], "dfn": sets["dfn"][x],
                    "off_prior": prior_o[x]["epa"] / prior_o[x]["plays"] if x in prior_o else 0.0,
                    "def_prior": prior_d[x]["epa"] / prior_d[x]["plays"] if x in prior_d else 0.0,
                    **{f"r_{k}": v[x] for k, v in rk.items()},
                })
    return pl.DataFrame(rows)


def fit(t):
    A = np.column_stack([t[f].to_numpy() for f in FEATURES] + [np.ones(t.height)])
    return np.linalg.lstsq(A, t["pd_next"].to_numpy(), rcond=None)[0]


def predict(t, coef):
    return np.column_stack([t[f].to_numpy() for f in FEATURES] + [np.ones(t.height)]) @ coef


def rate(mask):
    return float(np.mean(mask)) if len(mask) else None


rng = np.random.default_rng(13)


def cluster_ci(values, season, team, boots=2000):
    """95% range for a mean, resampling whole team-seasons: one team shows up at
    several checkpoints in a season, so its rows aren't independent."""
    keys = np.array([f"{s}-{t}" for s, t in zip(season, team)])
    groups = [values[keys == k] for k in np.unique(keys)]
    picks = (rng.integers(0, len(groups), len(groups)) for _ in range(boots))
    means = [np.concatenate([groups[i] for i in p]).mean() for p in picks]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    off, dfn = load_tables()
    t = build_rows(off, dfn)
    print(f"{t.height} team-checkpoints, {SEASONS[0]}-{SEASONS[-1]}")

    # Forecast: one set of weights per part of the season, each season held out to score it.
    weights, held = {}, np.zeros(t.height)
    index = t.with_row_index("i")
    for period in PERIODS:
        p = index.filter(pl.col("period") == period)
        weights[period] = dict(zip(FEATURES + ["intercept"], [round(float(x), 3) for x in fit(p)]))
        for season in SEASONS:
            te, tr = p.filter(pl.col("season") == season), p.filter(pl.col("season") != season)
            held[te["i"].to_numpy()] = predict(te, fit(tr))
    y = t["pd_next"].to_numpy()
    print(f"forecast corr with rest-of-season margin {np.corrcoef(held, y)[0, 1]:.3f} | margin so far alone {np.corrcoef(t['pd'].to_numpy(), y)[0, 1]:.3f}")
    print("weights:", weights)
    t = t.with_columns(pl.Series("forecast", held))
    t = t.with_columns(pl.col("forecast").rank("ordinal", descending=True).over(["season", "cut"]).alias("r_forecast"))

    # Hold rates: how often a unit ranked here stays top 10 over the rest of the season.
    buckets = [(1, 5), (6, 10), (11, 16), (17, 22), (23, 32)]
    hold = {}
    print("\nHold rates: rank now -> top 10 the rest of the way, average rank later")
    for unit, now, later in (("offense", "r_off", "r_off_later"), ("defense", "r_dfn", "r_dfn_later")):
        hold[unit] = {}
        for period in PERIODS:
            hold[unit][period] = []
            p = t.filter(pl.col("period") == period)
            for lo, hi in buckets:
                b = p.filter((pl.col(now) >= lo) & (pl.col(now) <= hi))
                row = {"lo": lo, "hi": hi, "n": b.height, "top10": round(rate((b[later] <= 10).to_numpy()), 3), "avg_rank": round(float(b[later].mean()), 1)}
                hold[unit][period].append(row)
                print(f"  {unit:<8} {period:<6} #{lo}-{hi}: top 10 later {row['top10']:.0%} | avg rank {row['avg_rank']} | n {row['n']}")

    # Do the stories change the odds? Top-10 units only.
    print("\nSplits among top-10 units: stays top 10 the rest of the way")
    splits = {}

    def split(name, frame, cond, later):
        a, b = frame.filter(cond), frame.filter(~cond)
        pa, pb = rate((a[later] <= 10).to_numpy()), rate((b[later] <= 10).to_numpy())
        # Resample whole team-seasons; each draw recomputes both rates.
        f2 = frame.with_columns(cond.alias("_c"), (pl.col(later) <= 10).cast(pl.Float64).alias("_y"))
        keys = np.array([f"{s}-{t}" for s, t in zip(f2["season"].to_list(), f2["team"].to_list())])
        cs, ys = f2["_c"].to_numpy(), f2["_y"].to_numpy()
        groups = [(cs[keys == k], ys[keys == k]) for k in np.unique(keys)]
        diffs = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            c_ = np.concatenate([groups[i][0] for i in pick])
            y_ = np.concatenate([groups[i][1] for i in pick])
            if c_.any() and (~c_).any():
                diffs.append(y_[c_].mean() - y_[~c_].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        splits[name] = {"with": round(pa, 3), "without": round(pb, 3), "n_with": a.height, "n_without": b.height, "diff_lo": round(float(lo), 3), "diff_hi": round(float(hi), 3)}
        print(f"  {name:<34} with {pa:.0%} (n {a.height}) | without {pb:.0%} (n {b.height}) | diff {pa - pb:+.0%} [{lo:+.0%},{hi:+.0%}]")

    top_def = t.filter(pl.col("r_dfn") <= 10)
    split("defense with top-10 QB hit rate", top_def, pl.col("r_hit") <= 10, "r_dfn_later")
    split("defense leaning on takeaways", top_def, (pl.col("r_def_no_to") - pl.col("r_dfn")) >= 5, "r_dfn_later")
    split("defense top 10 without takeaways", top_def, pl.col("r_def_no_to") <= 10, "r_dfn_later")
    top_off = t.filter(pl.col("r_off") <= 10)
    split("offense led by the pass", top_off, pl.col("r_off_pass") <= pl.col("r_off_rush"), "r_off_later")
    split("offense top 10 in both pass and run", top_off, (pl.col("r_off_pass") <= 10) & (pl.col("r_off_rush") <= 10), "r_off_later")

    # Verdicts, scored on held-out forecasts.
    labels = [verdict_for(a, b) for a, b in zip(t["r_pd"].to_list(), t["r_forecast"].to_list())]
    t = t.with_columns(pl.Series("verdict", labels))
    verdicts = {}
    print("\nVerdicts: how those teams actually finished")
    for v in sorted(set(labels)):
        b = t.filter(pl.col("verdict") == v)
        verdicts[v] = {
            "n": b.height, "margin_so_far": round(float(b["pd"].mean()), 1), "forecast": round(float(b["forecast"].mean()), 1), "margin_later": round(float(b["pd_next"].mean()), 1),
            "rank_so_far": round(float(b["r_pd"].mean()), 1), "rank_later": round(float(b["r_pd_next"].mean()), 1),
            "top10_later": round(rate((b["r_pd_next"] <= 10).to_numpy()), 3), "bottom10_later": round(rate((b["r_pd_next"] >= 23).to_numpy()), 3),
        }
        x = verdicts[v]
        print(f"  {v:<22} n {x['n']:>3} | margin {x['margin_so_far']:+.1f} -> {x['margin_later']:+.1f} (forecast {x['forecast']:+.1f}) | rank {x['rank_so_far']} -> {x['rank_later']} | top 10 later {x['top10_later']:.0%} | bottom 10 later {x['bottom10_later']:.0%}")

    # Winning teams by what's carrying them: same scoreboard, different odds?
    print("\nTop-10 teams by margin, by which unit is doing it")
    winners = t.filter(pl.col("r_pd") <= 10)
    carried = {}
    for name, cond in (("bottom_half_offense", pl.col("r_off") >= 17), ("bottom_half_defense", pl.col("r_dfn") >= 17), ("both_top_half", (pl.col("r_off") <= 16) & (pl.col("r_dfn") <= 16))):
        b = winners.filter(cond)
        nx = b["pd_next"].to_numpy()
        lo, hi = cluster_ci(nx, b["season"].to_list(), b["team"].to_list())
        carried[name] = {
            "n": b.height, "team_seasons": b.select("season", "team").unique().height, "margin_so_far": round(float(b["pd"].mean()), 1), "margin_later": round(float(nx.mean()), 1), "later_lo": round(lo, 1), "later_hi": round(hi, 1),
            "rank_so_far": round(float(b["r_pd"].mean()), 1), "rank_later": round(float(b["r_pd_next"].mean()), 1), "top10_later": round(rate((b["r_pd_next"] <= 10).to_numpy()), 3),
        }
        x = carried[name]
        print(f"  {name:<22} n {x['n']:>3} ({x['team_seasons']} team-seasons) | margin {x['margin_so_far']:+.1f} -> {x['margin_later']:+.1f} [{x['later_lo']:+.1f},{x['later_hi']:+.1f}] | rank {x['rank_so_far']} -> {x['rank_later']} | top 10 later {x['top10_later']:.0%}")

    # Does each verdict beat plain regression to the mean? Compare with a forecast from margin alone.
    print("\nVerdicts vs. a forecast from scoreboard margin alone (rest-of-season margin, held out)")
    margin_only = np.zeros(t.height)
    pd_all = t["pd"].to_numpy()
    for season in SEASONS:
        tr, te = t["season"].to_numpy() != season, t["season"].to_numpy() == season
        margin_only[te] = np.polyval(np.polyfit(pd_all[tr], y[tr], 1), pd_all[te])
    resid = y - margin_only
    for v in sorted(set(labels)):
        m = np.array(labels) == v
        lo, hi = cluster_ci(resid[m], t["season"].to_numpy()[m], t["team"].to_numpy()[m])
        verdicts[v]["vs_margin_only"] = round(float(resid[m].mean()), 2)
        verdicts[v]["vs_margin_only_lo"] = round(lo, 2)
        verdicts[v]["vs_margin_only_hi"] = round(hi, 2)
        print(f"  {v:<22} {resid[m].mean():+.2f} [{lo:+.2f},{hi:+.2f}]")

    with open(os.path.join(REF_DIR, "team_lab.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(), "seasons": [SEASONS[0], SEASONS[-1]], "checkpoints": t.height,
            "periods": PERIODS, "forecast_weights": weights, "hold_rates": hold, "splits": splits, "carried": carried, "verdicts": verdicts,
        }, f, indent=1)
    print("Wrote scripts/reference/team_lab.json")


if __name__ == "__main__":
    main()

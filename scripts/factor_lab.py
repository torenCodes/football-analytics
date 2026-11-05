"""Factor Lab: which situational factors -- motivation, playoff race, rest,
travel, revenge -- add anything to the Game Picks model, and which does
Vegas already price in?

Every factor is a home-minus-away feature built only from what was known
before kickoff (schedules, standings to date, last week's result), and is
tested two ways:

  model  -- does it explain the model's misses (result - base v2 model
            margin)? 2019-2026, the seasons the rating model can run.
            Leave-one-season-out: the effect is fit on the other seasons
            and scored on the held-out one, against no factor at all.
  market -- does it explain the closing spread's misses (result -
            spread)? 2006-2026. A factor the market already prices shows
            ~0 here even if it matters on the field; a nonzero effect here
            is the only kind that could ever beat the line.

Writes data/lab/factor_lab.json for the Data Breakdown page.

Local-only (not run in CI): python scripts/factor_lab.py
Needs data_cache/ from ingest_historical.py; downloads 2006-2026 schedules
once into data_cache/.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import betting_scan as bs
from shared import CACHE_DIR, TEAM_CODE_ALIASES, load_cache, primary_coach

MARKET_SEASONS = list(range(2006, 2027))
MODEL_SEASONS = list(range(2019, 2027))
LONG_SCHEDULES = os.path.join(CACHE_DIR, "schedules_2006_2026.parquet")
RACE_FROM_WEEK = 11  # playoff-race status only means something late in the year
BOOT = 2000
rng = np.random.default_rng(11)

# Home time zone per team (relocated codes included) for the body-clock test.
WEST = {"SEA", "SF", "LA", "LAC", "LV", "OAK", "SD", "ARI", "DEN"}
EAST = {"ATL", "BAL", "BUF", "CAR", "CIN", "CLE", "DET", "IND", "JAX", "MIA", "NE", "NYG", "NYJ", "PHI", "PIT", "TB", "WAS"}


def canon(team):
    return TEAM_CODE_ALIASES.get(team, team)


def long_schedules():
    if not os.path.exists(LONG_SCHEDULES):
        import nflreadpy as nfl
        nfl.load_schedules(seasons=MARKET_SEASONS).write_parquet(LONG_SCHEDULES)
    return pl.read_parquet(LONG_SCHEDULES).filter(pl.col("game_type") == "REG")


def conferences():
    import nflreadpy as nfl
    t = nfl.load_teams()
    return {canon(a): c for a, c in zip(t["team_abbr"].to_list(), t["team_conf"].to_list())}


def team_games(s):
    """One row per team per REG game, in schedule order, with what that team
    knew going in: last game's margin, record, rest, coach."""
    side = lambda me, op, sign: s.select(
        "game_id", "season", "week", "gameday", "gametime", "div_game",
        pl.col(f"{me}_team").alias("team"), pl.col(f"{op}_team").alias("opp"),
        (pl.col("result") * sign).alias("margin"), pl.col(f"{me}_rest").alias("rest"),
        pl.col(f"{me}_coach").alias("coach"), pl.col(f"{op}_coach").alias("opp_coach"),
        pl.lit(me == "home").alias("is_home"),
    )
    tg = pl.concat([side("home", "away", 1), side("away", "home", -1)]).sort(["season", "team", "week"])
    return tg.with_columns(pl.col("margin").shift(1).over(["season", "team"]).alias("last_margin"))


def race_status(tg, conf):
    """Late-season playoff picture before each week: games back of the last
    playoff seed (6 per conference through 2019, 7 since), by win% to date."""
    out = {}
    for (season, week), g in tg.filter(pl.col("week") >= RACE_FROM_WEEK).group_by(["season", "week"]):
        seeds = 7 if season >= 2020 else 6
        # Every team's record before this week, including teams on bye.
        prior = tg.filter((pl.col("season") == season) & (pl.col("week") < week) & pl.col("margin").is_not_null())
        rec = prior.group_by("team").agg((pl.col("margin") > 0).sum().alias("w"), (pl.col("margin") < 0).sum().alias("l"))
        rows = [(canon(t), w, l) for t, w, l in rec.iter_rows()]
        for c in ("AFC", "NFC"):
            for t, status in bs.classify_race([r for r in rows if conf.get(r[0]) == c], seeds).items():
                out[(season, week, t)] = status
    return out


def game_features(s, tg, conf):
    """Home-minus-away factor values per game."""
    race = race_status(tg, conf)
    q = bs.qb_quality(load_cache("player_stats_weekly"))  # player stats start in 2018; earlier QBs all read as equal
    tgi = {(r["season"], r["week"], r["team"]): r for r in tg.iter_rows(named=True)}
    # Each team's primary coach by season (the live model's majority-vote rule),
    # keyed by current team code so relocations (OAK->LV etc.) carry over.
    coach_of = {}
    for season in MARKET_SEASONS:
        for team, coach in primary_coach(s, season).items():
            coach_of[(season, canon(team))] = coach
    # Most recent meeting between two teams, this season included.
    meetings = {}
    # Starting QBs seen so far per team-season. schedules' qb ids are the actual
    # starters -- announced before kickoff in all but rare late scratches.
    qb_seen = {}
    rows = []
    for r in s.sort(["season", "week"]).iter_rows(named=True):
        h, a, season, week = r["home_team"], r["away_team"], r["season"], r["week"]
        th, ta = tgi[(season, week, h)], tgi[(season, week, a)]
        key = frozenset((h, a))
        last = meetings.get(key)
        f = {"game_id": r["game_id"], "season": season, "week": week, "home": h, "away": a,
             "result": r["result"], "spread": r["spread_line"], "neutral": r["location"] == "Neutral"}

        def team_vals(t, tr, opp, opp_coach, qb):
            lm = tr["last_margin"]
            seen = qb_seen.get((season, t), {})
            usual = max(seen, key=seen.get) if sum(seen.values()) >= 2 else None
            st = race.get((season, week, canon(t)))
            pc = coach_of.get((season - 1, canon(t)))
            return {
                "off_loss": float(lm is not None and lm < 0),
                "off_blowout_loss": float(lm is not None and lm <= -17),
                "off_blowout_win": float(lm is not None and lm >= 17),
                "lost_cause": float(st == "out"),
                "on_the_cusp": float(st == "cusp"),
                "playoff_safe": float(st == "safe"),
                "off_bye": float((tr["rest"] or 0) >= 13),
                "revenge": float(bool(r["div_game"]) and last is not None and last[0] == opp),
                "former_coach": float(pc is not None and pc == opp_coach and pc != tr["coach"]),
                "west_early": float(not tr["is_home"] and t in WEST and opp in EAST and (r["gametime"] or "99") < "14:00"),
                "backup_qb": float(usual is not None and qb is not None and qb != usual),
                "qb_downgrade": float(usual is not None and qb is not None and qb != usual and season >= 2019 and q(qb, season) < q(usual, season)),
                "qb_upgrade": float(usual is not None and qb is not None and qb != usual and season >= 2019 and q(qb, season) >= q(usual, season)),
            }

        vh = team_vals(h, th, a, r["away_coach"], r["home_qb_id"])
        va = team_vals(a, ta, h, r["home_coach"], r["away_qb_id"])
        for k in vh:
            f[k] = vh[k] - va[k]
        f["rest_diff"] = float(np.clip((r["home_rest"] or 7) - (r["away_rest"] or 7), -7, 7))
        rows.append(f)
        if r["result"] is not None:
            meetings[key] = (h if r["result"] > 0 else a if r["result"] < 0 else None, season)
            for t, qb in ((h, r["home_qb_id"]), (a, r["away_qb_id"])):
                if qb:
                    qb_seen.setdefault((season, t), {}).setdefault(qb, 0)
                    qb_seen[(season, t)][qb] += 1
    return pl.DataFrame(rows)


FACTORS = {
    "off_loss": "Coming off a loss",
    "off_blowout_loss": "Coming off a 17+ point loss",
    "off_blowout_win": "Coming off a 17+ point win",
    "lost_cause": "Playoff race: 3+ games out (Week 11+)",
    "on_the_cusp": "Playoff race: within a game of the last seed (Week 11+)",
    "playoff_safe": "Playoff race: 3+ games clear (Week 11+)",
    "off_bye": "Coming off a bye",
    "rest_diff": "Rest advantage (per day)",
    "revenge": "Revenge: lost the last division meeting",
    "former_coach": "Facing last season's head coach",
    "west_early": "West-coast team at a 1 PM ET kickoff (away)",
    "backup_qb": "Starting someone other than the usual QB",
    "qb_downgrade": "Usual QB replaced by a weaker track record",
    "qb_upgrade": "Usual QB replaced by an equal or better one",
}


def model_margins(schedules_cached):
    """Base model margin (production constants, no QB-out adjustment) for every
    2019-2026 REG game, walk-forward -- the same code the live board runs, with
    the QB who actually started."""
    team_stats = load_cache("team_stats").filter(pl.col("season") < bs.UPCOMING_SEASON)
    live = bs.load_live_team_stats(bs.UPCOMING_SEASON, allow_empty=True)
    if live is not None and not live.is_empty():
        team_stats = pl.concat([team_stats, live], how="diagonal_relaxed")
    eff = bs.team_game_efficiency(team_stats, schedules_cached)
    starter, primary = bs.qb_starts(schedules_cached)
    q = bs.qb_quality(load_cache("player_stats_weekly"))
    out = {}
    games = schedules_cached.filter(pl.col("season").is_in(MODEL_SEASONS) & (pl.col("game_type") == "REG"))
    for (season, week), g in games.group_by(["season", "week"]):
        starters = {t: starter.get((r["game_id"], t)) for r in g.iter_rows(named=True) for t in (r["home_team"], r["away_team"])}
        ratings = bs.blend(bs.rating_inputs_asof(eff, season, week), bs.qb_gaps_for_week(starters, season, primary, q))
        for r in g.iter_rows(named=True):
            m = bs.base_margin(ratings, r["home_team"], r["away_team"], r["location"] == "Neutral")
            if m is not None:
                out[r["game_id"]] = m
    return out


def ols_effect(x, y):
    """Slope of y on x (with intercept) and a bootstrap 95% CI."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0][1]
    n = len(y)
    boots = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return b, lo, hi


def cover_rate(x, result, spread):
    """When exactly one side has the factor: how often did that side cover?"""
    m = (x != 0) & (result != spread)
    side_home = x[m] > 0
    covered = (result[m] > spread[m]) == side_home
    n = int(m.sum())
    p = covered.mean() if n else float("nan")
    se = np.sqrt(p * (1 - p) / n) if n else float("nan")
    return p, n, p - 1.96 * se, p + 1.96 * se


def loso_gain(x, resid, seasons):
    """Leave-one-season-out MSE change from adding the factor (negative = helps), with a bootstrap CI."""
    diff = np.zeros_like(resid)
    for s in np.unique(seasons):
        tr, te = seasons != s, seasons == s
        X = np.column_stack([np.ones(tr.sum()), x[tr]])
        a, b = np.linalg.lstsq(X, resid[tr], rcond=None)[0]
        base = np.mean(resid[tr])
        diff[te] = (resid[te] - (a + b * x[te])) ** 2 - (resid[te] - base) ** 2
    n = len(diff)
    boots = [diff[rng.integers(0, n, n)].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff.mean(), lo, hi


def main():
    s = long_schedules()
    conf = conferences()
    print("Building pre-kickoff factors for 2006-2026...")
    tg = team_games(s)
    feats = game_features(s, tg, conf).filter(pl.col("result").is_not_null() & pl.col("spread").is_not_null())
    print("Walk-forward model margins for 2019-2026...")
    margins = model_margins(load_cache("schedules"))
    feats = feats.with_columns(pl.col("game_id").replace_strict(margins, default=None).alias("model"))

    mk = feats
    md = feats.filter(pl.col("model").is_not_null())
    print(f"\nmarket test: {mk.height} games 2006-2026 | model test: {md.height} games 2019-2026")
    res_mk = (mk["result"] - mk["spread"]).to_numpy()
    res_md = (md["result"] - md["model"]).to_numpy()
    print(f"model residual sd {res_md.std():.1f} | spread residual sd {res_mk.std():.1f}\n")
    print(f"{'factor':<58} {'games':>6} | {'model: pts [95% CI]':>24} {'OOS MSE':>8} | {'market: pts [95% CI]':>24} {'covers':>7}")
    rows = []
    for key, label in FACTORS.items():
        x_mk, x_md = mk[key].to_numpy(), md[key].to_numpy()
        n_on = int((x_mk != 0).sum())
        b1, l1, h1 = ols_effect(x_md, res_md)
        g, gl, gh = loso_gain(x_md, res_md, md["season"].to_numpy())
        b2, l2, h2 = ols_effect(x_mk, res_mk)
        p, n, pl_, ph = cover_rate(x_mk, mk["result"].to_numpy(), mk["spread"].to_numpy())
        flag = lambda lo, hi: "*" if lo > 0 or hi < 0 else " "
        print(f"{label:<58} {n_on:>6} | {b1:+5.2f} [{l1:+5.2f},{h1:+5.2f}]{flag(l1, h1)} {g:+8.2f} | {b2:+5.2f} [{l2:+5.2f},{h2:+5.2f}]{flag(l2, h2)} {p:6.1%}")
        rows.append({
            "key": key, "label": label, "games": n_on, "games_model": int((x_md != 0).sum()),
            "model": {"pts": round(b1, 2), "lo": round(l1, 2), "hi": round(h1, 2), "oos_mse": round(g, 2), "oos_lo": round(gl, 2), "oos_hi": round(gh, 2)},
            "market": {"pts": round(b2, 2), "lo": round(l2, 2), "hi": round(h2, 2)},
            "covers": {"rate": round(float(p), 3), "n": n, "lo": round(float(pl_), 3), "hi": round(float(ph), 3)},
        })
    print("\n* = 95% CI excludes zero. Effects are points for the side with the factor (home-minus-away feature);")
    print("  'covers' = how often the side with the factor covered when only one side had it (break-even 52.4%).")
    out_dir = os.path.join(bs.BASE_DIR, "data", "lab")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "factor_lab.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "market_games": mk.height, "market_seasons": [MARKET_SEASONS[0], MARKET_SEASONS[-1]],
            "model_games": md.height, "model_seasons": [MODEL_SEASONS[0], MODEL_SEASONS[-1]],
            "model_residual_sd": round(float(res_md.std()), 1), "market_residual_sd": round(float(res_mk.std()), 1),
            "break_even": 0.524, "factors": rows,
        }, f, indent=1)
    print("Wrote data/lab/factor_lab.json")


if __name__ == "__main__":
    main()

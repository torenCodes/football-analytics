"""Rating Lab: is the Game Picks rating built the best way?

The live rating is each team's offense and defense EPA/play to date (vs.
league average), every play weighted the same, blended with last season.
This tests standard refinements of that core, one at a time and combined,
walk-forward over 2019-2026 regular-season games:

  recency   -- weight this season's games by DECAY ** (weeks ago)
  opponent  -- adjust each offense for the defenses it faced and vice
               versa (iterated to convergence), this season and last
  rush_wt   -- weight rushing EPA below passing EPA (rushing efficiency
               is the noisier, less repeatable half)
  points    -- blend in point differential per game, same blending

Each variant's scale and home field are refit leave-one-season-out (fit on
seven seasons, predict the eighth), so every number is out of sample. The
baseline is the live structure refit the same way.

Writes data/lab/rating_lab.json for the Data Breakdown page.

Local-only (not run in CI): python scripts/rating_lab.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import betting_scan as bs
from shared import load_cache

SEASONS = list(range(2019, 2027))
EARLY = 6
BOOT = 2000
rng = np.random.default_rng(5)


def team_games():
    ts = load_cache("team_stats").filter(pl.col("season") < bs.UPCOMING_SEASON)
    live = bs.load_live_team_stats(bs.UPCOMING_SEASON, allow_empty=True)
    if live is not None and not live.is_empty():
        ts = pl.concat([ts, live], how="diagonal_relaxed")
    ts = ts.filter(pl.col("season_type") == "REG").with_columns(
        pl.col("attempts").fill_null(0), pl.col("carries").fill_null(0),
        pl.col("passing_epa").fill_null(0.0), pl.col("rushing_epa").fill_null(0.0),
    )
    sched = load_cache("schedules").filter(pl.col("game_type") == "REG")
    pts = pl.concat([
        sched.select("game_id", pl.col("home_team").alias("team"), pl.col("result").alias("pd")),
        sched.select("game_id", pl.col("away_team").alias("team"), (-pl.col("result")).alias("pd")),
    ])
    return ts.join(pts, on=["game_id", "team"], how="left").select(
        "season", "week", "team", "opponent_team", "attempts", "carries", "passing_epa", "rushing_epa", "pd")


def ratings(tg, season, week, decay=1.0, opponent=False, rush_wt=1.0):
    """Per-team centered off/def EPA/play this season (weeks < week, decayed)
    and last season, plus plays and point differential per game."""
    def period(rows, weights):
        teams = sorted(set(rows["team"].to_list()) | set(rows["opponent_team"].to_list()))
        if not teams:
            return {}, {}
        ix = {t: i for i, t in enumerate(teams)}
        off_t = np.array([ix[t] for t in rows["team"].to_list()])
        def_t = np.array([ix[t] for t in rows["opponent_team"].to_list()])
        plays = (rows["attempts"] + rows["carries"]).to_numpy().astype(float) * weights
        epa = (rows["passing_epa"] + rush_wt * rows["rushing_epa"]).to_numpy() * weights
        n = len(teams)
        mean = epa.sum() / plays.sum() if plays.sum() else 0.0
        off_p, def_p = np.bincount(off_t, plays, n), np.bincount(def_t, plays, n)
        off = np.divide(np.bincount(off_t, epa, n), off_p, out=np.zeros(n), where=off_p > 0) - mean
        dfn = np.divide(np.bincount(def_t, epa, n), def_p, out=np.zeros(n), where=def_p > 0) - mean
        if opponent:
            raw = epa - mean * plays  # centered EPA per row
            for _ in range(25):
                off_new = np.divide(np.bincount(off_t, raw - dfn[def_t] * plays, n), off_p, out=np.zeros(n), where=off_p > 0)
                dfn = np.divide(np.bincount(def_t, raw - off[off_t] * plays, n), def_p, out=np.zeros(n), where=def_p > 0)
                off = off_new
        pd_w = rows["pd"].fill_null(0).to_numpy() * weights
        games_w = np.bincount(off_t, weights, n)
        pdpg = np.divide(np.bincount(off_t, pd_w, n), games_w, out=np.zeros(n), where=games_w > 0)
        return ({t: (off[ix[t]], dfn[ix[t]], off_p[ix[t]], def_p[ix[t]], pdpg[ix[t]], games_w[ix[t]]) for t in teams}, ix)

    cur = tg.filter((pl.col("season") == season) & (pl.col("week") < week))
    w_cur = decay ** (week - 1 - cur["week"].to_numpy()) if cur.height else np.array([])
    pri = tg.filter(pl.col("season") == season - 1)
    cur_r, _ = period(cur, w_cur)
    pri_r, _ = period(pri, np.ones(pri.height))
    return cur_r, pri_r


def features(tg, decay, opponent, rush_wt, qb_carry=0.0, qb_swap=0.0):
    """Per game: the blended off/def gap (the live blend shape) and the
    blended point-differential gap, plus outcome columns."""
    sched = (load_cache("schedules").filter(pl.col("season").is_in(SEASONS) & (pl.col("game_type") == "REG"))
             .drop_nulls(["result", "spread_line"]).sort(["season", "week", "game_id"]))
    starter, primary = bs.qb_starts(load_cache("schedules"))
    q = bs.qb_quality(load_cache("player_stats_weekly"))
    rows, ids = [], []
    # Same game order for every variant, so per-game errors can be compared pairwise.
    for (season, week), g in sched.group_by(["season", "week"], maintain_order=True):
        cur, pri = ratings(tg, season, week, decay, opponent, rush_wt)

        def blended(t, game_id):
            c = cur.get(t, (0, 0, 0, 0, 0, 0))
            p = pri.get(t, (0, 0, 0, 0, 0, 0))
            # A new starting QB since last season: last year's offense says less.
            qb, last = starter.get((game_id, t)), primary.get((season - 1, t))
            changed = bool(qb and last and qb != last)
            rho_off = bs.RHO_OFF * (1 - qb_carry) if changed else bs.RHO_OFF
            # QB swap: shift last year's offense by the gap between the two QBs' own records.
            swap = qb_swap * (q(qb, season) - q(last, season)) if changed else 0.0
            w_off = c[2] / (c[2] + bs.K_PLAYS)
            w_def = c[3] / (c[3] + bs.K_PLAYS)
            w_pd = c[5] / (c[5] + bs.K_GAMES)
            return (w_off * c[0] + (1 - w_off) * (rho_off * p[0] + swap),
                    w_def * c[1] + (1 - w_def) * bs.RHO_DEF * p[1],
                    w_pd * c[4] + (1 - w_pd) * bs.RHO_PD * p[4])

        for r in g.iter_rows(named=True):
            h, a = blended(r["home_team"], r["game_id"]), blended(r["away_team"], r["game_id"])
            ids.append(r["game_id"])
            rows.append(((h[0] - a[0]) + (a[1] - h[1]), h[2] - a[2], float(r["location"] != "Neutral"),
                         r["result"], r["spread_line"], season, week))
    arr = np.array(rows, dtype=float)
    out = {k: arr[:, i] for i, k in enumerate(["gap", "pd_gap", "home", "result", "spread", "season", "week"])}
    out["game_id"] = np.array(ids)
    return out


def loso(F, use_pd, extra=()):
    pred = np.zeros_like(F["result"])
    for s in SEASONS:
        tr, te = F["season"] != s, F["season"] == s
        cols = lambda m: np.column_stack([F["gap"][m], F["home"][m]] + ([F["pd_gap"][m]] if use_pd else []) + [F[k][m] for k in extra])
        coef = np.linalg.lstsq(cols(tr), F["result"][tr], rcond=None)[0]
        pred[te] = cols(te) @ coef
    return pred


def summary(pred, F, base_se=None):
    y, sp, wk = F["result"], F["spread"], F["week"]
    se = (pred - y) ** 2
    m = np.round(pred, 1)
    g = (m != sp) & (y != sp)
    out = {
        "mse": se.mean(), "mse_early": se[wk <= EARLY].mean(),
        "corr": np.corrcoef(pred, y)[0, 1],
        "winners": np.mean((pred[(y != 0) & (pred != 0)] > 0) == (y[(y != 0) & (pred != 0)] > 0)),
        "ats": np.mean((m[g] > sp[g]) == (y[g] > sp[g])),
        "corr_2026": np.corrcoef(pred[F["season"] == 2026], y[F["season"] == 2026])[0, 1],
    }
    if base_se is not None:
        d = se - base_se
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
        out["diff"], out["lo"], out["hi"] = d.mean(), *np.percentile(boots, [2.5, 97.5])
    return out


def main():
    tg = team_games()
    import factor_lab as fl
    s_long = fl.long_schedules()
    factors = fl.game_features(s_long, fl.team_games(s_long), fl.conferences())
    fac = {k: dict(zip(factors["game_id"].to_list(), factors[k].to_list())) for k in ("backup_qb", "qb_downgrade", "rest_diff", "lost_cause")}
    L = (1.0, False, 1.0)
    variants = [("live structure", L, 0.0, False, ())]
    variants += [(f"recency {d}", (d, False, 1.0), 0.0, False, ()) for d in (0.95, 0.9, 0.85, 0.8)]
    variants += [("opponent-adjusted", (1.0, True, 1.0), 0.0, False, ())]
    variants += [(f"rushing EPA x{w}", (1.0, False, w), 0.0, False, ()) for w in (0.75, 0.5, 0.25)]
    variants += [("+ point differential", L, 0.0, True, ())]
    variants += [(f"new-QB carryover x{1 - q:g}", L, q, False, ()) for q in (0.25, 0.5, 0.75, 1.0)]
    variants += [("+ backup-QB flag", L, 0.0, False, ("backup_qb",))]
    variants += [(f"QB-swap x{b:g}", L, ("swap", b), False, ()) for b in (0.25, 0.5, 0.75, 1.0)]
    variants += [("v2: differential + QB swap", L, ("swap", bs.QB_SWAP_SHARE), True, ())]
    variants += [("v2 + backup-QB flag", L, ("swap", bs.QB_SWAP_SHARE), True, ("backup_qb",))]
    variants += [("v2 + QB downgrade flag", L, ("swap", bs.QB_SWAP_SHARE), True, ("qb_downgrade",))]
    variants += [("v2 + QB downgrade, rest, lost cause", L, ("swap", bs.QB_SWAP_SHARE), True, ("qb_downgrade", "rest_diff", "lost_cause"))]
    cache = {}
    results = []
    base = None
    for name, key, q, p, extra in variants:
        ck = key + (q,)
        if ck not in cache:
            print(f"  building {name}...")
            F = features(tg, *key, qb_carry=0.0 if isinstance(q, tuple) else q, qb_swap=q[1] if isinstance(q, tuple) else 0.0)
            for k in fac:
                F[k] = np.array([fac[k].get(g, 0.0) for g in F["game_id"]])
            cache[ck] = F
        F = cache[ck]
        pred = loso(F, p, extra)
        if base is None:
            base = (pred - F["result"]) ** 2
            results.append((name, summary(pred, F)))
        else:
            results.append((name, summary(pred, F, base)))
    print(f"\n{len(base)} games 2019-2026 ({int((cache[L + (0.0,)]['week'] <= EARLY).sum())} in weeks 1-{EARLY}), all out of sample")
    print(f"{"variant":<34} {'MSE':>7} {'vs live [95% CI]':>24} {'MSE wk1-6':>10} {'corr':>6} {'2026 corr':>9} {'winners':>8} {'ATS':>6}")
    for name, r in results:
        d = f"{r['diff']:+5.2f} [{r['lo']:+5.2f},{r['hi']:+5.2f}]" if "diff" in r else ""
        print(f"{name:<34} {r['mse']:7.2f} {d:>24} {r['mse_early']:10.2f} {r['corr']:6.3f} {r['corr_2026']:9.3f} {r['winners']:8.3f} {r['ats']:6.3f}")
    out_dir = os.path.join(bs.BASE_DIR, "data", "lab")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rating_lab.json"), "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games": len(base), "seasons": [SEASONS[0], SEASONS[-1]],
            "variants": [{"name": n, **{k: round(float(v), 3) for k, v in r.items()}} for n, r in results],
        }, f, indent=1)
    print("Wrote data/lab/rating_lab.json")


if __name__ == "__main__":
    main()

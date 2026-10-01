"""Does offensive-line continuity add anything to the Game Picks model?

Continuity per team-season (2021-2025) is shared.build_oline_continuity's
definition, measured at Week 1: the share of the Week 1 starting five who
were starters for the same team the season before (finished that season in
the starting five, or were listed as a starter for at least
REGULAR_STARTER_SHARE of its regular-season weeks).

Two questions:
1. Mechanism (team-season level): does continuity change how much of last
   season's offensive efficiency carries into this one -- or lift this
   season's offense on its own?
2. Model (game level): does adding it improve the walk-forward margin
   predictions? Three ways in, each scored leave-one-season-out over
   2021-2025 (fit on four seasons, predict the fifth) against the same
   baseline refit the same way:
     flat   -- points per unit of continuity gap, all season
     fading -- the same, scaled by how much of the ratings is still last
               season (gone once current-season data takes over)
     carry  -- continuity sets how much of last season's offense carries
               over: RHO_OFF + lambda * (continuity - average)

Local-only (not run in CI): python scripts/backtest_line_continuity.py
Needs data_cache/ populated by ingest_historical.py; downloads nflverse's
2020-2024 weekly depth charts once into data_cache/.
"""

import os

import numpy as np
import polars as pl

import betting_scan as bs
from shared import CACHE_DIR, OFFENSIVE_LINE_POS, REGULAR_STARTER_SHARE, load_cache

SEASONS = [2021, 2022, 2023, 2024, 2025]
HISTORY_PATH = os.path.join(CACHE_DIR, "depth_charts_2020_2024.parquet")
KICKOFF_2025 = "2025-09-04"  # 2025 charts are dated snapshots; Week 1 = the last one before kickoff
EARLY_WEEKS = 6
BOOT = 2000
rng = np.random.default_rng(7)


def weekly_starters():
    """Old-format (through 2024) weekly charts: one row per OL starter per team-week."""
    if not os.path.exists(HISTORY_PATH):
        import nflreadpy as nfl
        nfl.load_depth_charts(seasons=[2020, 2021, 2022, 2023, 2024]).write_parquet(HISTORY_PATH)
    return (
        pl.read_parquet(HISTORY_PATH)
        .filter((pl.col("game_type") == "REG") & (pl.col("depth_team") == "1")
                & pl.col("depth_position").is_in(OFFENSIVE_LINE_POS) & pl.col("gsis_id").is_not_null())
        .select(pl.col("season"), pl.col("week"), pl.col("club_code").alias("team"), pl.col("gsis_id"))
        .unique()
    )


def continuity_table():
    """(season, team) -> Week 1 O-line continuity vs. the season before."""
    old = weekly_starters()
    snaps = load_cache("depth_charts").filter(
        pl.col("pos_abb").is_in(OFFENSIVE_LINE_POS) & (pl.col("pos_rank") == 1) & (pl.col("dt") < KICKOFF_2025)
        & (pl.col("dt") >= "2025-08-01")
    )
    week1_2025 = snaps.join(snaps.group_by("team").agg(pl.col("dt").max()), on=["team", "dt"])
    out = {}
    for season in SEASONS:
        prior = old.filter(pl.col("season") == season - 1)
        weeks = prior.group_by("team").agg(pl.col("week").n_unique().alias("n"))
        regulars = (prior.group_by(["team", "gsis_id"]).agg(pl.col("week").n_unique().alias("listed"))
                    .join(weeks, on="team").filter(pl.col("listed") / pl.col("n") >= REGULAR_STARTER_SHARE))
        final = prior.join(prior.group_by("team").agg(pl.col("week").max()), on=["team", "week"])
        current = week1_2025 if season == 2025 else old.filter((pl.col("season") == season) & (pl.col("week") == 1))
        for team in current["team"].unique().to_list():
            cur = set(current.filter(pl.col("team") == team)["gsis_id"].drop_nulls().to_list())
            pool = set(regulars.filter(pl.col("team") == team)["gsis_id"].to_list()) | set(final.filter(pl.col("team") == team)["gsis_id"].to_list())
            if cur and pool:
                out[(season, team)] = len(cur & pool) / len(cur)
    return out


def boot_ci(stat, n):
    vals = [stat(rng.integers(0, n, n)) for _ in range(BOOT)]
    return np.percentile(vals, [2.5, 97.5])


def mechanism(eff, cont):
    """Team-season regression: this season's offense on last season's, continuity, and their interaction."""
    print("\n=== 1. Mechanism: does continuity change how last season's offense carries over? ===")
    for label, max_week in ((f"weeks 1-{EARLY_WEEKS}", EARLY_WEEKS), ("full season", 99)):
        rows = []
        for season in SEASONS:
            prior_off, _, prior_mean = bs._side_totals(eff.filter(pl.col("season") == season - 1))
            cur_off, _, cur_mean = bs._side_totals(eff.filter((pl.col("season") == season) & (pl.col("week") <= max_week)))
            for team, (e, p) in cur_off.items():
                if (season, team) in cont and team in prior_off and p:
                    pe, pp = prior_off[team]
                    rows.append((e / p - cur_mean, pe / pp - prior_mean, cont[(season, team)]))
        y, prior, c = (np.array(v) for v in zip(*rows))
        cc = c - c.mean()
        X = np.column_stack([np.ones_like(y), prior, cc, prior * cc])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        ci = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in (rng.integers(0, len(y), len(y)) for _ in range(BOOT))])
        lo, hi = np.percentile(ci, [2.5, 97.5], axis=0)
        print(f"\n{label}: {len(y)} team-seasons, continuity mean {c.mean():.2f} (sd {c.std():.2f})")
        print(f"  carryover of last season's offense   {coef[1]:+.3f}   95% CI [{lo[1]:+.3f}, {hi[1]:+.3f}]")
        print(f"  continuity, on its own (EPA/play)    {coef[2]:+.3f}   95% CI [{lo[2]:+.3f}, {hi[2]:+.3f}]"
              f"   ~ {bs.MARGIN_SCALE * coef[2] * 0.2:+.2f} pts per 20-point continuity edge")
        print(f"  continuity x carryover interaction   {coef[3]:+.3f}   95% CI [{lo[3]:+.3f}, {hi[3]:+.3f}]")


FEATURES = ["cur_off", "cur_off_plays", "prior_off", "cur_def", "cur_def_plays", "prior_def"]


def game_rows(eff, schedules, cont):
    rows = []
    for season in SEASONS:
        games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")).drop_nulls(["result", "spread_line"])
        for week in sorted(games["week"].unique().to_list()):
            inputs = bs.rating_inputs_asof(eff, season, week)
            for r in games.filter(pl.col("week") == week).iter_rows(named=True):
                h, a = inputs.get(r["home_team"]), inputs.get(r["away_team"])
                ch, ca = cont.get((season, r["home_team"])), cont.get((season, r["away_team"]))
                if h is None or a is None or ch is None or ca is None:
                    continue
                rows.append({**{f"h_{f}": h[f] for f in FEATURES}, **{f"a_{f}": a[f] for f in FEATURES},
                             "h_cont": ch, "a_cont": ca, "neutral": float(r["location"] == "Neutral"),
                             "result": r["result"], "spread": r["spread_line"], "season": season, "week": week})
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def raw_gap(G, idx, rho_off_h, rho_off_a):
    def side(p, part, rho):
        plays = G[f"{p}_cur_{part}_plays"][idx]
        w = plays / (plays + bs.K_PLAYS)
        return w * G[f"{p}_cur_{part}"][idx] + (1 - w) * rho * G[f"{p}_prior_{part}"][idx]
    return (side("h", "off", rho_off_h) - side("a", "off", rho_off_a)) + (side("a", "def", bs.RHO_DEF) - side("h", "def", bs.RHO_DEF))


def prior_share(G, idx):
    """How much of the two offenses' ratings is still last season (1 at Week 1)."""
    w = [G[f"{p}_cur_off_plays"][idx] / (G[f"{p}_cur_off_plays"][idx] + bs.K_PLAYS) for p in ("h", "a")]
    return 1 - (w[0] + w[1]) / 2


def design(G, idx, variant, lam=0.0, cbar=0.0):
    hfa = 1 - G["neutral"][idx]
    if variant == "carry":
        rho_h = np.clip(bs.RHO_OFF + lam * (G["h_cont"][idx] - cbar), 0, 1)
        rho_a = np.clip(bs.RHO_OFF + lam * (G["a_cont"][idx] - cbar), 0, 1)
        return np.column_stack([raw_gap(G, idx, rho_h, rho_a), hfa])
    cols = [raw_gap(G, idx, bs.RHO_OFF, bs.RHO_OFF), hfa]
    gap = G["h_cont"][idx] - G["a_cont"][idx]
    if variant == "flat":
        cols.append(gap)
    elif variant == "fading":
        cols.append(gap * prior_share(G, idx))
    return np.column_stack(cols)


def fit_predict(G, train, test, variant):
    """Least-squares fit on train (carry: also grid lambda on train), predict test."""
    cbar = G["h_cont"][train].mean()
    lams = np.round(np.arange(-1.0, 2.01, 0.1), 1) if variant == "carry" else [0.0]
    best = None
    for lam in lams:
        X = design(G, train, variant, lam, cbar)
        coef = np.linalg.lstsq(X, G["result"][train], rcond=None)[0]
        mse = np.mean((X @ coef - G["result"][train]) ** 2)
        if best is None or mse < best[0]:
            best = (mse, lam, coef)
    _, lam, coef = best
    return design(G, test, variant, lam, cbar) @ coef, lam, coef


def model_test(G):
    print("\n=== 2. Model: does it improve game predictions? (leave-one-season-out, 2021-2025) ===")
    n = len(G["result"])
    seasons = G["season"]
    preds = {v: np.zeros(n) for v in ("base", "flat", "fading", "carry")}
    fitted = {v: [] for v in preds}
    for s in SEASONS:
        train, test = np.where(seasons != s)[0], np.where(seasons == s)[0]
        for v in preds:
            p, lam, coef = fit_predict(G, train, test, v)
            preds[v][test] = p
            fitted[v].append((s, lam, coef))
    actual, spread, week = G["result"], G["spread"], G["week"]
    early = week <= EARLY_WEEKS
    print(f"{n} games ({early.sum()} in weeks 1-{EARLY_WEEKS}); continuity gap between opponents sd {np.std(G['h_cont'] - G['a_cont']):.2f}")
    print("\nFitted effect by held-out season (fit on the other four):")
    for s, lam, coef in fitted["flat"]:
        print(f"  {s}  flat: {coef[2]:+5.2f} pts per 1.0 of continuity gap", end="")
        f = [c for t, _, c in fitted["fading"] if t == s][0]
        l = [la for t, la, _ in fitted["carry"] if t == s][0]
        print(f" | fading: {f[2]:+5.2f} at Week 1 | carry lambda: {l:+.1f}")

    def ats(p, mask):
        m = np.round(p, 1)
        g = mask & (m != spread) & (actual != spread)
        return np.mean((m[g] > spread[g]) == (actual[g] > spread[g])), g.sum()

    print("\nOut-of-sample error vs. baseline (negative = continuity helps):")
    print(f"  {'variant':<8} {'MSE all':>9} {'diff [95% CI]':>24} {'MSE wk1-6':>11} {'diff [95% CI]':>24} {'ATS all':>8} {'ATS wk1-6':>10}")
    base_se = (preds["base"] - actual) ** 2
    for v in preds:
        se = (preds[v] - actual) ** 2
        d = se - base_se
        row = [f"  {v:<8} {se.mean():9.2f}"]
        for mask in (np.ones(n, bool), early):
            idx = np.where(mask)[0]
            if v == "base":
                row.append(f"{'':>24}" if mask.all() else f" {se[mask].mean():10.2f} {'':>24}")
                continue
            lo, hi = boot_ci(lambda i: d[idx][i].mean(), len(idx))
            cell = f"{d[mask].mean():+6.2f} [{lo:+.2f}, {hi:+.2f}]"
            row.append(f" {cell:>23}" if mask.all() else f" {se[mask].mean():10.2f} {cell:>24}")
        a_all, _ = ats(preds[v], np.ones(n, bool))
        a_early, ne = ats(preds[v], early)
        row.append(f" {a_all:8.3f} {a_early:9.3f}")
        print("".join(row))
    print(f"  (ATS graded on {ats(preds['base'], np.ones(n, bool))[1]} games, {ats(preds['base'], early)[1]} in weeks 1-{EARLY_WEEKS}; break-even 0.524)")
    side = lambda p: np.sign(np.round(p, 1) - spread)
    for v in ("flat", "fading", "carry"):
        flips = side(preds[v]) != side(preds["base"])
        print(f"  {v:<7} picks a different side of the spread than the baseline in {flips.sum()} of {n} games "
              f"({flips[early].sum()} of {early.sum()} in weeks 1-{EARLY_WEEKS}); "
              f"largest margin change {np.max(np.abs(preds[v] - preds['base'])):.1f} pts")


def main():
    schedules = load_cache("schedules")
    eff = bs.team_game_efficiency(load_cache("team_stats"))
    print("Computing Week 1 O-line continuity, 2021-2025...")
    cont = continuity_table()
    for s in SEASONS:
        vals = [v for (season, _), v in cont.items() if season == s]
        print(f"  {s}: {len(vals)} teams, mean {np.mean(vals):.2f}, min {min(vals):.1f}, max {max(vals):.1f}")
    mechanism(eff, cont)
    print("\nBuilding walk-forward game features...")
    model_test(game_rows(eff, schedules, cont))


if __name__ == "__main__":
    main()

"""Fit the Game Picks rating constants, then report how they hold up on data
the fit never saw.

Walk-forward over 2021-2024 regular-season games: each game is predicted
with only the games before it (plus the prior season), using the exact same
rating_inputs_asof/blend code betting_scan.py runs live. Grid over the blend
shape (RHO_OFF, RHO_DEF, K_PLAYS); for each point, MARGIN_SCALE and
HOME_FIELD_ADJ are solved exactly by least squares on final margins (not on
ATS wins -- margins are a far more stable target than a coin-flip outcome).
2025 is held out and reported separately with the constants currently in
betting_scan.py.

Local-only (not run in CI): python scripts/calibrate_game_model.py
Needs data_cache/ populated by ingest_historical.py.
"""

import numpy as np
import polars as pl

import betting_scan as bs
from shared import load_cache

FIT_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUT_SEASON = 2025
FEATURES = ["cur_off", "cur_off_plays", "prior_off", "cur_def", "cur_def_plays", "prior_def"]


def game_features(eff, schedules, seasons):
    rows = []
    for season in seasons:
        games = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")).drop_nulls(["result", "spread_line"])
        for week in sorted(games["week"].unique().to_list()):
            inputs = bs.rating_inputs_asof(eff, season, week)
            for r in games.filter(pl.col("week") == week).iter_rows(named=True):
                h, a = inputs.get(r["home_team"]), inputs.get(r["away_team"])
                if h is None or a is None:
                    continue
                rows.append([h[f] for f in FEATURES] + [a[f] for f in FEATURES]
                            + [r["location"] == "Neutral", r["result"], r["spread_line"], season, week])
    arr = np.array(rows, dtype=float)
    cols = {f"h_{f}": i for i, f in enumerate(FEATURES)}
    cols.update({f"a_{f}": i + len(FEATURES) for i, f in enumerate(FEATURES)})
    n = 2 * len(FEATURES)
    cols.update({"neutral": n, "result": n + 1, "spread": n + 2, "season": n + 3, "week": n + 4})
    return arr, cols


def raw_gap(X, c, rho_off, rho_def, k):
    def side(prefix, part, rho):
        plays = X[:, c[f"{prefix}_cur_{part}_plays"]]
        w = plays / (plays + k)
        return w * X[:, c[f"{prefix}_cur_{part}"]] + (1 - w) * rho * X[:, c[f"{prefix}_prior_{part}"]]
    return (side("h", "off", rho_off) - side("a", "off", rho_off)) + (side("a", "def", rho_def) - side("h", "def", rho_def))


def solve_scale_hfa(raw, X, c):
    A = np.column_stack([raw, 1 - X[:, c["neutral"]]])
    (scale, hfa), *_ = np.linalg.lstsq(A, X[:, c["result"]], rcond=None)
    return scale, hfa


def report(label, margin, X, c):
    actual, spread = X[:, c["result"]], X[:, c["spread"]]
    margin = np.round(margin, 1)
    nz = (actual != 0) & (margin != 0)
    push = actual == spread
    lean = margin != spread
    edge = margin - spread
    graded = lean & ~push
    ats = ((edge > 0) == (actual > spread))[graded]
    fav = np.where(spread[graded] != 0, (edge[graded] > 0) == (spread[graded] > 0), False)
    coef, *_ = np.linalg.lstsq(np.column_stack([spread, margin, np.ones(len(actual))]), actual, rcond=None)
    print(f"\n--- {label} ({len(actual)} games) ---")
    print(f"MAE      model {np.mean(np.abs(margin - actual)):5.2f} | spread {np.mean(np.abs(spread - actual)):5.2f}")
    print(f"corr     model {np.corrcoef(margin, actual)[0, 1]:5.3f} | spread {np.corrcoef(spread, actual)[0, 1]:5.3f}")
    print(f"result ~ {coef[0]:.2f}*spread + {coef[1]:.2f}*model + {coef[2]:.2f}   (model coef ~0 = adds nothing the spread lacks)")
    mfav = (spread != 0) & (actual != 0)
    print(f"outright winners  model {np.mean((margin[nz] > 0) == (actual[nz] > 0)):.3f} | market favorite {np.mean((spread[mfav] > 0) == (actual[mfav] > 0)):.3f}")
    print(f"ATS      {ats.sum()}-{len(ats) - ats.sum()} = {ats.mean():.3f}  (no lean {np.sum(~lean)}, pushes {np.sum(push)}; break-even at -110 is 0.524)")
    print(f"picks on the underdog: {np.mean(~fav):.2f} | ATS taking favorite {ats[fav].mean():.3f} ({fav.sum()}) | taking underdog {ats[~fav].mean():.3f} ({(~fav).sum()})")
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 99)):
        b = (np.abs(edge[graded]) >= lo) & (np.abs(edge[graded]) < hi)
        if b.sum():
            print(f"  |gap| {lo:>3}-{hi:<3}: {ats[b].mean():.3f} on {b.sum()} games")
    early = X[:, c["week"]] <= 4
    if early.sum() and (early & graded).sum():
        e = ((edge > 0) == (actual > spread))[early & graded]
        print(f"weeks 1-4 slice: corr {np.corrcoef(margin[early], actual[early])[0, 1]:.3f} | ATS {e.mean():.3f} on {len(e)}")


def main():
    schedules = load_cache("schedules")
    eff = bs.team_game_efficiency(load_cache("team_stats"))

    print(f"Building walk-forward features for {FIT_SEASONS} and {HOLDOUT_SEASON}...")
    Xf, c = game_features(eff, schedules, FIT_SEASONS)
    Xh, _ = game_features(eff, schedules, [HOLDOUT_SEASON])

    results = []
    for rho_off in np.arange(0, 1.01, 0.1):
        for rho_def in np.arange(0, 1.01, 0.1):
            for k in range(80, 1601, 40):
                raw = raw_gap(Xf, c, rho_off, rho_def, k)
                scale, hfa = solve_scale_hfa(raw, Xf, c)
                mse = np.mean((scale * raw + hfa * (1 - Xf[:, c["neutral"]]) - Xf[:, c["result"]]) ** 2)
                results.append((mse, round(rho_off, 1), round(rho_def, 1), k, scale, hfa))
    results.sort()
    print(f"\nTop 10 fits on {FIT_SEASONS[0]}-{FIT_SEASONS[-1]} (MSE, RHO_OFF, RHO_DEF, K_PLAYS, scale, HFA):")
    for mse, ro, rd, k, s, h in results[:10]:
        print(f"  {mse:7.2f}  rho_off {ro:.1f}  rho_def {rd:.1f}  K {k:>5}  scale {s:6.1f}  hfa {h:5.2f}")

    print(f"\nConstants currently in betting_scan.py: MARGIN_SCALE {bs.MARGIN_SCALE}, HOME_FIELD_ADJ {bs.HOME_FIELD_ADJ}, "
          f"RHO_OFF {bs.RHO_OFF}, RHO_DEF {bs.RHO_DEF}, K_PLAYS {bs.K_PLAYS}")
    for label, X in ((f"IN-SAMPLE {FIT_SEASONS[0]}-{FIT_SEASONS[-1]}", Xf), (f"HOLDOUT {HOLDOUT_SEASON} (never seen by the fit)", Xh)):
        raw = raw_gap(X, c, bs.RHO_OFF, bs.RHO_DEF, bs.K_PLAYS)
        report(label, bs.MARGIN_SCALE * raw + bs.HOME_FIELD_ADJ * (1 - X[:, c["neutral"]]), X, c)


if __name__ == "__main__":
    main()

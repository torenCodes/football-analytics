"""Fit the Game Picks rating constants, then report how they hold up on data
the fit never saw.

Walk-forward over 2021-2024 regular-season games (the same seasons as v1; 2020 had
no crowds and home field sat near zero): each game is predicted with
only the games before it (plus the prior season), from the same inputs
betting_scan.py runs live. Grids over the blend shape; for each point the
linear terms -- points per EPA, points per point of differential, the QB-swap
term, home field -- are solved exactly by least squares on final margins (not
on ATS wins: margins are a far more stable target than a coin-flip outcome).

2025 and 2026 are held out and scored with the constants in betting_scan.py,
next to v1 (the pre-Week 10 structure) and the Vegas line. One caveat travels
with those holdouts: the decision to try point differential and QB swaps came
from rating_lab.py, which did look at those seasons. The constants never did.

Writes data/lab/model_compare.json for the Data Breakdown page.

Local-only (not run in CI): python scripts/calibrate_game_model.py
Needs data_cache/ populated by ingest_historical.py.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import betting_scan as bs
from shared import load_cache

FIT_SEASONS = [2021, 2022, 2023, 2024]
HOLDOUTS = {"2025": [2025], "2026": [2026]}
V1 = {"scale": 50.0, "hfa": 2.0, "rho_off": 0.6, "rho_def": 0.2, "k": 480}
FEATURES = ["cur_off", "cur_off_plays", "prior_off", "cur_def", "cur_def_plays", "prior_def", "cur_pd", "cur_games", "prior_pd", "qb_gap"]
BOOT = 2000
LAB_DIR = os.path.join(bs.BASE_DIR, "data", "lab")
rng = np.random.default_rng(23)


def load_inputs():
    schedules = load_cache("schedules")
    team_stats = load_cache("team_stats").filter(pl.col("season") < bs.UPCOMING_SEASON)
    live = bs.load_live_team_stats(bs.UPCOMING_SEASON, allow_empty=True)
    if live is not None and not live.is_empty():
        team_stats = pl.concat([team_stats, live], how="diagonal_relaxed")
    return schedules, bs.team_game_efficiency(team_stats, schedules)


def game_features(eff, schedules, seasons):
    """One row per played game: each side's inputs plus the outcome columns."""
    starter, primary = bs.qb_starts(schedules)
    q = bs.qb_quality(load_cache("player_stats_weekly"))
    rows, ids = [], []
    for season in seasons:
        games = (schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))
                 .drop_nulls(["result", "spread_line"]).sort(["week", "game_id"]))
        for week in sorted(games["week"].unique().to_list()):
            inputs = bs.rating_inputs_asof(eff, season, week)
            week_games = games.filter(pl.col("week") == week)
            starters = {t: starter.get((r["game_id"], t)) for r in week_games.iter_rows(named=True) for t in (r["home_team"], r["away_team"])}
            gaps = bs.qb_gaps_for_week(starters, season, primary, q)
            for r in week_games.iter_rows(named=True):
                h, a = inputs.get(r["home_team"]), inputs.get(r["away_team"])
                if h is None or a is None:
                    continue
                side = lambda x, team: [x[f] for f in FEATURES[:-1]] + [gaps.get(team, 0.0)]
                rows.append(side(h, r["home_team"]) + side(a, r["away_team"])
                            + [r["location"] == "Neutral", r["result"], r["spread_line"], season, week])
                ids.append(r["game_id"])
    arr = np.array(rows, dtype=float)
    cols = {f"h_{f}": i for i, f in enumerate(FEATURES)}
    cols.update({f"a_{f}": i + len(FEATURES) for i, f in enumerate(FEATURES)})
    n = 2 * len(FEATURES)
    cols.update({"neutral": n, "result": n + 1, "spread": n + 2, "season": n + 3, "week": n + 4})
    return arr, cols, ids


def design(X, c, rho_off, rho_def, k, rho_pd, k_games):
    """The model's linear terms for one blend shape: EPA gap, differential gap,
    QB-swap gap, home field."""
    def share(prefix, part):
        plays = X[:, c[f"{prefix}_cur_{part}_plays"]]
        return plays / (plays + k)

    def rating(prefix, part, rho):
        w = share(prefix, part)
        return w * X[:, c[f"{prefix}_cur_{part}"]] + (1 - w) * rho * X[:, c[f"{prefix}_prior_{part}"]]

    def point_diff(prefix):
        g = X[:, c[f"{prefix}_cur_games"]]
        w = g / (g + k_games)
        return w * X[:, c[f"{prefix}_cur_pd"]] + (1 - w) * rho_pd * X[:, c[f"{prefix}_prior_pd"]]

    epa = (rating("h", "off", rho_off) - rating("a", "off", rho_off)) + (rating("a", "def", rho_def) - rating("h", "def", rho_def))
    pd_gap = point_diff("h") - point_diff("a")
    swap = (1 - share("h", "off")) * X[:, c["h_qb_gap"]] - (1 - share("a", "off")) * X[:, c["a_qb_gap"]]
    return epa, pd_gap, swap, 1 - X[:, c["neutral"]]


def predict_v1(X, c):
    epa, _, _, home = design(X, c, V1["rho_off"], V1["rho_def"], V1["k"], 0.0, 1.0)
    return V1["scale"] * epa + V1["hfa"] * home


def predict_v2(X, c):
    epa, pd_gap, swap, home = design(X, c, bs.RHO_OFF, bs.RHO_DEF, bs.K_PLAYS, bs.RHO_PD, bs.K_GAMES)
    return bs.MARGIN_SCALE * epa + bs.PD_WEIGHT * pd_gap + bs.MARGIN_SCALE * bs.QB_SWAP_SHARE * swap + bs.HOME_FIELD_ADJ * home


def scorecard(margin, X, c):
    actual, spread = X[:, c["result"]], X[:, c["spread"]]
    margin = np.round(margin, 1)
    graded = (margin != spread) & (actual != spread)
    wins = ((margin > spread) == (actual > spread))[graded]
    decided = (actual != 0) & (margin != 0)
    return {
        "games": len(actual),
        "corr": round(float(np.corrcoef(margin, actual)[0, 1]), 3),
        "mae": round(float(np.mean(np.abs(margin - actual))), 2),
        "winners": round(float(np.mean((margin[decided] > 0) == (actual[decided] > 0))), 3),
        "ats_wins": int(wins.sum()),
        "ats_losses": int(len(wins) - wins.sum()),
        "ats": round(float(wins.mean()), 3),
    }


def market_card(X, c):
    actual, spread = X[:, c["result"]], X[:, c["spread"]]
    decided = (spread != 0) & (actual != 0)
    return {
        "games": len(actual),
        "corr": round(float(np.corrcoef(spread, actual)[0, 1]), 3),
        "mae": round(float(np.mean(np.abs(spread - actual))), 2),
        "winners": round(float(np.mean((spread[decided] > 0) == (actual[decided] > 0))), 3),
    }


def paired_mse(a, b, y):
    d = (a - y) ** 2 - (b - y) ** 2
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 2), "lo": round(float(lo), 2), "hi": round(float(hi), 2)}


def calibration(margin, spread, y):
    """Average actual margin per bin of predicted margin -- away and home sides
    folded together by symmetry, so a bin reads "picked to win by about N"."""
    out = []
    edges = [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 6.5), (6.5, 9), (9, 40)]
    for name, pred in (("model", margin), ("vegas", spread)):
        sign = np.sign(pred)
        p, a = np.abs(pred), y * sign
        rows = []
        for lo, hi in edges:
            m = (p >= lo) & (p < hi) & (sign != 0)
            if m.sum() >= 15:
                se = a[m].std(ddof=1) / np.sqrt(m.sum())
                rows.append({"pred": round(float(p[m].mean()), 2), "actual": round(float(a[m].mean()), 2),
                             "lo": round(float(a[m].mean() - 1.96 * se), 2), "hi": round(float(a[m].mean() + 1.96 * se), 2), "n": int(m.sum())})
        out.append({"series": name, "bins": rows})
    return out


def versus_vegas(pred, X, c):
    """Does the model add anything Vegas doesn't already have?"""
    y, s, season = X[:, c["result"]], X[:, c["spread"]], X[:, c["season"]]
    m = np.round(pred, 1)

    def weight(mask):
        A = np.column_stack([s[mask], pred[mask], np.ones(mask.sum())])
        b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        draws = [np.linalg.lstsq(A[i], y[mask][i], rcond=None)[0] for i in (rng.integers(0, mask.sum(), mask.sum()) for _ in range(BOOT))]
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        return {"vegas": round(float(b[0]), 2), "vegas_lo": round(float(lo[0]), 2), "vegas_hi": round(float(hi[0]), 2), "model": round(float(b[1]), 2), "model_lo": round(float(lo[1]), 2), "model_hi": round(float(hi[1]), 2)}

    def rate(w):
        p = float(w.mean())
        se = float(np.sqrt(p * (1 - p) / len(w)))
        return {"rate": round(p, 3), "n": int(len(w)), "lo": round(p - 1.96 * se, 3), "hi": round(p + 1.96 * se, 3)}

    gaps = []
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 99)):
        g = (np.abs(m - s) >= lo) & (np.abs(m - s) < hi) & (m != s) & (y != s)
        gaps.append({"gap_from": lo, "gap_to": hi, **rate((m[g] > s[g]) == (y[g] > s[g]))})
    unseen = season >= 2025
    blend = [{"model_share": w, "squared_error": round(float(np.mean(((1 - w) * s[unseen] + w * pred[unseen] - y[unseen]) ** 2)), 2), "avg_miss": round(float(np.mean(np.abs((1 - w) * s[unseen] + w * pred[unseen] - y[unseen]))), 3)} for w in (0, 0.1, 0.2, 0.3, 0.5, 1.0)]
    differ = (np.sign(m) != np.sign(s)) & (m != 0) & (s != 0) & (y != 0)
    return {"weight_all": weight(season >= 2021), "weight_unseen": weight(unseen), "gaps": gaps, "blend": blend, "different_winner": rate(np.sign(m[differ]) == np.sign(y[differ]))}


def main():
    schedules, eff = load_inputs()
    print(f"Building walk-forward features for {FIT_SEASONS} and the holdouts...")
    Xf, c, _ = game_features(eff, schedules, FIT_SEASONS)
    y = Xf[:, c["result"]]

    print("\nBest blend shape for the two new terms (EPA blend held at the current constants):")
    results = []
    for rho_pd in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        for k_games in (2, 4, 6, 8, 10, 12):
            A = np.column_stack(design(Xf, c, bs.RHO_OFF, bs.RHO_DEF, bs.K_PLAYS, rho_pd, k_games))
            coef = np.linalg.lstsq(A, y, rcond=None)[0]
            results.append((float(np.mean((A @ coef - y) ** 2)), rho_pd, k_games, *coef))
    results.sort()
    for mse, rho_pd, k_games, scale, pdw, swap, hfa in results[:6]:
        print(f"  {mse:7.2f}  rho_pd {rho_pd:.1f}  k_games {k_games:>2}  scale {scale:5.1f}  pd {pdw:.3f}  swap {swap:5.1f} (share {swap / scale:+.2f})  hfa {hfa:.2f}")

    print("\nRe-checking the EPA blend shape with those terms in the model:")
    best = results[0]
    grid = []
    for rho_off in np.arange(0.2, 1.01, 0.1):
        for rho_def in np.arange(0.0, 0.61, 0.1):
            for k in range(240, 1001, 80):
                A = np.column_stack(design(Xf, c, rho_off, rho_def, k, best[1], best[2]))
                coef = np.linalg.lstsq(A, y, rcond=None)[0]
                grid.append((float(np.mean((A @ coef - y) ** 2)), round(float(rho_off), 1), round(float(rho_def), 1), k, *coef))
    grid.sort()
    for mse, ro, rd, k, scale, pdw, swap, hfa in grid[:6]:
        print(f"  {mse:7.2f}  rho_off {ro:.1f}  rho_def {rd:.1f}  K {k:>4}  scale {scale:5.1f}  pd {pdw:.3f}  swap {swap:5.1f}  hfa {hfa:.2f}")

    print(f"\nIn betting_scan.py: scale {bs.MARGIN_SCALE}, pd {bs.PD_WEIGHT}, swap share {bs.QB_SWAP_SHARE}, hfa {bs.HOME_FIELD_ADJ}, "
          f"rho {bs.RHO_OFF}/{bs.RHO_DEF}, K {bs.K_PLAYS}, rho_pd {bs.RHO_PD}, k_games {bs.K_GAMES}")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "fit_seasons": FIT_SEASONS, "holdouts": {}}
    samples = {"fit": Xf}
    for label, seasons in HOLDOUTS.items():
        samples[label] = game_features(eff, schedules, seasons)[0]
    for label, X in samples.items():
        v1, v2 = predict_v1(X, c), predict_v2(X, c)
        card = {"v1": scorecard(v1, X, c), "v2": scorecard(v2, X, c), "vegas": market_card(X, c),
                "v2_minus_v1_mse": paired_mse(v2, v1, X[:, c["result"]])}
        if X.size:
            card["through_week"] = int(X[:, c["week"]].max())
        report["holdouts"][label] = card
        print(f"\n--- {label} ({len(X)} games) ---")
        for k_, v in card.items():
            print(f"  {k_}: {v}")

    all_seasons = np.vstack([samples["fit"], samples["2025"], samples["2026"]])
    report["calibration"] = calibration(predict_v2(all_seasons, c), all_seasons[:, c["spread"]], all_seasons[:, c["result"]])
    report["calibration_games"] = len(all_seasons)
    report["vs_vegas"] = versus_vegas(predict_v2(all_seasons, c), all_seasons, c)
    report["constants"] = {
        "MARGIN_SCALE": bs.MARGIN_SCALE, "PD_WEIGHT": bs.PD_WEIGHT, "QB_SWAP_SHARE": bs.QB_SWAP_SHARE, "QB_OUT_PENALTY": bs.QB_OUT_PENALTY,
        "HOME_FIELD_ADJ": bs.HOME_FIELD_ADJ, "RHO_OFF": bs.RHO_OFF, "RHO_DEF": bs.RHO_DEF, "RHO_PD": bs.RHO_PD, "K_PLAYS": bs.K_PLAYS, "K_GAMES": bs.K_GAMES,
    }
    os.makedirs(LAB_DIR, exist_ok=True)
    with open(os.path.join(LAB_DIR, "model_compare.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\nWrote data/lab/model_compare.json")


if __name__ == "__main__":
    main()

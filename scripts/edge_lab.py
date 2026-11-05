"""Edge Lab: the wider search for anything the team rating and Vegas miss.

Everything is out of sample: leave-one-season-out for fitted models, and for
simple market rules a discovery era (2006-2018) checked against 2019-2026.

  ratings   -- richer team ratings built from full play-by-play, vs. v2 and Vegas
  misses    -- does a team's past miss predict its next one?
  explained -- how much of a miss is what happened in that game (turnovers etc.)?
  lasts     -- which stats repeat, half to half and year to year?
  totals    -- our own over/under model vs. the line
  rules     -- simple spread and totals rules, and how often luck matches them
  moneyline -- flat-bet return by price
  combined  -- every signal together against the spread

Writes data/lab/edge_lab.json for the Data Breakdown page.
Local-only (not run in CI): python scripts/edge_lab.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import betting_scan as bs
import calibrate_game_model as cal
import factor_lab as fl
from shared import TEAM_CODE_ALIASES
from team_lab import load_tables

rng = np.random.default_rng(77)
canon = lambda t: TEAM_CODE_ALIASES.get(t, t)
COLS = ["plays", "epa", "epa_no_to", "plays_no_to", "epa_neutral", "plays_neutral", "epa_early", "plays_early", "success", "explosive",
        "turnovers", "dropbacks", "sacks", "qb_hits", "pass_epa", "pass_plays", "rush_epa", "rush_plays", "g"]
VARS = {"base": ("epa", "plays"), "no_to": ("epa_no_to", "plays_no_to"), "neutral": ("epa_neutral", "plays_neutral"), "early": ("epa_early", "plays_early"),
        "success": ("success", "plays"), "explosive": ("explosive", "plays"), "pass": ("pass_epa", "pass_plays"), "rush": ("rush_epa", "rush_plays"),
        "to": ("turnovers", "plays"), "sack": ("sacks", "dropbacks")}
LASTS = {"EPA/play": ("epa", "plays"), "EPA excl. turnovers": ("epa_no_to", "plays_no_to"), "Success rate": ("success", "plays"), "Pass EPA/play": ("pass_epa", "pass_plays"),
         "Rush EPA/play": ("rush_epa", "rush_plays"), "Explosive rate": ("explosive", "plays"), "Turnover rate": ("turnovers", "plays"), "Sack rate": ("sacks", "dropbacks"), "QB hit rate": ("qb_hits", "dropbacks")}


def cube(tbl, cols, ix):
    out = {}
    for season in tbl["season"].unique().to_list():
        arr = {c: np.zeros((len(ix), 20)) for c in cols}
        for r in tbl.filter(pl.col("season") == season).iter_rows(named=True):
            for c in cols:
                arr[c][ix[r["team"]], r["week"]] += float(r[c] or 0.0)
        out[season] = {c: np.cumsum(v, axis=1) for c, v in arr.items()}
    return out


def ratio(n, d):
    m = n.sum() / d.sum() if d.sum() else 0.0
    return np.divide(n, d, out=np.full_like(n, m), where=d > 0) - m


def rate(C, season, week, num, den, rho):
    c, p = C[season], C[season - 1]
    w = c["plays"][:, week - 1] / (c["plays"][:, week - 1] + bs.K_PLAYS)
    return w * ratio(c[num][:, week - 1], c[den][:, week - 1]) + (1 - w) * rho * ratio(p[num][:, -1], p[den][:, -1])


def summary(w):
    p = float(np.mean(w))
    se = float(np.sqrt(p * (1 - p) / len(w)))
    return {"rate": round(p, 3), "n": int(len(w)), "lo": round(p - 1.96 * se, 3), "hi": round(p + 1.96 * se, 3)}


def boot_slope(x, y, boots=1000):
    A = np.column_stack([x, np.ones(len(x))])
    b = np.linalg.lstsq(A, y, rcond=None)[0][0]
    draws = [np.linalg.lstsq(A[i], y[i], rcond=None)[0][0] for i in (rng.integers(0, len(y), len(y)) for _ in range(boots))]
    return {"slope": round(float(b), 3), "lo": round(float(np.percentile(draws, 2.5)), 3), "hi": round(float(np.percentile(draws, 97.5)), 3)}


def main():
    off, dfn = load_tables()
    add_g = lambda t: t.with_columns(pl.lit(1.0).alias("g"))
    off, dfn = add_g(off), add_g(dfn)
    teams = sorted(set(off["team"].to_list()))
    ix = {t: i for i, t in enumerate(teams)}
    CO, CD = cube(off, COLS + ["st"], ix), cube(dfn, COLS, ix)

    sch, eff = cal.load_inputs()
    X, c, ids = cal.game_features(eff, sch, list(range(2019, 2027)))
    Y, S, SEAS, WK = X[:, c["result"]], X[:, c["spread"]], X[:, c["season"]], X[:, c["week"]]
    epa_gap, pd_gap, swap, home = cal.design(X, c, bs.RHO_OFF, bs.RHO_DEF, bs.K_PLAYS, bs.RHO_PD, bs.K_GAMES)
    rows = {r["game_id"]: r for r in sch.iter_rows(named=True)}
    s_long = fl.long_schedules()
    feats = fl.game_features(s_long, fl.team_games(s_long), fl.conferences())
    down_of = dict(zip(feats["game_id"].to_list(), feats["qb_downgrade"].to_list()))
    DOWN = np.array([down_of.get(g, 0.0) for g in ids])

    F = {f"{v}_{k}": np.zeros(len(ids)) for v in VARS for k in ("off", "def")}
    F["st"] = np.zeros(len(ids))
    T = {k: np.zeros(len(ids)) for k in ("sum_off", "sum_def", "pace", "wind", "temp", "indoor", "line", "total")}
    memo = {}
    for i, g in enumerate(ids):
        r = rows[g]
        s, w, h, a = r["season"], r["week"], ix[canon(r["home_team"])], ix[canon(r["away_team"])]
        if (s, w) not in memo:
            memo[(s, w)] = {v: (rate(CO, s, w, n, d, bs.RHO_OFF), rate(CD, s, w, n, d, bs.RHO_DEF)) for v, (n, d) in VARS.items()}
            memo[(s, w)]["st"] = rate(CO, s, w, "st", "g", 0.3)
            memo[(s, w)]["pace"] = rate(CO, s, w, "plays", "g", 0.5)
        k = memo[(s, w)]
        for v in VARS:
            o, d = k[v]
            F[f"{v}_off"][i], F[f"{v}_def"][i] = o[h] - o[a], d[a] - d[h]
        F["st"][i] = k["st"][h] - k["st"][a]
        o, d = k["base"]
        T["sum_off"][i], T["sum_def"][i], T["pace"][i] = o[h] + o[a], d[h] + d[a], k["pace"][h] + k["pace"][a]
        indoor = r["roof"] in ("dome", "closed")
        T["wind"][i] = 0.0 if indoor or r["wind"] is None else r["wind"]
        T["temp"][i] = 70.0 if indoor or r["temp"] is None else r["temp"]
        T["indoor"][i] = float(indoor)
        T["line"][i] = r["total_line"] if r["total_line"] is not None else np.nan
        T["total"][i] = r["total"] if r["total"] is not None else np.nan

    def loso(cols, y=Y):
        A = np.column_stack(cols)
        p = np.zeros(len(y))
        for s in np.unique(SEAS):
            tr, te = SEAS != s, SEAS == s
            p[te] = A[te] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        return p

    def score(name, p, base=None):
        m = np.round(p, 1)
        g = (m != S) & (Y != S)
        d = (m != 0) & (Y != 0)
        se = (p - Y) ** 2
        A = np.column_stack([S, p, np.ones(len(p))])
        wt = np.linalg.lstsq(A, Y, rcond=None)[0][1]
        draws = [np.linalg.lstsq(A[i], Y[i], rcond=None)[0][1] for i in (rng.integers(0, len(p), len(p)) for _ in range(800))]
        out = {"name": name, "mse": round(float(se.mean()), 2), "corr": round(float(np.corrcoef(p, Y)[0, 1]), 3), "miss": round(float(np.mean(np.abs(p - Y))), 2),
               "winners": round(float(np.mean((m[d] > 0) == (Y[d] > 0))), 3), "ats": round(float(np.mean(((m > S) == (Y > S))[g])), 3),
               "weight": round(float(wt), 2), "weight_lo": round(float(np.percentile(draws, 2.5)), 2), "weight_hi": round(float(np.percentile(draws, 97.5)), 2)}
        if base is not None:
            x = se - base
            bo = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
            out.update(diff=round(float(x.mean()), 2), lo=round(float(np.percentile(bo, 2.5)), 2), hi=round(float(np.percentile(bo, 97.5)), 2))
        print(out)
        return out

    gap = lambda v: F[f"{v}_off"] + F[f"{v}_def"]
    core = [pd_gap, swap, home, DOWN]
    v2 = loso([epa_gap] + core)
    base_se = (v2 - Y) ** 2
    variants = [score("v2 as it runs", v2)]
    for name, cols in [
        ("EPA without turnovers", [gap("no_to")] + core),
        ("Neutral-game EPA (no garbage time)", [gap("neutral")] + core),
        ("Early-down EPA", [gap("early")] + core),
        ("Success rate", [gap("success")] + core),
        ("Success rate + EPA", [gap("success"), epa_gap] + core),
        ("No-turnover EPA + turnovers apart", [gap("no_to"), F["to_off"], F["to_def"]] + core),
        ("Offense and defense weighted apart", [F["base_off"], F["base_def"]] + core),
        ("Pass and rush weighted apart", [F["pass_off"], F["pass_def"], F["rush_off"], F["rush_def"]] + core),
        ("v2 + special teams", [epa_gap, F["st"]] + core),
        ("v2 + explosive plays", [epa_gap, gap("explosive")] + core),
        ("v2 + sack rates", [epa_gap, F["sack_off"], F["sack_def"]] + core),
        ("Everything", [gap("no_to"), gap("success"), F["pass_off"], F["pass_def"], F["to_off"], F["to_def"], F["sack_off"], F["sack_def"], F["st"], gap("explosive")] + core),
    ]:
        variants.append(score(name, loso(cols), base_se))

    # Do past misses predict the next one?
    order = np.lexsort((np.arange(len(ids)), WK, SEAS))
    trail_m, trail_v, last3_m, last3_v = (np.zeros(len(ids)) for _ in range(4))
    hist_m, hist_v = {}, {}
    for i in order:
        r = rows[ids[i]]
        s, h, a = r["season"], canon(r["home_team"]), canon(r["away_team"])

        def mean_of(d, t, n=None):
            v = d.get((s, t), [])
            return float(np.mean(v[-n:] if n else v)) if len(v) >= 2 else 0.0

        trail_m[i], trail_v[i] = mean_of(hist_m, h) - mean_of(hist_m, a), mean_of(hist_v, h) - mean_of(hist_v, a)
        last3_m[i], last3_v[i] = mean_of(hist_m, h, 3) - mean_of(hist_m, a, 3), mean_of(hist_v, h, 3) - mean_of(hist_v, a, 3)
        rm, rv = Y[i] - v2[i], Y[i] - S[i]
        hist_m.setdefault((s, h), []).append(rm)
        hist_m.setdefault((s, a), []).append(-rm)
        hist_v.setdefault((s, h), []).append(rv)
        hist_v.setdefault((s, a), []).append(-rv)
    misses = [
        {"name": "Our misses so far, our next miss", **boot_slope(trail_m, Y - v2)},
        {"name": "Our last 3 misses, our next miss", **boot_slope(last3_m, Y - v2)},
        {"name": "Vegas misses so far, Vegas's next miss", **boot_slope(trail_v, Y - S)},
        {"name": "Vegas's last 3 misses, Vegas's next miss", **boot_slope(last3_v, Y - S)},
        {"name": "Our misses so far, Vegas's next miss", **boot_slope(trail_m, Y - S)},
    ]

    # How much of a miss is what happened in that game?
    d_ = {(r["game_id"], r["team"]): r for r in off.iter_rows(named=True)}
    xs = []
    for gid in ids:
        r = rows[gid]
        h, a = d_.get((gid, canon(r["home_team"]))), d_.get((gid, canon(r["away_team"])))
        xs.append([a["turnovers"] - h["turnovers"], (h["st"] or 0) - (a["st"] or 0), (h["epa_no_to"] or 0) - (a["epa_no_to"] or 0)] if h and a else [np.nan] * 3)
    xs = np.array(xs, float)
    ok = ~np.isnan(xs).any(axis=1)

    def r2(miss, cols):
        A = np.column_stack([xs[ok][:, cols], np.ones(ok.sum())])
        return round(float(1 - np.var(miss - A @ np.linalg.lstsq(A, miss, rcond=None)[0]) / miss.var()), 3)

    explained = {}
    for name, miss in (("model", (Y - v2)[ok]), ("vegas", (Y - S)[ok])):
        explained[name] = {"turnovers": r2(miss, [0]), "plus_special_teams": r2(miss, [0, 1]), "plus_offense": r2(miss, [0, 1, 2])}
    coef = np.linalg.lstsq(np.column_stack([xs[ok], np.ones(ok.sum())]), (Y - v2)[ok], rcond=None)[0]
    explained["points_per_turnover"] = round(float(coef[0]), 1)

    # What lasts?
    lasts = {}
    for label, C in (("offense", CO), ("defense", CD)):
        lasts[label] = []
        for name, (n, dd) in LASTS.items():
            a, b, y1, y2 = [], [], [], []
            for s in range(2019, 2026):
                x = C[s]
                a += list(x[n][:, 9] / x[dd][:, 9])
                b += list((x[n][:, -1] - x[n][:, 9]) / (x[dd][:, -1] - x[dd][:, 9]))
                if s < 2025:
                    y1 += list(x[n][:, -1] / x[dd][:, -1])
                    y2 += list(C[s + 1][n][:, -1] / C[s + 1][dd][:, -1])
            lasts[label].append({"stat": name, "half": round(float(np.corrcoef(a, b)[0, 1]), 2), "year": round(float(np.corrcoef(y1, y2)[0, 1]), 2)})

    # Our own over/under model.
    tot, line = T["total"], T["line"]
    good = ~np.isnan(line) & ~np.isnan(tot)
    A = np.column_stack([T["sum_off"], T["sum_def"], T["pace"], T["wind"], T["temp"], T["indoor"], np.ones(len(tot))])
    pt = np.zeros(len(tot))
    for s in np.unique(SEAS):
        tr, te = (SEAS != s) & good, SEAS == s
        pt[te] = A[te] @ np.linalg.lstsq(A[tr], tot[tr], rcond=None)[0]
    totals = {"line_corr": round(float(np.corrcoef(line[good], tot[good])[0, 1]), 3), "line_miss": round(float(np.mean(np.abs(line[good] - tot[good]))), 2),
              "model_corr": round(float(np.corrcoef(pt[good], tot[good])[0, 1]), 3), "model_miss": round(float(np.mean(np.abs(pt[good] - tot[good]))), 2), "gaps": []}
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 99)):
        m = good & (np.abs(pt - line) >= lo) & (np.abs(pt - line) < hi) & (tot != line)
        totals["gaps"].append({"gap_from": lo, "gap_to": hi, **summary((pt[m] > line[m]) == (tot[m] > line[m]))})

    # Simple market rules, found in one era and checked in the other.
    s0 = pl.read_parquet(fl.LONG_SCHEDULES).filter((pl.col("game_type") == "REG") & pl.col("result").is_not_null() & pl.col("spread_line").is_not_null())
    season, week = s0["season"].to_numpy(), s0["week"].to_numpy()
    rs, sp = s0["result"].to_numpy().astype(float), s0["spread_line"].to_numpy().astype(float)
    tl, tt = s0["total_line"].cast(pl.Float64).to_numpy(), s0["total"].cast(pl.Float64).to_numpy()
    wind, temp = s0["wind"].cast(pl.Float64).to_numpy(), s0["temp"].cast(pl.Float64).to_numpy()
    roof = np.array(s0["roof"].fill_null("").to_list())
    day, gt = np.array(s0["weekday"].to_list()), np.array(s0["gametime"].fill_null("13:00").to_list())
    div = s0["div_game"].fill_null(0).to_numpy()
    outdoor, prime = ~np.isin(roof, ["dome", "closed"]), gt >= "19:00"
    old = season <= 2018
    cover, under = rs > sp, tt < tl
    spread_push, total_push = rs == sp, (tt == tl) | np.isnan(tl)
    spread_rules = {
        "Home underdogs": (sp < 0, cover), "Road underdogs": (sp > 0, ~cover), "Home underdogs of 7+": (sp <= -7, cover), "Road underdogs of 7+": (sp >= 7, ~cover),
        "Home favorites of 10+": (sp >= 10, cover), "Division underdogs": ((div == 1) & (sp != 0), cover == (sp < 0)), "Primetime underdogs": (prime & (sp != 0), cover == (sp < 0)),
        "Thursday underdogs": ((day == "Thursday") & (sp != 0), cover == (sp < 0)), "Week 1 underdogs": ((week == 1) & (sp != 0), cover == (sp < 0)),
    }
    total_rules = {
        "Wind 10+ mph": outdoor & (wind >= 10), "Wind 15+ mph": outdoor & (wind >= 15), "Wind 20+ mph": outdoor & (wind >= 20), "32 degrees or colder": outdoor & (temp <= 32),
        "Primetime": prime, "Thursday": day == "Thursday", "Monday": day == "Monday", "Sunday night": (day == "Sunday") & prime, "Division games": div == 1,
        "Total of 50+": tl >= 50, "Total of 40 or less": tl <= 40, "Week 13 on": week >= 13, "Weeks 1-4": week <= 4, "Indoors": ~outdoor,
    }

    def eras(hit, mask):
        return {"old": summary(hit[mask & old]), "new": summary(hit[mask & ~old])}

    rules = {"spread": [{"name": k, **eras(h, m & ~spread_push)} for k, (m, h) in spread_rules.items()],
             "totals": [{"name": k, **eras(under, m & ~total_push)} for k, m in total_rules.items()]}

    def strength(cov, und):
        out = {}
        for k, (m, h) in spread_rules.items():
            side = cov if h is cover else (~cov if k in ("Road underdogs", "Road underdogs of 7+") else cov == (sp < 0))
            u = m & ~spread_push
            out[k] = min((side[u & e].mean() - 0.5) * 2 * np.sqrt((u & e).sum()) for e in (old, ~old))
        for k, m in total_rules.items():
            u = m & ~total_push
            out[k] = min((und[u & e].mean() - 0.5) * 2 * np.sqrt((u & e).sum()) for e in (old, ~old))
        return out

    observed = strength(cover, under)
    nulls = []
    for _ in range(2000):
        c2, u2 = cover.copy(), under.copy()
        for s in np.unique(season):
            k = np.where(season == s)[0]
            c2[k] = c2[rng.permutation(k)]
            k2 = k[~total_push[k]]
            u2[k2] = u2[rng.permutation(k2)]
        nulls.append(max(strength(c2, u2).values()))
    nulls = np.array(nulls)
    rules["luck"] = {k: round(float(np.mean(nulls >= v)), 3) for k, v in sorted(observed.items(), key=lambda x: -x[1])[:4]}
    print("rules", rules["luck"])

    # Moneylines.
    ml_h, ml_a = s0["home_moneyline"].cast(pl.Float64).to_numpy(), s0["away_moneyline"].cast(pl.Float64).to_numpy()
    pay = lambda ml: np.where(ml > 0, ml / 100, 100 / np.abs(ml))
    moneyline = []
    for name, lo, hi in (("Favorites −2000 to −400", -2000, -400), ("Favorites −400 to −250", -400, -250), ("Favorites −250 to −150", -250, -150), ("Favorites −150 to −100", -150, -100),
                         ("Underdogs +100 to +150", 100, 150), ("Underdogs +150 to +250", 150, 250), ("Underdogs +250 to +400", 250, 400), ("Underdogs +400 and up", 400, 2000)):
        row = {"name": name}
        for key, sel in (("old", old), ("new", ~old)):
            parts = []
            for ml, won in ((ml_h, rs > 0), (ml_a, rs < 0)):
                m = sel & (ml >= lo) & (ml < hi) & (rs != 0)
                parts.append(np.where(won[m], pay(ml[m]), -1.0))
            r_ = np.concatenate(parts)
            row[key] = {"roi": round(float(r_.mean()), 3), "n": int(len(r_))}
        moneyline.append(row)

    # Every signal at once against the spread.
    feats_all = [gap("no_to"), gap("success"), F["pass_off"], F["pass_def"], F["to_off"], F["to_def"], F["sack_off"], F["sack_def"], F["st"], gap("explosive"), pd_gap, swap, DOWN, trail_v, last3_v, v2 - S]
    Z = np.column_stack(feats_all)
    Z = np.column_stack([(Z - Z.mean(0)) / Z.std(0), np.ones(len(Y))])
    R = Y - S

    def ridge(tr, te, target=R, lam=3000):
        return Z[te] @ np.linalg.solve(Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1]), Z[tr].T @ target[tr])

    p_out = np.zeros(len(Y))
    for s in np.unique(SEAS):
        p_out[SEAS == s] = ridge(SEAS != s, SEAS == s)
    p_past = np.full(len(Y), np.nan)
    for s in range(2021, 2027):
        p_past[SEAS == s] = ridge(SEAS < s, SEAS == s)
    g = R != 0
    m = g & ~np.isnan(p_past)
    combined = {
        "leave_one_season_out": summary((p_out[g] > 0) == (R[g] > 0)),
        "trained_on_the_past": summary((p_past[m] > 0) == (R[m] > 0)),
        "by_season": [{"season": int(s), **summary((p_past[m & (SEAS == s)] > 0) == (R[m & (SEAS == s)] > 0))} for s in range(2021, 2027)],
        "always_underdog": summary((S[g & (S != 0)] < 0) == (R[g & (S != 0)] > 0)),
    }
    # Its most confident picks.
    conf = np.abs(np.nan_to_num(p_past))
    combined["top_decile"] = summary(((p_past > 0) == (R > 0))[m & (conf >= np.quantile(conf[m], 0.9))])
    print("combined", combined["trained_on_the_past"], combined["top_decile"])

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory": {"team_games": off.height, "plays": int(off["plays"].sum()), "pbp_seasons": [int(off["season"].min()), int(off["season"].max())], "market_games": s0.height, "market_seasons": [int(season.min()), int(season.max())], "model_games": len(ids)},
        "ratings": {"vegas": {"mse": round(float(np.mean((S - Y) ** 2)), 2), "corr": round(float(np.corrcoef(S, Y)[0, 1]), 3), "miss": round(float(np.mean(np.abs(S - Y))), 2)}, "games": len(ids), "variants": variants},
        "misses": misses, "explained": explained, "lasts": lasts, "totals": totals, "rules": rules, "moneyline": moneyline, "combined": combined,
    }
    with open(os.path.join(bs.BASE_DIR, "data", "lab", "edge_lab.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("Wrote data/lab/edge_lab.json")


if __name__ == "__main__":
    main()

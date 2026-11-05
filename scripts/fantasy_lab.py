"""Fantasy Lab: do our rankings know anything FantasyPros consensus doesn't?

Two walk-forward tests over 2021-2026, each graded on what actually happened:

  draft  -- our preseason baseline (2025-style: recency-weighted last-season
            PPG) vs. the last consensus snapshot before Week 1, graded on that
            season's PPG (4+ games), plus rank blends of the two.
  weekly -- candidate inputs to the Start Score vs. plain recent PPG and the
            FantasyPros weekly consensus, graded on that week's points. A
            player whose team played and who didn't is graded on zero points
            -- that's what sitting costs you, and it's the only way injury
            information can earn credit. Each input's weight is fit by least
            squares per position, leave-one-season-out.

FantasyPros history comes from nflreadpy (scraped weekly since late 2020).
Writes data/lab/fantasy_lab.json for the Data Breakdown page, and the fitted
weights (2021-2025 only, so 2026 stays held out) to
scripts/reference/start_score_weights.json, which fantasy_scan.py reads.

Local-only (not run in CI): python scripts/fantasy_lab.py
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

import fantasy_scan as fs
from shared import CACHE_DIR, TEAM_CODE_ALIASES, load_cache

SEASONS = list(range(2021, 2027))
HISTORY = os.path.join(CACHE_DIR, "ff_rankings_history.parquet")
ROSTERS_WEEKLY = os.path.join(CACHE_DIR, "rosters_weekly.parquet")
POOL = {"QB": 24, "RB": 36, "WR": 48, "TE": 18, "K": 18, "DST": 18}  # roughly the startable pool in a 12-team league
POSITIONS = list(POOL)
FP_TEAM = {"JAC": "JAX", "LAR": "LA", "WSH": "WAS", "LVR": "LV"}
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIOR_K = 2  # games' worth of last season's PPG mixed into a player's recent form
BOOT = 2000
rng = np.random.default_rng(29)
WEIGHTS_PATH = os.path.join(fs.REF_DIR, "start_score_weights.json")
# The hand-set Start Score used through Week 8 of 2026, kept here to grade against.
V1 = {"matchup": 0.5, "questionable": -2.0, "doubtful": -10.0, "out": -20.0, "qb_changed": -2.0, "former_team": 1.0}


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def load_history():
    if not os.path.exists(HISTORY):
        import nflreadpy as nfl
        pages = [f"{kind}-{pos}" for kind in ("redraft", "weekly") for pos in ("qb", "rb", "wr", "te", "k", "dst")]
        (nfl.load_ff_rankings(type="all").filter(pl.col("page_type").is_in(pages))
         .select("page_type", "fp_page", "player", "id", "pos", "team", "ecr", "scrape_date").write_parquet(HISTORY))
    return pl.read_parquet(HISTORY)


def load_rosters_weekly():
    if not os.path.exists(ROSTERS_WEEKLY):
        import nflreadpy as nfl
        (nfl.load_rosters_weekly(seasons=SEASONS).select("season", "week", "team", "gsis_id", "status").drop_nulls("gsis_id").write_parquet(ROSTERS_WEEKLY))
    return pl.read_parquet(ROSTERS_WEEKLY)


def load_stats():
    raw = load_cache("player_stats_weekly")
    teams = load_cache("team_stats")
    cur_p = fs.load_current_season_player_stats(fs.UPCOMING_SEASON, allow_empty=True)
    cur_t = fs.load_current_season_team_stats(fs.UPCOMING_SEASON, allow_empty=True)
    raw = pl.concat([raw, cur_p], how="diagonal_relaxed")
    teams = pl.concat([teams, cur_t], how="diagonal_relaxed")
    return pl.concat([fs.build_skill_weekly(raw), fs.build_kicker_weekly(raw), fs.build_dst_weekly(teams, load_cache("schedules"))]).filter(pl.col("season_type") == "REG")


def fp_to_player_id(stats):
    """Maps a FantasyPros row to our player_id: the crosswalk for QB/RB/WR/TE,
    normalized name for kickers (not in the crosswalk), team for defenses."""
    import nflreadpy as nfl
    ids = nfl.load_ff_playerids().select("fantasypros_id", "gsis_id").drop_nulls()
    by_fp = dict(zip(ids["fantasypros_id"].to_list(), ids["gsis_id"].to_list()))
    kickers = stats.filter(pl.col("position") == "K").select("player_id", "player_display_name").unique()
    by_name = {fs._normalize_name(n): i for i, n in zip(kickers["player_id"].to_list(), kickers["player_display_name"].to_list())}

    def mapper(pos, fp_id, name, team):
        if pos == "K":
            return by_name.get(fs._normalize_name(name))
        if pos == "DST":
            t = FP_TEAM.get(team, team)
            return f"{TEAM_CODE_ALIASES.get(t, t)}_DST"
        try:
            return by_fp.get(int(fp_id))
        except (TypeError, ValueError):
            return None
    return mapper


def snapshot(history, page, date, mapper):
    """{player_id: consensus rank} for one page on one scrape date."""
    rows = history.filter((pl.col("page_type") == page) & (pl.col("scrape_date") == date) & ~pl.col("fp_page").str.contains("ros-"))
    out = {}
    for r in rows.iter_rows(named=True):
        pid = mapper(r["pos"], r["id"], r["player"], r["team"])
        if pid:
            out[pid] = min(out.get(pid, 1e9), r["ecr"])
    return out


# ---- Draft test ----------------------------------------------------------


def baseline_ppg(stats, season):
    """The Draft Board's baseline, PPR, from one finished season: half full-season
    PPG, half last-six-games PPG, with its 4-game minimum."""
    reg = stats.filter(pl.col("season") == season).sort("week")
    out = {}
    for (pid,), g in reg.group_by(["player_id"]):
        pts = g.sort("week")["fpts_ppr"].to_numpy()
        if len(pts) >= fs.MIN_GAMES_FOR_DRAFT_BOARD:
            out[pid] = (fs.RECENCY_WEIGHT * pts[-6:].mean() + (1 - fs.RECENCY_WEIGHT) * pts.mean(), g["position"][0])
    return out


def draft_test(stats, history, schedules, mapper):
    rows = []
    for season in SEASONS:
        first_game = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG"))["gameday"].min()
        dates = [d for d in history.filter(pl.col("page_type") == "redraft-wr")["scrape_date"].unique().to_list() if d < first_game and d[:4] == str(season)]
        if not dates:
            continue
        date = max(dates)
        ours = baseline_ppg(stats, season - 1)
        played = stats.filter(pl.col("season") == season).group_by("player_id").agg(pl.col("fpts_ppr").mean().alias("ppg"), pl.len().alias("g")).filter(pl.col("g") >= fs.MIN_GAMES_FOR_DRAFT_BOARD)
        actual = dict(zip(played["player_id"].to_list(), played["ppg"].to_list()))
        for pos in POSITIONS:
            market = snapshot(history, f"redraft-{pos.lower()}", date, mapper)
            ranked = sorted(((pid, v[0]) for pid, v in ours.items() if v[1] == pos), key=lambda x: -x[1])[:50]
            keep = [(pid, i + 1) for i, (pid, _) in enumerate(ranked) if pid in market and pid in actual]
            if len(keep) < 8:
                continue
            o = rank(np.array([k[1] for k in keep], float))
            m = rank(np.array([market[k[0]] for k in keep], float))
            y = -np.array([actual[k[0]] for k in keep])
            row = {"season": season, "position": pos, "n": len(keep), "snapshot": date}
            for w in BLEND_WEIGHTS:
                row[f"w{w}"] = spearman((1 - w) * o + w * m, y)
            rows.append(row)
            print(f"  draft {season} {pos:<3} n={len(keep):>2} ours {row['w0.0']:.3f}  consensus {row['w1.0']:.3f}  50/50 {row['w0.5']:.3f}")
    return rows


# ---- Weekly test ---------------------------------------------------------


def game_lines(schedules):
    """(season, week, team) -> (opponent, the team's Vegas implied points).
    implied = (total + the team's margin by the spread) / 2."""
    out = {}
    for r in schedules.filter(pl.col("game_type") == "REG").iter_rows(named=True):
        total, spread = r["total_line"], r["spread_line"]
        known = total is not None and spread is not None
        out[(r["season"], r["week"], r["home_team"])] = (r["away_team"], (total + spread) / 2 if known else None)
        out[(r["season"], r["week"], r["away_team"])] = (r["home_team"], (total - spread) / 2 if known else None)
    return out


def week_for_scrape(schedules, season):
    """(week, last gameday) pairs: a Friday snapshot ranks that week's Sunday games."""
    g = schedules.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")).group_by("week").agg(pl.col("gameday").max().alias("last")).sort("week")
    return list(zip(g["week"].to_list(), g["last"].to_list()))


def qb_starter_by_week(schedules):
    starter = {}
    for r in schedules.filter(pl.col("game_type") == "REG").iter_rows(named=True):
        for side in ("home", "away"):
            if r[f"{side}_qb_id"]:
                starter[(r["season"], r["week"], r[f"{side}_team"])] = r[f"{side}_qb_id"]
    return starter


def player_former_teams(rosters):
    """gsis_id -> every team this player was ACTIVE on in any season."""
    active = rosters.filter((pl.col("status") == "ACT") & pl.col("gsis_id").is_not_null()).with_columns(pl.col("team").replace(TEAM_CODE_ALIASES).alias("team"))
    by_player = active.group_by("gsis_id").agg(pl.col("team").unique().alias("teams"))
    return dict(zip(by_player["gsis_id"].to_list(), (set(t) for t in by_player["teams"].to_list())))


def player_week_table(stats, usage, history, schedules, mapper, points):
    """One row per pooled player per week: what was known going in, plus what
    he scored. Pool = each position's top recent-form players whose team plays,
    chosen before any adjustment, so every variant ranks the same players."""
    lines = game_lines(schedules)
    starter = qb_starter_by_week(schedules)
    injuries = pl.concat([load_cache("injuries"), fs.load_current_season_injuries(fs.UPCOMING_SEASON, allow_empty=True)], how="diagonal_relaxed").filter(pl.col("report_status").is_not_null())
    report = {(int(r["season"]), int(r["week"]), r["gsis_id"]): r["report_status"] for r in injuries.select("season", "week", "gsis_id", "report_status").iter_rows(named=True)}
    reserve = {(r["season"], r["week"], r["gsis_id"]) for r in load_rosters_weekly().filter(pl.col("status") == "RES").iter_rows(named=True)}
    former = player_former_teams(load_cache("rosters"))
    ppg_key = "ppr_ppg" if points == "fpts_ppr" else "standard_ppg"

    by_date = {}
    for d in history.filter(pl.col("page_type") == "weekly-wr")["scrape_date"].unique().to_list():
        by_date.setdefault(d[:4], []).append(d)

    table = []
    for season in SEASONS:
        weeks = week_for_scrape(schedules, season)
        season_stats = stats.filter(pl.col("season") == season)
        prior = stats.filter(pl.col("season") == season - 1).group_by("player_id").agg(pl.col(points).mean().alias("p"))
        prior_ppg = dict(zip(prior["player_id"].to_list(), prior["p"].to_list()))
        fallback, fallback_avg = fs._defense_vs_position(stats, season - 1)
        ecr_dates = {}
        for week, last in weeks:
            prev_last = next((l for w, l in weeks if w == week - 1), "0000")
            c = [d for d in by_date.get(str(season), []) if prev_last <= d <= last]
            if c:
                ecr_dates[week] = max(c)
        use = usage.filter(pl.col("season") == season)
        for week, _ in weeks:
            done = season_stats.filter(pl.col("week") < week)
            now = season_stats.filter(pl.col("week") == week)
            if week < 2 or done.is_empty() or now.is_empty():
                continue
            actual = dict(zip(now["player_id"].to_list(), now[points].to_list()))
            cur_def, cur_avg = fs._defense_vs_position(done, season)
            def_games = {}
            for (team, _pos), row in cur_def.items():
                def_games[team] = max(def_games.get(team, 0), row["games"])
            ecr_by_pos = {pos: snapshot(history, f"weekly-{pos.lower()}", ecr_dates[week], mapper) if week in ecr_dates else {} for pos in POSITIONS}
            done_use = use.filter(pl.col("week") < week).group_by("player_id").agg(pl.col("targets").tail(6).mean(), pl.col("carries").tail(6).mean())
            usage_of = {r["player_id"]: (r["targets"], r["carries"]) for r in done_use.iter_rows(named=True)}

            usual = {}
            for r in done.filter(pl.col("position") == "QB").group_by(["team", "player_id"]).agg(pl.len().alias("n"), pl.col(points).sum().alias("p")).sort(["n", "p"]).iter_rows(named=True):
                usual[r["team"]] = r["player_id"]

            players = {}
            for (pid,), g in done.group_by(["player_id"]):
                g = g.sort("week")
                pts = g[points].to_numpy()
                players[pid] = {"pos": g["position"][-1], "team": g["team"][-1], "name": g["player_display_name"][-1],
                                "recent": fs.RECENCY_WEIGHT * pts[-6:].mean() + (1 - fs.RECENCY_WEIGHT) * pts.mean(), "games": len(pts)}
            for pos in POSITIONS:
                pool = sorted((p for p in players.items() if p[1]["pos"] == pos and (season, week, p[1]["team"]) in lines), key=lambda x: -x[1]["recent"])[:POOL[pos]]
                if len(pool) < 8:
                    continue
                worst_ecr = max(ecr_by_pos[pos].values(), default=99) + 1
                for pid, p in pool:
                    team = p["team"]
                    opp, implied = lines[(season, week, team)]
                    opp_implied = lines[(season, week, opp)][1]
                    adj = 0.0
                    if pos in fs.MATCHUP_POS:
                        if def_games.get(opp, 0) >= fs.MIN_GAMES_FOR_DEFENSE_SAMPLE and (opp, pos) in cur_def:
                            adj = cur_def[(opp, pos)][ppg_key] - cur_avg.get(pos, {}).get(ppg_key, 0)
                        elif (opp, pos) in fallback:
                            adj = fallback[(opp, pos)][ppg_key] - fallback_avg.get(pos, {}).get(ppg_key, 0)
                    status = report.get((season, week, pid))
                    starting = starter.get((season, week, team))
                    tg, cr = usage_of.get(pid, (0.0, 0.0))
                    pr = prior_ppg.get(pid)
                    table.append({
                        "season": season, "week": week, "position": pos, "player_id": pid, "y": actual.get(pid, 0.0),
                        "recent": p["recent"], "games": p["games"], "prior": pr if pr is not None else p["recent"], "has_prior": float(pr is not None),
                        "matchup": adj,
                        "implied": (implied - 22.5) if implied is not None else 0.0,
                        "opp_implied": (opp_implied - 22.5) if opp_implied is not None else 0.0,
                        "targets": tg or 0.0, "carries": cr or 0.0,
                        "questionable": float(status == "Questionable"), "doubtful": float(status == "Doubtful"), "out": float(status == "Out"),
                        "reserve": float(status is None and (season, week - 1, pid) in reserve),
                        "not_starter": float(pos == "QB" and bool(starting) and starting != pid),
                        "qb_changed": float(pos in ("WR", "TE") and bool(starting) and bool(usual.get(team)) and starting != usual[team]),
                        "former_team": float(opp in former.get(pid, set())),
                        "ecr": ecr_by_pos[pos].get(pid, worst_ecr), "has_ecr": bool(ecr_by_pos[pos]),
                    })
        print(f"  {season}: {sum(1 for t in table if t['season'] == season)} player-weeks")
    return pl.DataFrame(table)


def form_of(t, k=PRIOR_K):
    """This season's form with k games' worth of last season mixed in."""
    return (t["games"] * t["recent"] + k * t["prior"]) / (t["games"] + k)


def design(t, pos, terms):
    cols = [np.ones(t.height)]
    for term in terms:
        if term == "form":
            cols.append(form_of(t).to_numpy())
        elif term == "team_total":
            cols.append((t["opp_implied"] * -1 if pos == "DST" else t["implied"]).to_numpy().astype(float))
        else:
            cols.append(t[term].to_numpy().astype(float))
    return np.column_stack(cols)


TERMS = {
    "QB": ["form", "team_total", "matchup", "carries", "questionable", "doubtful", "out", "reserve", "not_starter"],
    "RB": ["form", "team_total", "matchup", "targets", "carries", "questionable", "doubtful", "out", "reserve"],
    "WR": ["form", "team_total", "matchup", "targets", "questionable", "doubtful", "out", "reserve"],
    "TE": ["form", "team_total", "matchup", "targets", "questionable", "doubtful", "out", "reserve"],
    "K": ["form", "team_total", "matchup", "questionable", "doubtful", "out", "reserve"],
    "DST": ["team_total"],  # a defense's own fantasy form adds nothing once its opponent's Vegas total is known
}


def fit_predict(t, pos, terms, train, test):
    """Least squares on `train`, scored on `test`. Every in-sample report below
    is leave-one-season-out, so a season is never ranked by a fit that saw it."""
    coef = np.linalg.lstsq(design(train, pos, terms), train["y"].to_numpy().astype(float), rcond=None)[0]
    return design(test, pos, terms) @ coef, coef


def weekly_test(stats, usage, history, schedules, mapper, points):
    t = player_week_table(stats, usage, history, schedules, mapper, points)
    variants = {
        "form": lambda pos: ["form"],
        "availability": lambda pos: ["form"] + [x for x in TERMS[pos] if x in ("questionable", "doubtful", "out", "reserve", "not_starter")],
        "vegas": lambda pos: ["form", "team_total"],
        "matchup": lambda pos: ["form", "matchup"] if "matchup" in TERMS[pos] else ["form"],
        "usage": lambda pos: ["form"] + [x for x in TERMS[pos] if x in ("targets", "carries")],
        "team_qb": lambda pos: ["form", "qb_changed"] if pos in ("WR", "TE") else ["form"],
        "former_team": lambda pos: ["form", "former_team"] if pos != "DST" else ["form"],
        "fitted": lambda pos: TERMS[pos],
    }
    rows = []
    preds = {}
    for pos in POSITIONS:
        tp = t.filter(pl.col("position") == pos)
        for season in SEASONS:
            te, tr = tp.filter(pl.col("season") == season), tp.filter(pl.col("season") != season)
            out = {name: fit_predict(tp, pos, f(pos), tr, te)[0] for name, f in variants.items()}
            for (week,), wk in te.with_row_index("i").group_by(["week"]):
                idx = wk["i"].to_numpy()
                y = wk["y"].to_numpy().astype(float)
                row = {"season": season, "week": week, "position": pos, "n": wk.height, "has_ecr": bool(wk["has_ecr"][0])}
                row["recent_ppg"] = spearman(wk["recent"].to_numpy(), y)
                v1 = (wk["recent"] + sum(V1[k] * wk[k] for k in V1)).to_numpy()
                row["start_score_v1"] = spearman(v1, y)
                for name in variants:
                    row[name] = spearman(out[name][idx], y)
                if row["has_ecr"]:
                    row["ecr"] = spearman(-wk["ecr"].to_numpy(), y)
                    row["blend"] = spearman(rank(out["fitted"][idx]) + rank(-wk["ecr"].to_numpy()), y)
                rows.append(row)
    coefs = {}
    for pos in POSITIONS:
        tp = t.filter(pl.col("position") == pos)
        train = tp.filter(pl.col("season") < fs.UPCOMING_SEASON)
        _, coef = fit_predict(tp, pos, TERMS[pos], train, train)
        coefs[pos] = {"intercept": round(float(coef[0]), 3), **{k: round(float(v), 3) for k, v in zip(TERMS[pos], coef[1:])}, "n": train.height}
    return rows, coefs


def clustered_diff(rows, a, b):
    """Mean within-week difference a - b and a bootstrap CI that resamples
    season-weeks, since one week's positions share one slate of games."""
    weeks = sorted({(r["season"], r["week"]) for r in rows})
    by_week = {w: [] for w in weeks}
    for r in rows:
        if r.get(a) is not None and r.get(b) is not None:
            by_week[(r["season"], r["week"])].append(r[a] - r[b])
    means = np.array([np.mean(v) for v in by_week.values() if v])
    if not len(means):
        return None
    boots = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(means.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3), "weeks": len(means)}


def mean_of(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(v)), 3) if v else None


def main():
    stats = load_stats()
    schedules = load_cache("schedules")
    history = load_history()
    mapper = fp_to_player_id(stats)

    usage = load_cache("player_stats_weekly").select("player_id", "season", "week", "season_type", "targets", "carries")
    live = fs.load_current_season_player_stats(fs.UPCOMING_SEASON, allow_empty=True).select("player_id", "season", "week", "season_type", "targets", "carries")
    usage = pl.concat([usage, live], how="diagonal_relaxed").filter(pl.col("season_type") == "REG").sort(["season", "week"])

    print("Draft test...")
    draft = draft_test(stats, history, schedules, mapper)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons": [SEASONS[0], SEASONS[-1]]}

    # Draft: 2021-2025 sets the blend; 2026 is scored separately.
    report["draft"] = {"by_position": {}}
    print("\nDraft, mean rank correlation with that season's PPG")
    print(f"{'pos':<4} {'seasons':>7} " + " ".join(f"w={w:<5}" for w in BLEND_WEIGHTS) + "   2026: ours / consensus / 50-50")
    for pos in POSITIONS:
        hist = [r for r in draft if r["position"] == pos and r["season"] < 2026]
        cur = [r for r in draft if r["position"] == pos and r["season"] == 2026]
        means = {str(w): mean_of(hist, f"w{w}") for w in BLEND_WEIGHTS}
        beat = sum(1 for r in hist if r["w1.0"] > r["w0.0"])
        report["draft"]["by_position"][pos] = {
            "seasons": len(hist), "means": means, "consensus_beat_ours": beat,
            "by_season": [{"season": r["season"], "ours": round(r["w0.0"], 3), "consensus": round(r["w1.0"], 3), "blend": round(r["w0.75"], 3), "n": r["n"]} for r in hist + cur],
            "season_2026": {str(w): round(cur[0][f"w{w}"], 3) for w in BLEND_WEIGHTS} if cur else None,
        }
        c = report["draft"]["by_position"][pos]["season_2026"] or {}
        print(f"{pos:<4} {len(hist):>7} " + " ".join(f"{means[str(w)]:<7}" for w in BLEND_WEIGHTS) + f"   {c.get('0.0')} / {c.get('1.0')} / {c.get('0.75')}  (consensus beat ours {beat} of {len(hist)})")

    keys = ["recent_ppg", "start_score_v1", "form", "availability", "vegas", "matchup", "usage", "team_qb", "former_team", "fitted", "ecr", "blend"]
    report["weekly"] = {}
    for fmt, points in (("ppr", "fpts_ppr"), ("standard", "fpts_standard")):
        print(f"\nWeekly test, {fmt}...")
        weekly, coefs = weekly_test(stats, usage, history, schedules, mapper, points)
        rep = {"position_weeks": len(weekly), "coefficients": coefs, "by_position": {}, "overall": {}, "season_2026": {}, "effects": {}}
        print(f"{'variant':<16} " + " ".join(f"{p:>6}" for p in POSITIONS) + "    all    2026")
        for k in keys:
            cells = [mean_of([r for r in weekly if r["position"] == p], k) for p in POSITIONS]
            rep["overall"][k] = mean_of(weekly, k)
            rep["season_2026"][k] = mean_of([r for r in weekly if r["season"] == 2026], k)
            for p, c in zip(POSITIONS, cells):
                rep["by_position"].setdefault(p, {})[k] = c
            print(f"{k:<16} " + " ".join(f"{c if c is not None else '':>6}" for c in cells) + f" {rep['overall'][k]:>6} {rep['season_2026'][k]}")
        print("Each input's gain over form alone [95% CI, resampling weeks]")
        for k in keys:
            if k != "form":
                rep["effects"][k] = clustered_diff(weekly, k, "form")
                print(f"  {k:<16} {rep['effects'][k]}")
        with_ecr = [r for r in weekly if r.get("ecr") is not None]
        rep["vs_consensus"] = {"fitted_minus_ecr": clustered_diff(with_ecr, "fitted", "ecr"), "blend_minus_ecr": clustered_diff(with_ecr, "blend", "ecr"), "fitted_minus_v1": clustered_diff(weekly, "fitted", "start_score_v1")}
        print(rep["vs_consensus"])
        rep["weeks_2026"] = [
            {"week": w, **{k: mean_of([r for r in weekly if r["season"] == 2026 and r["week"] == w], k) for k in ("recent_ppg", "start_score_v1", "fitted", "ecr")}}
            for w in sorted({r["week"] for r in weekly if r["season"] == 2026})
        ]
        print("Coefficients (fit on 2021-2025):")
        for pos, c in coefs.items():
            print(" ", pos, c)
        report["weekly"][fmt] = rep

    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "_comment": "Written by scripts/fantasy_lab.py. Start Score weights per format and position, least squares on 2021-2025 "
                        "player-weeks. form = this season's recent PPG with PRIOR_GAMES games of last season mixed in; team_total = "
                        "Vegas implied points minus 22.5 (the opponent's, negated, for DST); matchup = points allowed to the position "
                        "vs. league average; targets and carries per game over the last six.",
            "prior_games": PRIOR_K,
            "ppr": report["weekly"]["ppr"]["coefficients"],
            "standard": report["weekly"]["standard"]["coefficients"],
        }, f, indent=1)
    out_dir = os.path.join(fs.BASE_DIR, "data", "lab")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fantasy_lab.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("Wrote data/lab/fantasy_lab.json")


if __name__ == "__main__":
    main()

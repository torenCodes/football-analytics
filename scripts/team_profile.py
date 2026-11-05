"""Team profiles for the Game Picks cards: "is this team for real?"

For each team, from this season's play-by-play: where its offense and defense
rank and what they're built on, where its scoreboard margin ranks, and a
forecast of its margin from here on. Verdicts and the odds quoted beside them
come from scripts/team_lab.py, which measured how teams in the same spot
actually finished in 2019-2025 (reference/team_lab.json).

Writes data/betting/team_profiles.json.
"""

import json
import os
from datetime import datetime, timezone

import polars as pl

from shared import REF_DIR, TEAM_CODE_ALIASES, json_safe

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(BASE_DIR, "data", "betting", "team_profiles.json")
SEASON = 2026
MIN_GAMES = 3  # below this a rank is noise; cards say so instead of guessing

VERDICT_TEXT = {
    "for_real": "For real",
    "better_than_record": "Better than the record",
    "sell_high": "Regression watch",
    "real_problems": "Real problems",
    "middle": "About what it looks like",
}


def verdict_for(margin_rank, forecast_rank):
    """Rank by scoreboard margin so far against rank by forecast from here on."""
    if forecast_rank <= 10 and margin_rank <= 12:
        return "for_real"
    if margin_rank - forecast_rank >= 6:
        return "better_than_record"
    if forecast_rank - margin_rank >= 6:
        return "sell_high"
    if forecast_rank >= 23 and margin_rank >= 21:
        return "real_problems"
    return "middle"


def team_game_tables(pbp):
    """Per team-game offense and defense totals from raw play-by-play."""
    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & ((pl.col("pass").fill_null(0) == 1) | (pl.col("rush").fill_null(0) == 1))
        & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    ).with_columns(
        ((pl.col("interception").fill_null(0) == 1) | (pl.col("fumble_lost").fill_null(0) == 1)).alias("to"),
        (pl.col("yards_gained").fill_null(0) >= 20).alias("explosive"),
        pl.col("wp").is_between(0.1, 0.9).fill_null(False).alias("neutral"),
        (pl.col("down").fill_null(0) <= 2).alias("early"),
    )

    def agg(by):
        return plays.group_by(["game_id", "season", "week", by]).agg(
            pl.len().alias("plays"),
            pl.col("epa").sum().alias("epa"),
            pl.col("epa").filter(~pl.col("to")).sum().alias("epa_no_to"),
            (~pl.col("to")).sum().alias("plays_no_to"),
            pl.col("epa").filter(pl.col("neutral")).sum().alias("epa_neutral"),
            pl.col("neutral").sum().alias("plays_neutral"),
            pl.col("epa").filter(pl.col("early")).sum().alias("epa_early"),
            pl.col("early").sum().alias("plays_early"),
            pl.col("success").fill_null(0).sum().alias("success"),
            pl.col("explosive").sum().alias("explosive"),
            pl.col("to").sum().alias("turnovers"),
            pl.col("qb_dropback").fill_null(0).sum().alias("dropbacks"),
            pl.col("sack").fill_null(0).sum().alias("sacks"),
            pl.col("qb_hit").fill_null(0).sum().alias("qb_hits"),
            pl.col("epa").filter(pl.col("pass") == 1).sum().alias("pass_epa"),
            (pl.col("pass") == 1).sum().alias("pass_plays"),
            pl.col("epa").filter(pl.col("rush") == 1).sum().alias("rush_epa"),
            (pl.col("rush") == 1).sum().alias("rush_plays"),
        ).rename({by: "team"})

    special = pbp.filter((pl.col("season_type") == "REG") & (pl.col("special_teams_play").fill_null(0) == 1) & pl.col("epa").is_not_null() & pl.col("posteam").is_not_null())
    st_for = special.group_by(["game_id", "posteam"]).agg(pl.col("epa").sum().alias("a")).rename({"posteam": "team"})
    st_against = special.group_by(["game_id", "defteam"]).agg((-pl.col("epa")).sum().alias("b")).rename({"defteam": "team"})
    offense = agg("posteam").join(st_for, on=["game_id", "team"], how="left").join(st_against, on=["game_id", "team"], how="left").with_columns(
        (pl.col("a").fill_null(0) + pl.col("b").fill_null(0)).alias("st")).drop("a", "b")
    return offense, agg("defteam")


def _rank(values, best_high=True):
    order = sorted(values, key=lambda k: -values[k] if best_high else values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def _totals(table):
    canon = table.with_columns(pl.col("team").replace(TEAM_CODE_ALIASES))
    return {r["team"]: r for r in canon.group_by("team").agg(pl.exclude("game_id", "season", "week", "team").sum(), pl.len().alias("games")).iter_rows(named=True)}


def period_of(lab, week):
    for name, (lo, hi) in lab["periods"].items():
        if lo <= week <= hi:
            return name
    return "late"


def hold_rate(lab, unit, period, rank):
    for row in lab["hold_rates"][unit][period]:
        if row["lo"] <= rank <= row["hi"]:
            return row
    return None


def built_on(rank, parts):
    """The one or two things a unit does best, as prose: only ones ranked in the
    top 10, and only when the unit itself is above average."""
    if rank > 16:
        return ""
    best = sorted((r, name) for name, r in parts.items() if r <= 10)[:2]
    return " and ".join(f"{name} (#{r})" for r, name in best)


def _signed(x):
    return f"{'+' if x >= 0 else '−'}{abs(x):.1f}"


def verdict_sentence(verdict, record, margin, forecast_rank, margin_rank, seen):
    """What the numbers say, and how teams in this spot actually finished."""
    lead = f"{record}, {_signed(margin)} a game"
    if verdict == "for_real":
        return f"{lead}, and the underlying numbers agree. Teams in this spot have averaged {_signed(seen['margin_later'])} a game from here, and {seen['top10_later']:.0%} stayed top 10."
    if verdict == "better_than_record":
        return f"{lead}, but the numbers rank them #{forecast_rank} going forward, not #{margin_rank}. Teams with this kind of gap have climbed about {seen['rank_so_far'] - seen['rank_later']:.0f} spots from here."
    if verdict == "sell_high":
        return f"{lead}, but the numbers rank them #{forecast_rank} going forward, not #{margin_rank}. Teams with this kind of gap have slid about {seen['rank_later'] - seen['rank_so_far']:.0f} spots from here."
    if verdict == "real_problems":
        return f"{lead}, and the underlying numbers agree. {seen['bottom10_later']:.0%} of teams in this spot stayed bottom 10 from here."
    return f"{lead}, and the numbers land about there."


def unit_sentence(off_rank, off_built, def_rank, def_built):
    describe = lambda name, rank, built: f"{name} #{rank}" + (f", built on {built}" if built else "")
    return f"So far this season: {describe('offense', off_rank, off_built)}; {describe('defense', def_rank, def_built)}."


def odds_sentence(lab, period, x):
    """One tested odds line for the team's best unit, or nothing."""
    pressure, balance = lab["splits"]["defense with top-10 QB hit rate"], lab["splits"]["offense top 10 in both pass and run"]
    weak_off, weak_def = lab["carried"]["bottom_half_offense"], lab["carried"]["bottom_half_defense"]
    if x["margin"] <= 10 and x["off"] >= 17:
        return (f"Winning teams carried by defense tend to fade: with a bottom-half offense they've slid from about #{weak_off['rank_so_far']:.0f} to #{weak_off['rank_later']:.0f} "
                f"the rest of the way, while winners with a bottom-half defense held near #{weak_def['rank_later']:.0f}.")
    if x["margin"] <= 10 and x["def"] >= 17:
        return (f"Winning with offense has been the sturdier path: winners with a bottom-half defense have held near #{weak_def['rank_later']:.0f} the rest of the way, "
                f"while winners with a bottom-half offense slid to about #{weak_off['rank_later']:.0f}.")
    if x["def"] <= 10 and x["hit"] <= 10:
        return f"Top-10 defenses with pressure like this have stayed top 10 {pressure['with']:.0%} of the time, against {pressure['without']:.0%} without it."
    if x["off"] <= 10 and x["pass"] <= 10 and x["rush"] <= 10:
        return f"Offenses top 10 both passing and running have stayed top 10 {balance['with']:.0%} of the time, against {balance['without']:.0%} for other top-10 offenses."
    if x["off"] <= 5:
        return f"Top-5 offenses at this point have stayed top 10 {hold_rate(lab, 'offense', period, x['off'])['top10']:.0%} of the time."
    if x["def"] <= 5:
        return f"Top-5 defenses at this point have stayed top 10 {hold_rate(lab, 'defense', period, x['def'])['top10']:.0%} of the time."
    return ""


def build_profiles(off, dfn, schedules, prior_off, prior_dfn, lab):
    """off/dfn: this season's team-game tables; prior_*: last season's totals by team."""
    o, d = _totals(off), _totals(dfn)
    teams = sorted(set(o) & set(d))
    games = schedules.filter((pl.col("season") == SEASON) & (pl.col("game_type") == "REG") & pl.col("result").is_not_null())
    if games.is_empty() or len(teams) < 30:
        return {"season": SEASON, "available": False}
    through = int(games["week"].max())
    record = {t: {"w": 0, "l": 0, "t": 0, "pts": 0.0, "g": 0} for t in teams}
    for r in games.iter_rows(named=True):
        for team, margin in ((r["home_team"], r["result"]), (r["away_team"], -r["result"])):
            team = TEAM_CODE_ALIASES.get(team, team)
            if team in record:
                x = record[team]
                x["pts"] += margin
                x["g"] += 1
                x["w" if margin > 0 else "l" if margin < 0 else "t"] += 1

    share = lambda t, n, den: {x: t[x][n] / max(t[x][den], 1) for x in teams}
    off_epa, def_epa = share(o, "epa", "plays"), share(d, "epa", "plays")
    margin = {x: record[x]["pts"] / max(record[x]["g"], 1) for x in teams}
    period = period_of(lab, through)
    w = lab["forecast_weights"][period]
    forecast = {
        x: w["intercept"] + w["pd"] * margin[x] + w["off"] * off_epa[x] + w["dfn"] * def_epa[x]
        + w["off_prior"] * prior_off.get(x, 0.0) + w["def_prior"] * prior_dfn.get(x, 0.0)
        for x in teams
    }
    r_off, r_def = _rank(off_epa), _rank(def_epa, best_high=False)
    r_margin, r_forecast = _rank(margin), _rank(forecast)
    r_pass, r_rush = _rank(share(o, "pass_epa", "pass_plays")), _rank(share(o, "rush_epa", "rush_plays"))
    r_succ, r_expl = _rank(share(o, "success", "plays")), _rank(share(o, "explosive", "plays"))
    r_dpass, r_drush = _rank(share(d, "pass_epa", "pass_plays"), False), _rank(share(d, "rush_epa", "rush_plays"), False)
    r_hit, r_sack = _rank(share(d, "qb_hits", "dropbacks")), _rank(share(d, "sacks", "dropbacks"))
    r_take = _rank(share(d, "turnovers", "plays"))

    profiles = {}
    for x in teams:
        rec = record[x]
        if rec["g"] < MIN_GAMES:
            continue
        verdict = verdict_for(r_margin[x], r_forecast[x])
        off_parts = {"passing": r_pass[x], "running": r_rush[x], "staying on schedule": r_succ[x], "big plays": r_expl[x]}
        def_parts = {"pass defense": r_dpass[x], "run defense": r_drush[x], "QB hits": r_hit[x], "sacks": r_sack[x], "takeaways": r_take[x]}
        built_off, built_def = built_on(r_off[x], off_parts), built_on(r_def[x], def_parts)
        record_text = f"{rec['w']}-{rec['l']}" + (f"-{rec['t']}" if rec["t"] else "")
        story = " ".join(s for s in (
            verdict_sentence(verdict, record_text, margin[x], r_forecast[x], r_margin[x], lab["verdicts"][verdict]),
            unit_sentence(r_off[x], built_off, r_def[x], built_def),
            odds_sentence(lab, period, {"margin": r_margin[x], "off": r_off[x], "def": r_def[x], "hit": r_hit[x], "pass": r_pass[x], "rush": r_rush[x]}),
        ) if s)
        profiles[x] = {
            "record": record_text,
            "story": story,
            "margin": round(margin[x], 1),
            "margin_rank": r_margin[x],
            "forecast": round(forecast[x], 1),
            "forecast_rank": r_forecast[x],
            "verdict": verdict,
            "verdict_label": VERDICT_TEXT[verdict],
            "offense_rank": r_off[x],
            "defense_rank": r_def[x],
            "offense_epa": round(off_epa[x], 3),
            "defense_epa": round(def_epa[x], 3),
            "ranks": {
                "pass": r_pass[x], "rush": r_rush[x], "success": r_succ[x], "explosive": r_expl[x],
                "def_pass": r_dpass[x], "def_rush": r_drush[x], "hits": r_hit[x], "sacks": r_sack[x], "takeaways": r_take[x],
            },
        }
    history = {k: lab[k] for k in ("seasons", "checkpoints", "hold_rates", "splits", "carried", "verdicts")}
    return {"season": SEASON, "available": True, "through_week": through, "period": period, "teams": profiles, "history": history}


def main():
    import nflreadpy as nfl

    from shared import load_cache

    with open(os.path.join(REF_DIR, "team_lab.json"), encoding="utf-8") as f:
        lab = json.load(f)
    off, dfn = team_game_tables(nfl.load_pbp(seasons=[SEASON]))
    prior_o, prior_d = team_game_tables(nfl.load_pbp(seasons=[SEASON - 1]))
    per_play = lambda t: {k: v["epa"] / max(v["plays"], 1) for k, v in _totals(t).items()}
    result = build_profiles(off, dfn, load_cache("schedules"), per_play(prior_o), per_play(prior_d), lab)
    result = {"meta": {"generated_at": datetime.now(timezone.utc).isoformat()}, **result}
    with open(OUT_PATH, "w") as f:
        json.dump(json_safe(result), f, indent=1)
    print(f"Wrote team_profiles.json: {len(result.get('teams', {}))} teams, through Week {result.get('through_week')}")


if __name__ == "__main__":
    main()

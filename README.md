# Football Forecast

**An NFL fantasy and individual-game research platform — where the disagreement with expert consensus is the product, not the ranking itself.**

🔗 **Live site: [footballforecast.net](https://footballforecast.net)**

> A personal research project built on public nflverse data: every projection is explainable (no black-box ML) and is placed directly beside the market's own number — FantasyPros consensus on the fantasy side, the real Vegas line on the game-pick side. Educational only — not betting or financial advice.

---

## What it does

Six times a week — right after Monday Night Football wraps on Tuesday (with two backup attempts), Wednesday evening after the first official injury report, and twice Friday evening after the final one — two GitHub Actions workflows pull fresh play-by-play efficiency, rosters, injuries and schedules from [nflverse](https://github.com/nflverse), re-score every relevant player and game, and commit the results. That commit is also what deploys the site, since Render redeploys on every push.

| Page | What it surfaces |
|---|---|
| **[Fantasy](https://footballforecast.net/fantasy.html)** | A frozen preseason Draft Board (VORP-based snake-draft simulation) that pivots to weekly in-season content once Week 1 starts: Weekly Rankings (start/sit), a Waiver Wire Watch for likely-undrafted players outperforming expectations, and an accountability section grading the preseason board against how the season actually went |
| **[Game Picks](https://footballforecast.net/betting.html)** | A weekly against-the-spread board: team efficiency blended from last season into this one, plus a few situational adjustments, with each pick explained in plain English beside the real line. The season-long track record is shown both against the spread and on outright winners, next to a walk-forward backtest |

Both pages pull their "why" from the same shared logic (see `scripts/shared.py` below) rather than maintaining two independent, silently-drifting copies of it.

---

## The part I am most willing to be judged on

Two weeks into the 2026 season, the Game Picks record looked respectable (60%, then 50%). I graded every frozen pick against the final scores anyway, and the receipts didn't add up: the model's projected margins had almost nothing to do with what actually happened.

The cause was a sign error. A team's defensive number is the EPA per play it *allows*, so a bad defense should help the offense facing it — but the formula subtracted it. Offense counted forward, defense counted backward, and the two mostly cancelled. Across 2025, the model's margins correlated **0.06** with the real ones and it picked outright winners **50.6%** of the time: a coin flip. The early "good" weeks were luck.

Fixing it surfaced three more problems, all fixed in the same pass:
- **Margins were about twice too large.** The model routinely disagreed with Vegas by 10–20 points. That was miscalibration, not an edge.
- **Picks and grades didn't match.** The live board picked outright winners but graded them against the spread.
- **The backtest peeked ahead.** Later games in a season leaked into earlier predictions.

Team ratings now blend last season into this one by plays played. Every constant is fitted walk-forward on 2021–2024 (`scripts/calibrate_game_model.py`) and tested on 2025, a season the fit never saw:

| 2025 holdout, 272 games | Model | Vegas |
|---|---|---|
| Correlation with final margin | 0.37 | 0.50 |
| Outright winners | 62.6% | 65.3% (the favorite) |
| Against the spread | 47.6% | — |

The honest conclusion: the model now predicts games reasonably well, and it still doesn't beat the spread — everything it knows, the market already prices in. About two-thirds of its spread picks land on underdogs, a structural side effect of margins that vary less than betting lines, not an edge. Bigger gaps from the line didn't win more often in the holdout either, so the site doesn't label any pick "strong."

What it does instead is show both records, publish the backtest on the page, and fail its own scheduled run if a regression like this ever drops the backtest's correlation below 0.2 — so the site keeps its last good data rather than quietly publishing a broken model.

---

## The part I'd point a data team at

**1. One shared "constitution," not two copies of the same logic.**
`scripts/shared.py` centralizes every signal both models need — offensive/defensive-line continuity, head-coach and coordinator continuity, and situational motivation (a team facing the coach who ran them last season, a player facing a team they used to play for) — so `betting_scan.py` and `fantasy_scan.py` compute a team's or player's "story" from the exact same facts instead of two independently-derived, driftable versions of the same idea.

**2. Frozen numbers, graded later — never re-run with hindsight.**
Before each week's games kick off, that week's model margins and lines (and the preseason Draft Board, once Week 1 starts) are archived as a point-in-time snapshot. The track record is computed from those frozen numbers against final results. When the pick rule changed from outright winners to against-the-spread, every past week was re-graded by applying the new rule to its frozen pre-kickoff margin and spread — the archive files themselves are untouched. Five displayed Week 1–2 picks changed, and the record went from 17–14 to 15–15.

**3. A real production incident, root-caused and fixed.**
A scheduled GitHub Actions run silently didn't fire — no success, no failure, no log entry, the trigger itself never fired. Root-caused to GitHub's own documented advice that cron jobs scheduled on round minutes (`:00`/`:15`/`:30`/`:45`) face the worst scheduling contention, since every other repo's cron fires at those same marks. Fixed by pinning dependency versions (an unbounded `>=` had let a scheduled run silently resolve a different package version than what was tested locally), staggering both workflows off round minutes, and adding redundant backup runs rather than relying on a single scheduled trigger succeeding.

**4. Explainable adjustments, calibrated against each other — not a black box.**
Every signal is a small, named point adjustment layered onto a base production/efficiency number, deliberately sized relative to its neighbors (a "team facing their former coach" motivation nudge is calibrated softer than a concrete "starting QB is out" injury penalty, for instance) — see the table below. Nothing here is a hidden ML weight; every number in a projection traces back to a commented constant.

---

## Signals both models are built from

| Signal | Where | Magnitude | Idea |
|---|---|---|---|
| Team efficiency rating | Game Picks | 50 pts per 1.0 EPA/play of combined gap | Offense and defense EPA/play vs. league average; last season (×0.6 offense, ×0.2 defense) blended into this season by plays played. Fitted walk-forward on 2021–24 |
| Home-field | Game Picks | +2.0 pts, 0 at neutral sites | Fitted on 2021–24 final margins |
| Starting QB out/doubtful | Game Picks | ±4.0 pts | Published estimates of a backup QB's scoring impact |
| Revenge game (lost the last meeting) | Game Picks | ±1.0 pt | Division games only — where a real prior result exists |
| Facing a former head coach | Both | ±1.0 pt (picks) / +0.5 pt (fantasy, skill positions) | The coach who ran this team last season is now coaching the opponent |
| Facing a former team | Fantasy | +1.0 pt | A player individually facing a team they used to play for |
| O-line / D-line continuity | Fantasy (scored); shown as a factor tag in Game Picks, not scored there | 1.5 × (continuity% − 60%) | Returning starters vs. a rebuilt unit, centered on a 60% baseline |
| Head coach / coordinator continuity | Fantasy | +0.2 / −0.5 pts | Scheme continuity vs. a new system to learn |
| Matchup vs. league-average defense | Fantasy | ×0.5 of the raw gap | Softened, not full-strength, points-allowed differential |
| Injury status (Questionable/Doubtful/Out) | Fantasy | −2.0 / −10.0 / −20.0 pts | Soft nudge, not a hard exclusion — an "Out" player still shows, clearly flagged |

The Game Picks situational nudges (QB out, revenge, former coach) are in points on top of the fitted rating. They aren't part of the backtest, because they can't be reconstructed historically without leaking later news.

---

## Architecture

```
GitHub Actions (2 scheduled workflows, 6 runs/week: Tue ~1/4/6am
                right after MNF, Wed ~6pm, Fri ~6/9pm ET)
        │
        ├─ Refresh nflverse cache (nflreadpy: schedules, weekly stats,
        │  rosters, depth charts, injuries, snap counts, team stats --
        │  gitignored, rebuilt fresh every run, not stored in git)
        ├─ Score: betting_scan.py + fantasy_scan.py, both reading
        │         shared.py for continuity/motivation signals
        └─ Commit JSON  ──────────────┐
                                      │ push triggers auto-deploy
                                      ▼
                            Render (static site, no build step)
                            └─ index.html / fantasy.html / betting.html
                                      │  fetch their own data/*.json
                                      ▼
                            footballforecast.net
```

**Design note:** the scan scripts are fully decoupled from the web tier. They write JSON into the repo; the commit itself is the deployment trigger. No database, no server to keep running, no runtime API calls from the browser — pages load pre-computed JSON instantly and can't fail because an upstream data source is briefly down.

**Tech:** Python (polars, [nflreadpy](https://github.com/nflverse/nflreadpy)) · vanilla JS + HTML/CSS (no frontend framework, no build step) · GitHub Actions · Render · [nflverse](https://github.com/nflverse) / [FantasyPros ECR](https://www.fantasypros.com/) / National Weather Service as data sources

---

## Repository layout

```
├── index.html                # homepage -- season-aware teaser, waiver wire sneak peek
├── fantasy.html               # Draft Board, Dream Team, Weekly Rankings, Waiver Wire, accountability
├── betting.html                # Game Picks board, storylines, ATS track record
├── styles.css
├── app.py                     # local-dev file server only -- Render serves this same folder statically
├── render.yaml                 # Render blueprint (static site, no build step)
├── scripts/
│   ├── ingest_historical.py     # refreshes the nflverse parquet cache (gitignored, rebuilt every run)
│   ├── shared.py                 # the shared "constitution" -- continuity + motivation signals
│   ├── betting_scan.py            # game picks: model margin vs. the real Vegas line, storylines, backtest
│   ├── calibrate_game_model.py     # fits the Game Picks constants walk-forward (local-only, not in CI)
│   ├── fantasy_scan.py             # Draft Board, Dream Team, Weekly Rankings, Waiver Wire
│   ├── requirements.txt             # exact-pinned deps -- CI has no lockfile, so floors aren't safe
│   └── reference/                    # hand-maintained data (coordinator continuity, stadium geocoords)
├── data/
│   ├── betting/                       # game_board.json, accountability.json, best_calls_backtest.json, archive/
│   └── fantasy/                        # draft_board.json, weekly_rankings.json, waiver_wire.json, archive/
└── .github/workflows/                   # the two scheduled scans
```

Each scan script's output is the site's entire runtime data layer — nothing is computed client-side beyond simple rendering, and nothing is queried live from a database.

---

## A note on the commit history

A large share of the commits in this repository read `chore: betting scan …` / `chore: fantasy scan …` and were made by `github-actions[bot]`. That's the pipeline working as intended — every scheduled run that finds a real diff (a completed game, an updated injury report, a fresh line) commits it, and that commit is what deploys the change. Human-authored commits carry a plain descriptive message and are where the actual feature/architecture work lives.

---

## Disclaimer

A personal research and educational project. Nothing here is betting or financial advice — projections are one tunable model's output, not a guarantee, and the site says so directly. Always do your own research.

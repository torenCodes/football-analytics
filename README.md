# Football Forecast

**An NFL fantasy and individual-game research platform — where the disagreement with expert consensus is the product, not the ranking itself.**

🔗 **Live site: [footballforecast.net](https://footballforecast.net)**

> A personal research project built on public nflverse data: every projection is explainable (no black-box ML) and is placed directly beside the market's own number — FantasyPros consensus on the fantasy side, the real Vegas line on the game-pick side. Educational only — not betting or financial advice.

---

## What it does

Each Tuesday, right after Monday Night Football wraps (with two backup attempts a few hours later in case nflverse's own data hasn't synced yet), two GitHub Actions workflows pull fresh play-by-play efficiency, rosters, injuries and schedules from [nflverse](https://github.com/nflverse), re-score every relevant player and game, and commit the results — which is also what deploys the site, since Render redeploys on every push.

| Page | What it surfaces |
|---|---|
| **[Fantasy](https://footballforecast.net/fantasy.html)** | A frozen preseason Draft Board (VORP-based snake-draft simulation) that pivots to weekly in-season content once Week 1 starts: Weekly Rankings (start/sit), a Waiver Wire Watch for likely-undrafted players outperforming expectations, and an accountability section grading the preseason board against how the season actually went |
| **[Game Picks](https://footballforecast.net/betting.html)** | A weekly model-vs-market spread board: matchup efficiency, rest, weather, rivalry/revenge and motivation context, synthesized into a plain-English storyline per game, plus a season-long against-the-spread track record |

Both pages pull their "why" from the same shared logic (see `scripts/shared.py` below) rather than maintaining two independent, silently-drifting copies of it.

---

## The part I am most willing to be judged on

The site's own backtest says its game-pick model, run against the spread over the full 2025 season (271 graded games), hit **48.7%** — worse than a coin flip, and well under the ~52.4% break-even most sportsbooks require after the vig.

That number is published on the site itself, not hidden. It's a leave-one-out EPA-per-play model: for each game, every team's offensive/defensive efficiency is recomputed with *that specific game excluded*, so the backtest can't quietly cheat by letting a game leak into its own inputs. Against the spread is also a much harder bar than picking straight-up winners — a model can correctly call the better team and still lose the ATS bet if it doesn't also correctly judge *by how much*.

This season's actual picks are running warmer — **54.8%** (17 of 31 graded games) through the first two weeks — but that's a tiny sample, and the site says so rather than leading with the flattering number. The point of publishing both, and of the archive system that makes it possible (see below), is that the model's actual hit rate is a real, checkable number, not a marketing claim.

---

## The part I'd point a data team at

**1. One shared "constitution," not two copies of the same logic.**
`scripts/shared.py` centralizes every signal both models need — offensive/defensive-line continuity, head-coach and coordinator continuity, and situational motivation (a team facing the coach who ran them last season, a player facing a team they used to play for) — so `betting_scan.py` and `fantasy_scan.py` compute a team's or player's "story" from the exact same facts instead of two independently-derived, driftable versions of the same idea.

**2. Frozen picks, graded later, never revised after the fact.**
Before each week's games kick off, that week's picks (and the preseason Draft Board, once Week 1 starts) are archived as a point-in-time snapshot. The site's track record is computed by grading those frozen archives against final results — not by re-running the model with hindsight and calling it a track record.

**3. A real production incident, root-caused and fixed.**
A scheduled GitHub Actions run silently didn't fire — no success, no failure, no log entry, the trigger itself never fired. Root-caused to GitHub's own documented advice that cron jobs scheduled on round minutes (`:00`/`:15`/`:30`/`:45`) face the worst scheduling contention, since every other repo's cron fires at those same marks. Fixed by pinning dependency versions (an unbounded `>=` had let a scheduled run silently resolve a different package version than what was tested locally), staggering both workflows off round minutes, and adding redundant backup runs rather than relying on a single scheduled trigger succeeding.

**4. Explainable adjustments, calibrated against each other — not a black box.**
Every signal is a small, named point adjustment layered onto a base production/efficiency number, deliberately sized relative to its neighbors (a "team facing their former coach" motivation nudge is calibrated softer than a concrete "starting QB is out" injury penalty, for instance) — see the table below. Nothing here is a hidden ML weight; every number in a projection traces back to a commented constant.

---

## Signals both models are built from

| Signal | Where | Magnitude | Idea |
|---|---|---|---|
| Home-field | Game Picks | +1.5 pts | Standard analytics convention |
| Starting QB out/doubtful | Game Picks | ±4.0 pts | Backtested estimates of a backup QB's scoring impact |
| Revenge game (lost the last meeting) | Game Picks | ±1.0 pt | Division games only — where a real prior result exists |
| Facing a former head coach | Both | ±1.0 pt (picks) / +0.5 pt (fantasy, skill positions) | The coach who ran this team last season is now coaching the opponent |
| Facing a former team | Fantasy | +1.0 pt | A player individually facing a team they used to play for |
| O-line / D-line continuity | Fantasy (scored); shown as a factor tag in Game Picks, not scored there | 1.5 × (continuity% − 60%) | Returning starters vs. a rebuilt unit, centered on a 60% baseline |
| Head coach / coordinator continuity | Fantasy | +0.2 / −0.5 pts | Scheme continuity vs. a new system to learn |
| Matchup vs. league-average defense | Fantasy | ×0.5 of the raw gap | Softened, not full-strength, points-allowed differential |
| Injury status (Questionable/Doubtful/Out) | Fantasy | −2.0 / −10.0 / −20.0 pts | Soft nudge, not a hard exclusion — an "Out" player still shows, clearly flagged |

---

## Architecture

```
GitHub Actions (2 scheduled workflows, ~1am/4am/6am ET Tuesdays
                -- clustered right after Monday Night Football)
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
│   ├── betting_scan.py            # game picks: model margin vs. the real Vegas line, storylines
│   ├── fantasy_scan.py             # Draft Board, Dream Team, Weekly Rankings, Waiver Wire
│   ├── requirements.txt             # exact-pinned deps -- CI has no lockfile, so floors aren't safe
│   └── reference/                    # hand-maintained data (coordinator continuity, stadium geocoords)
├── data/
│   ├── betting/                       # game_board.json, accountability.json, archive/ (frozen weekly picks)
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

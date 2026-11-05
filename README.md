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
| **[Game Picks](https://footballforecast.net/betting.html)** | A weekly board: team efficiency and scoring margin blended from last season into this one, plus the QB situation, with each pick explained beside the real line. Each card answers "is this team for real?" from how similar teams finished, and Wind Watch flags over/unders with 10+ mph of forecast wind as a live test. The season-long track record is shown by model version, next to a walk-forward backtest |
| **[Data Breakdown](https://footballforecast.net/data.html)** | Under the hood: the evidence behind Wind Watch, team verdicts and defense streaming, the models with their real weights, explorers that show each margin and Start Score being built, and a log of everything tried and rejected |

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

**Tested and left out: offensive-line continuity.** It seems obvious that a team bringing back its offensive line should get points for it. `scripts/backtest_line_continuity.py` checked that on 2021–2025 (1,359 games, each season predicted by a fit on the other four):
- A flat "returning line = more points" bonus made predictions slightly *worse* out of sample.
- Continuity didn't lift a team's offense on its own.
- What it may do is make last season's numbers a steadier guide. That version pointed the same way in all five seasons, but its improvement was too small to separate from noise, and it would have changed about 1 pick in 27.

It stays a context tag on the cards until more seasons say otherwise.

**Week 9 review: motivation, and what Vegas already knows.** `scripts/factor_lab.py` tests each situational idea on 5,320 games back to 2006, using only what was known before kickoff, two ways: does it explain this model's misses, and does it explain the closing line's misses?
- Teams out of the playoff race run about 1.5 points worse than the model expects, but the range includes zero and Vegas has moved about half of it. "On the cusp, keep trying" and bounce-backs after a blowout loss showed nothing. Revenge showed nothing across 1,896 games.
- The one factor that clearly holds up is a step down at QB: a team replacing its usual QB with a weaker track record runs 2.6 points worse than expected (95% CI 1.3 to 3.8). An equal or better replacement runs about even, which is why a flat "backup" penalty was the wrong shape. Vegas prices this in too.
- No factor gets the side that has it meaningfully past the 52.4% break-even.

`scripts/rating_lab.py` tested the standard refinements to the rating on 1,992 games, each season predicted by a fit on the others. Weighting recent games more, adjusting for opponents and discounting rushing all made predictions worse. Point differential and a shift for a new starting QB helped, and together with the step-down flag they cut squared error by 3.0 (95% CI 0.8 to 5.2). That's **Game Picks v2**, live from Week 10, with revenge and former-coach games demoted to context. On the 2025 holdout, v2 correlates 0.40 with final margins against v1's 0.37 and Vegas's 0.50. It predicts better and still doesn't beat the spread.

**Two more bugs the review turned up.** Every Friday scan rewrote that week's frozen file from a board that only lists unplayed games, which silently dropped the Thursday game: six games, Weeks 3 through 8, were never graded. `scripts/restore_thursday_games.py` put them back from their pre-kickoff commits (the record moved from 57–51 to 61–53), and games now lock one at a time. And Weekly Rankings kept ranking players on reserve lists, who drop off the injury report: Jalen Hurts was QB15 and Davante Adams WR10 with no flag.

**Fantasy, against the experts.** `scripts/fantasy_lab.py` grades our rankings against seven years of FantasyPros history. Consensus beat a stats-only draft board in 29 of 30 position-seasons from 2021 to 2025, so the 2027 board leans 75% on consensus. (A 2026-only look suggested we beat consensus at kickers and defenses. Five more seasons said that was noise.) For weekly rankings, counting a player who sat out as zero points, the biggest gain by far is knowing who's playing (+0.18 rank correlation), then Vegas team totals (+0.09). Matchups and usage add about 0.01 each, and facing a former team or a receiver's QB being out add nothing. The Start Score is now fitted on those inputs instead of hand-set: 0.43 overall against the old score's 0.22 and FantasyPros' 0.46.

**Looking for an edge.** A dozen richer team ratings built from full play-by-play (turnovers removed, garbage time removed, success rate, special teams) all landed within noise of v2, and a team's past misses don't predict its next one, so the rating sits on a plateau. The search moved to spots the rating doesn't touch:
- `scripts/wind_lab.py`: totals under-react to wind. With 10+ mph forecast at an open-air stadium, games went under 58.5% of the time (147 games, 2022 to 2026), and recorded wind points the same way in both 2006-18 and 2019-26. Promising, not proven: the 95% range reaches 50.5%, below the 52.4% break-even. So it runs as Wind Watch, a rule fixed in advance and graded live.
- `scripts/streaming_lab.py`: among defenses rostered in half of leagues or fewer, the one facing the lowest Vegas total has averaged 8.8 fantasy points a week against 7.7 for FantasyPros' top pick from the same group (87 weeks, +1.1, 95% CI 0.2 to 2.1).
- `scripts/team_lab.py`: "is this team for real?", tested. Top-10 defenses with a top-10 QB hit rate stay top 10 52% of the time against 40% without one. Takeaway-heavy defenses don't fade any faster, which goes against the usual luck story.

---

## The part I'd point a data team at

**1. One shared "constitution," not two copies of the same logic.**
`scripts/shared.py` centralizes every signal both models need — offensive/defensive-line continuity, head-coach and coordinator continuity, and situational motivation (a team facing the coach who ran them last season, a player facing a team they used to play for) — so `betting_scan.py` and `fantasy_scan.py` compute a team's or player's "story" from the exact same facts instead of two independently-derived, driftable versions of the same idea.

**2. Frozen numbers, graded later — never re-run with hindsight.**
Before each week's games kick off, that week's model margins and lines (and the preseason Draft Board, once Week 1 starts) are archived as a point-in-time snapshot. The track record is computed from those frozen numbers against final results. When the pick rule changed from outright winners to against-the-spread, every past week was re-graded by applying the new rule to its frozen pre-kickoff margin and spread — the archive files themselves are untouched. Five displayed Week 1–2 picks changed, and the record went from 17–14 to 15–15.

**3. A real production incident, root-caused and fixed.**
A scheduled GitHub Actions run silently didn't fire — no success, no failure, no log entry, the trigger itself never fired. Root-caused to GitHub's own documented advice that cron jobs scheduled on round minutes (`:00`/`:15`/`:30`/`:45`) face the worst scheduling contention, since every other repo's cron fires at those same marks. Fixed by pinning dependency versions (an unbounded `>=` had let a scheduled run silently resolve a different package version than what was tested locally), staggering both workflows off round minutes, and adding redundant backup runs rather than relying on a single scheduled trigger succeeding.

**4. Explainable inputs, each one tested — not a black box.**
Every signal is a named input with a weight fitted on past seasons or checked out of sample, and an idea that doesn't hold up on games it wasn't fit to gets dropped or demoted to a context tag, however good the story (revenge games, facing a former coach, a returning offensive line). See the table below. Every number in a projection traces back to a commented constant or the fitted weights file, and every game card shows the pieces its margin adds up from.

---

## Signals both models are built from

| Signal | Where | Magnitude | Idea |
|---|---|---|---|
| Team efficiency rating | Game Picks | 37 pts per 1.0 EPA/play of combined gap | Offense and defense EPA/play vs. league average; last season (×0.6 offense, ×0.2 defense) blended into this season by plays played. Fitted walk-forward on 2021–24 |
| Point differential | Game Picks | 0.25 pts per 1 pt/game of gap | Picks up what EPA misses on the scoreboard; last season (×0.6) blended in by games played |
| New starting QB | Game Picks | About 15 pts per 1.0 EPA/dropback at the start of a season, fading | Last year's offense, shifted by the gap between the new starter's record and last season's primary QB |
| Home-field | Game Picks | +2.0 pts, 0 at neutral sites | Fitted on 2021–24 final margins |
| Step down at QB | Game Picks | ±2.5 pts | Starter ruled out, or usual QB replaced by a weaker track record. Backtested on 2019–26 |
| Start Score | Fantasy | Fitted per position | Recent form, Vegas team total, the opposing defense, targets and carries, and availability, weighted by least squares on 2021–25 (`reference/start_score_weights.json`) |
| Consensus blend | Fantasy Draft Board, from 2027 | 75% consensus, 25% our rank | Consensus out-predicted a stats-only board in 29 of 30 position-seasons |
| Wind Watch | Game Picks, over/unders only | Lean under at 10+ mph forecast, open-air stadiums | Fixed before any live game was graded; frozen at each kickoff |
| O-line / D-line continuity | Fantasy (scored); shown as a factor tag in Game Picks, not scored there | 1.5 × (continuity% − 60%) | Returning starters (last season's regulars or its final starters) vs. a rebuilt unit, centered on a 60% baseline. Backtested for Game Picks and left out: no measurable gain |
| Head coach / coordinator continuity | Fantasy | +0.2 / −0.5 pts | Scheme continuity vs. a new system to learn |

The QB-out adjustment is in points on top of the fitted rating. The published backtest doesn't include it, because the injury report as it stood before kickoff isn't kept historically; `factor_lab.py` estimates it from who actually started.

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
                            └─ index.html / fantasy.html / betting.html / data.html
                                      │  fetch their own data/*.json
                                      ▼
                            footballforecast.net
```

**Design note:** the scan scripts are fully decoupled from the web tier. They write JSON into the repo; the commit itself is the deployment trigger. No database, no server to keep running, no runtime API calls from the browser — pages load pre-computed JSON instantly and can't fail because an upstream data source is briefly down.

**Tech:** Python (polars, [nflreadpy](https://github.com/nflverse/nflreadpy)) · vanilla JS + HTML/CSS (no frontend framework, no build step) · GitHub Actions · Render · [nflverse](https://github.com/nflverse) / [FantasyPros ECR](https://www.fantasypros.com/) / National Weather Service / [Open-Meteo](https://open-meteo.com) as data sources

---

## Repository layout

```
├── index.html                # homepage -- season-aware teaser, waiver wire sneak peek
├── fantasy.html               # Draft Board, Dream Team, Weekly Rankings, Waiver Wire, accountability
├── betting.html                # Game Picks board, storylines, ATS track record
├── data.html, data.js           # Data Breakdown -- every chart is plain SVG from the site's own JSON
├── tokens.css                 # design tokens (colors per theme, spacing, type) from the Claude Design system
├── styles.css                 # component styles -- the design system's bundle.css plus a short site-additions block
├── app.py                     # local-dev file server only -- Render serves this same folder statically
├── render.yaml                 # Render blueprint (static site, no build step)
├── scripts/
│   ├── ingest_historical.py     # refreshes the nflverse parquet cache (gitignored, rebuilt every run)
│   ├── shared.py                 # the shared "constitution" -- continuity + motivation signals
│   ├── betting_scan.py            # game picks: model margin vs. the real Vegas line, storylines, backtest
│   ├── calibrate_game_model.py     # fits the Game Picks constants walk-forward (local-only, not in CI)
│   ├── factor_lab.py                # tests situational and motivation factors against the model and Vegas (local-only)
│   ├── rating_lab.py                 # tests refinements to the rating, leave-one-season-out (local-only)
│   ├── fantasy_lab.py                 # grades the rankings against FantasyPros history, fits the Start Score (local-only)
│   ├── wind_lab.py, streaming_lab.py, team_lab.py  # the evidence behind Wind Watch, defense streaming and team verdicts (local-only)
│   ├── edge_lab.py                      # richer ratings, market rules, moneylines, what lasts (local-only)
│   ├── wind_watch.py, team_profile.py  # Wind Watch spots and "is this team for real?" (in CI)
│   ├── backtest_line_continuity.py # tests whether O-line continuity improves the picks (local-only; it doesn't, yet)
│   ├── restore_thursday_games.py, backfill_weekly_rankings.py  # one-time repairs from git history
│   ├── fantasy_scan.py             # Draft Board, Dream Team, Weekly Rankings, Waiver Wire, weekly report card
│   ├── requirements.txt             # exact-pinned deps -- CI has no lockfile, so floors aren't safe
│   └── reference/                    # hand-maintained data (coordinator continuity, stadium geocoords)
├── data/
│   ├── betting/                       # game_board.json, accountability.json, best_calls_backtest.json, archive/
│   ├── fantasy/                        # draft_board.json, weekly_rankings.json, waiver_wire.json, weekly_accountability.json, archive/
│   └── lab/                             # results of the local test scripts, read by the Data Breakdown page
└── .github/workflows/                   # the two scheduled scans
```

Each scan script's output is the site's entire runtime data layer — nothing is computed client-side beyond simple rendering, and nothing is queried live from a database.

---

## A note on the commit history

A large share of the commits in this repository read `chore: betting scan …` / `chore: fantasy scan …` and were made by `github-actions[bot]`. That's the pipeline working as intended — every scheduled run that finds a real diff (a completed game, an updated injury report, a fresh line) commits it, and that commit is what deploys the change. Human-authored commits carry a plain descriptive message and are where the actual feature/architecture work lives.

---

## Disclaimer

A personal research and educational project. Nothing here is betting or financial advice — projections are one tunable model's output, not a guarantee, and the site says so directly. Always do your own research.

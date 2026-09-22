# Football Forecast — context for the portfolio site project

This file exists for the **Data by Toren** portfolio-website Claude Code project
(repo: `DataByToren`, live at databytoren.com), which features this project as a
case study at `projects/football-forecast.html`. That page already carries the
full narrative copy — question, approach, pipeline diagram, results, what's
next — written from this project's own README and code. **Don't duplicate that
prose here.** This doc's job is the stuff that page can't know on its own:
this repo's current standing, its GitHub layout, and specific numbers worth
double-checking before the portfolio goes live. Pull from here, not from
guessing at this repo's state.

Generated 2026-09-22, from the live repository at that date. Numbers below are
snapshots — re-verify anything load-bearing (a stat quoted in headline copy)
against the live site or repo before publishing, rather than trusting this
file to still be current.

## Standing, right now

- **Live and actively maintained**, in-season (2026 NFL season, currently
  Week 3). Not a finished/archived portfolio piece — it updates on its own
  every Tuesday during the season via scheduled GitHub Actions.
- Started 2026-08-12. 42 commits total as of this snapshot; roughly 40% are
  automated `chore: … scan` commits from the scheduled pipeline itself, the
  rest are human-authored feature/architecture work.
- A full `README.md` now exists at the repo root (added 2026-09-22, same pass
  that produced this file) — point at it directly for anything technical
  rather than re-deriving from this summary. It covers architecture, the
  signal/constant table, repo layout, and an honest accuracy write-up.

## GitHub layout — the "funnel"

- **Repo:** `torenCodes/football-analytics` — **currently private.** Toren
  plans to make it public in a pre-launch pass before the portfolio site
  goes live; this doc assumes that hasn't happened yet unless told otherwise.
- **Default branch:** `main`. No other branches in normal use — all work
  (including this session's) commits straight to `main` and pushes.
- **Live site:** https://footballforecast.net (Render static site, auto-deploys
  on every push to `main`) — also reachable at the Render subdomain
  `football-analytics-9gnl.onrender.com`, but the custom domain is the one
  to link.
- **The football-forecast.html case study currently links "View on GitHub"
  to `https://github.com/torenCodes`** (the profile, not this repo) — per
  DataByToren's own `CLAUDE.md` TODO list, this is a known placeholder
  pending the repo going public. Once it is, that link should become
  `https://github.com/torenCodes/football-analytics` specifically, same as
  the pattern presumably used for The Invest Lab's repo link.
- No other cross-linking exists yet (no badge, no "view source" link from
  footballforecast.net itself back to GitHub or to the portfolio site).
  That's a real gap if a two-way funnel (portfolio → repo → live site → back
  to portfolio) is the goal — currently it's one-directional (portfolio →
  live site / GitHub profile only).

## Tools and methods (factual backup for the case-study prose)

- **Language/data stack:** Python + [polars](https://pola.rs/) (not pandas,
  despite pandas being a listed dependency — polars is the actual dataframe
  library used throughout both scan scripts). Data pulled via
  [`nflreadpy`](https://github.com/nflverse/nflreadpy), the official Python
  client for [nflverse](https://github.com/nflverse) (the open-source project
  behind `nflfastR`).
- **No ML, no black-box model.** Every projection is prior production plus a
  series of small, named, commented point adjustments (home-field, injury
  status, continuity, matchup difficulty, motivational signals). This is a
  deliberate, stated design choice, not a limitation — see the README's
  signal table for exact magnitudes.
- **No database, no backend framework in production.** The scan scripts run
  in GitHub Actions, write JSON straight into the repo, and that commit is
  what deploys the site (Render's static-site hosting just serves the repo
  folder — `render.yaml` sets `buildCommand: echo "static site, no build
  step"`). `app.py` (Flask) exists only for local dev preview and is not used
  in production.
- **Frontend is plain HTML/CSS/vanilla JS** — no framework, no build step,
  matching the portfolio site's own stack philosophy (DataByToren is also
  framework-free HTML/CSS).
- **Market/comparison baselines**, used the same way both here and in the
  case-study's "gap vs. consensus" framing: FantasyPros expert consensus
  rankings (fantasy side) and the real Vegas closing line, sourced from
  nflverse's own schedule feed at no cost (game-picks side) — no paid odds
  API.
- **One hand-maintained input:** offensive/defensive coordinator continuity
  (`scripts/reference/coordinators_2026.json`), researched each offseason
  since no free feed tracks OC/DC moves. Flagged in this repo's own code
  comments as the most likely thing to go quietly stale — worth knowing if
  the case study wants to name a concrete "what I'd improve" item (it
  already does, independently, in the "coordinator continuity... needs a
  freshness check" paragraph).

## Numbers worth verifying before publishing

The portfolio's `football-forecast.html` currently states (in "What it turned
into"): *"the value-gap in the draft table had sixteen players graded at
least 13 slots above their consensus rank, topping out at a 41-slot gap."*
DataByToren's own `CLAUDE.md` already flags this as unverified pending
launch. As of this snapshot:

**That number has moved. The current frozen preseason Draft Board (permanent
for the season — it's archived and never recomputed once Week 1 starts) shows
48 players with a market-consensus gap of at least 13 slots, topping out at
an 80.6-slot gap (Tyreek Hill, our rank 25 vs. FantasyPros consensus rank
~106).** This is the archived, stable number — it won't drift further this
season, so it's safe to cite going forward, but it does not match "sixteen…
topping out at 41" and that line needs updating (or rewording to avoid a
specific count that can look wrong to a technical reader who checks).

Other numbers that may be useful for the case study, pulled from the live
site's own self-graded track record (not previously in the case-study copy):

- **Game-pick backtest (2025 season, retrospective, leave-one-out):** 48.7%
  against the spread across 271 graded games — worse than a coin flip. This
  is published on the site itself, unprompted. It could strengthen the "I'd
  point a data team at" honesty angle the case study already gestures at in
  its "What I'd improve next" section, similar in spirit to The Invest Lab
  case study's insider-model story.
- **This season's actual picks (Weeks 1–2, live, not backtested):** 54.8%
  against the spread (17 of 31 graded games) — better, but a small sample,
  and the site is explicit about that being a small sample rather than
  leading with the flattering number.

## Open items relevant to the portfolio integration

- **Repo visibility:** still private as of this snapshot. The GitHub-link
  fix above can't happen until it's public.
- **No screenshots exist in this repo yet** (`docs/screenshots/` the way
  The Invest Lab's repo has them) — the two images the case-study page uses
  (`football-analytics_site.jpg`, `football-analytics_site_fantasy_list.jpg`)
  live only in the DataByToren repo's own `images/`, not here.
- The site's schedule just changed (2026-09-22, same day as this file) —
  scans now run ~1am/4am/6am ET on Tuesdays instead of Tuesday
  mid-morning/Wednesday, specifically so fresh data lands shortly after
  Monday Night Football rather than half a day later. Not case-study-worthy
  on its own, but relevant if the case study ever claims a specific
  refresh cadence.

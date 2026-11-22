# Dashboard

A live overview of the GitHub side of the project in one static page: `index.html` plus `live.js`, `pipeline.js`,
`activity.js` and `triage.js`. No build step, no dependencies, no server code: it reads public GitHub data in the visitor's
browser. An example running from a fork's GitHub Pages: https://clhelper.github.io/Leonida/

## What it shows

- **PR pipeline**: every open pull request as one dot in the column of its stage (draft, waiting for CI, CI running,
  CI failing, needs a rebase, CI green, ready), then `main` and the PRs merged in the last two hours. A dot moves when
  its stage changes and flies into `main` when it is merged. Hover a dot for the PR, click to open it. Too narrow for
  every column (phones), the stages become rows with `main` as a lane across. `tests/test_dashboard_pipeline.cjs`
  checks that every stage keeps room for its dots at any width.
- **Merge queue**: open PRs by CI result, the bottleneck they point at, the oldest green and failing PRs, rebases by
  author and merge chains from the coordination issue.
- **Triage**: items that open PRs from more than one author work on (possible duplicates), and with a token the next
  merge round (green, mergeable PRs that share no file, so merging them all makes none of them conflict) and the files
  the most open PRs change. `tests/test_dashboard_triage.cjs` covers it (`node --test`).
- **Live feed**: PRs opened, merged and closed, claims and other commands on the coordination issue, merge rounds and
  reverts on `main`.
- **Activity**: merged and opened PRs per hour over the last 24 hours, and who merged most.
- **Road to playable**: a weighted milestone score from the README's status table and the D3D12 tracker, the
  blockers, the graphics layer's method coverage and the work board.

## Viewing and hosting

- Locally: open `dashboard/index.html`, or serve the repository with any static server (for example
  `python -I -m http.server` in the repository root) and open `/dashboard/`.
- GitHub Pages: publish a branch that has this folder and open `/dashboard/`. The icon is `../assets/logo.png`.
- `?repo=owner/name&branch=name` points it at a fork.

## Where the data comes from

From the browser, GET requests only, except the GraphQL sweep with a token:

- `raw.githubusercontent.com`: the README (no rate limit).
- `api.github.com`: the coordination issue and its comments, the D3D12 tracker, GitHub search (open and merged PRs,
  CI results), the latest commits on `main` and the repository's star and contributor counts.
- `api.github.com/graphql`, only with a token: every open PR with its changed files, CI result and merge state, about
  a dozen read-only queries every 2 minutes while the tab is in use. It replaces the PR searches, so the pipeline
  has no 1,000-result cap and conflicts are read live. A PR's description is fetched only when the PR changed.

It parses the coordination issue and the tracker as they are written today. When their format changes, a section
shows less until the parser follows. Text from GitHub is shown with the game's title masked.

## Rate limits

Without a token GitHub allows 10 searches a minute and 60 other requests an hour per network, shared by everyone on
it. A first visit fills in over a few minutes; a line in the top bar shows what still loads and, in short, the
requests made and each limit (hover it for the reset times), and sections that wait for a limit say so with a countdown. After that the page keeps
itself current from one search a minute while someone is looking, every 5 minutes when nobody touches it, and not at
all in a hidden tab. An optional token with no scopes raises the limits and turns on the GraphQL sweep (about 11 of its 5,000 points an
hour per sweep); it is kept in the browser's local storage
and sent only to `api.github.com`.

## What it keeps in the browser

Local storage under `ldash:v2:<repo>:`: the last GitHub answers, the last pipeline load, the merged PRs of the last
24 hours and the feed, so a reload shows them at once. Clearing the site's data resets it.

## No game data

The page reads GitHub only. It contains no game data and no files from the game.

# Dashboard

A live overview of the GitHub side of the project in one static page: `index.html` plus `live.js`, `pipeline.js`,
`activity.js`, `triage.js`, `security.js` and `proof.js`. No build step, no dependencies, no server code: it reads public GitHub data in the visitor's
browser. An example running from a fork's GitHub Pages: https://clhelper.github.io/Leonida/

## What it shows

- **PR pipeline**: every open pull request as one dot in the column of its stage (draft, waiting for CI, CI running,
  CI failing, needs a rebase, CI green, ready), then `main` and the PRs merged in the last two hours. A dot moves when
  its stage changes and flies into `main` when it is merged. Hover a dot for the PR, click to open it. Too narrow for
  every column (phones), the stages become rows with `main` as a lane across. `tests/test_dashboard_pipeline.cjs`
  checks that every stage keeps room for its dots at any width.
- **Merge queue**: open PRs by CI result, the bottleneck they point at, the oldest green and failing PRs, rebases by
  author and merge chains from the coordination issue (5 shown, 20 more per click).
- **Triage**: items that open PRs from more than one author work on (possible duplicates), and with a token the next
  merge round (green, mergeable PRs that share no file, so merging them all makes none of them conflict) and the files
  the most open PRs change. `tests/test_dashboard_triage.cjs` covers it (`node --test`).
- **Security reviews**: the shared table of PRs that volunteers have read for malicious code
  ([Etoile-Bleu/leonida-reviews](https://github.com/Etoile-Bleu/leonida-reviews)): one box per verdict (do not merge,
  needs a second look, quick scan only, could not be reviewed, being reviewed, clean) that filters the list, search,
  reviewer and order filters, and an Open / Closed tag per PR once the pipeline has the open PRs. The list starts with
  5 reviews and shows 20 more per click (`security.js`; `tests/test_dashboard_security.cjs` covers its checks).
- **Live feed**: PRs opened, merged and closed, claims and other commands on the coordination issue, merge rounds and
  reverts on `main`.
- **Activity**: merged and opened PRs per hour over the last 24 hours, and who merged most.
- **Road to playable**: a weighted milestone score from the README's status table and the D3D12 tracker, the
  blockers, the graphics layer's method coverage and the work board.
- **How sure are we?**: the translator's contract rows in `docs/design.md` by how they were confirmed (a GPU, an
  LLVM/Mesa reference, a corpus count, a CPU test, or not stated), the rows still waiting for a GPU test, and from
  `docs/runtime_imports.md` the imports with no stand-in. It counts with the rules of `tools/progress_dashboard.py`
  (`proof.js`; `tests/test_dashboard_proof.cjs` covers the parsers).

## Viewing and hosting

- Locally: open `dashboard/index.html`, or serve the repository with any static server (for example
  `python -I -m http.server` in the repository root) and open `/dashboard/`.
- GitHub Pages: https://itsnotrexed.github.io/Leonida/dashboard/, which the main README links. Pages is a repository
  setting, not something a merge turns on: a maintainer enables it once under Settings → Pages → Build and deployment
  → Deploy from a branch, `main`, `/ (root)`. The empty `.nojekyll` in the repository root makes Pages serve the files
  as they are, without running every Markdown file through Jekyll. A fork works the same way at
  `<owner>.github.io/Leonida/dashboard/`. The icon is `../assets/logo.png`.
- `?repo=owner/name&branch=name` points it at a fork.

## Where the data comes from

From the browser, GET requests only, except the GraphQL sweep with a token:

- `raw.githubusercontent.com`: the README, every 30 minutes `docs/design.md` and `docs/runtime_imports.md`, and every
  15 minutes the review table's `docs/reviews.json` from `Etoile-Bleu/leonida-reviews` (no rate limit). Every field
  of the review table is type-checked and escaped before it is shown.
- `api.github.com`: the coordination issue and its comments, the D3D12 tracker, GitHub search (open and merged PRs,
  CI results), the latest commits on `main` and the repository's star and contributor counts.
- `api.github.com/graphql`, only with a token: every open PR with its changed files, CI result and merge state. A full
  sweep (about a dozen read-only queries) runs every 10 minutes and after a merge lands on `main`; in between, every
  2 minutes while the tab is in use, one query reads only the PRs updated since the last sweep (CI results and
  conflicts don't change a PR's update time, so those wait for the next full sweep). After the first, sweeps run in
  the background behind the list already shown. It replaces the PR searches, so the pipeline
  has no 1,000-result cap and conflicts are read live. A PR's description is fetched only when the PR changed.

Every request gives up after 30 seconds (a GraphQL call after 60), so a call GitHub never answers cannot keep the
page updating forever; a sweep page that times out is asked again with fewer PRs.

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
24 hours, the feed, the counts of the two documents (not the documents) and the review table packed (each scope text
once, about 0.3 MB), so a reload shows them at once. Clearing the site's data resets it.

## No game data

The page reads GitHub only. It contains no game data and no files from the game.

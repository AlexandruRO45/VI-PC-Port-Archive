# GTA6 PC Port Archive

Archived copies of three open-source projects that worked on running GTA VI on PC, saved on 2026-11-23.
No game files are included: each project kept game data out of its repository.

| Folder | Project | Original | License |
|---|---|---|---|
| [`leonida/`](leonida) | Leonida: native PC recompilation of the Xbox version | `ItsNotRexed/Leonida` (removed from GitHub on 2026-11-23) | GPL-3.0 |
| [`xcl/`](xcl) | xcl: Xbox Series compatibility layer for Windows | `ivall/xcl` | GPL-2.0 |
| [`victory/`](victory) | VIctory: an attempt at running GTA VI on PC | `Marvisak/VIctory` | GPL-3.0 |
| [`github-export/`](github-export) | each project's issues, pull requests and comments | | |

Each project folder keeps its own license file. Leonida's `third_party/rpf9-explorer` submodule
(`breadbbbb/rpf9-explorer`) has no license, so it is not included.

## Branches

The folders on `main` were merged in with their history. Each project's original history, with its original commit
hashes and file paths, is also on its own branches:

| Project | Code | Pull requests | Issues and comments |
|---|---|---|---|
| Leonida | `leonida/main` | `leonida/pr/<number>` (9,425) | `leonida/github-export` |
| xcl | `xcl/main` | `xcl/pr/<number>` (361) | `xcl/github-export` |
| VIctory | `victory/main` | `victory/pr/<number>` (145) | `victory/github-export` |

Get one project on its own:

```sh
git clone --branch leonida/main --single-branch https://github.com/Guernica4u/GTA6-PC-Port-Archive.git leonida
```

## Issues and comments

`github-export/<project>/data/` holds gzipped JSON Lines straight from the GitHub API: one issue, pull request or
comment per line. Each folder's README lists the files and counts.

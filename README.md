# GTA6 PC Port Archive

Archived copies of three open-source projects that worked on running GTA VI on PC, saved on 2026-11-23.
No game files are included: each project kept game data out of its repository.

Each project keeps its own history, so it lives on its own set of branches. This `main` branch only holds this index.

| Project | Original | License | Code | Pull requests | Issues and comments |
|---|---|---|---|---|---|
| Leonida: native PC recompilation of the Xbox version | `ItsNotRexed/Leonida` (removed from GitHub on 2026-11-23) | GPL-3.0 | `leonida/main` | `leonida/pr/<number>` (9,425) | `leonida/github-export` |
| xcl: Xbox Series compatibility layer for Windows | `ivall/xcl` | GPL-2.0 | `xcl/main` | `xcl/pr/<number>` (361) | `xcl/github-export` |
| VIctory: an attempt at running GTA VI on PC | `Marvisak/VIctory` | GPL-3.0 | `victory/main` | `victory/pr/<number>` (145) | `victory/github-export` |

Each project keeps its own license file on its code branch. Leonida's `third_party/rpf9-explorer` submodule
(`breadbbbb/rpf9-explorer`) has no license, so it is not included here.

Get one project:

```sh
git clone --branch leonida/main --single-branch https://github.com/Guernica4u/GTA6-PC-Port-Archive.git leonida
```

The `github-export` branches hold gzipped JSON Lines straight from the GitHub API: one issue, pull request or comment
per line. Their READMEs list the files and counts.

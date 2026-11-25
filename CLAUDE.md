# CLAUDE.md

Notes for Claude Code (and anyone else) working in this repository. The README
is the status log; `docs/architecture.md` is the map. Read those for detail.

## Rule #1: nothing from the game, ever

No game files, and nothing derived from them, may enter the repository: assets,
executable bytes, shader dumps, disassembly, screenshots, extracted tables, logs
with unscrubbed title names. `docs/asset-policy.md` lists what is allowed.

- `cargo xtask policy` is the authority. It is an allowlist (file type, size
  under 512 KiB, no binary, no long hex/base64 runs) and it checks **untracked
  files that are not ignored too**. `cargo test` runs it as
  `repository_passes_policy`, so a stray local file fails the tests.
- Any tool that writes an index or cache into the tree (for example
  `.codegraph/`) must be ignored by git before you run the tests.
- Never paste game-derived output (shader disassembly, title strings, memory
  dumps) into a commit, a test or a doc, even as an example.

## Build and check

The toolchain is pinned in `rust-toolchain.toml` (1.98.1, target
`x86_64-pc-windows-msvc`). The runtime only builds and runs on Windows.

```powershell
cargo xtask install-hooks                       # pre-commit runs the policy check
cargo test --workspace --locked                 # needs no game data
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo build --release -p victory
cargo build --profile diag -p victory --features diagnostics   # debugging aids
```

CI (`.github/workflows/ci.yml`) runs the two clippy lines and the tests on
`windows-latest`. Keep both clippy runs clean: the `diagnostics` feature
compiles extra code paths. Formatting: `rustfmt.toml` (max width 100).

## Layout

| Crate | Role | Size |
|-------|------|------|
| `crates/victory-gpu` | D3D12.X objects, command lists, GPU memory → host D3D12 | ~60k lines |
| `crates/victory-shader` | console shader containers → DXIL | ~43k lines |
| `crates/victory-hle` | console API implementations, host pass-through | ~19k lines |
| `crates/victory-loader` | prepare, map, bind and start the title image | ~4k lines |
| `crates/victory-formats` | texture layouts, tiling, BCPack, audio codecs | ~4k lines |
| `crates/victory-pe`, `-core`, `-pipeline`, `-game` | PE reading, config/logging, cache, title specifics | small |
| `apps/victory` | command-line front end | ~2k lines |
| `xtask` | asset-policy check, git hooks | small |

Some files are very large (`victory-shader/src/emit/lower.rs` ~12k lines,
`victory-gpu/src/host.rs` ~9k). Prefer go-to-definition / find-references or a
targeted search over reading such a file whole.

## Conventions

- Runtime behaviour is steered by `VICTORY_*` environment switches (about 250,
  read where they are used, mostly in `victory-gpu`). New behaviour goes in
  **off by default** behind a switch until a test run shows it helps. Name the
  switch in the commit message and the README.
- `unsafe_op_in_unsafe_fn` and `unused_must_use` are denied workspace-wide.
  Write a `// SAFETY:` comment on new unsafe blocks.
- Upstream is [Marvisak/VIctory](https://github.com/Marvisak/VIctory). Keep
  fork changes easy to tell apart and credit ported patches in the README.
- Results are measured on one Windows machine (see the README). Don't claim a
  speed-up or a fix without a run that shows it; say "not measured yet" instead.

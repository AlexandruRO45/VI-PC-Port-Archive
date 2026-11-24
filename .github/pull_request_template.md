## What and why

<!-- One change per PR. Link the issue or describe the bug/feature. -->

## How it was tested

<!-- Paste the relevant `cargo test` output, or the game/fixture you ran. -->

## Checklist

- [ ] Rebased on the latest `main`, no conflicts
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings` passes
- [ ] `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` passes
- [ ] `cargo test --workspace --locked` passes
- [ ] A test covers the change (or I explain why not)
- [ ] No game assets (see `docs/asset-policy.md`); the pre-commit hook is installed
- [ ] I searched open PRs and this is not a duplicate

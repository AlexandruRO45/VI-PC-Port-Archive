# Mutation testing

Every code change in this repository must come with automated tests that
would fail if the change were removed or inverted. Mutation testing checks
that mechanically: it changes the code in small ways ("mutants") and runs the
tests. A mutant that the tests still accept is a missing test, not a warning.

The policy is part of review: a pull request whose changed lines have
surviving mutants is not ready.

## Running it on a pull request

From the branch, with `main` fetched:

```powershell
git fetch origin main
git diff origin/main...HEAD > pr.diff
cargo mutants --in-diff pr.diff --timeout 300 --jobs 2
```

The repository config (`.cargo/mutants.toml`) disables the target-directory
copy and caps lints, so a stale build cannot decide the result. CI runs the
same command in `.github/workflows/mutants.yml`.

A docs-only or config-only pull request may report no mutants; that is fine.
A green run means every mutant in the changed lines was caught.

## When a mutant survives

Add a test that fails on the mutant. Prefer a table test over a single case:
`wants_wave_size` (`crates/victory-shader/src/emit/lower.rs`) exists because
the condition was buried in `Lower::new`, where `cargo-mutants` generates no
mutants. Extracting a pure helper made the rule testable and the tool found
nine mutants in it, all caught.

Do not exclude a mutant unless it is genuinely equivalent or unreachable.
Write the reason in the code next to the exclusion, or in this file if the
exclusion is repository-wide.

## Tests that need a GPU or the game

The GPU differential tests run on a real adapter when one is present and skip
otherwise. On a machine with several adapters (or a flaky display driver):

```powershell
cargo test -p victory-shader --lib -- --test-threads=1
```

Serial execution avoids the parallel GPU-test crash seen on the RX 6700 XT.
The mutation runner uses the normal test command; on CI without a GPU the
hardware tests skip, so the CPU tests are what decide a mutant.

## Known limits

- `cargo-mutants` generates no mutants inside `Lower::new` in this tree
  (large function, many branches). Extract the rule into a helper and test
  the helper.
- `crates/victory-shader/src/emit/lower.rs` exceeds the `cargo xtask policy`
  source-size limit. That is a known issue in the README and does not excuse
  missing tests.
- The baseline must be green before mutation results mean anything. Known
  baseline failures are tracked in issues #2 and #8; #93 fixes #8.

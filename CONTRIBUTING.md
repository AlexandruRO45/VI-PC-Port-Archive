# Contributing to VIctory from Vice-Tay

VIctory reviews changes from this fork as focused pull requests against
its current `main`. Give each PR one independently reviewable fix, with
its own description and tests.
[Maintainer guidance](https://github.com/Marvisak/VIctory/pull/2257#issuecomment-7163922886)

## Prepare the fix

1. Reproduce the problem on the current
   [VIctory main branch](https://github.com/Marvisak/VIctory/tree/main)
   and look for an existing issue, fix, or open PR.
2. Start a fresh branch from that revision. Port the relevant changes from
   Vice-Tay. For a commit with several changes, carry over only the parts
   needed to fix this problem and test it.
3. Use VIctory's project name, CI workflow, lint settings, and source
   conventions. Keep Vice-Tay branding, benchmarks, and fork design notes
   in the fork.
4. Follow
   [VIctory's asset policy](https://github.com/Marvisak/VIctory/blob/main/docs/asset-policy.md).
   Use synthetic fixtures. Review the diff and the PR text for game
   assets, game strings, shader names, dumps, and screenshots.

## Verify and describe the change

- Add or update a regression test that reproduces the failure and passes
  with the fix. Record the upstream base, test commands, and results.
- Review the final diff and run the relevant tests. Run
  `cargo xtask policy --staged` with the intended files staged. Use the
  [upstream Windows CI workflow](https://github.com/Marvisak/VIctory/blob/main/.github/workflows/ci.yml)
  for the full checks on the submitted revision.
- Write the commit and PR in English. Describe the problem, the change,
  and the verification. Credit the source commit or contributor and
  disclose AI assistance when relevant.
- Update the documentation with the fix. In Vice-Tay, keep usage,
  affected behavior, sources, and verification current where relevant.
  For VIctory, document the change in its PR description and update
  upstream files when the fix changes their documented behavior.
- Push the focused branch to Vice-Tay and target `Marvisak/VIctory:main`
  in the PR. Verify the base, changed files, and check results on GitHub.
  [XGameSave example](https://github.com/Marvisak/VIctory/pull/2716)

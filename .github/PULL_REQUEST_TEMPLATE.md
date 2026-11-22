## Change

Describe the observable change and the task or milestone it advances, or the
independent problem it solves. Link related issues with `Refs #123` or
`Fixes #123`. If the change touches a shared interface, name it and link any
overlapping work or PR that must land first. For a documentation patch,
identify the contributor task the edit helps someone complete.

## Verification

Give the commit or build, command and result. For GPU changes include the
adapter, backend, first mismatch if any, and capability skips. If you could not
run a check, say so and identify the missing prerequisite.

## Evidence and follow-up

For new fixtures, describe their source or generation method and license.
Identify assumptions that affect the result and link any work another
contributor can pick up.

If this changes what contributors can run, link the updated README entry or
build step and the reproduction command. For a local-data result, identify the
part the public fixture verifies.

## Checklist

- [ ] This pull request covers one reviewable behavior.
- [ ] I included the exact command and result for every check I ran.
- [ ] I considered a negative control or failure case for a new fixture.
- [ ] I updated README/docs when a capability or build requirement changed.
- [ ] I did not include game files, dumps, distribution links or secrets.

If this change was AI-assisted, say so in the description. Recent contributions
follow this practice.

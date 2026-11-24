import unittest

from pr_check import check


HEAD = "1234567890abcdef"
BODY = "**Tested**\nSynthetic fixtures\n**Not tested**\nRetail titles\n**AI use**\nNone\n"


def changed(path, binary=False):
    return (path, "modified", binary, lambda: 10)


class TestOnlyApprovalTests(unittest.TestCase):
    def test_runtime_path_requires_smoke_until_approved(self):
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        problems = check(BODY, HEAD, files, lambda _: False)
        self.assertEqual(2, len(problems))
        self.assertTrue(all("Regression run" in problem for problem in problems))
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False, test_only=True))

    def test_approval_keeps_description_and_asset_checks(self):
        files = [
            changed("crates/xcl/src/hle/d3d12x/vram.rs"),
            changed("assets/content.rpf"),
            changed("notes/attachment.txt", binary=True),
        ]
        body = "**Tested**\nSynthetic fixtures\n**AI use**\nQwen\n"
        problems = check(body, HEAD, files, lambda _: False, test_only=True)
        self.assertEqual(4, len(problems))
        for reason in ("Not tested", "AI use", "content.rpf", "attachment.txt"):
            self.assertTrue(any(reason in problem for problem in problems), problems)

    def test_test_paths_do_not_need_approval(self):
        files = [changed("crates/xcl/tests/integration.rs"), changed("crates/xcl/src/gpu/recomp/tests.rs")]
        self.assertEqual([], check(BODY, HEAD, files, lambda _: False))


class SmokeBlockCommitTests(unittest.TestCase):
    """Which `xcl-smoke` block counts as the run on the PR's head commit."""

    HEAD = "1234567890abcdef1234567890abcdef12345678"
    MAIN = "feedfacefeedfacefeedfacefeedfacefeedface"
    NEEDS_HEAD = "needs an `xcl-smoke` block made on this PR's head commit"

    def problems(self, commit):
        block = ("```\nxcl-smoke v1\ncommit: {}\ntitle: t\nseconds: 60\nframes shown: 100\n"
                 "peak commit: 100 MB\ndevice removals: 0\ncrashes: 0\nexit: ok\n```\n")
        body = BODY + "**Regression run**\n" + block.format(self.MAIN) + block.format(commit)
        files = [changed("crates/xcl/src/hle/d3d12x/vram.rs")]
        return check(body, self.HEAD, files, lambda sha: sha == self.MAIN)

    def test_a_block_on_the_head_commit_is_the_prs_run(self):
        for commit in (self.HEAD, self.HEAD[:7], self.HEAD[:12]):
            self.assertEqual([], self.problems(commit), commit)

    def test_a_short_first_word_names_no_commit(self):
        # The whole value was measured, but only its first word compared: "1 some other"
        # passed as a run on any head that starts with 1.
        for commit in ("1 some other", "12 not a hash", "123456", "123456 (uncommitted changes)"):
            problems = self.problems(commit)
            self.assertEqual(1, len(problems), (commit, problems))
            self.assertIn(self.NEEDS_HEAD, problems[0])

    def test_another_commit_is_not_the_prs_run(self):
        problems = self.problems("abcdef0123456")
        self.assertEqual(1, len(problems), problems)
        self.assertIn(self.NEEDS_HEAD, problems[0])

    def test_uncommitted_changes_are_refused(self):
        problems = self.problems(self.HEAD[:12] + " (uncommitted changes)")
        self.assertEqual(1, len(problems), problems)
        self.assertIn(self.NEEDS_HEAD, problems[0])


if __name__ == "__main__":
    unittest.main()

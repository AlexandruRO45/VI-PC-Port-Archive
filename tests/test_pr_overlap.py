"""scripts/pr_overlap.py on synthetic pull requests (no network, no game data).

    python -I tests/test_pr_overlap.py

`gh` is replaced by a stand-in that answers the two GraphQL queries and `gh pr diff` from a table of made-up pull
requests, so the tests cover the paging, the diff parser, the line comparison and the saved list without GitHub.
The `--branch` case runs the real `git diff` on a throwaway repository.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import pr_overlap as po  # noqa: E402


def hunk(path, old_start, before, removed, added, after):
    """One file's unified diff: `removed` lines replaced by `added` at old line `old_start`, with context."""
    first = old_start - len(before)
    body = [" " + x for x in before] + ["-" + x for x in removed] + ["+" + x for x in added] + [" " + x for x in after]
    return "\n".join([f"diff --git a/{path} b/{path}", "index 1111111..2222222 100644", f"--- a/{path}",
                      f"+++ b/{path}",
                      f"@@ -{first},{len(before) + len(removed) + len(after)} "
                      f"+{first},{len(before) + len(added) + len(after)} @@"] + body) + "\n"


CSS_FIX = hunk("viewer/page.html", 94, ["a", "b", "c"], ["#models-view { width: 100%; display: flex; }"],
               ["#models-view { width: 100%; }"], ["d", "e", "f"])
# The same one-line change by someone else, described in other words, plus an unrelated hunk further down.
CSS_FIX_TWIN = CSS_FIX + hunk("viewer/page.html", 600, ["x"], ["function resize() {"], ["function onResize() {"],
                              ["y"])[
    len("diff --git a/viewer/page.html b/viewer/page.html\nindex 1111111..2222222 100644\n"
        "--- a/viewer/page.html\n+++ b/viewer/page.html\n"):]
ELSEWHERE = hunk("viewer/page.html", 365, ["s"], ['<script src="/vendor/three.min.js">'],
                 ['<script src="/vendor/three.module.js">'], ["t"])

PULLS = {
    10: {"title": "Studio: show one full-width tab", "author": "first", "diff": CSS_FIX_TWIN},
    11: {"title": "Studio: load three.js from the CDN", "author": "second", "diff": ELSEWHERE},
    12: {"title": "docs: a note", "author": "third",
         "diff": hunk("docs/note.md", 3, [], ["old words here"], ["new words here"], [])},
    13: {"title": "studio: the Textures tab shows the gallery", "author": "me", "diff": CSS_FIX},
    14: {"title": "a pull request too large to diff", "author": "big", "diff": None,
         "files": ["viewer/page.html"] + [f"runtime/gen/file{i:03}.h" for i in range(230)]},
}


class FakeGh:
    """Stands in for pr_overlap.run: `gh` from PULLS, everything else (git) for real."""

    def __init__(self, pulls=PULLS, page=2):
        self.pulls, self.page, self.calls = pulls, page, []
        self.real = po.run

    def files(self, number):
        pull = self.pulls[number]
        return pull.get("files") or sorted(po.changed_lines(pull["diff"]))

    def __call__(self, program, *args):
        if program != "gh":
            return self.real(program, *args)
        self.calls.append(args)
        if args[:2] == ("pr", "diff"):
            diff = self.pulls[int(args[2])]["diff"]
            if diff is None:
                raise po.ToolError("gh pr: PullRequest.diff too_large")
            return diff
        variables = dict(a.split("=", 1) for a in args[3::2])
        query = variables.pop("query")
        start = int(variables.get("cursor", 0))
        if "pullRequest(number" in query:
            files = self.files(int(variables["number"]))
            chunk, more = files[start:start + po.PAGE], start + po.PAGE < len(files)
            reply = {"pullRequest": {"files": {"pageInfo": {"hasNextPage": more, "endCursor": str(start + po.PAGE)},
                                               "nodes": [{"path": p} for p in chunk]}}}
        else:
            numbers = sorted(self.pulls)
            nodes = []
            for number in numbers[start:start + self.page]:
                files = self.files(number)
                nodes.append({"number": number, "title": self.pulls[number]["title"],
                              "author": {"login": self.pulls[number]["author"]} if number != 12 else None,
                              "files": {"totalCount": len(files),
                                        "pageInfo": {"hasNextPage": len(files) > po.PAGE, "endCursor": str(po.PAGE)},
                                        "nodes": [{"path": p} for p in files[:po.PAGE]]}})
            reply = {"pullRequests": {"totalCount": len(numbers), "nodes": nodes,
                                      "pageInfo": {"hasNextPage": start + self.page < len(numbers),
                                                   "endCursor": str(start + self.page)}}}
        return json.dumps({"data": {"repository": reply}})


class Case(unittest.TestCase):
    def setUp(self):
        self.gh = FakeGh()
        patcher = mock.patch.object(po, "run", self.gh)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = po.main(list(argv))
            except SystemExit as e:                        # argparse usage errors
                code = e.code
        return code, out.getvalue(), err.getvalue()


class DiffParser(unittest.TestCase):
    def test_a_replaced_line_is_its_old_line_and_both_texts(self):
        entry = po.changed_lines(CSS_FIX)["viewer/page.html"]
        self.assertEqual(entry["ranges"], [(94, 94)])
        self.assertEqual(entry["lines"],
                         {"#models-view { width: 100%; display: flex; }", "#models-view { width: 100%; }"})

    def test_an_insertion_is_the_two_old_lines_it_lands_between(self):
        diff = hunk("a.py", 20, ["x", "y"], [], ["inserted line one", "inserted line two"], ["z"])
        self.assertEqual(po.changed_lines(diff)["a.py"]["ranges"], [(19, 20)])
        top = "--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n+a new first line\n first\n second\n"
        self.assertEqual(po.changed_lines(top)["a.py"]["ranges"], [(0, 1)])

    def test_two_runs_in_one_hunk_are_two_ranges_and_short_lines_are_not_compared(self):
        diff = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -5,7 +5,6 @@\n ctx\n-removed number one\n-}\n"
                " ctx\n ctx\n-removed number two\n+replacement number two\n ctx\n\\ No newline at end of file\n")
        entry = po.changed_lines(diff)["a.py"]
        self.assertEqual(entry["ranges"], [(6, 7), (10, 10)])
        self.assertEqual(entry["lines"], {"removed number one", "removed number two", "replacement number two"})

    def test_new_deleted_renamed_binary_and_quoted_files(self):
        diff = ("diff --git a/new file.txt b/new file.txt\nnew file mode 100644\n--- /dev/null\n+++ b/new file.txt\n"
                "@@ -0,0 +1,2 @@\n+first added line\n+second added line\n"
                "diff --git a/gone.txt b/gone.txt\ndeleted file mode 100644\n--- a/gone.txt\n+++ /dev/null\n"
                "@@ -1,2 +0,0 @@\n-first removed line\n-second removed line\n"
                "diff --git a/old/name.py b/new/name.py\nsimilarity index 90%\nrename from old/name.py\n"
                "rename to new/name.py\n--- a/old/name.py\n+++ b/new/name.py\n@@ -3,1 +3,1 @@\n-before the rename\n"
                "+after the rename\n"
                "diff --git a/logo.png b/logo.png\nBinary files a/logo.png and b/logo.png differ\n"
                'diff --git "a/docs/caf\\303\\251.md" "b/docs/caf\\303\\251.md"\n--- "a/docs/caf\\303\\251.md"\n'
                '+++ "b/docs/caf\\303\\251.md"\n@@ -7,1 +7,1 @@\n-a quoted path line\n+its replacement line\n')
        files = po.changed_lines(diff)
        self.assertEqual({p: e["ranges"] for p, e in files.items()},
                         {"new file.txt": [(0, 0)], "gone.txt": [(1, 2)], "new/name.py": [(3, 3)], "logo.png": [],
                          "docs/café.md": [(7, 7)]})

    def test_hunks_within_the_slack_are_the_same_lines(self):
        self.assertTrue(po.close([(94, 94)], [(94, 94)], 0))
        self.assertTrue(po.close([(94, 94)], [(97, 99)], 3))
        self.assertFalse(po.close([(94, 94)], [(98, 99)], 3))
        self.assertFalse(po.close([(94, 94)], [], 3))
        self.assertEqual(po.show_ranges([(5, 5), (1, 2), (3, 3), (9, 12), (20, 20), (30, 31), (40, 40)]),
                         "1-3, 5, 9-12, 20, ...")


class Snapshot(Case):
    def test_every_page_of_pull_requests_and_of_a_long_file_list_is_read(self):
        snapshot = po.fetch_snapshot("owner/name")
        self.assertEqual([p["number"] for p in snapshot["pulls"]], [10, 11, 12, 13, 14])
        self.assertEqual(snapshot["open"], 5)
        self.assertEqual(len(snapshot["pulls"][4]["files"]), 231)            # 100 + two more requests
        self.assertEqual(snapshot["pulls"][2]["author"], "ghost")            # a deleted account has no author
        cursors = [dict(a.split("=", 1) for a in call[3::2]).get("cursor") for call in self.gh.calls]
        self.assertEqual(cursors, [None, "2", "4", "100", "200"])
        self.assertIn(("-F", "number=14"), [call[i:i + 2] for call in self.gh.calls for i in range(len(call))])

    def test_the_saved_list_is_reused_and_refresh_reads_again(self):
        with tempfile.TemporaryDirectory(prefix="leonida pr_overlap ") as tmp:
            cache = os.path.join(tmp, "open.json")
            self.assertEqual(self.run_tool("viewer/page.html", "--cache", cache)[0], 1)
            with open(cache, "rb") as f:
                self.assertNotIn(b"\r", f.read())
            calls = len(self.gh.calls)
            code, out, _ = self.run_tool("viewer/page.html", "--cache", cache)
            self.assertEqual((code, len(self.gh.calls)), (1, calls))        # no request the second time
            self.assertIn("(from " + cache + ")", out)
            self.run_tool("viewer/page.html", "--cache", cache, "--refresh")
            self.assertGreater(len(self.gh.calls), calls)

    def test_a_saved_list_that_cannot_be_used_is_one_error_line(self):
        with tempfile.TemporaryDirectory(prefix="leonida pr_overlap ") as tmp:
            cache = os.path.join(tmp, "open.json")
            for text in ("not json", '{"repo": "ItsNotRexed/Leonida"}', '{"repo": "other/repo", "pulls": []}'):
                with open(cache, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                code, out, err = self.run_tool("viewer/page.html", "--cache", cache)
                self.assertEqual((code, out, err.count("\n")), (2, "", 1), text)
                self.assertTrue(err.startswith("pr_overlap: "), err)
                self.assertEqual(self.gh.calls, [])


class Paths(Case):
    def test_paths_list_who_changes_each_file_and_which_nobody_does(self):
        code, out, _ = self.run_tool("viewer\\page.html", "./docs/note.md", "LICENSE")
        self.assertEqual(code, 1)
        self.assertIn("5 open pull requests of ItsNotRexed/Leonida", out)
        self.assertIn("5 of them change 2 of 3 paths.", out)
        page = out.split("\nviewer/page.html\n")[1].split("\n\n")[0]
        self.assertEqual([line.split()[0] for line in page.splitlines()], ["#10", "#11", "#13", "#14"])
        self.assertIn("#12", out.split("\ndocs/note.md\n")[1].split("\n\n")[0])
        self.assertIn("No open pull request changes: LICENSE", out)
        self.assertFalse(any(call[:2] == ("pr", "diff") for call in self.gh.calls))   # no diff of yours to compare

    def test_a_folder_covers_everything_under_it_and_names_the_file(self):
        code, out, _ = self.run_tool("runtime/gen/", "--json")
        report = json.loads(out)
        self.assertEqual((code, report["pull_requests"], report["unchanged"]), (1, [14], []))
        self.assertEqual(len(report["paths"]["runtime/gen/"]), 230)
        self.assertEqual(report["paths"]["runtime/gen/"][0]["path"], "runtime/gen/file000.h")

    def test_nothing_shared_is_exit_0(self):
        code, out, _ = self.run_tool("LICENSE", "README.md")
        self.assertEqual(code, 0)
        self.assertIn("No other open pull request changes these files.", out)


class LineComparison(Case):
    def test_the_same_change_in_other_words_is_found_by_its_lines(self):
        code, out, _ = self.run_tool("--pr", "13", "--json")
        report = json.loads(out)
        self.assertEqual((code, report["checked"], report["same_lines"]), (1, "#13", [10]))
        rows = {row["pr"]: row for row in report["paths"]["viewer/page.html"]}
        self.assertNotIn(13, rows)                                           # not compared with itself
        self.assertEqual((rows[10]["verdict"], rows[10]["identical"]), ("same lines", 2))
        self.assertEqual(rows[10]["lines"], [[94, 94], [600, 600]])
        self.assertEqual((rows[11]["verdict"], rows[11]["identical"], rows[11]["lines"]),
                         ("same file", 0, [[365, 365]]))
        # A diff GitHub refuses is listed, not an error that hides the others.
        self.assertEqual(rows[14]["verdict"], "same file, lines not compared")
        self.assertEqual([row["pr"] for row in report["paths"]["viewer/page.html"]], [10, 11, 14])

    def test_the_text_report_puts_the_same_lines_first(self):
        code, out, _ = self.run_tool("--pr", "13")
        self.assertEqual(code, 1)
        self.assertIn("3 of them change 1 of 1 path; 1 on the same lines.", out)
        self.assertIn("viewer/page.html  (yours: lines 94)", out)
        first = out.split("(yours: lines 94)\n")[1].splitlines()[0]
        self.assertRegex(first,
                         r"^  #10\s+same lines, 2 identical\s+lines 94, 600\s+first: Studio: show one full-width tab$")

    def test_slack_max_diffs_and_skip_author(self):
        rows = json.loads(self.run_tool("--pr", "11", "--slack", "0", "--json")[1])["paths"]["viewer/page.html"]
        self.assertEqual({row["pr"]: row["verdict"] for row in rows},
                         {10: "same file", 13: "same file", 14: "same file, lines not compared"})
        wide = json.loads(self.run_tool("--pr", "11", "--slack", "300", "--json")[1])
        self.assertEqual(wide["same_lines"], [10, 13])
        limited = json.loads(self.run_tool("--pr", "13", "--max-diffs", "1", "--json")[1])
        self.assertEqual({row["pr"]: row["verdict"] for row in limited["paths"]["viewer/page.html"]},
                         {10: "same lines", 11: "same file, lines not compared", 14: "same file, lines not compared"})
        skipped = json.loads(self.run_tool("--pr", "13", "--skip-author", "FIRST", "--json")[1])
        self.assertEqual(skipped["pull_requests"], [11, 14])

    def test_a_branch_is_compared_through_git_diff(self):
        with tempfile.TemporaryDirectory(prefix="leonida pr_overlap ") as repo:
            env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
            env.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})

            def git(*args):
                subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                                "-c", "core.autocrlf=false", *args], check=True, capture_output=True, env=env)

            os.makedirs(os.path.join(repo, "viewer"))
            page = os.path.join(repo, "viewer", "page.html")
            lines = [f"line {i}" for i in range(1, 121)]
            lines[93] = "#models-view { width: 100%; display: flex; }"
            with open(page, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
            git("init", "-q", "-b", "main")
            git("add", "-A")
            git("commit", "-q", "-m", "base")
            git("checkout", "-q", "-b", "fix")
            lines[93] = "#models-view { width: 100%; }"
            with open(page, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
            git("commit", "-q", "-am", "fix")
            cwd = os.getcwd()
            os.chdir(repo)
            try:
                with mock.patch.dict(os.environ, env, clear=True):
                    code, out, err = self.run_tool("--branch", "--main", "main", "--json")
                    self.assertEqual(code, 1, err)
                    report = json.loads(out)
                    self.assertEqual((report["checked"], report["same_lines"]), ("HEAD against main", [10, 13]))
                    self.assertEqual(report["paths"]["viewer/page.html"][0]["identical"], 2)
                    code, out, err = self.run_tool("--branch", "main", "--main", "main")
                    self.assertEqual((code, out), (2, ""))
                    self.assertEqual(err, "pr_overlap: main against main changes no file\n")
            finally:
                os.chdir(cwd)                      # Windows cannot remove a folder that is a working directory


class Errors(Case):
    def test_usage_errors(self):
        for argv in ([], ["a.py", "--pr", "3"], ["--branch", "--pr", "3"], ["a.py", "--repo", "nope"],
                     ["a.py", "--slack", "-1"], ["a.py", "--max-diffs", "-1"], ["--pr", "0"]):
            code, out, err = self.run_tool(*argv)
            self.assertEqual((code, out), (2, ""), argv)
            self.assertIn("pr_overlap.py: error: ", err)
            self.assertEqual(self.gh.calls, [])

    def test_gh_failures_are_one_error_line(self):
        def missing(program, *args):
            raise po.ToolError(f"cannot run {program}: No such file or directory")

        with mock.patch.object(po, "run", missing):
            code, out, err = self.run_tool("--pr", "13")
        self.assertEqual((code, out, err), (2, "", "pr_overlap: cannot run gh: No such file or directory\n"))
        with mock.patch.object(po, "run", lambda *a: "<html>rate limited</html>"):
            code, out, err = self.run_tool("a.py")
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(err, "pr_overlap: gh api graphql: the reply is not the JSON this tool asked for\n")

    def test_run_reports_a_program_that_is_missing_or_fails(self):
        with mock.patch.object(po, "run", self.gh.real):
            with self.assertRaisesRegex(po.ToolError, "^cannot run leonida-no-such-program: "):
                po.run("leonida-no-such-program", "x")
            with self.assertRaisesRegex(po.ToolError, "^git rev-parse: "):
                po.run("git", "rev-parse", "--verify", "refs/heads/leonida-no-such-branch^{commit}")


if __name__ == "__main__":
    unittest.main()

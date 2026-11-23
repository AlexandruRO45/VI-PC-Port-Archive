"""The pull request template prompts for a `Depends on:` line and for whether the work was AI-assisted.

    python -I tests/test_docs_pr_template.py

The coordination issue builds its merge chains from `Depends on: #N` lines, and AGENTS.md asks a pull request to
say that it was AI-assisted; without a prompt in `.github/pull_request_template.md` those are the lines authors
leave out. This reads the template only, so rewording AGENTS.md or CONTRIBUTING.md cannot fail it. No game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(".github", "pull_request_template.md")
# A prompt is a line of its own that starts with the words an author fills in.
PROMPTS = ("Depends on:", "AI-assisted:")


def missing_prompts(template):
    """The prompts of PROMPTS that no visible line of `template` starts with."""
    visible = re.sub(r"<!--.*?-->", lambda m: "\n" * m[0].count("\n"), template, flags=re.S)
    return [p for p in PROMPTS if not re.search("^" + re.escape(p), visible, re.M)]


class PullRequestTemplateTest(unittest.TestCase):
    def test_prompts_inside_a_comment_are_missing(self):
        template = "<!--\nDepends on: #42\nAI-assisted: yes\n-->\n"
        self.assertEqual(missing_prompts(template), ["Depends on:", "AI-assisted:"])

    def test_prompt_after_a_multiline_comment_starts_its_own_line(self):
        template = "Instructions <!--\nDepends on: #42\n-->AI-assisted: yes\n"
        self.assertEqual(missing_prompts(template), ["Depends on:"])

    def test_comments_after_visible_prompts_are_allowed(self):
        template = "Depends on: <!-- #N -->\nAI-assisted: <!-- yes or no -->\n"
        self.assertEqual(missing_prompts(template), [])

    def test_template_has_both_prompts(self):
        with open(os.path.join(ROOT, TEMPLATE), encoding="utf-8") as f:
            self.assertEqual(missing_prompts(f.read()), [])

    def test_a_template_without_a_prompt_is_reported(self):
        self.assertEqual(missing_prompts("### Tested\n"), ["Depends on:", "AI-assisted:"])
        # A mention inside another sentence is not a prompt: the line has to start with it.
        self.assertEqual(missing_prompts("no Depends on: here\nAI-assisted: no\n"), ["Depends on:"])


if __name__ == "__main__":
    unittest.main()

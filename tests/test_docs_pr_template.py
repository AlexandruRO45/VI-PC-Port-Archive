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


def without_comments(template):
    """Remove comments, keeping line boundaries and treating fence contents as literal text."""
    parts = []
    fence = ""
    comment_end = 0
    # Inspect the fence info without consuming a comment opener on the same line.
    tokens = r"^ {0,3}(`{3,}|~{3,})(?=([^\r\n]*))|(^ {0,3})?(?P<opener><!--)"
    for match in re.finditer(tokens, template, re.M):
        if match.start() < comment_end:
            continue
        marker, rest, block_indent = match.group(1, 2, 3)
        if marker:
            if fence:
                if marker.startswith(fence) and not rest.strip(" \t"):
                    fence = ""
            elif marker[0] == "~" or "`" not in rest:
                fence = marker
        elif not fence:
            close = template.find("-->", match.end())
            if close >= 0:
                parts.append(template[comment_end:match.start("opener")])
                parts.append("\n" * template.count("\n", match.end(), close))
                comment_end = close + 3
            elif block_indent is not None:
                parts.append(template[comment_end:match.start()])
                return "".join(parts)
    parts.append(template[comment_end:])
    return "".join(parts)


def missing_prompts(template):
    """The prompts of PROMPTS that no visible line of `template` starts with."""
    visible = without_comments(template)
    return [p for p in PROMPTS if not re.search("^" + re.escape(p), visible, re.M)]


class PullRequestTemplateTest(unittest.TestCase):
    def test_invalid_backtick_fence_leaves_comment_opener_visible_to_checker(self):
        template = "```info` <!--\nDepends on: #42\n-->\nAI-assisted: yes\n"
        self.assertEqual(missing_prompts(template), ["Depends on:"])

    def test_unclosed_block_comment_hides_prompts_through_eof(self):
        for indent in range(4):
            with self.subTest(indent=indent):
                template = "Depends on: #1\r\n" + " " * indent + "<!-- instructions\r\nAI-assisted: yes\r\n"
                self.assertEqual(missing_prompts(template), ["AI-assisted:"])
        self.assertEqual(missing_prompts("<!--\nDepends on: #42\nAI-assisted: yes"), ["Depends on:", "AI-assisted:"])

    def test_fenced_literal_does_not_consume_a_later_placeholder(self):
        template = "~~~html\n<!--\n~~~\n\nDepends on: <!-- #N -->\nAI-assisted: yes"
        self.assertEqual(missing_prompts(template), [])

    def test_literal_openers_leave_following_prompts_visible(self):
        for example in ("Some text <!--", "Use `<!--` here", "    <!--",
                        "```html\n<!--\n```", "~~~~\n~~~\n<!--\n~~~~"):
            with self.subTest(example=example):
                self.assertEqual(missing_prompts(example + "\n\nDepends on: #1\nAI-assisted: yes"), [])

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

# cool_fox GitHub export

Issues, pull requests and comments from `codec-d/cool_fox`, exported from the GitHub REST API on 2026-11-26.

Each file in `data/` is gzipped JSON Lines: one API object per line, unchanged.

| File | Contents | Items |
|---|---|---|
| `issues_and_prs.jsonl.gz` | every issue and PR (title, body, labels, state, author) | 7 |
| `pulls.jsonl.gz` | PR details (head and base branch, merge state) | 4 |
| `issue_comments.jsonl.gz` | conversation comments on issues and PRs | 8 |
| `review_comments.jsonl.gz` | inline code-review comments | 0 |
| `labels.jsonl.gz` | labels | 10 |

Not included: PR approval reviews and images attached to comments (only their links).

`export.py` made these files (`python -I export.py`, needs the `gh` CLI signed in).

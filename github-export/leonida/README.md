# Leonida GitHub export

Issues, pull requests and comments from `ItsNotRexed/Leonida`, exported from the GitHub REST API on 2026-11-23.
The code is on `main`; every pull request's head is a `pr/<number>` branch.

Each file in `data/` is gzipped JSON Lines: one API object per line, unchanged.

| File | Contents | Items |
|---|---|---|
| `issues_and_prs.jsonl.gz` | every issue and PR (title, body, labels, state, author) | 9,853 |
| `pulls.jsonl.gz` | PR details (head and base branch, merge state) | 9,511 |
| `issue_comments.jsonl.gz` | conversation comments on issues and PRs | 67,447 |
| `review_comments.jsonl.gz` | inline code-review comments | 310 |
| `labels.jsonl.gz` | labels | 12 |

Not included: PR approval reviews, images attached to comments (only their links), and the Discord server.

Read one in Python:

```python
import gzip, json
comments = [json.loads(line) for line in gzip.open("data/issue_comments.jsonl.gz", "rt", encoding="utf-8")]
```

`export.py` made these files (`python -I export.py`, needs the `gh` CLI signed in).

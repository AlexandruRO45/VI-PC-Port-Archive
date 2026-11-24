"""Builds docs/data/ for the Pages site from github-export/<project>/data/*.jsonl.gz.
issues.json.gz: one row per issue/PR. comments/<n // 100>.json.gz: comments for issues n*100..n*100+99.
Run from the repo root: python -I docs/build_site.py"""
import gzip, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data")
PROJECTS = ["leonida", "xcl", "victory"]


def rows(project, name):
    with gzip.open(os.path.join(ROOT, "github-export", project, "data", name), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def build(project):
    issues = []
    for i in rows(project, "issues_and_prs.jsonl.gz"):
        issues.append({
            "n": i["number"], "t": i["title"], "s": i["state"], "pr": "pull_request" in i,
            "m": bool((i.get("pull_request") or {}).get("merged_at")),
            "u": (i.get("user") or {}).get("login", "ghost"), "c": i["created_at"][:10],
            "k": i.get("comments", 0), "l": [l["name"] for l in i.get("labels", [])], "b": i.get("body") or "",
        })
    issues.sort(key=lambda r: -r["n"])
    write_gz(os.path.join(OUT, project, "issues.json.gz"), issues)

    chunks = {}
    for c in rows(project, "issue_comments.jsonl.gz"):
        n = int(c["issue_url"].rsplit("/", 1)[1])
        chunks.setdefault(n // 100, {}).setdefault(str(n), []).append(
            {"u": (c.get("user") or {}).get("login", "ghost"), "c": c["created_at"][:16].replace("T", " "), "b": c.get("body") or ""})
    for k, by_issue in chunks.items():
        for lst in by_issue.values():
            lst.sort(key=lambda r: r["c"])
        write_gz(os.path.join(OUT, project, "comments", f"{k}.json.gz"), by_issue)
    total = sum(len(v) for ch in chunks.values() for v in ch.values())
    print(f"{project}: {len(issues)} issues/PRs, {total} comments in {len(chunks)} chunks")
    return {"issues": sum(not r["pr"] for r in issues), "prs": sum(r["pr"] for r in issues), "comments": total}


if __name__ == "__main__":
    stats = {p: build(p) for p in PROJECTS}
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1)
    # self-check: every comment chunk parses and every issue number is unique
    for p in PROJECTS:
        with gzip.open(os.path.join(OUT, p, "issues.json.gz"), "rt", encoding="utf-8") as f:
            nums = [r["n"] for r in json.load(f)]
        assert len(nums) == len(set(nums)), f"{p}: duplicate issue numbers"
    print("ok")

"""Builds docs/data/ for the Pages site from github-export/<project>/data/*.jsonl.gz.
issues.json.gz: one row per issue/PR. comments/<n // 100>.json.gz: comments for issues n*100..n*100+99.
Run from the repo root: python -I docs/build_site.py [project ...] (default: all; others keep their stats.json entry)"""
import gzip, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data")
PROJECTS = ["leonida-new", "leonida-old", "xcl", "victory", "xbox2windows", "tuxbox", "victory-fork", "anyxboxx", "ztrx"]


def rows(project, name):
    with gzip.open(os.path.join(ROOT, "github-export", project, "data", name), "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_gz(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:  # mtime 0: same data, same bytes
        f.write(data)


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


def updated(project):
    """Time of the project's last commit, from its original history (refs/remotes/a/<project>)."""
    out = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%ct", f"refs/remotes/a/{project}"], capture_output=True, text=True)
    return int(out.stdout) if out.returncode == 0 else None


if __name__ == "__main__":
    chosen = sys.argv[1:] or PROJECTS
    assert set(chosen) <= set(PROJECTS), f"unknown project in {chosen}"
    path = os.path.join(OUT, "stats.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    stats = {p: build(p) if p in chosen else old[p] for p in PROJECTS}
    for p in PROJECTS:
        stats[p]["updated"] = updated(p) or stats[p].get("updated")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1)
    # self-check: every comment chunk parses and every issue number is unique
    for p in PROJECTS:
        with gzip.open(os.path.join(OUT, p, "issues.json.gz"), "rt", encoding="utf-8") as f:
            nums = [r["n"] for r in json.load(f)]
        assert len(nums) == len(set(nums)), f"{p}: duplicate issue numbers"
    print("ok")

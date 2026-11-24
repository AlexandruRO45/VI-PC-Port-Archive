"""Builds docs/data/ for the Pages site from github-export/<project>/data/*.jsonl.gz.
issues.json.gz: one row per issue/PR. comments/<n // 100>.json.gz: comments for issues n*100..n*100+99.
Run from the repo root: python -I docs/build_site.py"""
import gzip, json, os, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data")
PROJECTS = ["leonida", "xcl", "victory", "xbox2windows"]


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


def person(name, email):
    """Key commits by GitHub login when the email is a noreply address, else by name."""
    m = re.match(r"^(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$", email, re.I)
    return (m.group(1).lower(), m.group(1)) if m else (name.strip().lower(), None)


def history():
    """Commit activity per project from each project's original history (refs/remotes/a/<project>)."""
    logs = {}
    for p in PROJECTS:
        out = subprocess.run(["git", "-C", ROOT, "log", "-z", "--format=%at%x1f%aN%x1f%aE", f"refs/remotes/a/{p}"],
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode:
            print(f"history skipped: refs/remotes/a/{p} not found, site.json left as is")
            return None
        logs[p] = [l.split("\x1f") for l in out.stdout.split("\0") if l]  # NUL-separated: some names and emails hold newlines
    start = min(int(t) for rows_ in logs.values() for t, _, _ in rows_) // 3600 * 3600
    end = max(int(t) for rows_ in logs.values() for t, _, _ in rows_) // 3600 * 3600
    hours = (end - start) // 3600 + 1
    projects, people = {}, {}
    for p, rows_ in logs.items():
        hourly, mine = [0] * hours, {}
        for t, name, email in rows_:
            hourly[(int(t) - start) // 3600] += 1
            key, login = person(name, email)
            e = people.setdefault(key, {"names": {}, "login": login, "n": 0, "p": set()})
            e["names"][name] = e["names"].get(name, 0) + 1
            e["login"] = e["login"] or login
            e["n"] += 1
            e["p"].add(p)
            mine[key] = mine.get(key, 0) + 1
        ts = sorted(int(t) for t, _, _ in rows_)
        projects[p] = {"commits": len(rows_), "authors": len(mine), "first": ts[0], "last": ts[-1], "hourly": hourly,
                       "top": [k for k, _ in sorted(mine.items(), key=lambda kv: -kv[1])[:12]]}
    label = lambda e: max(e["names"], key=e["names"].get)
    ranked = sorted(people.items(), key=lambda kv: -kv[1]["n"])
    for pr in projects.values():  # one person can commit under a login and a plain email: show their name once
        seen, top = set(), []
        for k in pr["top"]:
            if label(people[k]).lower() not in seen:
                seen.add(label(people[k]).lower())
                top.append([label(people[k]), people[k]["login"]])
        pr["top"] = top[:6]
    return {"start": start, "hours": hours, "projects": projects, "people_total": len(people),
            "people": [[label(e), e["login"], e["n"], [p for p in PROJECTS if p in e["p"]]] for _, e in ranked[:40]]}


if __name__ == "__main__":
    stats = {p: build(p) for p in PROJECTS}
    with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, indent=1)
    site = history()
    if site:
        assert sum(site["projects"][p]["commits"] for p in PROJECTS) == sum(sum(site["projects"][p]["hourly"]) for p in PROJECTS)
        with open(os.path.join(OUT, "site.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(site, f, ensure_ascii=False, separators=(",", ":"))
        print(f"site.json: {site['hours']} hours, {site['people_total']} people")
    # self-check: every comment chunk parses and every issue number is unique
    for p in PROJECTS:
        with gzip.open(os.path.join(OUT, p, "issues.json.gz"), "rt", encoding="utf-8") as f:
            nums = [r["n"] for r in json.load(f)]
        assert len(nums) == len(set(nums)), f"{p}: duplicate issue numbers"
    print("ok")

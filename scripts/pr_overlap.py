"""Before a claim or a pull request: which open pull requests already change the same files, and the same lines?

    python -I scripts/pr_overlap.py PATH [PATH ...]     # the open pull requests that change these files
    python -I scripts/pr_overlap.py --branch [REF]      # ... the files REF (default HEAD) changes against --main
    python -I scripts/pr_overlap.py --pr N              # ... the files pull request N changes

GitHub's search reads titles and descriptions, not diffs, so a pull request that makes the same change in other
words is not found by searching for yours. This lists them by what they change. PATH is a file, or a folder for
everything under it, relative to the repository root. With `--branch` or `--pr` there is a diff to compare, so each
pull request that shares a file is also marked:

- `same lines`: one of its hunks is within `--slack` lines (default 3, a diff's context) of one of yours. Such a
  pair conflicts or repeats when both merge. `N identical` counts the changed lines the two diffs share once
  stripped (at least 8 characters, added or removed): the same fix written twice shows up here.
- `same file`: it changes the file elsewhere.

Line numbers are those of each diff's own base. If main changed a file between the two bases they can be shifted,
so read `same lines` as "look at this one", not as proof. Only `--max-diffs` diffs are fetched (default 25, the
pull requests sharing the most files first); the others are listed as `same file, lines not compared`.

The list of open pull requests and their files is read with `gh api graphql`, 100 per request (about two seconds
each). `--cache FILE` keeps it: the file is used when it exists (`--refresh` reads again), so several checks cost
one read, and a saved file works offline. Read-only: nothing is written to GitHub or to the repository.

Exit 0 when no other open pull request changes the files, 1 when one does, 2 on a usage, `gh` or `git` error.
Standard library only; needs the GitHub CLI (`gh`, logged in) unless a `--cache` file exists, and `git` for
`--branch`.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

DEFAULT_REPO = "ItsNotRexed/Leonida"
PAGE = 100
MIN_IDENTICAL = 8      # a shorter stripped line ("}", "else:", "return 0") is shared by unrelated changes

PULLS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: %d, after: $cursor, orderBy: {field: CREATED_AT, direction: ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        number title author { login }
        files(first: %d) { totalCount pageInfo { hasNextPage endCursor } nodes { path } }
      }
    }
  }
}""" % (PAGE, PAGE)

FILES_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      files(first: %d, after: $cursor) { pageInfo { hasNextPage endCursor } nodes { path } }
    }
  }
}""" % PAGE


class ToolError(Exception):
    """`gh` or `git` could not be run, or answered with an error."""


def run(program, *args):
    try:
        r = subprocess.run([program, *args], capture_output=True)
    except OSError as e:
        raise ToolError(f"cannot run {program}: {e.strerror or e}") from None
    if r.returncode != 0:
        detail = r.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ToolError(f"{program} {args[0]}: {detail[-1] if detail else 'exit %d' % r.returncode}")
    return r.stdout.decode("utf-8", "replace")


def graphql(query, **variables):
    args = ["api", "graphql", "-f", "query=" + query]
    for key, value in variables.items():
        if value is not None:
            # -F sends a number as a number, which $number: Int! needs; -f keeps a string a string.
            args += ["-F" if isinstance(value, int) else "-f", f"{key}={value}"]
    try:
        reply = json.loads(run("gh", *args))
        return reply["data"]["repository"]
    except (ValueError, KeyError, TypeError):
        raise ToolError("gh api graphql: the reply is not the JSON this tool asked for") from None


def fetch_snapshot(repo):
    """Every open pull request of `repo` with the paths it changes, oldest first."""
    owner, _, name = repo.partition("/")
    pulls, cursor, total = [], None, 0
    while True:
        page = graphql(PULLS_QUERY, owner=owner, name=name, cursor=cursor)["pullRequests"]
        total = page["totalCount"]
        for node in page["nodes"]:
            files = node.get("files") or {"nodes": [], "pageInfo": {"hasNextPage": False}}
            paths, info = [f["path"] for f in files["nodes"]], files["pageInfo"]
            # A list cut at 100 files would hide an overlap, which is the one thing this tool is for.
            while info["hasNextPage"]:
                more = graphql(FILES_QUERY, owner=owner, name=name, number=node["number"],
                               cursor=info["endCursor"])["pullRequest"]["files"]
                paths += [f["path"] for f in more["nodes"]]
                info = more["pageInfo"]
            pulls.append({"number": node["number"], "title": node["title"],
                          "author": (node.get("author") or {}).get("login") or "ghost", "files": paths})
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {"repo": repo, "fetched_at": now, "open": total, "pulls": pulls}


def load_snapshot(repo, cache, refresh):
    """(snapshot, came from the cache file)."""
    if cache and not refresh and os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                snapshot = json.load(f)
            if snapshot["repo"] != repo:
                raise ToolError(f"{cache} holds {snapshot['repo']}, not {repo}: pass --refresh or another --cache")
            for pull in snapshot["pulls"]:
                pull["number"], pull["title"], pull["author"], list(pull["files"])
            return snapshot, True
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ToolError(f"cannot use {cache}: {e}: delete it or pass --refresh") from None
    snapshot = fetch_snapshot(repo)
    if cache:
        try:
            with open(cache, "w", encoding="utf-8", newline="\n") as f:
                json.dump(snapshot, f)
        except OSError as e:
            raise ToolError(f"cannot write {cache}: {e.strerror or e}") from None
    return snapshot, False


def unquote(path):
    """A path as `git diff` prints it: quoted, with octal escapes, when it holds a byte outside printable ASCII."""
    if len(path) < 2 or path[0] != '"' or path[-1] != '"':
        return path
    out, i, body = bytearray(), 0, path[1:-1]
    simple = {"n": 10, "t": 9, "r": 13, "a": 7, "b": 8, "f": 12, "v": 11, '"': 34, "\\": 92}
    while i < len(body):
        if body[i] == "\\" and re.fullmatch(r"[0-7]{3}", body[i + 1:i + 4]):
            out.append(int(body[i + 1:i + 4], 8))
            i += 4
        elif body[i] == "\\" and body[i + 1:i + 2] in simple:
            out.append(simple[body[i + 1]])
            i += 2
        else:
            out += body[i].encode("utf-8")
            i += 1
    return out.decode("utf-8", "replace")


HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
SAME_NAME = re.compile(r'^diff --git ("?)a/(.*) \1b/\2$')


def changed_lines(diff):
    """{path: {"ranges": [(first, last), ...], "lines": {stripped changed line, ...}}} of a unified diff.

    A range is in the old file's line numbers, both ends included. Added lines have no old number, so an insertion
    is the pair of old lines it lands between; (0, 0) is a new file. A renamed file is listed under its new name,
    a deleted one under its old name, and a binary or mode-only change with no ranges."""
    files, current, in_hunk, old, removed, added = {}, None, False, 0, [], False

    def flush():
        nonlocal removed, added
        if removed:
            current["ranges"].append((removed[0], removed[-1]))
        elif added:
            current["ranges"].append((old - 1, old) if old else (0, 0))
        removed, added = [], False

    def finish():
        if current is None:
            return
        flush()
        names = [n[2:] for n in (current["new"], current["old"]) if n and n != "/dev/null"]
        path = names[0] if names else current["header"]
        if path:
            entry = files.setdefault(path, {"ranges": [], "lines": set()})
            entry["ranges"] += current["ranges"]
            entry["lines"] |= {text for text in current["lines"] if len(text) >= MIN_IDENTICAL}

    for line in diff.splitlines():
        if line.startswith("diff --git ") or (current is None and line.startswith("--- ")):
            finish()
            # A binary or mode-only change has no ---/+++ pair; the header names the file when both sides agree.
            same = SAME_NAME.match(line)
            header = (unquote('"' + same[2] + '"') if same[1] else same[2]) if same else None
            current, in_hunk = {"header": header, "old": None, "new": None, "ranges": [], "lines": set()}, False
        if current is None:
            continue
        if not in_hunk and line.startswith(("--- ", "+++ ")):
            current["old" if line[0] == "-" else "new"] = unquote(line[4:].split("	")[0])
        elif HUNK.match(line):
            flush()
            in_hunk, old = True, int(HUNK.match(line)[1])
        elif in_hunk and line.startswith("-"):
            removed.append(old)
            old += 1
            current["lines"].add(line[1:].strip())
        elif in_hunk and line.startswith("+"):
            added = True
            current["lines"].add(line[1:].strip())
        elif in_hunk and not line.startswith("\\"):      # not "\ No newline at end of file"
            flush()
            old += 1
    finish()
    return files


def close(mine, theirs, slack):
    return any(a <= d + slack and c <= b + slack for a, b in mine for c, d in theirs)


def show_ranges(ranges, limit=4):
    merged = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    text = [str(a) if a == b else f"{a}-{b}" for a, b in merged[:limit]]
    return ", ".join(text) + (", ..." if len(merged) > limit else "")


def repo_path(path, root):
    """PATH as the repository spells it: forward slashes, relative to the root; a folder ends in '/'."""
    full = os.path.abspath(path)
    folder = path.endswith(("/", "\\")) or os.path.isdir(full)
    if root and os.path.normcase(full).startswith(os.path.normcase(os.path.join(root, ""))):
        path = os.path.relpath(full, root)
    path = path.replace("\\", "/").rstrip("/")
    while path.startswith("./"):
        path = path[2:]
    return path + "/" if folder else path


def find_overlaps(snapshot, mine, skip, skip_authors, slack, max_diffs, their_diff):
    """mine: {path or 'folder/': None or changed_lines() entry}. Returns ({path: [row, ...]}, paths nobody changes)."""
    folders = [p for p in mine if p.endswith("/")]
    hits = {}                                           # pull number -> (pull, [(their path, my key)])
    for pull in snapshot["pulls"]:
        if pull["number"] in skip or pull["author"].lower() in skip_authors:
            continue
        shared = []
        for path in pull["files"]:
            key = path if path in mine else next((f for f in folders if path.startswith(f)), None)
            if key is not None:
                shared.append((path, key))
        if shared:
            hits[pull["number"]] = (pull, shared)
    compare = any(entry is not None for entry in mine.values())
    order = sorted(hits, key=lambda n: (-len(hits[n][1]), n))
    diffs = {}
    for number in order[:max_diffs] if compare else []:
        try:
            diffs[number] = changed_lines(their_diff(number))
        except ToolError:
            pass                                        # e.g. GitHub refuses a very large diff: listed, not compared
    report = {}
    for number in sorted(hits):
        pull, shared = hits[number]
        for path, key in shared:
            row = {"pr": number, "author": pull["author"], "title": pull["title"], "verdict": "same file",
                   "identical": 0, "lines": []}
            if key != path:
                row["path"] = path
            ours, theirs = mine[key], diffs.get(number, {}).get(path)
            if compare and number not in diffs:
                row["verdict"] = "same file, lines not compared"
            elif ours is not None and theirs is not None:
                row["lines"] = [list(r) for r in sorted(theirs["ranges"])]
                row["identical"] = len(ours["lines"] & theirs["lines"])
                if close(ours["ranges"], theirs["ranges"], slack):
                    row["verdict"] = "same lines"
            report.setdefault(key, []).append(row)
    for rows in report.values():
        rows.sort(key=lambda r: (r["verdict"] != "same lines", r["verdict"].endswith("not compared"), -r["identical"],
                                 r["pr"]))
    return report, sorted(p for p in mine if p not in report)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("paths", nargs="*", metavar="PATH", help="files, or folders for everything under them")
    ap.add_argument("--branch", nargs="?", const="HEAD", metavar="REF",
                    help="check what REF (default HEAD) changes against --main, with the line comparison")
    ap.add_argument("--pr", type=int, metavar="N", help="check what pull request N changes, with the line comparison")
    ap.add_argument("--main", default="origin/main", help="the branch --branch is compared with (default origin/main)")
    ap.add_argument("--repo", default=DEFAULT_REPO, metavar="OWNER/NAME", help=f"default {DEFAULT_REPO}")
    ap.add_argument("--cache", metavar="FILE", help="keep the list of open pull requests here and reuse it")
    ap.add_argument("--refresh", action="store_true", help="read the list again even when --cache exists")
    ap.add_argument("--skip-author", action="append", default=[], metavar="LOGIN",
                    help="leave out this author's pull requests, e.g. your own (repeatable)")
    ap.add_argument("--slack", type=int, default=3, help="hunks this many lines apart count as the same lines (3)")
    ap.add_argument("--max-diffs", type=int, default=25, help="pull request diffs fetched for the line comparison (25)")
    ap.add_argument("--json", action="store_true", help="one JSON report instead of text")
    args = ap.parse_args(argv)
    if (bool(args.paths) + (args.branch is not None) + (args.pr is not None)) != 1:
        ap.error("give PATHs, or --branch, or --pr: exactly one of the three")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        ap.error("--repo is OWNER/NAME")
    if args.slack < 0 or args.max_diffs < 0:
        ap.error("--slack and --max-diffs are not negative")
    if args.pr is not None and args.pr < 1:
        ap.error("--pr is a pull request number")

    def pr_diff(number):
        return run("gh", "pr", "diff", str(number), "--repo", args.repo)

    try:
        skip = set()
        if args.branch is not None:
            # core.quotepath=off keeps a non-ASCII path readable; -U3 is the context `gh pr diff` uses too.
            mine = changed_lines(run("git", "-c", "core.quotepath=off", "diff", "--no-color", "--no-ext-diff",
                                     "-U3", f"{args.main}...{args.branch}"))
            what = f"{args.branch} against {args.main}"
        elif args.pr is not None:
            mine, skip, what = changed_lines(pr_diff(args.pr)), {args.pr}, f"#{args.pr}"
        else:
            try:
                root = os.path.normpath(run("git", "rev-parse", "--show-toplevel").strip())
            except ToolError:
                root = None                              # not in a checkout: the paths are taken as written
            mine, what = {repo_path(p, root): None for p in args.paths}, "the paths given"
        if not mine:
            print(f"pr_overlap: {what} changes no file", file=sys.stderr)
            return 2
        snapshot, cached = load_snapshot(args.repo, args.cache, args.refresh)
        report, free = find_overlaps(snapshot, mine, skip, {a.lower() for a in args.skip_author}, args.slack,
                                     args.max_diffs, pr_diff)
    except ToolError as e:
        print(f"pr_overlap: {e}", file=sys.stderr)
        return 2

    pulls = {row["pr"] for rows in report.values() for row in rows}
    same = {row["pr"] for rows in report.values() for row in rows if row["verdict"] == "same lines"}
    if args.json:
        print(json.dumps({"repo": snapshot["repo"], "fetched_at": snapshot["fetched_at"], "cached": cached,
                          "open": len(snapshot["pulls"]), "checked": what, "paths": report, "unchanged": free,
                          "pull_requests": sorted(pulls), "same_lines": sorted(same)}, indent=2))
        return 1 if pulls else 0
    print(f"{len(snapshot['pulls']):,} open pull requests of {snapshot['repo']}, read {snapshot['fetched_at']}"
          f"{' (from ' + args.cache + ')' if cached else ''}; checked: {what}.")
    if not pulls:
        print("No other open pull request changes these files.")
        return 0
    print(f"{len(pulls)} of them change{'s' if len(pulls) == 1 else ''} {len(report)} of {len(mine)} "
          f"path{'' if len(mine) == 1 else 's'}" + (f"; {len(same)} on the same lines" if same else "") + ".")
    for path in sorted(report):
        ours = mine[path]
        print(f"\n{path}" + (f"  (yours: lines {show_ranges(ours['ranges'])})" if ours and ours["ranges"] else ""))
        for row in report[path]:
            verdict = row["verdict"] + (f", {row['identical']} identical" if row["identical"] else "")
            where = (row.get("path") or "") + (" lines " + show_ranges([tuple(r) for r in row["lines"]])
                                               if row["lines"] else "")
            title = row["title"] if len(row["title"]) <= 72 else row["title"][:69] + "..."
            print(f"  #{row['pr']:<5} {verdict:<30} {where.strip():<24} {row['author']}: {title}".rstrip())
    if free:
        print("\nNo open pull request changes: " + ", ".join(free))
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

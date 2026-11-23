"""Before a merge round: is a pull request already on main, and would merging it again duplicate text?

    python -I scripts/merge_check.py [--main REF] [--json] [--min-length N] PR=REF [PR=REF ...]

PR is the pull request number and REF its head (a branch, a fetched `refs/pull/PR/head`, a commit). For each one,
against `--main` (default `origin/main`), using one history scan for the whole batch:

- `merges`: first-parent commits on main whose subject is "Merge pull request #PR" or ends with " (#PR)" (squash
  merges; "Revert ..." subjects are ignored), newest first. A number that is already there is the case the repeated
  merges come from: a rebased copy merged again keeps both copies of what it added.
- `commits`: the PR's commits (merge base..REF) and how many `git cherry` finds on main already (same patch).
- `duplicates`: runs of three or more consecutive lines the merge would add (each at least `--min-length`
  characters once stripped) that already stand in the same order in the same file on main. This is what a second
  merge leaves behind as a doubled paragraph, table or function, whether or not the commits match by patch; a single
  repeated line is not reported, since code and tables repeat lines legitimately. Only text files (.md, .txt,
  .rst) make the verdict `duplicates`; a block in code is listed, since equal lines in two functions are ordinary.
- `conflict`: whether `git merge-tree` reports a conflict (the duplicate check is then on the conflicted tree).
- `on_main_outside_conflicts`: with a conflict, whether the merge with every conflicting hunk resolved to main's side
  (`merge-tree -X ours`) leaves main as it is, so that the PR adds nothing outside its conflicts: what a PR that
  landed inside another merge looks like once main has edited its lines since. It stays a `conflict`, since a PR
  whose every change conflicts looks the same. `null` without a conflict, and where git cannot make that merge.

Verdict per PR: `already-merged` (a prior merge exists and either every commit is on main or the conflict-free merged
tree matches main, as after a multi-commit squash merge), `already-on-main` (no merge of this PR is found, but the
conflict-free merged tree matches main: everything it adds landed another way, such as a stacked PR squash-merged
inside the PR above it, so merging it changes nothing), `duplicates`, `conflict` or `clean`.
Exit 0 when every PR is clean, 1 when any is not, 2 on a usage or git error. Read-only: `merge-tree --write-tree`
writes only objects, no refs. Standard library only; needs git 2.38 or newer for `merge-tree --write-tree`.
"""
import argparse
import codecs
import json
import re
import subprocess
import sys
from collections import Counter

MERGE_SUBJECT = re.compile(r"^Merge pull request #(\d+)\b")
SQUASH_SUBJECT = re.compile(r"(?:^|\s)\(#(\d+)\)\s*$")
REVERT_SUBJECT = re.compile(r"^Revert\b", re.IGNORECASE)
# Where a doubled block is the repeated-merge damage. In code, three equal lines in two functions are ordinary, so a
# block there is listed but does not change the verdict.
TEXT_SUFFIXES = (".md", ".txt", ".rst")


class GitError(Exception):
    pass


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True)
    # Decode without universal-newline translation: a bare CR inside a Git line is content.
    r.stdout = r.stdout.decode("utf-8", "replace")
    r.stderr = r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip() or r.returncode}")
    return r


def _merged_pr_number(subject):
    m = MERGE_SUBJECT.match(subject)
    if m:
        return int(m.group(1))
    if REVERT_SUBJECT.match(subject):
        return None
    m = SQUASH_SUBJECT.search(subject)
    return int(m.group(1)) if m else None


def prior_merges(repo, main, numbers, depth):
    out = git(repo, "log", main, "--first-parent", f"-n{depth}", "--format=%H%x09%s").stdout
    hits = {number: [] for number in numbers}
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        number = _merged_pr_number(subject)
        if number in hits:
            hits[number].append(sha[:12])
    return hits


def cherry(repo, main, ref):
    """(commits on the PR side, those whose patch is already on main)."""
    lines = [l for l in git(repo, "cherry", main, ref).stdout.splitlines() if l.strip()]
    return len(lines), sum(1 for l in lines if l.startswith("-"))


def rev_tree(repo, rev):
    return git(repo, "rev-parse", f"{rev}^{{tree}}").stdout.strip()


def merged_tree(repo, main, ref):
    r = git(repo, "merge-tree", "--write-tree", main, ref, check=False)
    if r.returncode not in (0, 1) or not r.stdout.strip():
        raise GitError(f"git merge-tree {main} {ref}: {r.stderr.strip() or r.returncode}")
    return r.stdout.split()[0], r.returncode == 1


def merged_tree_keeping_main(repo, main, ref):
    """The merged tree with every conflicting hunk resolved to main's side, or None where git cannot make it.

    `-X ours` settles content conflicts only; a conflict it leaves (a file deleted on one side, for one) exits 1, and
    a git whose `merge-tree` takes no `-X` exits with a usage error. Either way there is no such tree to compare.
    """
    r = git(repo, "merge-tree", "--write-tree", "-X", "ours", main, ref, check=False)
    return r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None


def patch_destination(header):
    """Decode Git's +++ header. An unquoted path containing spaces can have a trailing tab."""
    name = header[4:].removesuffix("\t")
    if name.startswith('"') and name.endswith('"'):
        # Git's octal escapes describe filesystem bytes, not Unicode code points.
        name = codecs.escape_decode(name[1:-1].encode("utf-8"))[0].decode("utf-8", "surrogateescape")
    return name[2:] if name.startswith("b/") else None  # Deleted files say /dev/null.


def duplicates(repo, main, tree, min_length, block=3):
    """{path: [line, ...]} for runs of `block` or more consecutive added lines (each at least `min_length` characters
    once stripped; blank lines skipped) that already stand in the same order in the same file on main: a paragraph,
    table or function body merged twice. A single repeated line is not reported: code and tables repeat lines
    legitimately."""
    found = {}
    hunks, path, added = {}, None, None
    patch = git(repo, "-c", "core.quotePath=true", "diff", "--unified=0", "--no-color", "--no-renames",
                "--src-prefix=a/", "--dst-prefix=b/", "--no-relative", "--no-ext-diff", "--no-textconv",
                main, tree).stdout
    for line in patch.split("\n"):
        if line.startswith("diff --git "):
            path, added = None, None
        elif added is None and line.startswith("+++ "):
            path = patch_destination(line)
        elif line.startswith("@@ ") and path:
            added = []
            hunks.setdefault(path, []).append(added)
        elif added is not None and line.startswith("+"):
            text = line[1:].strip()
            if text:
                added.append(text)
    def windows(rev, path):
        r = git(repo, "show", f"{rev}:{path}", check=False)
        lines = [l.strip() for l in r.stdout.split("\n") if l.strip()] if r.returncode == 0 else []
        return r.returncode == 0, Counter(tuple(lines[i:i + block]) for i in range(len(lines) - block + 1))

    for path, groups in hunks.items():
        exists, before = windows(main, path)
        if not exists:
            continue  # a new file: nothing to duplicate
        _, after = windows(tree, path)
        same = []
        for added in groups:
            marked = [False] * len(added)
            for i in range(len(added) - block + 1):
                run = tuple(added[i:i + block])
                # Doubled: on main already, and more copies after the merge. A block the PR moves (deleted where
                # it was, added elsewhere) keeps its count and is not reported.
                if all(len(x) >= min_length for x in run) and before[run] and after[run] > before[run]:
                    for j in range(i, i + block):
                        marked[j] = True
            same += [x for x, m in zip(added, marked) if m]
        if same:
            found[path] = same
    return found


def check(repo, main, number, ref, min_length, merges):
    total, on_main = cherry(repo, main, ref)
    tree, conflict = merged_tree(repo, main, ref)
    main_tree = rev_tree(repo, main)
    unchanged = not conflict and tree == main_tree
    outside = None
    if conflict:
        kept = merged_tree_keeping_main(repo, main, ref)
        outside = None if kept is None else kept == main_tree
    dups = {} if unchanged else duplicates(repo, main, tree, min_length)
    text_dups = {p: v for p, v in dups.items() if p.lower().endswith(TEXT_SUFFIXES)}
    if merges and (total == on_main or unchanged):
        verdict = "already-merged"
    elif unchanged:
        # No merge of its own, yet merging changes nothing: a stacked PR that went in with the one above it.
        verdict = "already-on-main"
    elif text_dups:
        verdict = "duplicates"
    elif conflict:
        verdict = "conflict"
    else:
        verdict = "clean"
    return {"pr": number, "ref": ref, "verdict": verdict, "merges": merges,
            "commits": {"total": total, "on_main": on_main}, "conflict": conflict,
            "on_main_outside_conflicts": outside,
            "duplicates": {p: len(v) for p, v in dups.items()},
            "duplicate_samples": {p: v[:3] for p, v in dups.items()}}


def parse_pair(text):
    number, sep, ref = text.partition("=")
    if not sep or not number.lstrip("#").isdigit() or not ref:
        raise argparse.ArgumentTypeError(f"expected PR=REF, got {text!r}")
    return int(number.lstrip("#")), ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("pairs", nargs="+", type=parse_pair, metavar="PR=REF")
    ap.add_argument("--main", default="origin/main", help="the branch the round merges into (default origin/main)")
    ap.add_argument("--repo", default=".", help="repository folder (default: the current one)")
    ap.add_argument("--min-length", type=int, default=40, help="shortest stripped line counted as a duplicate (40)")
    ap.add_argument("--depth", type=int, default=5000, help="first-parent commits searched for earlier merges (5000)")
    ap.add_argument("--json", action="store_true", help="one JSON report instead of text")
    args = ap.parse_args(argv)
    try:
        history = prior_merges(args.repo, args.main, (n for n, _ in args.pairs), args.depth)
        results = [check(args.repo, args.main, n, ref, args.min_length, history[n]) for n, ref in args.pairs]
    except GitError as e:
        print(f"merge_check: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            c = r["commits"]
            print(f"#{r['pr']} {r['verdict']}: {c['on_main']}/{c['total']} commit(s) already on main"
                  + (", merged tree matches main"
                     if r["verdict"] == "already-merged" and c["on_main"] != c["total"] else "")
                  + (", merged tree matches main, though no merge of it is found"
                     if r["verdict"] == "already-on-main" else "")
                  + (f", merged before at {', '.join(r['merges'])}" if r["merges"] else "")
                  + (", conflicts" if r["conflict"] else "")
                  + (", and outside them merging changes nothing (already on main another way?)"
                     if r["on_main_outside_conflicts"] else ""))
            for path, n in sorted(r["duplicates"].items()):
                note = "" if path.lower().endswith(TEXT_SUFFIXES) else " (code: listed, not counted)"
                print(f"  {path}: {n} added line(s) already there in the same order{note}, "
                      f"e.g. {r['duplicate_samples'][path][0][:90]!r}")
    return 0 if all(r["verdict"] == "clean" for r in results) else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

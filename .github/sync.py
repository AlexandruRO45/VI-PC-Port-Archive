"""Keep the archive in step with the projects that are still on GitHub.
Run from the repo root: python -I .github/sync.py [--push] [project ...]  (no projects: all of them)

For each project: fetch its code and pull requests, refresh its folder on main, export its issues and comments
again, and rebuild the site data. Nothing is ever deleted: a project that can't be fetched (taken down, renamed,
made private) is skipped and its archived copy stays as it is.
--push also pushes each project's branches to origin (<p>/main, <p>/pr/<n>, <p>/branch/<name>, <p>/github-export)
and opens an issue when the README part the site's progress steps come from has changed.
Committing main is left to the caller (.github/workflows/sync.yml)."""
import datetime, gzip, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# folder: (GitHub repo, README line where its "how far it gets" part starts)
PROJECTS = {
    "leonida-new": ("ubiquitarianism/leonida", r"## Project status"),
    "xcl": ("ivall/xcl", r"## Games"),
    "victory": ("Marvisak/VIctory", r"## Status"),
    "xbox2windows": ("R0Y00/Xbox2Windows", r"> \*\*Current status\.\*\*"),
    "tuxbox": ("brcly/TuxBox", r"\*\*GTA VI \(ProjectScarlett\)\*\*"),
    "victory-gta6": ("fxceless/victory-fork",  # renamed from victory-gta6; folder and branches keep the old name
                     r"## Status of this fork"),
}
PUSH_BUDGET = 300  # seconds of branch pushing per project per run; the rest waits for the next run
TODAY = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def git(*args, check=True):
    p = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True, encoding="utf-8")
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {p.stderr.strip()}")
    return p


def status_part(text, start):
    """The README lines from `start` up to the next ## heading."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        if re.match(start, line):
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
            return "\n".join(lines[i:end]).strip()
    return None


assert status_part("x\n## Status (1 Dec)\nok\n\n## Next\nno", r"## Status") == "## Status (1 Dec)\nok"


def fetch(p, repo):
    """Fetch every branch and pull request of `repo` into refs/sync/<p>/. Returns its default branch, or None."""
    url = f"https://github.com/{repo}.git"
    head = git("ls-remote", "--symref", url, "HEAD", check=False)
    m = re.search(r"^ref: refs/heads/(\S+)\tHEAD", head.stdout, re.M)
    if head.returncode or not m:
        return None
    git("fetch", "-q", "--no-tags", "--force", url, f"+refs/heads/*:refs/sync/{p}/heads/*", f"+refs/pull/*/head:refs/sync/{p}/pull/*")
    return m.group(1)


def refresh(p, src):
    """Replace folder <p>/ on main with the tree of `src`."""
    git("rm", "-r", "-q", "-f", "--ignore-unmatch", "--", p)
    git("read-tree", f"--prefix={p}/", "-u", src)
    for line in git("ls-files", "-s", "--", p).stdout.splitlines():
        if line.startswith("160000"):  # submodules point at other repos and are not archived
            git("rm", "-q", "--cached", "--", line.split("\t", 1)[1])
    git("update-ref", f"refs/remotes/a/{p}", src)  # docs/build_site.py reads the last-change date from here


def merge(path, old):
    """Union of the new export and the last one, by id: issues and comments deleted upstream stay archived."""
    items = {}
    for blob in (old, open(path, "rb").read()):
        for line in gzip.decompress(blob).decode("utf-8").splitlines() if blob else []:
            if line:
                items[json.loads(line)["id"]] = line
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:  # mtime 0: same items, same bytes
        f.write("".join(items[k] + "\n" for k in sorted(items)).encode("utf-8"))
    return len(items)


def export(p):
    """Export issues, PRs and comments again and merge them into the last export; on failure keep the last one."""
    d = os.path.join(ROOT, "github-export", p)
    data = os.path.join(d, "data")
    for f in os.listdir(data):
        if f.endswith((".done", ".part")):
            os.remove(os.path.join(data, f))
    if subprocess.run([sys.executable, "-I", "export.py"], cwd=d).returncode:
        git("checkout", "HEAD", "--", f"github-export/{p}/data", check=False)
        raise RuntimeError("issue export failed, kept the last one")
    counts = {}
    for f in os.listdir(data):
        if f.endswith(".jsonl.gz"):
            old = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:github-export/{p}/data/{f}"], capture_output=True).stdout
            counts[f[:-len(".jsonl.gz")]] = merge(os.path.join(data, f), old)
    readme = os.path.join(d, "README.md")
    text = open(readme, encoding="utf-8").read()
    text = re.sub(r"^(\| `(\w+)\.jsonl\.gz` \|.*\| )[\d,]+ \|$", lambda m: f"{m.group(1)}{counts[m.group(2)]:,} |", text, flags=re.M)
    text = re.sub(r"(exported from the GitHub REST API on )[\d-]+", rf"\g<1>{TODAY}", text)
    open(readme, "w", encoding="utf-8", newline="\n").write(text)


def push(p, repo, default):
    """Push the project's branches that changed, a few at a time. Returns the pushes GitHub refused."""
    want = {}  # branch on origin: commit it should point at
    for line in git("for-each-ref", "--format=%(objectname) %(refname)", f"refs/sync/{p}/heads/", f"refs/sync/{p}/pull/").stdout.splitlines():
        sha, ref = line.split(" ", 1)
        kind, name = ref.split("/", 4)[3:]
        want[f"{p}/main" if kind == "heads" and name == default else f"{p}/branch/{name}" if kind == "heads" else f"{p}/pr/{name}"] = sha
    # the issue export also lives on its own branch, with the folder as the branch's root
    git("add", "--", f"github-export/{p}")
    tree = git("write-tree", f"--prefix=github-export/{p}/").stdout.strip()
    git("fetch", "-q", "--depth=1", "origin", f"+refs/heads/{p}/github-export:refs/sync/{p}/export", check=False)
    parent = git("rev-parse", "-q", "--verify", f"refs/sync/{p}/export^{{commit}}", check=False).stdout.strip()
    if not parent or git("rev-parse", f"{parent}^{{tree}}").stdout.strip() != tree:
        msg = f"GitHub export of {repo} issues, PRs and comments ({TODAY})"
        want[f"{p}/github-export"] = git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg).stdout.strip()
    have = {}
    for line in git("ls-remote", "origin", f"refs/heads/{p}/*").stdout.splitlines():
        sha, ref = line.split("\t")
        have[ref[len("refs/heads/"):]] = sha
    specs = [f"+{sha}:refs/heads/{b}" for b, sha in sorted(want.items()) if have.get(b) != sha]
    print(f"{p}: pushing {len(specs)} changed branches", flush=True)
    deadline = time.monotonic() + PUSH_BUDGET
    def rejected(*s):
        # GitHub sometimes never answers a push; stop waiting at the project's deadline, the next run tries again
        left = deadline - time.monotonic()
        if left < 10:
            return [f"!\t{x.lstrip('+')}\t[skipped] (out of time this run)" for x in s]
        try:
            out = subprocess.run(["git", "-C", ROOT, "push", "--no-tags", "--porcelain", "origin", *s],
                                 capture_output=True, text=True, encoding="utf-8", timeout=min(180, left)).stdout
        except subprocess.TimeoutExpired:
            return [f"!\t{x.lstrip('+')}\t[skipped] (no answer from GitHub)" for x in s]
        return [l for l in out.splitlines() if l.startswith("!")]
    refused = []
    for i in range(0, len(specs), 20):  # GitHub's workflow-file check times out on pushes with many branches
        for line in rejected(*specs[i:i + 20]):
            # GitHub's own "timed out" refusals usually pass when the branch is pushed alone
            refused += rejected("+" + line.split("\t")[1]) if "timed out" in line.lower() else [line]
    return refused


def flag(p, repo):
    """Open an issue (once) saying the README's status part changed, so the site's progress steps get checked."""
    title = f"Check the progress steps for {p}"
    gh = lambda *a: subprocess.run(["gh", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    found = gh("issue", "list", "--state", "open", "--search", f'"{title}" in:title', "--json", "title", "--jq", ".[].title")
    if found.returncode == 0 and title not in found.stdout.splitlines():
        gh("issue", "create", "--title", title, "--body",
           f"The status part of [{repo}'s README](https://github.com/{repo}#readme) changed in the sync on {TODAY}.\n\n"
           "Check that the progress steps on the site still match it: `PROJECTS` in `docs/index.html`.")


def check_merge():
    import tempfile
    z = lambda *rows: gzip.compress("".join(json.dumps({"id": i, "v": v}) + "\n" for i, v in rows).encode())
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "x.jsonl.gz")
        open(f, "wb").write(z((3, "new"), (2, "new")))
        assert merge(f, z((1, "old"), (2, "old"))) == 3
        assert [json.loads(l)["v"] for l in gzip.open(f, "rt")] == ["old", "new", "new"]
        assert merge(f, b"") == 3


if __name__ == "__main__":
    check_merge()
    pushing = "--push" in sys.argv[1:]
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert set(only) <= set(PROJECTS), f"unknown project in {only}"
    synced, problems, skipped = [], [], []
    for p, (repo, start) in PROJECTS.items():
        if only and p not in only:
            continue
        try:
            default = fetch(p, repo)
            if not default:
                problems.append(f"{p}: can't reach {repo}, skipped; the archived copy stays as it is")
                continue
            src = f"refs/sync/{p}/heads/{default}"
            old = git("show", f"HEAD:{p}/README.md", check=False).stdout
            new = git("show", f"{src}:README.md", check=False).stdout
            refresh(p, src)
            synced.append(p)
            print(f"{p}: fetched", flush=True)
            export(p)
            print(f"{p}: issues exported", flush=True)
            if pushing:
                # GitHub refuses branches that edit .github/workflows files (the workflow's token may not touch those)
                # and sometimes times out; both are retried next run, so they're warnings, not failures
                skipped += [f"{p}: GitHub refused {line}" for line in push(p, repo, default)]
                if status_part(old, start) != status_part(new, start):
                    flag(p, repo)
            print(f"{p}: synced", flush=True)
        except Exception as e:
            problems.append(f"{p}: {e}")
    if synced:
        subprocess.run([sys.executable, "-I", os.path.join(ROOT, "docs", "build_site.py"), *synced], check=True)
    for line in skipped:
        print("::warning::" + line)
    for line in problems:
        print("PROBLEM", line)
    sys.exit(1 if problems else 0)

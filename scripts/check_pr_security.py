"""Scan what a pull request adds for changes that hide or run code, and fail on the ones that are never fine here.

    python3 -I scripts/check_pr_security.py BASE HEAD             # the changes HEAD makes since its merge base with BASE
    python3 -I scripts/check_pr_security.py --strict BASE HEAD    # also exit 1 for findings that only need a review
    python3 -I scripts/check_pr_security.py --github BASE HEAD    # GitHub Actions annotations and a step summary

Many contributors here are coding agents, and a pull request is reviewed in a batch with others, so a workflow that
force-pushes `main`, a download piped into a shell in a build script or a bidi control that makes code read differently from how
it runs is easy to miss. This reads the diff between the merge base of BASE and HEAD, and HEAD, from git objects
only: it never checks out, imports or runs anything the pull request contains, so it is safe to run on untrusted
changes (.github/workflows/pr-security.yml runs the copy on `main` against each pull request).

Each finding is one of two levels:

* `block`: never fine in this repository (a hidden Unicode control, a download piped into a shell, an unpinned
  action, a workflow that gets secrets or write access, a compiled executable, a credential). Exit status 1.
* `review`: fine sometimes, but a maintainer should read it before merging (a new network call, a changed submodule,
  an edit to the files coding agents take instructions from, a file that runs code by being present, a change to
  this check). Reported, exit status 0 unless --strict.

Only lines the pull request adds are matched, so what is already on `main` does not count again. Prints one line per
finding, `<path>:<line>: <level> <rule>: <message>`, followed by the line itself with control characters escaped.
Exit status 2 when git fails. Standard library only; needs git; no game files.
"""
import argparse
import html
import os
import posixpath
import re
import subprocess
import sys

BLOCK = "block"
REVIEW = "review"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
MAX_SCAN_BYTES = 8 * 1024 * 1024  # larger text is reported as unscanned rather than read
MAX_PER_RULE_AND_FILE = 20        # past this, one line says how many more there were

# ---------------------------------------------------------------------------------------------------------------------
# What a file is


CODE_SUFFIXES = (".py", ".pyw", ".ps1", ".psm1", ".psd1", ".sh", ".bash", ".zsh", ".bat", ".cmd", ".js", ".cjs",
                 ".mjs", ".ts", ".html", ".htm", ".cs", ".csproj", ".props", ".targets", ".cpp", ".cc", ".cxx", ".c",
                 ".h", ".hpp", ".inl", ".rs", ".toml", ".cmake", ".yml", ".yaml", ".json", ".mk", ".rb", ".pl", ".lua",
                 ".vbs", ".asm", ".s")
CODE_NAMES = ("cmakelists.txt", "makefile", "dockerfile", "justfile")


def is_code(path):
    name = posixpath.basename(path).lower()
    return name in CODE_NAMES or name.endswith(CODE_SUFFIXES)


def is_workflow(path):
    lower = path.lower()
    return (lower.startswith(".github/workflows/") and lower.endswith((".yml", ".yaml"))) or (
        lower.startswith(".github/actions/") and posixpath.basename(lower) in ("action.yml", "action.yaml"))


# Compiled code, by its first bytes. PE images are also caught by check_no_game_data.py; listing them here keeps
# this check complete on its own.
EXECUTABLE_MAGICS = (
    (b"MZ", "a Windows PE executable"),
    (b"\x7fELF", "an ELF executable"),
    (b"\xcf\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "a Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "a Mach-O executable"),
    (b"\xfe\xed\xfa\xce", "a Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "a Mach-O universal binary or Java class"),
    (b"\x00asm", "a WebAssembly module"),
)
EXECUTABLE_SUFFIXES = (".exe", ".dll", ".sys", ".scr", ".com", ".msi", ".so", ".dylib", ".pyc", ".pyo", ".pyd",
                       ".jar", ".class", ".whl", ".egg", ".node", ".wasm", ".appimage", ".deb", ".rpm", ".apk", ".lnk")

# Files that make a tool run code just by being there: Python loads them at startup or under pytest, MSBuild imports
# them into every project below them, editors and git hooks run them when a folder opens or a commit is made.
STARTUP_HOOKS = (  # (pattern on the lower-case path, level, why)
    (r"(^|/)(sitecustomize|usercustomize)\.py$", BLOCK, "Python runs it at every interpreter start"),
    (r"\.pth$", BLOCK, "Python runs its import lines at every interpreter start"),
    (r"(^|/)conftest\.py$", REVIEW, "pytest imports it before any test in its folder"),
    (r"(^|/)directory\.(build|packages)\.(props|targets)$", REVIEW, "MSBuild imports it into every project below it"),
    (r"(^|/)nuget\.config$", REVIEW, "it can point NuGet at another package source"),
    (r"(^|/)build\.rs$", REVIEW, "cargo runs it on every build"),
    (r"(^|/)\.cargo/config(\.toml)?$", REVIEW, "it can change how cargo builds and where it fetches crates"),
    (r"(^|/)(package\.json|\.npmrc|\.yarnrc(\.yml)?)$", REVIEW, "npm runs its install scripts and reads its registry"),
    (r"(^|/)(setup\.py|setup\.cfg|pyproject\.toml|pip\.conf|pip\.ini)$", REVIEW, "pip runs or reads it on install"),
    (r"(^|/)requirements[^/]*\.txt$", REVIEW, "it changes what pip installs"),
    (r"(^|/)cargo\.(toml|lock)$", REVIEW, "it changes which crates are built"),
    (r"(^|/)packages\.lock\.json$", REVIEW, "it changes which NuGet packages are restored"),
    (r"^\.vscode/", REVIEW, "VS Code can run its tasks when the folder opens"),
    (r"^\.devcontainer/", REVIEW, "a dev container runs its commands on start"),
    (r"^\.idea/", REVIEW, "JetBrains IDEs read run configurations from it"),
    (r"(^|/)\.envrc$", REVIEW, "direnv runs it when a shell enters the folder"),
    (r"^(\.githooks|\.husky)/|(^|/)\.pre-commit-config\.ya?ml$", REVIEW, "it runs on git commands"),
    (r"^\.gitattributes$|/\.gitattributes$", REVIEW, "it can name diff, merge and filter drivers"),
)

# Files coding agents read as instructions. Most contributors here are agents, so an edit to these steers all of them.
AGENT_INSTRUCTIONS = re.compile(
    r"^(agents\.md|claude\.md|gemini\.md|contributing\.md|\.cursorrules|\.windsurfrules|\.clinerules|"
    r"\.github/copilot-instructions\.md|\.github/(instructions|prompts)/.*|\.cursor/.*|\.claude/.*)$|(^|/)agents\.md$")

# This check, the other checks it backs up, and who reviews them.
SECURITY_CHECKS = (
    ".github/codeowners", "codeowners", "docs/codeowners", ".github/dependabot.yml",
    ".github/workflows/pr-security.yml", ".github/workflows/no-game-data.yml", ".github/workflows/jactionlint.yml",
    "scripts/check_pr_security.py", "scripts/check_no_game_data.py", "tests/test_check_pr_security.py",
)

# ---------------------------------------------------------------------------------------------------------------------
# Line rules: (rule, level, pattern, message). A pattern is matched against one added line at a time.

HIDDEN_BIDI = re.compile("[\u202a-\u202e\u2066-\u2069]")
HIDDEN_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\u180e\u00ad]|(?<=.)\ufeff")

DOWNLOAD = r"(?:\bcurl\b|\bwget\b|\biwr\b|\birm\b|Invoke-WebRequest|Invoke-RestMethod|DownloadString|DownloadData)"
SHELLS = r"(?:sudo\s+(?:-\S+\s+)*)?(?:(?:ba|z|da|k|fi)?sh|pwsh|powershell|python[\d.]*|perl|ruby|node|iex|Invoke-Expression|cmd)\b"
DECODE = (r"(?:b64decode|b32decode|a85decode|FromBase64String|base64\s+(?:-d|--decode|-D)\b|zlib\.decompress|"
          r"lzma\.decompress|bz2\.decompress|gzip\.decompress|bytes\.fromhex|codecs\.decode|marshal\.loads|"
          r"\[char\]\s*0x|-join\s*\[char|String\.fromCharCode|atob\s*\(|unhexlify)")
RUN = (r"(?:(?<![\w.])(?-i:exec|eval|compile)\s*\(|Invoke-Expression|(?<![\w-])iex\b|\b(?:ba|z)?sh\s+-c\b|"
       r"new\s+Function\s*\(|\bFunction\s*\(|Start-Process|os\.system\s*\(|subprocess\.)")

ALL_TEXT_RULES = (
    ("download-piped-to-shell", BLOCK,
     re.compile(DOWNLOAD + r"[^\n|]*\|\s*" + SHELLS, re.I),
     "a download is piped straight into an interpreter; download to a file, check its hash, then run it"),
    ("download-then-run", BLOCK,
     re.compile(r"(?:Invoke-Expression|(?<![\w-])iex\b)[^\n]*" + DOWNLOAD + "|" + DOWNLOAD +
                r"[^\n]*(?:\|\s*|;\s*|&&\s*)(?:Invoke-Expression|(?<![\w-])iex\b|\.\s+\S)", re.I),
     "fetched text is run as code"),
    ("encoded-command", BLOCK,
     re.compile(r"(?:powershell|pwsh)(?:\.exe)?\b[^\n]*\s-(?:e|ec|enc|encodedcommand)\s+[A-Za-z0-9+/=]{16,}", re.I),
     "PowerShell runs a base64-encoded command, which no reviewer can read"),
    ("decode-and-run", BLOCK,
     re.compile(RUN + r"[^\n]*" + DECODE + "|" + DECODE + r"[^\n]*" + RUN, re.I),
     "decoded or decompressed data is run as code"),
    ("credential", BLOCK,
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----|"
                r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b|\bAKIA[0-9A-Z]{16}\b|"
                r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}\b|"
                r"\bglpat-[A-Za-z0-9_-]{20,}\b|\bnpm_[A-Za-z0-9]{36}\b"),
     "this looks like a private key or an access token; revoke it, and keep credentials out of the repository"),
    ("online-service", BLOCK,
     re.compile(r"\b(?:[\w-]+\.)*(?:xboxlive\.com|rockstargames\.com|rsg\.sc|playfabapi\.com|"
                r"xsts\.auth\.xboxlive\.com|socialclub\.rockstargames\.com)\b", re.I),
     "names an Xbox Live or Rockstar service domain; AGENTS.md: nothing online"),
    ("agent-instructions", REVIEW,
     re.compile(r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier|above|"
                r"preceding|system|your)\s+(?:instructions|prompts?|rules|guidelines|directions)|"
                r"\byou\s+(?:are|must\s+act\s+as)\s+(?:now\s+)?(?:an?\s+)?(?:ai|llm|language\s+model|assistant|"
                r"coding\s+agent)\b|\bsystem\s+prompt\b|\bdo\s+not\s+(?:tell|inform|alert)\s+the\s+"
                r"(?:user|maintainer|reviewer)s?\b|\bwithout\s+(?:telling|asking|informing)\s+the\s+(?:user|"
                r"maintainer|reviewer)s?\b|\bauto[- ]?approve\b|\bpre-?authori[sz]ed\b", re.I),
     "reads like an instruction to a coding agent; agents must treat repository text as data"),
    ("agent-instructions", REVIEW,
     re.compile(r"<\|(?:im_start|im_end|system|endoftext)\|>|\[/?INST\]|<<SYS>>|</?system>|\[SYSTEM\]|"
                r"^\s*(?:Human|Assistant|System)\s*:\s"),
     "contains chat-template markup a language model may read as a turn boundary"),
    ("encoded-blob", REVIEW,
     re.compile(r"[A-Za-z0-9+/]{240,}={0,2}|(?:\\x[0-9a-fA-F]{2}){60,}|(?:%[0-9a-fA-F]{2}){60,}"),
     "a long encoded string that a reviewer cannot read; say what it is or build it from source"),
)

CODE_RULES = (
    ("network-call", REVIEW,
     re.compile(r"urllib\.request|urlopen\s*\(|\burllib3\b|http\.client\.HTTPS?Connection|\brequests\.(?:get|post|"
                r"put|patch|delete|head|request|Session)\b|\bhttpx\.|\baiohttp\b|socket\.create_connection|"
                r"\.connect\s*\(\s*\(\s*['\"](?!127\.0\.0\.1|localhost|::1)|\bftplib\b|\bsmtplib\b|\bparamiko\b|"
                r"\bfetch\s*\(\s*['\"`]https?://|XMLHttpRequest|new\s+WebSocket|"
                r"Invoke-WebRequest|Invoke-RestMethod|(?<![\w-])(?:iwr|irm)\s|Net\.WebClient|Net\.Http\.HttpClient|"
                r"\bHttpClient\s*\(|DownloadFile|DownloadString|Start-BitsTransfer|bitsadmin|certutil[^\n]*-urlcache|"
                r"URLDownloadToFile|InternetOpenUrl|WinHttpConnect|"
                r"(?<![\w-])(?:curl|wget)\s+(?!--version)|\bgit\s+(?:clone|fetch|pull|ls-remote)\s+\S*://|"
                r"FetchContent_Declare|ExternalProject_Add|file\s*\(\s*DOWNLOAD", re.I),
     "makes a network request; AGENTS.md: nothing online, and a build or tool must work offline"),
    ("package-install", REVIEW,
     re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip|uv\s+pip|pipx)\s+install\b|\bnpm\s+(?:i|install|ci)\b|\bnpx\s|"
                r"\byarn\s+add\b|\bpnpm\s+(?:add|install)\b|\bcargo\s+install\b|\bgo\s+install\b|"
                r"\bdotnet\s+tool\s+install\b|\bgem\s+install\b|Install-Module|Install-Package|\bchoco\s+install\b|"
                r"\bwinget\s+install\b|\bscoop\s+install\b|\bbrew\s+install\b|\bapt(?:-get)?\s+install\b", re.I),
     "installs a package; pin the version and say why it is needed"),
    ("package-source", BLOCK,
     re.compile(r"--(?:extra-)?index-url\s+(?!https://pypi\.org/)|--trusted" r"-host\b|"
                r"\bgit\+(?:https?|ssh)://|\bregistry\s*=\s*['\"]?https?://(?!registry\.npmjs\.org)", re.I),
     "installs from a package source other than the public registry"),
    ("hidden-process", REVIEW,
     re.compile(r"-WindowStyle\s+Hidden|-w\s+hidden\b|\bnohup\b|"
                r"\bdisown\b|setsid\s|schtasks\s|crontab\s|New-ScheduledTask|Register-ScheduledTask|"
                r"CurrentVersion\\+Run(?:Once)?\b|"
                r"Set-MpPreference|Add-MpPreference|DisableRealtimeMonitoring|ExclusionPath", re.I),
     "starts a hidden or persistent process, or changes Defender settings"),
    ("credential-read", REVIEW,
     re.compile(r"\.ssh[\\/]|id_rsa|id_ed25519|\.aws[\\/]credentials|\.git-credentials|\.netrc\b|"
                r"[\\/](?:Login Data|Local State|Cookies)\b|\bkeyring\.get_password|CredRead|CredEnumerate|"
                r"\bwallet\.dat|\.npmrc|\bGH_TOKEN\b|\bGITHUB_TOKEN\b|ACTIONS_RUNTIME_TOKEN|"
                r"ACTIONS_ID_TOKEN_REQUEST", re.I),
     "reads a credential store or token"),
    ("dynamic-code", REVIEW,
     re.compile(r"(?<![\w.])(?-i:exec|eval)\s*\((?!\s*\))|(?<![\w.])__import__\s*\(|\bimportlib\.import_module\s*\(\s*"
                r"(?!['\"])|\bmarshal\.loads\b|\bpickle\.loads?\b|Invoke-Expression|(?<![\w-])iex\s|"
                r"\[ScriptBlock\]::Create|Add-Type\s+-TypeDefinition|new\s+Function\s*\(", re.I),
     "runs code built at run time"),
)

UNTRUSTED_EXPRESSION = re.compile(
    r"\$\{\{[^}]*\b(?:github\.head_ref|github\.event\.(?:pull_request|issue|comment|review|review_comment|"
    r"discussion|discussion_comment|head_commit|commits|pages|workflow_run)\.[\w.\[\]*]*(?:title|body|ref|label|"
    r"message|name|email|page_name|default_branch|head_branch|display_title)\b|inputs\.)", re.I)

WORKFLOW_RULES = (
    ("dangerous-trigger", BLOCK,
     re.compile(r"(?:^|[\s\[,{])(?:pull_request_target|workflow_run)\b"),
     "pull_request_target and workflow_run give a fork's code the base repository's token and secrets; "
     "a maintainer adds these, with the code they run kept out of the pull request"),
    ("unpinned-action", BLOCK,
     re.compile(r"^\s*(?:-\s*)?uses\s*:\s*['\"]?(?!\./|docker://[^\s'\"]+@sha256:[0-9a-f]{64})"
                r"(?![^\s'\"#]+@[0-9a-f]{40}(?:['\"]?\s*(?:#.*)?)?$)[^\s'\"#]+", re.I),
     "an action must be pinned to a full commit SHA (`owner/repo@<40 hex> # vX.Y.Z`); a tag can be moved"),
    ("workflow-secret", BLOCK,
     re.compile(r"\bsecrets\s*(?:\.\s*(?!GITHUB_TOKEN\b)\w+|\[|:\s*inherit)", re.I),
     "no workflow here needs a secret; a pull request that adds one is asking for credentials"),
    ("workflow-token", REVIEW,
     re.compile(r"\bsecrets\.GITHUB_TOKEN\b|\bgithub\.token\b|persist-credentials\s*:\s*true", re.I),
     "hands the job's token to a step"),
    ("write-permission", BLOCK,
     re.compile(r"^\s*(?:permissions\s*:\s*write-all|(?!security-events\b)[\w-]+\s*:\s*write)\s*(?:#.*)?$", re.I),
     "gives the workflow write access; CI here only reads"),
    ("write-permission", REVIEW,
     re.compile(r"^\s*security-events\s*:\s*write\s*(?:#.*)?$", re.I),
     "lets the workflow upload code scanning results (CodeQL needs it); it cannot change the repository"),
    ("script-injection", BLOCK,
     UNTRUSTED_EXPRESSION,
     "expands text an outsider controls (a title, a branch name, a body) into the workflow; pass it in `env:`"),
    ("self-hosted-runner", BLOCK,
     re.compile(r"\bself-hosted\b", re.I),
     "a self-hosted runner would run a fork's code on someone's machine"),
    ("unsafe-runner-setting", BLOCK,
     re.compile(r"ACTIONS_ALLOW_UNSECURE_COMMANDS|ACTIONS_STEP_DEBUG|ACTIONS_RUNNER_DEBUG", re.I),
     "turns on deprecated workflow commands or debug output that prints secrets"),
    ("pushes-from-ci", BLOCK,
     re.compile(r"\bgit\s+push\b|\bgh\s+(?:pr\s+(?:merge|close|edit)|release\s+create|repo\s+(?:delete|edit)|"
                r"api\s+[^\n]*(?:-X|--method)\s*(?:POST|PUT|PATCH|DELETE))|\bmethod\s*:\s*['\"](?:DELETE|PUT|PATCH)",
                re.I),
     "CI pushes, merges, publishes or deletes; nothing here should change the repository from a workflow"),
    ("scheduled-workflow", REVIEW,
     re.compile(r"^\s*(?:-\s*)?(?:schedule\s*:|cron\s*:)", re.I),
     "runs on a timer, without a pull request in front of it"),
    ("artifact-or-cache-write", REVIEW,
     re.compile(r"\bGITHUB_(?:ENV|PATH|OUTPUT)\b[^\n]*\$\{\{|actions/cache/save", re.I),
     "writes an expression into the job environment or a cache another run reads"),
)


# ---------------------------------------------------------------------------------------------------------------------
# git


class GitError(Exception):
    pass


def git(repo, *args):
    command = ["git", "-C", repo, "-c", "core.quotePath=false", "-c", "diff.external=", "-c", "core.fsmonitor=false",
               *args]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise GitError("git " + " ".join(args) + ": " + result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def resolve(repo, rev):
    if rev == EMPTY_TREE:
        return rev
    return git(repo, "rev-parse", "--verify", "--end-of-options", rev + "^{commit}").decode().strip()


def merge_base(repo, base, head):
    if base == EMPTY_TREE:
        return base
    return git(repo, "merge-base", base, head).decode().strip()


def changed_files(repo, old, new):
    """[(status, old_mode, new_mode, old_sha, new_sha, path)] for every path that differs, renames as delete + add."""
    out = git(repo, "diff", "--raw", "-z", "--no-renames", "--no-ext-diff", "--no-textconv", "--abbrev=40",
              "--ignore-submodules=none", old, new, "--")
    fields = out.split(b"\0")
    entries = []
    i = 0
    while i + 1 < len(fields) and fields[i]:
        meta = fields[i].decode("ascii").lstrip(":").split(" ")
        path = fields[i + 1].decode("utf-8", "surrogateescape")
        old_mode, new_mode, old_sha, new_sha, status = meta[0], meta[1], meta[2], meta[3], meta[4][:1]
        entries.append((status, old_mode, new_mode, old_sha, new_sha, path))
        i += 2
    return entries


def blob(repo, sha):
    return git(repo, "cat-file", "blob", sha)


def blob_size(repo, sha):
    return int(git(repo, "cat-file", "-s", sha).decode().strip())


HUNK = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def added_lines(repo, old, new, path):
    """[(line number in the new file, text)] for each line the change adds to one text file."""
    out = git(repo, "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--text", "--no-renames", "-U0",
              old, new, "--", ":(literal)" + path)
    lines = []
    number = None
    in_hunks = False
    for raw in out.split(b"\n"):
        match = HUNK.match(raw)
        if match:
            number = int(match.group(1))
            in_hunks = True
            continue
        if not in_hunks or number is None:
            continue
        if raw.startswith(b"+"):
            lines.append((number, raw[1:].rstrip(b"\r").decode("utf-8", "replace")))
            number += 1
        elif raw.startswith(b" "):
            number += 1
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# Checks


class Finding:
    __slots__ = ("path", "line", "level", "rule", "message", "text")

    def __init__(self, path, line, level, rule, message, text=""):
        self.path, self.line, self.level, self.rule, self.message, self.text = path, line, level, rule, message, text

    def key(self):
        return (0 if self.level == BLOCK else 1, self.path, self.line or 0, self.rule)


def visible(text, limit=200):
    """The text with control, bidi and zero-width characters spelled out, so a log shows what is really there."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "\t":
            out.append(" ")
        elif code < 0x20 or code == 0x7f or 0x80 <= code < 0xa0 or HIDDEN_BIDI.match(ch) or \
                HIDDEN_ZERO_WIDTH.match(ch) or ch in "\u2028\u2029\ufeff\u00ad":
            out.append("\\u%04x" % code)
        else:
            out.append(ch)
    s = "".join(out).strip()
    return s if len(s) <= limit else s[:limit - 3] + "..."


def check_path(status, old_mode, new_mode, path, head_bytes):
    """Findings that come from what a changed path is, not from its lines."""
    found = []
    lower = path.lower()
    gone = status == "D"
    if new_mode == "160000" or old_mode == "160000":
        what = "removes a submodule" if gone else ("adds a submodule" if old_mode != "160000" else
                                                   "moves a submodule to another commit")
        found.append(Finding(path, None, REVIEW, "submodule", what + "; check the commit it points at"))
        return found
    if new_mode == "120000" and not gone:
        target = head_bytes.decode("utf-8", "replace")
        escapes = posixpath.isabs(target) or re.match(r"^[A-Za-z]:[\\/]", target) or \
            posixpath.normpath(posixpath.join(posixpath.dirname(path), target)).startswith("..")
        found.append(Finding(path, None, BLOCK if escapes else REVIEW, "symlink",
                             "a symbolic link that points outside the repository" if escapes else
                             "a symbolic link; tools that follow it read another file", "-> " + target))
    if not gone:
        for magic, what in EXECUTABLE_MAGICS:
            if head_bytes.startswith(magic):
                found.append(Finding(path, None, BLOCK, "executable", "adds " + what +
                                     "; build it from source in CI instead of committing it"))
                break
        else:
            if lower.endswith(EXECUTABLE_SUFFIXES):
                found.append(Finding(path, None, BLOCK, "executable",
                                     "adds a compiled or packaged program; build it from source instead"))
    for pattern, level, why in STARTUP_HOOKS:
        if re.search(pattern, lower):
            if gone and level == BLOCK:
                continue
            found.append(Finding(path, None, level, "runs-when-present",
                                 ("removes" if gone else "changes" if status == "M" else "adds") +
                                 " a file that runs code by being present: " + why))
            break
    if AGENT_INSTRUCTIONS.search(lower):
        found.append(Finding(path, None, REVIEW, "agent-instructions-file",
                             "changes instructions coding agents follow; read it for text that steers them"))
    if lower in SECURITY_CHECKS:
        found.append(Finding(path, None, REVIEW, "security-check-changed",
                             ("removes" if gone else "changes") + " a security check or who reviews it"))
    elif is_workflow(path) and status != "M":
        found.append(Finding(path, None, REVIEW, "workflow-file",
                             ("removes" if gone else "adds") + " a GitHub Actions workflow"))
    if lower == ".gitmodules" and not gone:
        found.append(Finding(path, None, REVIEW, "submodule", "changes where a submodule is fetched from"))
    return found


def check_line(path, number, text):
    """Findings for one added line of a text file."""
    found = []
    if HIDDEN_BIDI.search(text):
        found.append(Finding(path, number, BLOCK, "hidden-unicode",
                             "a bidirectional control character makes the line read differently from how it runs",
                             text))
    if HIDDEN_ZERO_WIDTH.search(text):
        level = REVIEW if path.lower().startswith("docs/i18n/") else BLOCK
        found.append(Finding(path, number, level, "hidden-unicode",
                             "an invisible character (zero-width, soft hyphen or BOM) hides what the line says", text))
    rules = list(ALL_TEXT_RULES)
    if is_code(path):
        rules += CODE_RULES
    if is_workflow(path):
        rules += WORKFLOW_RULES
    for rule, level, pattern, message in rules:
        if pattern.search(text):
            if rule == "network-call" and re.search(r"127\.0\.0\.1|\blocalhost\b|\[::1\]", text) and \
                    not re.search(r"https?://(?!127\.0\.0\.1|localhost|\[::1\])", text):
                continue
            found.append(Finding(path, number, level, rule, message, text))
    return found


def scan(repo, base, head):
    """(findings, number of files looked at)."""
    old = merge_base(repo, resolve(repo, base), resolve(repo, head)) if base != EMPTY_TREE else EMPTY_TREE
    new = resolve(repo, head)
    findings = []
    entries = changed_files(repo, old, new)
    for status, old_mode, new_mode, _old_sha, new_sha, path in entries:
        gone = status == "D"
        is_blob = new_mode not in ("160000",) and not gone
        size = blob_size(repo, new_sha) if is_blob else 0
        head_bytes = blob(repo, new_sha)[:8192] if is_blob else b""
        findings += check_path(status, old_mode, new_mode, path, head_bytes)
        if gone or not is_blob or new_mode == "120000" or b"\0" in head_bytes:
            continue
        if size > MAX_SCAN_BYTES:
            findings.append(Finding(path, None, REVIEW, "not-scanned",
                                    "a text file over %d MiB that this check does not read" % (MAX_SCAN_BYTES >> 20)))
            continue
        per_rule = {}
        for number, text in added_lines(repo, old, new, path):
            for finding in check_line(path, number, text):
                count = per_rule.get(finding.rule, 0) + 1
                per_rule[finding.rule] = count
                if count <= MAX_PER_RULE_AND_FILE:
                    findings.append(finding)
        for rule, count in per_rule.items():
            if count > MAX_PER_RULE_AND_FILE:
                level = next(f.level for f in findings if f.path == path and f.rule == rule)
                findings.append(Finding(path, None, level, rule, "%d more lines like this in the file" %
                                        (count - MAX_PER_RULE_AND_FILE)))
    findings.sort(key=Finding.key)
    return findings, len(entries)


# ---------------------------------------------------------------------------------------------------------------------
# Output


def location(finding):
    return finding.path + (":%d" % finding.line if finding.line else "")


def plain_lines(findings):
    lines = []
    for f in findings:
        lines.append("%s: %s %s: %s" % (visible(location(f), 300), f.level, f.rule, f.message))
        if f.text:
            lines.append("    > " + visible(f.text))
    return lines


def escape_data(s):
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(s):
    return escape_data(s).replace(":", "%3A").replace(",", "%2C")


def annotation(finding):
    """A GitHub workflow command; every piece of pull request text in it is escaped so it cannot start another."""
    kind = "error" if finding.level == BLOCK else "warning"
    props = "file=" + escape_property(finding.path)
    if finding.line:
        props += ",line=%d" % finding.line
    props += ",title=" + escape_property("%s %s" % (finding.level, finding.rule))
    message = finding.message + ("\n" + visible(finding.text) if finding.text else "")
    return "::%s %s::%s" % (kind, props, escape_data(message))


def cell(s):
    return html.escape(s, quote=True).replace("|", "&#124;").replace("`", "&#96;").replace("\n", " ")


def summary(findings, files):
    blocks = sum(f.level == BLOCK for f in findings)
    reviews = len(findings) - blocks
    out = ["## PR security scan", ""]
    if not findings:
        out.append("Nothing to report in %d changed file%s." % (files, "" if files == 1 else "s"))
        return "\n".join(out) + "\n"
    out.append("%d to fix (`block`), %d to read before merging (`review`), in %d changed file%s. "
               "The check never runs the pull request's code; see `scripts/check_pr_security.py`."
               % (blocks, reviews, files, "" if files == 1 else "s"))
    out += ["", "| Level | Rule | Where | What | Line |", "|---|---|---|---|---|"]
    for f in findings:
        out.append("| %s | %s | <code>%s</code> | %s | %s |" % (
            f.level, f.rule, cell(visible(location(f), 300)), cell(f.message),
            "<code>%s</code>" % cell(visible(f.text, 160)) if f.text else ""))
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Scan the lines a pull request adds for changes that hide or run code. Reads git objects only.")
    parser.add_argument("base", help="the branch the pull request goes into (its merge base with HEAD is compared)")
    parser.add_argument("head", help="the pull request's head commit")
    parser.add_argument("--repo", default=".", help="the git repository (default: the current folder)")
    parser.add_argument("--strict", action="store_true", help="exit 1 for review findings too")
    parser.add_argument("--github", action="store_true",
                        help="also print GitHub Actions annotations and write $GITHUB_STEP_SUMMARY")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    try:
        findings, files = scan(args.repo, args.base, args.head)
    except GitError as error:
        print("check_pr_security: %s" % visible(str(error), 2000), file=sys.stderr)
        return 2
    # Every printed line starts with a fixed prefix or a workflow command this script built, so a file name or a
    # line of the pull request can never be read by the runner as a command of its own.
    for line in plain_lines(findings):
        print("| " + line)
    blocks = sum(f.level == BLOCK for f in findings)
    print("| %d block, %d review, %d changed file%s" % (blocks, len(findings) - blocks, files,
                                                       "" if files == 1 else "s"))
    if args.github:
        for finding in findings:
            print(annotation(finding))
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a", encoding="utf-8", newline="\n") as out:
                out.write(summary(findings, files))
    if blocks or (args.strict and findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

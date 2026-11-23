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
    r"\.github/copilot-instructions\.md|\.github/(instructions|prompts)/.*|\.cursor/.*|\.claude/.*)$|(^|/)agents\.md$|"
    # an instruction file under another name, anywhere (tests/semantic/MASTER_PROMPT.md, agent_directives.md)
    r"(^|/)[^/]*(prompt|instruction|directive|system[-_]?message)[^/]*\.(md|mdc|txt|ya?ml|json)$")

# This check, the other checks it backs up, and who reviews them.
SECURITY_CHECKS = (
    ".github/codeowners", "codeowners", "docs/codeowners", ".github/dependabot.yml",
    ".github/workflows/pr-security.yml", ".github/workflows/no-game-data.yml", ".github/workflows/jactionlint.yml",
    "scripts/check_pr_security.py", "scripts/check_no_game_data.py", "tests/test_check_pr_security.py",
)

# ---------------------------------------------------------------------------------------------------------------------
# Line rules: (rule, level, pattern, message). A pattern is matched against one added line at a time.

HIDDEN_BIDI = re.compile("[\u202a-\u202e\u2066-\u2069]")
HIDDEN_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\u180e\u00ad]")
HIDDEN_BOM = re.compile("\ufeff")  # listed, not blocked: a BOM changes no meaning

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


# ---------------------------------------------------------------------------------------------------------------------
# Look-alike letters, submodule sources, pull request text, pinned actions, the maintainer override

# Letters of other scripts that render like a Latin letter in common fonts: the part of Unicode's confusables list
# that has a Latin twin. They are listed as code points, so this file contains none of them.
LATIN_LOOKALIKES = frozenset(map(chr, (
    # Cyrillic: A B E K M H O P C T X S I J Y h Q W, a e o p c y x s i j h l d q w y
    0x0410, 0x0412, 0x0415, 0x041A, 0x041C, 0x041D, 0x041E, 0x0420, 0x0421, 0x0422, 0x0425, 0x0405, 0x0406,
    0x0408, 0x04AE, 0x04BA, 0x051A, 0x051C, 0x0430, 0x0435, 0x043E, 0x0440, 0x0441, 0x0443, 0x0445, 0x0455,
    0x0456, 0x0458, 0x04BB, 0x04CF, 0x0501, 0x051B, 0x051D, 0x04AF,
    # Greek: A B E Z H I K M N O P T Y X, o v i k p
    0x0391, 0x0392, 0x0395, 0x0396, 0x0397, 0x0399, 0x039A, 0x039C, 0x039D, 0x039F, 0x03A1, 0x03A4, 0x03A5,
    0x03A7, 0x03BF, 0x03BD, 0x03B9, 0x03BA, 0x03C1,
    # Armenian: o u h n
    0x0585, 0x057D, 0x0570, 0x0578,
)))
IDENTIFIER = re.compile(r"[^\W\d_]\w*")


def spelled(text):
    """Text with every non-ASCII letter as \\uXXXX, so a log shows which letters are not Latin."""
    return "".join(ch if ch.isascii() else chr(92) + "u%04x" % ord(ch) for ch in text)


def script_of(ch):
    if ch.isascii():
        return "LATIN" if ch.isalpha() else ""
    import unicodedata
    return unicodedata.name(ch, "").split(" ", 1)[0]


def lookalike_findings(path, number, text):
    """A code identifier with a letter that only looks Latin: password spelled with a Cyrillic a is another name."""
    if text.isascii() or not is_code(path):
        return []
    for token in IDENTIFIER.findall(text):
        if token.isascii():
            continue
        letters = [ch for ch in token if ch.isalpha()]
        twins = [ch for ch in letters if ch in LATIN_LOOKALIKES]
        fullwidth = [ch for ch in letters if 0xFF21 <= ord(ch) <= 0xFF5A]
        if not twins and not fullwidth:
            continue
        latin = [ch for ch in letters if script_of(ch) == "LATIN"]
        foreign = [ch for ch in letters if script_of(ch) != "LATIN"]
        # It reads as a Latin name when every letter that isn't Latin looks Latin. `GUEST_` and a Cyrillic word with
        # a letter that has no Latin twin is plainly Cyrillic; a word that is all twins is often real Cyrillic text
        # (a Russian UI string), so that is only listed.
        reads_latin = all(ch in LATIN_LOOKALIKES or 0xFF21 <= ord(ch) <= 0xFF5A for ch in foreign)
        if not reads_latin:
            continue
        level = BLOCK if latin or fullwidth else REVIEW
        import unicodedata
        names = ", ".join(sorted({"U+%04X %s" % (ord(ch), unicodedata.name(ch, "?")) for ch in twins + fullwidth}))
        return [Finding(path, number, level, "lookalike-letters",
                        "the name %s is spelled with letters that only look Latin (%s), so it is not the name it "
                        "reads as" % (spelled(token), names), spelled(text))]
    return []


GITHUB_SUBMODULE = re.compile(r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/|\.\.?/)", re.I)


def gitmodules_findings(path, number, text):
    if path.lower() != ".gitmodules":
        return []
    url = re.match(r"^\s*url\s*=\s*(\S+)", text, re.I)
    if url and not GITHUB_SUBMODULE.match(url.group(1)):
        return [Finding(path, number, BLOCK, "submodule-host",
                        "a submodule fetched from outside github.com over https or ssh; every submodule here is on "
                        "GitHub, where its history can be read", text)]
    if re.match(r"^\s*update\s*=\s*!", text, re.I):
        return [Finding(path, number, BLOCK, "submodule-command",
                        "a submodule update that runs a shell command", text)]
    return []


# What a pull request's text says to the agents and maintainers reading it.
APPROVAL_CLAIM = re.compile(
    r"\b(?:pre-?approved|already\s+(?:been\s+)?(?:reviewed|approved|audited|vetted)|approved\s+by\s+(?:the\s+)?"
    r"(?:maintainer|owner|admin|coordinator)s?|safe\s+to\s+(?:auto-?)?merge|merge\s+(?:this\s+|it\s+)?(?:without|"
    r"before)\s+(?:a\s+|any\s+)?(?:review|ci|checks?|testing)|(?:skip|bypass|ignore|disable)\s+(?:the\s+)?(?:code\s+)?"
    r"(?:review|security|checks?|ci|scan)|no\s+review\s+(?:is\s+)?(?:needed|required)|security-reviewed)\b", re.I)
TEXT_RULES = tuple(r for r in ALL_TEXT_RULES if r[0] in ("credential", "agent-instructions")) + (
    ("approval-claim", REVIEW, APPROVAL_CLAIM,
     "claims the change is approved or asks for review to be skipped; only a maintainer's own review approves"),)


def check_text(where, text):
    """Findings for text that is not a file: a pull request's title or description, a commit message."""
    found = []
    for number, line in enumerate((text or "").splitlines(), 1):
        if HIDDEN_BIDI.search(line) or HIDDEN_ZERO_WIDTH.search(line):
            found.append(Finding(where, number, BLOCK, "hidden-unicode",
                                 "an invisible or bidirectional control character changes what the text appears to "
                                 "say", line))
        elif HIDDEN_BOM.search(line):
            found.append(Finding(where, number, REVIEW, "byte-order-mark",
                                 "a byte-order mark (an invisible character) in the text", line))
        for rule, level, pattern, message in TEXT_RULES:
            if pattern.search(line):
                found.append(Finding(where, number, level, rule, message, line))
    return found


def commit_messages(repo, old, new, limit=500):
    """[(sha, message)] of the commits in old..new (all of new's history when old is the empty tree)."""
    span = [new] if old == EMPTY_TREE else [old + ".." + new]
    out = git(repo, "log", "--no-color", "--no-show-signature", "-n", str(limit), "--format=%H%x00%B%x1e", *span, "--")
    commits = []
    for record in out.decode("utf-8", "replace").split("\x1e"):
        sha, _, message = record.strip("\n").partition("\x00")
        if sha:
            commits.append((sha, message))
    return commits


# `uses: owner/repo[/path]@<40 hex> # v1.2.3`
PINNED_USE = re.compile(r"^\s*(?:-\s*)?uses\s*:\s*['\"]?([A-Za-z0-9][\w.-]*)/([A-Za-z0-9_][\w.-]*)(?:/[^\s'\"@]*)?"
                        r"@([0-9a-f]{40})['\"]?\s*(?:#\s*(\S+))?")
RELEASE_TAG = re.compile(r"^[A-Za-z0-9][\w.+-]{0,99}$")
ACTION_REMOTE = "https://github.com/{owner}/{repo}.git"


def tag_commit(url, tag):
    """The commit a tag of a remote repository points at (peeled), or None when there is no such tag."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_ASKPASS="")
    result = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", "--tags", "--", url,
                             "refs/tags/" + tag, "refs/tags/" + tag + "^{}"],
                            capture_output=True, env=env, timeout=60)
    if result.returncode != 0:
        raise GitError("git ls-remote %s: %s" % (url, result.stderr.decode("utf-8", "replace").strip()))
    refs = dict(reversed(line.split("\t", 1)) for line in result.stdout.decode().splitlines() if "\t" in line)
    return refs.get("refs/tags/" + tag + "^{}") or refs.get("refs/tags/" + tag)


def check_pins(pins, remote=ACTION_REMOTE):
    """Each pinned action line the pull request adds names the commit its `# vX.Y.Z` comment says it does."""
    found = []
    seen = {}
    for path, number, owner, repo, sha, tag, text in pins:
        where = "%s/%s" % (owner, repo)
        if not tag:
            found.append(Finding(path, number, REVIEW, "pin-without-version",
                                 "a pinned action without a `# vX.Y.Z` comment, so nobody can tell which release "
                                 "it is meant to be", text))
            continue
        if not RELEASE_TAG.match(tag):
            found.append(Finding(path, number, REVIEW, "pin-without-version",
                                 "the comment after the pin is not a release tag", text))
            continue
        key = (owner.lower(), repo.lower(), tag)
        if key not in seen:
            try:
                seen[key] = (tag_commit(remote.format(owner=owner, repo=repo), tag), None)
            except (GitError, subprocess.TimeoutExpired, OSError) as error:
                seen[key] = (None, str(error) or "timed out")
        actual, error = seen[key]
        if error:
            found.append(Finding(path, number, REVIEW, "pin-unverified",
                                 "could not read the tags of %s to check this pin: %s" % (where, error[:200]), text))
        elif actual is None:
            found.append(Finding(path, number, BLOCK, "pin-tag-mismatch",
                                 "%s has no tag %s, so the comment does not say what this commit is" % (where, tag),
                                 text))
        elif actual != sha:
            found.append(Finding(path, number, BLOCK, "pin-tag-mismatch",
                                 "%s %s is commit %s, not the %s pinned here; the comment hides which code runs"
                                 % (where, tag, actual, sha), text))
    return found


# A maintainer who has read a change marks that exact commit as reviewed with a label named for it. The label names
# the commit, so a push after the review is not covered by it, and a fork can't add labels to its own pull request.
# At least 16 hex digits: a commit whose SHA starts with the same 7 can be ground out in minutes, 16 can't. (GitHub
# caps a label at 50 characters, so the full 40 don't fit after the prefix.) Who added the label is checked too: see
# override_authorized().
OVERRIDE_LABEL = re.compile(r"^security-reviewed:([0-9a-f]{16,40})$")


def codeowners(path):
    """The user logins in a CODEOWNERS file (teams, `@org/team`, can't be checked without more access, so they don't
    count). The file is read from the base branch's checkout, which a pull request can't change."""
    owners = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                for word in line.split("#", 1)[0].split()[1:]:
                    if re.match(r"^@[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$", word):
                        owners.add(word[1:].lower())
    except OSError:
        pass
    return owners


def read_label_events(path):
    """[(login, label, time)] of the `labeled` events in a file of `gh api --paginate` issue-event pages."""
    import json
    with open(path, encoding="utf-8") as f:
        text = f.read()
    decoder, index, events = json.JSONDecoder(), 0, []
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        page, index = decoder.raw_decode(text, index)
        for event in page if isinstance(page, list) else [page]:
            if isinstance(event, dict) and event.get("event") == "labeled":
                events.append((str((event.get("actor") or {}).get("login") or ""),
                               str((event.get("label") or {}).get("name") or ""), str(event.get("created_at") or "")))
    return events


def override_authorized(label, events, owners):
    """(ok, who): whether the latest time `label` was added, a code owner added it."""
    added = sorted((when, who) for who, name, when in events if name == label)
    if not added:
        return False, ""
    who = added[-1][1]
    return who.lower() in owners, who


def override_label(event):
    """The `security-reviewed:<head sha>` label of the pull request in a GitHub event, if it names its head."""
    pr = event.get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pr, dict):
        return None
    head = str((pr.get("head") or {}).get("sha") or "")
    if not re.match(r"^[0-9a-f]{40}$", head):
        return None
    for label in pr.get("labels") or []:
        name = label.get("name") if isinstance(label, dict) else None
        match = OVERRIDE_LABEL.match(name or "")
        if match and head.startswith(match.group(1)):
            return name
    return None


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
    """One finding. `path` is a file, or `<pull request title>` / `<commit 0123abcd message>` for text that isn't."""
    __slots__ = ("path", "line", "level", "rule", "message", "text", "overridden")

    def __init__(self, path, line, level, rule, message, text=""):
        self.path, self.line, self.level, self.rule, self.message, self.text = path, line, level, rule, message, text
        self.overridden = False

    def key(self):
        return (0 if self.level == BLOCK else 1, self.path, self.line or 0, self.rule)

    def in_file(self):
        return not self.path.startswith("<")

    def fails(self):
        return self.level == BLOCK and not self.overridden


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


# Rules that documentation and tests trip by quoting what they warn about or reject (a page explaining why the
# install doesn't pipe a download into a shell; a test that the forbidden domains are refused). There they are listed.
QUOTED_IN_PROSE = ("download-piped-to-shell", "download-then-run", "online-service")

# The runtime's shims that implement network APIs by refusing every connection: they name those APIs on purpose.
OFFLINE_SHIMS = ("runtime/shims/winhttp_offline.cpp", "runtime/shims/xgameruntime_offline.cpp")


def is_doc_or_test(path):
    lower = path.lower()
    return lower.endswith((".md", ".rst", ".txt")) or lower.startswith(("docs/", "tests/")) or "/tests/" in lower or \
        "/test/" in lower or lower.startswith("runtime/test/")


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
                             "an invisible character (zero-width or soft hyphen) hides what the line says", text))
    # One byte-order mark at the start of a file is how some editors save it. Anywhere else it is listed, not
    # blocked: it is invisible, but unlike the characters above it can't split a name or reorder a line.
    rest = text[1:] if number == 1 and text.startswith(chr(0xFEFF)) else text
    if HIDDEN_BOM.search(rest):
        found.append(Finding(path, number, REVIEW, "byte-order-mark",
                             "a byte-order mark inside the file (an invisible character); usually an editor's doing",
                             text))
    found += lookalike_findings(path, number, text)
    found += gitmodules_findings(path, number, text)
    rules = list(ALL_TEXT_RULES)
    if is_code(path):
        rules += CODE_RULES
    if is_workflow(path) and not text.lstrip().startswith("#"):  # a YAML comment never runs
        rules += WORKFLOW_RULES
    for rule, level, pattern, message in rules:
        if pattern.search(text):
            if rule == "network-call" and re.search(r"127\.0\.0\.1|\blocalhost\b|\[::1\]", text) and \
                    not re.search(r"https?://(?!127\.0\.0\.1|localhost|\[::1\])", text):
                continue
            if rule == "network-call" and path in OFFLINE_SHIMS:
                continue
            if level == BLOCK and rule in QUOTED_IN_PROSE and is_doc_or_test(path):
                level, message = REVIEW, message + " (listed, not blocked, in documentation and tests, which quote it)"
            found.append(Finding(path, number, level, rule, message, text))
    return found


def safe_relative(path):
    """The path if it can be written under another folder as it is: relative, no `..`, no drive, no `.git`."""
    parts = path.split("/")
    if not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path) or "\\" in path or \
            any(p in ("", ".", "..") or p.lower() == ".git" or ":" in p for p in parts):
        return None
    return path


def export_changed(repo, base, head, out_dir):
    """Write the text files the change adds or modifies under out_dir, as data for Gitleaks.

    Only regular files are written (no symbolic links or submodules), from git objects, under their repository
    path; nothing is checked out and nothing written is ever run. Returns the number of files written."""
    old = merge_base(repo, resolve(repo, base), resolve(repo, head)) if base != EMPTY_TREE else EMPTY_TREE
    new = resolve(repo, head)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for status, _old_mode, new_mode, _old_sha, new_sha, path in changed_files(repo, old, new):
        rel = safe_relative(path)
        if status == "D" or new_mode not in ("100644", "100755") or rel is None:
            continue
        if blob_size(repo, new_sha) > MAX_SCAN_BYTES:
            continue
        data = blob(repo, new_sha)
        if b"\0" in data[:8192]:
            continue
        target = os.path.join(out_dir, *rel.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(data)
        written += 1
    return written


# The documents every reader starts from: the repository's top folder, and the docs index.
TOP_DOC = re.compile(r"^[^/]+\.(md|rst|txt)$|^docs/readme\.md$|^licen[cs]e", re.I)
LINK_HOST = re.compile(r"(?i)\b(?:https?://|www\.)((?:[a-z0-9-]+\.)+[a-z]{2,})(/[^\s)\"'<>]*)?")
KNOWN_LINK_HOSTS = frozenset(("github.com", "docs.github.com", "gist.github.com"))
# Hosts that serve any repository's files: a link there is this repository's own when its path starts with it.
REPO_FILE_HOSTS = frozenset(("raw.githubusercontent.com", "media.githubusercontent.com"))
OWN_REPOSITORY = "/itsnotrexed/leonida/"


def link_hosts(text):
    hosts = set()
    for match in LINK_HOST.finditer(text):
        host = match.group(1).lower()
        host = host[4:] if host.startswith("www.") else host
        if host in REPO_FILE_HOSTS and (match.group(2) or "").lower().startswith(OWN_REPOSITORY):
            continue  # this repository's own files, like the install scripts
        hosts.add(host)
    return hosts


def doc_wipe(repo, old, new, status, path, old_sha):
    """A top-level document the change deletes, or empties or replaces (half or more of its lines removed)."""
    if status not in ("M", "D") or not TOP_DOC.match(path):
        return []
    before = blob(repo, old_sha).decode("utf-8", "replace").count("\n")
    if before < 20:
        return []
    if status == "D":
        removed = before
    else:
        counts = git(repo, "diff", "--numstat", "--no-renames", "--no-ext-diff", "--no-textconv", old, new, "--",
                     ":(literal)" + path).decode("utf-8", "replace").split()
        removed = int(counts[1]) if len(counts) >= 2 and counts[1].isdigit() else 0
    if removed * 2 < before:
        return []
    return [Finding(path, None, REVIEW, "doc-wipe",
                    "removes %d of the %d lines of a top-level document; check it isn't being emptied or replaced"
                    % (removed, before))]


def scan(repo, base, head, check_tags=False, action_remote=ACTION_REMOTE, messages=True, external=None):
    """(findings, number of files looked at).

    `external` is a list of findings from another scanner (Gitleaks) on the exported files; only those on a
    line the change adds are kept, with that line as their text."""
    old = merge_base(repo, resolve(repo, base), resolve(repo, head)) if base != EMPTY_TREE else EMPTY_TREE
    new = resolve(repo, head)
    findings = []
    pins = []
    added = {}  # path -> {line number: text}
    entries = changed_files(repo, old, new)
    for status, old_mode, new_mode, old_sha, new_sha, path in entries:
        gone = status == "D"
        is_blob = new_mode not in ("160000",) and not gone
        size = blob_size(repo, new_sha) if is_blob else 0
        head_bytes = blob(repo, new_sha)[:8192] if is_blob else b""
        findings += check_path(status, old_mode, new_mode, path, head_bytes)
        if old_mode in ("100644", "100755"):
            findings += doc_wipe(repo, old, new, status, path, old_sha)
        # A file no scanner here can read is listed, so a person looks at it (the xz backdoor hid in binary test
        # files). An executable is already blocked above.
        if is_blob and new_mode in ("100644", "100755") and b"\0" in head_bytes and \
                not any(f.path == path and f.rule == "executable" for f in findings):
            findings.append(Finding(path, None, REVIEW, "not-inspected",
                                    "a binary file (%d bytes) that none of the scanners read; check what it is and "
                                    "why it is needed" % size))
        if gone or not is_blob or new_mode == "120000" or b"\0" in head_bytes:
            continue
        if size > MAX_SCAN_BYTES:
            findings.append(Finding(path, None, REVIEW, "not-scanned",
                                    "a text file over %d MiB that this check does not read" % (MAX_SCAN_BYTES >> 20)))
            continue
        per_rule = {}
        # A top-level document that starts linking another site (a "the project moved to ..." notice, say) is listed:
        # every reader and every agent starts from these files.
        top_doc = bool(TOP_DOC.match(path))
        hosts_before = KNOWN_LINK_HOSTS
        if top_doc and status == "M" and old_mode in ("100644", "100755"):
            hosts_before = hosts_before | link_hosts(blob(repo, old_sha).decode("utf-8", "replace"))
        for number, text in added_lines(repo, old, new, path):
            added.setdefault(path, {})[number] = text
            if top_doc:
                for host in sorted(link_hosts(text) - hosts_before):
                    findings.append(Finding(path, number, REVIEW, "new-link-host",
                                            "a top-level document links %s, a site it didn't link before" % host,
                                            text))
            if is_workflow(path) and not text.lstrip().startswith("#"):
                pin = PINNED_USE.match(text)
                if pin:
                    pins.append((path, number) + pin.groups() + (text,))
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
    if messages:
        for sha, message in commit_messages(repo, old, new):
            findings += check_text("<commit %s message>" % sha[:12], message)
    if check_tags:
        findings += check_pins(pins, action_remote)
    seen = set()
    for finding in external or ():
        if not finding.in_file():
            findings.append(finding)  # a scanner that did not run, reported as such
            continue
        text = added.get(finding.path, {}).get(finding.line)
        key = (finding.path, finding.line, finding.rule)
        if text is not None and key not in seen:
            seen.add(key)
            finding.text = text
            findings.append(finding)
    findings.sort(key=Finding.key)
    return findings, len(entries)


# ---------------------------------------------------------------------------------------------------------------------
# Semgrep rules: what malicious code does, matched on the syntax tree, so renaming a variable or splitting a line
# doesn't hide it the way it hides from the line rules above. They live here as data rather than as a .yml file in
# the repository: Windows antivirus quarantines a rule file that reads like the code it looks for, which would break
# a contributor's checkout. --write-semgrep-rules writes them out for the workflow. Written for this repository, so
# no registry rules, no login, no download and no rule licence. ERROR blocks; WARNING is listed for the reviewer.


def _either(*patterns):
    return {"pattern-either": [{"pattern": p} for p in patterns]}


def _not(*patterns):
    return [{"pattern-not": p} for p in patterns]


SEMGREP_RULES = [
    {"id": "decode-then-run", "languages": ["python"], "severity": "ERROR", "mode": "taint",
     "message": "decoded, decompressed or unpickled data reaches exec, eval or a shell; this is how a payload hides",
     "pattern-sources": [_either(
         "base64.$F(...)",
         "binascii.$F(...)",
         "codecs.decode(...)",
         "zlib.decompress(...)",
         "lzma.decompress(...)",
         "bz2.decompress(...)",
         "gzip.decompress(...)",
         "bytes.fromhex(...)",
         "marshal.loads(...)",
         "pickle.loads(...)")],
     "pattern-sinks": [_either(
         "exec(...)",
         "eval(...)",
         "compile(...)",
         "__import__(...)",
         "os.system(...)",
         "os.popen(...)",
         "subprocess.$F(...)",
         "types.FunctionType(...)")]},
    {"id": "environment-to-network", "languages": ["python"], "severity": "ERROR", "mode": "taint",
     "message": "an environment variable (where CI keeps tokens) is sent over the network",
     "pattern-sources": [_either(
         "os.environ",
         "os.environ[...]",
         "os.environ.get(...)",
         "os.getenv(...)")],
     "pattern-sinks": [_either(
         "urllib.request.urlopen(...)",
         "urllib.request.Request(...)",
         "requests.$M(...)",
         "http.client.$C(...).request(...)",
         "$S.sendall(...)",
         "$S.sendto(...)")]},
    {"id": "run-built-code", "languages": ["python"], "severity": "WARNING",
     "message": "runs code built at run time (exec, eval, a shell command made from a variable, a dynamic import)",
     "patterns": [_either(
         "exec($X, ...)",
         "eval($X, ...)",
         "__import__($X, ...)",
         "importlib.import_module($X, ...)",
         "getattr(__builtins__, $X)",
         "getattr(builtins, $X)",
         "subprocess.$F($X, ..., shell=True, ...)",
         "os.system($X)")] + _not(
         'exec("...", ...)',
         'eval("...", ...)',
         '__import__("...", ...)',
         'importlib.import_module("...", ...)',
         'subprocess.$F("...", ..., shell=True, ...)',
         'os.system("...")')},
    {"id": "js-decode-then-run", "languages": ["javascript"], "severity": "ERROR", "mode": "taint",
     "message": "decoded data reaches eval, Function or a child process; this is how a payload hides",
     "pattern-sources": [_either(
         "atob(...)",
         'Buffer.from($X, "base64")',
         'Buffer.from($X, "hex")',
         "String.fromCharCode(...)",
         "unescape(...)")],
     "pattern-sinks": [_either(
         "eval(...)",
         "Function(...)",
         "new Function(...)",
         "setTimeout($CODE, ...)",
         "$CP.exec(...)",
         "$CP.execSync(...)")]},
    {"id": "js-run-built-code", "languages": ["javascript"], "severity": "WARNING",
     "message": "runs code built at run time, or starts a process from JavaScript",
     "patterns": [_either(
         "eval($X)",
         "new Function(...)",
         'require("child_process")',
         'require("node:child_process")',
         'import $M from "child_process"',
         'import $M from "node:child_process"',
         'import { ... } from "child_process"',
         'import { ... } from "node:child_process"')] + _not('eval("...")')},
    {"id": "js-environment-to-network", "languages": ["javascript"], "severity": "ERROR", "mode": "taint",
     "message": "an environment variable (where CI keeps tokens) is sent over the network",
     "pattern-sources": [{"pattern": "process.env"}],
     "pattern-sinks": [_either(
         "fetch(...)",
         "$H.request(...)",
         "new WebSocket(...)")]},
]


def write_semgrep_rules(path):
    """Write SEMGREP_RULES as a Semgrep config (JSON, which Semgrep reads as YAML)."""
    import json
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"rules": SEMGREP_RULES}, f, indent=1)
        f.write("\n")


# ---------------------------------------------------------------------------------------------------------------------
# Results of Semgrep and Gitleaks. Their reports are untrusted too: a pull request controls the paths and lines in them.


def tool_path(path, root):
    """A scanner's path for an exported file, back to its repository path (or None when it is outside the export)."""
    path = str(path or "").replace("\\", "/")
    root = root.replace("\\", "/").rstrip("/")
    if root and path.startswith(root + "/"):
        path = path[len(root) + 1:]
    while path.startswith("./"):
        path = path[2:]
    return safe_relative(path)


def read_semgrep(path, root=""):
    """Findings from `semgrep --json`. A rule with severity ERROR blocks; the others are listed for review."""
    import json
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        results = report["results"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Fail closed: a scan that didn't happen is not a pass. Rerun the job, or a maintainer overrides it.
        return [Finding("<semgrep>", None, BLOCK, "scanner-did-not-run",
                        "Semgrep's results could not be read (%s), so its rules did not check this change; rerun "
                        "the job" % type(error).__name__)]
    found = []
    errors = report.get("errors") if isinstance(report, dict) else None
    if isinstance(errors, list) and errors:
        found.append(Finding("<semgrep>", None, REVIEW, "scanner-incomplete",
                             "Semgrep could not analyse everything (%d error%s, e.g. a file it couldn't parse or a "
                             "timeout); read those files by hand" % (len(errors), "" if len(errors) == 1 else "s")))
    for result in results if isinstance(results, list) else ():
        try:
            rel = tool_path(result["path"], root)
            line = int(result["start"]["line"])
            extra = result.get("extra") or {}
            rule = str(result["check_id"]).rsplit(".", 1)[-1]
            message = str(extra.get("message") or rule).strip()
            level = BLOCK if str(extra.get("severity", "")).upper() == "ERROR" else REVIEW
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if rel:
            found.append(Finding(rel, line, level, "semgrep:" + re.sub(r"[^\w.-]", "", rule)[:60], message[:300]))
    return found


def read_gitleaks(path, root=""):
    """Findings from a Gitleaks JSON report (written with --redact, so the secret itself is never in it)."""
    import json
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        if not isinstance(report, list):
            raise TypeError("not a list")
    except (OSError, ValueError, TypeError) as error:
        return [Finding("<gitleaks>", None, BLOCK, "scanner-did-not-run",
                        "Gitleaks's results could not be read (%s), so it did not check this change; rerun the job"
                        % type(error).__name__)]
    found = []
    for leak in report:
        try:
            rel = tool_path(leak["File"], root)
            line = int(leak["StartLine"])
            rule = re.sub(r"[^\w.-]", "", str(leak.get("RuleID") or "secret"))[:60]
            what = str(leak.get("Description") or rule)[:120]
        except (KeyError, TypeError, ValueError):
            continue
        if rel:
            found.append(Finding(rel, line, BLOCK, "gitleaks:" + rule,
                                 "Gitleaks finds a secret here (%s); revoke it, and keep credentials out of the "
                                 "repository" % what))
    return found


def read_event(path):
    """The pull request title, description and override label from a GitHub event file (all of it untrusted text)."""
    import json
    with open(path, encoding="utf-8") as f:
        event = json.load(f)
    pr = event.get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pr, dict):
        return "", "", None
    return str(pr.get("title") or ""), str(pr.get("body") or ""), override_label(event)


# ---------------------------------------------------------------------------------------------------------------------
# Output


def location(finding):
    return finding.path + (":%d" % finding.line if finding.line else "")


def level_of(finding):
    return finding.level + (" (overridden)" if finding.overridden else "")


def plain_lines(findings):
    lines = []
    for f in findings:
        lines.append("%s: %s %s: %s" % (visible(location(f), 300), level_of(f), f.rule, f.message))
        if f.text:
            lines.append("    > " + visible(f.text))
    return lines


def escape_data(s):
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(s):
    return escape_data(s).replace(":", "%3A").replace(",", "%2C")


def annotation(finding):
    """A GitHub workflow command; every piece of pull request text in it is escaped so it cannot start another."""
    kind = "error" if finding.fails() else "warning"
    props = []
    if finding.in_file():
        props.append("file=" + escape_property(finding.path))
        if finding.line:
            props.append("line=%d" % finding.line)
    where = "" if finding.in_file() else visible(location(finding), 120) + ": "
    props.append("title=" + escape_property("%s %s" % (level_of(finding), finding.rule)))
    message = where + finding.message + ("\n" + visible(finding.text) if finding.text else "")
    return "::%s %s::%s" % (kind, ",".join(props), escape_data(message))


def cell(s):
    return html.escape(s, quote=True).replace("|", "&#124;").replace("`", "&#96;").replace("\n", " ")


def summary(findings, files, override=None):
    blocks = sum(f.level == BLOCK for f in findings)
    reviews = len(findings) - blocks
    out = ["## PR security scan", ""]
    if override:
        out += ["A maintainer marked this head commit as reviewed with the label <code>%s</code>: the `block` findings "
                "below are listed but do not fail the check. A push after the review needs a new label." %
                cell(override), ""]
    if not findings:
        out.append("Nothing to report in %d changed file%s." % (files, "" if files == 1 else "s"))
        return "\n".join(out) + "\n"
    out.append("%d to fix (`block`), %d to read before merging (`review`), in %d changed file%s and the pull "
               "request's text. The check never runs the pull request's code; see `scripts/check_pr_security.py`. "
               "A maintainer who has read a `block` finding and accepts it adds the label "
               "`security-reviewed:<head sha>`." % (blocks, reviews, files, "" if files == 1 else "s"))
    out += ["", "| Level | Rule | Where | What | Line |", "|---|---|---|---|---|"]
    for f in findings:
        out.append("| %s | %s | <code>%s</code> | %s | %s |" % (
            level_of(f), f.rule, cell(visible(location(f), 300)), cell(f.message),
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
    parser.add_argument("--github-event", metavar="PATH",
                        help="a pull_request event file ($GITHUB_EVENT_PATH): scan its title and description, and "
                             "honour a security-reviewed:<head sha> label added by a code owner (needs --label-events)")
    parser.add_argument("--label-events", metavar="FILE",
                        help="the pull request's issue events (gh api --paginate .../issues/N/events), to check who "
                             "added the override label")
    parser.add_argument("--codeowners", default=os.path.join(".github", "CODEOWNERS"), metavar="FILE",
                        help="who may add the override label (default: %(default)s, from the base branch)")
    parser.add_argument("--check-action-tags", action="store_true",
                        help="check that each pinned action the change adds is the commit its # vX.Y.Z comment names "
                             "(reads the action repository's tags with git ls-remote)")
    parser.add_argument("--export", metavar="DIR",
                        help="only write the text files the change adds or modifies under DIR, for Gitleaks, and exit")
    parser.add_argument("--gitleaks-json", metavar="FILE", help="add a Gitleaks JSON report on the exported files")
    parser.add_argument("--semgrep-json", metavar="FILE", help="add Semgrep's --json results on the exported files")
    parser.add_argument("--write-semgrep-rules", metavar="FILE",
                        help="only write this script's Semgrep rules to FILE, and exit (BASE and HEAD are ignored)")
    parser.add_argument("--tool-root", default="", metavar="PATH",
                        help="the folder the exported files were under when Gitleaks read them (e.g. /src)")
    parser.add_argument("--action-remote", default=ACTION_REMOTE, metavar="TEMPLATE",
                        help="where an action's repository is, with {owner} and {repo} (default: %(default)s)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    override = None
    if args.write_semgrep_rules:
        write_semgrep_rules(args.write_semgrep_rules)
        print("| wrote %d Semgrep rules" % len(SEMGREP_RULES))
        return 0
    if args.export:
        try:
            written = export_changed(args.repo, args.base, args.head, args.export)
        except (GitError, OSError) as error:
            print("check_pr_security: %s" % visible(str(error), 2000), file=sys.stderr)
            return 2
        print("| wrote %d changed text file%s for the other scanners" % (written, "" if written == 1 else "s"))
        return 0
    external = []
    if args.semgrep_json:
        external += read_semgrep(args.semgrep_json, args.tool_root)
    if args.gitleaks_json:
        external += read_gitleaks(args.gitleaks_json, args.tool_root)
    try:
        findings, files = scan(args.repo, args.base, args.head, args.check_action_tags, args.action_remote,
                               external=external)
        if args.github_event:
            title, body, override = read_event(args.github_event)
            findings = check_text("<pull request title>", title) + check_text("<pull request description>", body) + \
                findings
    except GitError as error:
        print("check_pr_security: %s" % visible(str(error), 2000), file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print("check_pr_security: %s" % visible(str(error), 2000), file=sys.stderr)
        return 2
    if override:
        # Only a code owner's label waives anything; without the events to check that, nothing is waived.
        try:
            ok, who = override_authorized(override, read_label_events(args.label_events), codeowners(args.codeowners)) \
                if args.label_events else (False, "")
        except (OSError, ValueError) as error:
            ok, who = False, ""
            print("check_pr_security: label events: %s" % visible(str(error), 300), file=sys.stderr)
        if not ok:
            findings.append(Finding("<pull request labels>", None, REVIEW, "override-not-authorized",
                                    "the label %s was not added by someone in .github/CODEOWNERS (%s), so it waives "
                                    "nothing" % (override, ("added by " + who) if who else "who added it is unknown")))
            override = None
        else:
            override = "%s (added by %s)" % (override, who)
    if override:
        for finding in findings:
            if finding.level == BLOCK:
                finding.overridden = True
    # Every printed line starts with a fixed prefix or a workflow command this script built, so a file name or a
    # line of the pull request can never be read by the runner as a command of its own.
    for line in plain_lines(findings):
        print("| " + line)
    blocks = sum(f.level == BLOCK for f in findings)
    failing = sum(f.fails() for f in findings)
    print("| %d block%s, %d review, %d changed file%s" % (
        blocks, " (overridden by %s)" % visible(override, 80) if override and blocks else "", len(findings) - blocks,
        files, "" if files == 1 else "s"))
    if args.github:
        for finding in findings:
            print(annotation(finding))
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if path:
            with open(path, "a", encoding="utf-8", newline="\n") as out:
                out.write(summary(findings, files, override))
    if failing or (args.strict and findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

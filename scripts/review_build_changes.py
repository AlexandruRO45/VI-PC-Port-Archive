"""List added build-script lines that merit review. Run against two local Git trees."""
import argparse
import re
import subprocess
from urllib.parse import quote_from_bytes


BUILD_FILES = [
    "make_game.py", "tools/**/*.py", "scripts/**/*.py", "**/*.bat", "**/*.cmd",
    "**/*.ps1", "**/*.sh", "**/CMakeLists.txt", "**/*.cmake", "**/*.csproj",
    "**/*.props", "**/*.targets", "**/build.rs",
]
INDICATORS = {
    "Network access": rb"\b(?:urlopen|urlretrieve|requests\.(?:get|post|put|request)|"
                      rb"socket\.create_connection|Invoke-WebRequest|Invoke-RestMethod|"
                      rb"curl(?:\.exe)?|wget|Start-BitsTransfer|Download(?:String|File)|"
                      rb"FetchContent_Declare)\b|\bfile\s*\(\s*DOWNLOAD\b",
    "User directories": rb"\b(?:expanduser|USERPROFILE|APPDATA|LOCALAPPDATA)\b|"
                         rb"\bPath\.home\s*\(|\$HOME\b|%HOME%|\$env:HOME\b",
    "Dynamic execution": rb"\b(?:Invoke-Expression|EncodedCommand|iex)\b|"
                          rb"\b(?:exec|eval)\s*\(|\|\s*(?:bash|sh|pwsh|powershell)\b",
}
PATTERNS = {label: re.compile(pattern, re.I) for label, pattern in INDICATORS.items()}
HUNK = re.compile(rb"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
LIMIT = 50


def git(*args):
    return subprocess.check_output(["git", *args])


def findings(base, head):
    files = git("diff", "--no-renames", "--name-only", "--diff-filter=AMT", "-z", base, head,
                "--", *[":(glob)" + path for path in BUILD_FILES])
    for path in filter(None, files.split(b"\0")):
        patch = git("--literal-pathspecs", "diff", "--no-color", "--no-ext-diff", "--no-textconv",
                    "--no-renames", "-U0", base, head, "--", path)
        line_number = None
        for line in patch.split(b"\n"):
            hunk = HUNK.match(line)
            if hunk:
                line_number = int(hunk[1])
            elif line_number is not None and line.startswith(b"+"):
                labels = [label for label, pattern in PATTERNS.items() if pattern.search(line[1:])]
                if labels:
                    yield path, line_number, ", ".join(labels)
                line_number += 1
            elif line_number is not None and line.startswith(b" "):
                line_number += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("repository_url")
    args = parser.parse_args()
    # Resolve to immutable commits before constructing links or passing revisions to diff.
    base, head = [git("rev-parse", "--verify", "--end-of-options", rev + "^{commit}").decode().strip()
                  for rev in (args.base, args.head)]
    print("### Build-script changes for review\n")
    count = 0
    for path, line, labels in findings(base, head):
        count += 1
        if count <= LIMIT:
            encoded = quote_from_bytes(path)
            print(f"- {labels} in [`{encoded}:{line}`]({args.repository_url}/blob/{head}/{encoded}#L{line})")
    if count > LIMIT:
        print(f"\nShowing the first {LIMIT} of {count} matching lines.")
    if not count:
        print("No added lines matched the configured indicators.")


if __name__ == "__main__":
    main()

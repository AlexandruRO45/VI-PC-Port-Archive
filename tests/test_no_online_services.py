"""No file the repository tracks names an Xbox Live or Rockstar service domain.

    python -I tests/test_no_online_services.py

AGENTS.md's hard rule is "Nothing online: no Xbox Live or Rockstar services or domains", and nothing checked it:
`scripts/check_no_game_data.py` covers game data and the workflows cover file types, not host names, so a shim or tool
that called an Xbox Live or Rockstar service would pass CI today. This reads every file `git ls-files` lists (the same
boundary check_no_game_data.py uses, so a contributor's own work folder, reports and logs are not scanned), skips the
binary files by their NUL byte, and fails on one that names a service host.

Only host names count, not the words: the SDK's own method names (`XSystemGetXboxLiveSandboxId`) and prose about the
rule stay legal, and so does a domain that merely ends in one of these names. No game files, no network.
"""
import os
import re
import shutil
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This file has to spell the hosts it refuses, in the sample text the detector is checked on, so the scan skips it.
SELF = "tests/test_no_online_services.py"
# The service hosts the rule names. A host name, so the leading part is a subdomain chain (prod.xboxservices.com) and
# the match may not sit inside a longer name (notxboxlive.com is another host, not this one).
SERVICES = re.compile(
    r"(?<![\w.-])(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)*"
    r"(?:xboxlive\.com|xboxservices\.com|rockstargames\.com)(?![\w.-])",
    re.IGNORECASE,
)


def service_hosts(text):
    """[(1-based line number, host)] for every service host name in `text`."""
    found = []
    for number, line in enumerate(text.split("\n"), 1):
        found.extend((number, match.group(0)) for match in SERVICES.finditer(line))
    return found


def tracked_files(root):
    """The paths `git ls-files` lists for the checkout at `root`, relative to it."""
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"], check=True, capture_output=True).stdout
    return [path for path in out.decode("utf-8", "surrogateescape").split("\0") if path]


def offending_files(root):
    """`["<path>:<line>: <host>"]` for each service host name in a tracked file, in `git ls-files` order."""
    found = []
    for relative in tracked_files(root):
        if relative == SELF:
            continue  # the sample text above spells the hosts on purpose
        path = os.path.join(root, relative)
        if os.path.islink(path) or not os.path.isfile(path):
            continue  # a submodule, or a file deleted in the working tree
        with open(path, "rb") as f:
            raw = f.read()
        if b"\0" in raw:
            continue  # no host name to read in a binary file
        found.extend(f"{relative}:{number}: {host}"
                     for number, host in service_hosts(raw.decode("utf-8", "replace")))
    return found


@unittest.skipUnless(shutil.which("git"), "needs git to list the tracked files")
class NoOnlineServicesTest(unittest.TestCase):
    def test_detector_reads_hosts_not_the_words(self):
        text = "\n".join((
            "https://socialclub.rockstargames.com/games",      # 1
            "XSystemGetXboxLiveSandboxId",                     # 2 the SDK's method name, not a host
            "https://XBOXLive.com",                            # 3
            "https://prod.xboxservices.com/v1",                # 4
            "notxboxlive.com",                                 # 5 another host
            "rockstargames.com.example.org",                   # 6 a host that only starts like one
        ))
        self.assertEqual(service_hosts(text), [(1, "socialclub.rockstargames.com"), (3, "XBOXLive.com"),
                                               (4, "prod.xboxservices.com")])

    def test_no_tracked_file_names_a_service_host(self):
        found = offending_files(ROOT)
        self.assertEqual(found, [], "tracked files naming an Xbox Live or Rockstar service domain "
                                    "(nothing online: AGENTS.md): " + ", ".join(found))


if __name__ == "__main__":
    unittest.main()

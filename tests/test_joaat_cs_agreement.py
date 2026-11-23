"""One case-sensitive joaat rule across tools: every ``joaat_cs`` copy encodes a name the same way.

    python -I tests/test_joaat_cs_agreement.py

``tests/test_joaat_agreement.py`` pins the lower-cased copies (``def joaat``/``def joaat32``) to
``tools/hashdict.py``; its scan skips the ``joaat_cs`` family by name, so the case-sensitive copies the
0x7a schema tables hash with went unchecked. They have the same failure mode: nothing stopped one of them
from encoding the name differently. One did -- navmesh_decode hashed the ASCII bytes, so a schema name with
a non-ASCII byte raised ``UnicodeEncodeError`` instead of the hash fog_volumes and world_archetypes (and the
engine) compute from the Latin-1 bytes, and an accented struct or member would never resolve.

The reference here is ``hashdict.joaat`` without the lower-casing step: the same Jenkins one-at-a-time over
the Latin-1 bytes. The first test below pins that to ``hashdict.joaat`` on an already-lower-case name, so the
two families stay one hash.

Synthetic names only; no game data.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for path in (ROOT, TOOLS):
    if path not in sys.path:
        sys.path.insert(0, path)

import hashdict  # noqa: E402

# Schema-style identifiers (what the 0x7a tables actually carry), then names with a non-ASCII Latin-1
# character: "é" is one Latin-1 byte (0xE9), and "ÉLÉGIE" also tells a case-sensitive hash apart from a
# lower-cased one. Every character is a single Latin-1 byte, so a copy that forgot the encoding is the
# only one that differs.
NAMES = [
    "CMapTypes", "CCoverPointRegion", "eCoverFlags", "rage__spdAABB", "COVUSE_WALLTOLEFT",
    "café", "Café", "Señor", "Crème Brûlée", "ÉLÉGIE",
]

# The modules whose name a reviewer would expect here: a rename must not turn the loop below into a
# test of nothing.
EXPECTED = ("fog_volumes", "navmesh_decode", "world_archetypes")


def joaat_cs_reference(text):
    """The pinned case-sensitive hash: hashdict.joaat over the Latin-1 bytes, without the lower-casing."""
    h = 0
    for c in text.encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def case_sensitive_copies():
    """{module name: function} for every ``def joaat_cs(`` in tools/ that takes a string."""
    copies = {}
    for filename in sorted(os.listdir(TOOLS)):
        if not filename.endswith(".py"):
            continue
        with open(os.path.join(TOOLS, filename), encoding="utf-8") as handle:
            source = handle.read()
        if re.search(r"^def joaat_cs\(\s*[A-Za-z_]\w*", source, re.M):
            module = __import__(filename[:-3])
            copies[filename[:-3]] = module.joaat_cs
    return copies


class JoaatCsAgreementTest(unittest.TestCase):
    def test_reference_is_hashdict_without_lower_casing(self):
        # Anchor the reference to the pinned dictionary: on a name that is already lower case the two
        # agree, so the case-sensitive family is the same hash, not a second one that happens to look like it.
        for lowered in ("ccoverpointregion", "rage__spdaabb", "plain_name", "café"):
            self.assertEqual(joaat_cs_reference(lowered), hashdict.joaat(lowered),
                             f"reference disagrees with hashdict.joaat on the lower-case name {lowered!r}")
        # Known values, so a future edit to the reference is caught on its own.
        self.assertEqual(joaat_cs_reference("CMapTypes"), 0xD98BB561)
        self.assertEqual(joaat_cs_reference("café"), 0xEEFBF723)

    def test_every_copy_matches_the_reference(self):
        copies = case_sensitive_copies()
        for module in EXPECTED:
            self.assertIn(module, copies)
        for module, function in sorted(copies.items()):
            for name in NAMES:
                with self.subTest(copy=module, name=name):
                    self.assertEqual(function(name), joaat_cs_reference(name),
                                     f"{module}.joaat_cs hashes {name!r} differently from the pinned reference")

    def test_the_hash_is_case_sensitive(self):
        # The point of the _cs copies: unlike hashdict.joaat they keep the case, so a struct and its
        # lower-cased spelling land on different ids.
        for module, function in sorted(case_sensitive_copies().items()):
            with self.subTest(copy=module):
                self.assertNotEqual(function("CMapTypes"), function("cmaptypes"),
                                    f"{module}.joaat_cs lost the case distinction")


if __name__ == "__main__":
    unittest.main()

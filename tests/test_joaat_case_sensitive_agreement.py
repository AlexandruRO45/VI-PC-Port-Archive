"""The case-sensitive joaat copy in world_names uses the same Latin-1 bytes as the 0x7a schema copies.

    python -I tests/test_joaat_case_sensitive_agreement.py

The default (lowercased) copy is covered by tests/test_joaat_agreement.py. This covers the exact-bytes
variant used for schema struct and member names, where ASCII names hid the UTF-8/Latin-1 split.
Synthetic names only; no game data.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for path in (ROOT, TOOLS):
    if path not in sys.path:
        sys.path.insert(0, path)

import fog_volumes  # noqa: E402
import hashdict  # noqa: E402
import world_archetypes  # noqa: E402
import world_names  # noqa: E402

NAMES = ("CMapData", "archetypeName", "café", "Café", "ÉLÉGIE", "Señor")


class CaseSensitiveJoaat(unittest.TestCase):
    def test_world_names_matches_the_schema_copies(self):
        for name in NAMES:
            with self.subTest(name=name):
                expected = world_archetypes.joaat_cs(name)
                self.assertEqual(fog_volumes.joaat_cs(name), expected)
                self.assertEqual(world_names.joaat(name, case_sensitive=True), expected)

    def test_default_lowercased_hash_is_unchanged(self):
        for name in NAMES:
            with self.subTest(name=name):
                self.assertEqual(world_names.joaat(name), hashdict.joaat(name))

    def test_case_remains_significant(self):
        self.assertNotEqual(world_names.joaat("PLAYER", case_sensitive=True),
                            world_names.joaat("player", case_sensitive=True))


if __name__ == "__main__":
    unittest.main()

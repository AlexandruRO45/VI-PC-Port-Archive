"""The skinned-GLB docstring and the character-skeleton prose name the measured palette offsets.

    python -I tests/test_character_skeleton_offsets.py
"""
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaletteOffsetDocs(unittest.TestCase):
    def test_skinned_glb_docstring_names_the_measured_offsets(self):
        text = (ROOT / "tools" / "skinned_glb.py").read_text(encoding="utf-8")
        self.assertNotIn("grmModel + 0x20", text)
        self.assertNotIn("grmModel + 0x0C", text)
        self.assertIn("grmModel + 0x28", text)
        self.assertIn("grmModel + 0x14", text)

    def test_character_skeleton_prose_names_the_measured_offsets(self):
        text = (ROOT / "docs" / "character_skeleton.md").read_text(encoding="utf-8")
        self.assertNotIn("grmModel + 0x20", text)
        self.assertIn("grmModel + 0x28", text)
        self.assertIn("| `+0x14` | local bone count |", text)


if __name__ == "__main__":
    unittest.main()

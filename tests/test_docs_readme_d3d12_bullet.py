"""The README's D3D12 "want to help" bullet names an open piece the plan actually lists.

    python -I tests/test_docs_readme_d3d12_bullet.py

"Indirect draw state" is not a term docs/d3d12_vulkan.md uses: it records `ExecuteIndirect` as
implemented, with only its bundle replay still missing. A help bullet that names the vague term
points contributors at work that is already done. This test pins the bullet to the plan's wording.
No game files, keys or build tools.
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def d3d12_bullet(text):
    """The Xbox D3D12 bullet in README.md, with its continuation lines joined."""
    lines = text.splitlines()
    for n, line in enumerate(lines):
        if line.startswith("- **Xbox D3D12"):
            parts = [line.strip()]
            for following in lines[n + 1:]:
                if following.startswith("- ") or following.startswith("## "):
                    break
                if not following.strip():
                    break
                parts.append(following.strip())
            return " ".join(parts)
    raise AssertionError("README.md has no Xbox D3D12 help bullet")


class ReadmeD3d12Bullet(unittest.TestCase):
    def test_plan_records_the_open_bundle_replay(self):
        with open(os.path.join(ROOT, "docs", "d3d12_vulkan.md"), encoding="utf-8") as f:
            plan = f.read()
        self.assertIn("Not deferred yet", plan)
        self.assertIn("`ExecuteIndirect` in a bundle", plan)

    def test_bullet_names_the_missing_piece_not_a_vague_term(self):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            bullet = d3d12_bullet(f.read())
        self.assertNotIn("indirect draw state", bullet)
        self.assertIn("`ExecuteIndirect` in bundles", bullet)


if __name__ == "__main__":
    unittest.main()

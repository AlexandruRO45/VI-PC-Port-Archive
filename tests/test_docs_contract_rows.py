"""docs/design.md contract rows must not contradict the code they describe.

Each row named below once carried a stale sentence from before a fix
landed, left beside its corrected version. Pin the corrected contract so a
later edit cannot silently resurrect the old text.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def row(name):
    for line in (ROOT / "docs" / "design.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cell = line.split("|", 2)[1].strip().strip("`")
        if cell == name:
            return line
    raise AssertionError(f"row {name!r} not found in docs/design.md")


class ContractRowsTest(unittest.TestCase):
    def test_gather4_border_modes(self):
        # gather_addr_modes refuses only modes 4 and 5 on a literal S#;
        # modes 6 and 7 return the S# border colour instead.
        text = row("Other additions")
        self.assertIn("modes 4 and 5 act as clamp-to-edge", text)
        self.assertNotIn("4-7: half-border, border colour", text)
        self.assertNotIn("modes 4-7", text)

    def test_s_cmov_b64_not_unsupported(self):
        # The translator handles s_cmov_b64 for plain data pairs and
        # refuses only lane-mask operands.
        text = row("Scalar select and pack (`s_cmov_b32`, `s_cmovk_i32`, `s_pack_ll/lh/hh_b32_b16`)")
        self.assertIn("translates plain 64-bit data pairs", text)
        self.assertNotIn("`s_cmov_b64`, stay `Unsupported`", text)

    def test_fma_mix_measured_and_fixed(self):
        # #1774 (SHADER-296) landed: nearest-mode v_fma_mix* rounds the
        # exact fma once via Gen.fma_mix_odd, and all seven functions are
        # measured.
        text = row("Known approximations")
        self.assertIn("Gen.fma_mix_odd", text)
        self.assertNotIn("still narrows an f32 `Fma`", text)
        self.assertNotIn("still rounds twice", text)
        self.assertNotIn("are unmeasured", text)

    def test_atomic_x2_exchange_can_tear(self):
        # An exchange is not covered by the bitwise-op argument: two lanes
        # swapping into one element can interleave dwords, and the bounds
        # check on the 8-byte element is measured all-or-nothing.
        text = row("buffer_atomic_{and,or,xor,swap}_x2")
        self.assertIn("an exchange can tear", text)
        self.assertNotIn("or an exchange leaves the same memory", text)
        self.assertNotIn("only partly in range is **not measured**", text)

    def test_swizzle_is_modelled(self):
        # SWIZZLE_ENABLE / ADD_TID_ENABLE are modelled; the stale "not
        # modelled, silently wrong" sentence is gone.
        text = row("Buffer bounds checking")
        self.assertIn("are modelled; see the Swizzled buffer addressing row", text)
        self.assertNotIn("are not modelled", text)
        self.assertNotIn("silently wrong", text)


if __name__ == "__main__":
    unittest.main()

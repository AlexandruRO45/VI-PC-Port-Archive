"""A pixel shader's EXEC at entry holds the covered pixels only; helper lanes join at s_wqm (TEST-363).

RDNA2 starts a pixel wave with EXEC = the covered pixels. A helper lane of a partly covered 2x2 quad is off until
s_wqm turns its quad on, as LLVM's prologue does before anything that needs derivatives (`s_mov s[n], exec` then
`s_wqm exec, exec`, and EXEC restored from s[n] before exports and stores). The translator starts each invocation's
EXEC bit as !HelperInvocation. GPU-checked by w64_graphics.py's @partial checks on an RTX 3070 Ti; docs/design.md,
"Pixel-shader EXEC at entry". Hand-written programs, no tools needed.
"""
import unittest

import support
from spirv_wave_interp import run_wave

EXP = ["v_mov_b32_e32 v2, s4", "exp mrt0 v2, v2, v2, v2 done vm"]
END = ["s_endpgm"]


def sized(lines):
    """The lines with a placeholder encoding of the right size (exports are 8 bytes, the rest 4 here)."""
    return [f"{line} ; encoding: [" + ",".join(["0x00"] * (8 if line.startswith("exp ") else 4)) + "]"
            for line in lines]


def count_seen(body, helpers):
    """Runs a wave32 pixel shader that leaves a lane count in s4 and exports it: {lane: exported word, or None if the
    lane was killed}. `helpers` are the lanes that are helper invocations."""
    text = support.translate_lines("ps_entry_exec", sized(body + EXP + END))
    out = run_wave(text, wave=32, builtins={"HelperInvocation": lambda lane: lane in helpers})["outputs"]
    return {lane: None if o is None else o["%out_mrt0"][0] for lane, o in enumerate(out)}


# Lane 4n + 3 of every quad is a helper, and quad 7 (lanes 28-31) is all helpers: 24 - 3 = 21 covered pixels.
HELPERS = {4 * q + 3 for q in range(8)} | {28, 29, 30, 31}
COVERED = sorted(set(range(32)) - HELPERS)


class EntryExec(unittest.TestCase):
    def test_the_entry_exec_counts_covered_pixels_only(self):
        seen = count_seen(["s_bcnt1_i32_b32 s4, exec_lo"], HELPERS)
        self.assertEqual(len(COVERED), 21)
        self.assertEqual({lane: seen[lane] for lane in COVERED}, dict.fromkeys(COVERED, 21))

    def test_whole_quad_mode_brings_in_the_helpers_of_covered_quads(self):
        # s_wqm turns on the 7 quads with a covered pixel (28 lanes), not quad 7, which has none.
        body = ["s_mov_b32 s10, exec_lo", "s_wqm_b32 exec_lo, exec_lo", "s_bcnt1_i32_b32 s4, exec_lo",
                "s_mov_b32 exec_lo, s10"]
        seen = count_seen(body, HELPERS)
        self.assertEqual({lane: seen[lane] for lane in COVERED}, dict.fromkeys(COVERED, 28))

    def test_a_valu_write_does_not_land_in_a_helper_lane(self):
        # v_mov v2, s4 runs with the helper off, so it exports v2's start value, 0 (the GPU drops a helper's output).
        seen = count_seen(["s_bcnt1_i32_b32 s4, exec_lo"], HELPERS)
        self.assertEqual({lane: seen[lane] for lane in HELPERS}, dict.fromkeys(HELPERS, 0))

    def test_without_helpers_every_lane_is_in_the_entry_exec(self):
        seen = count_seen(["s_bcnt1_i32_b32 s4, exec_lo"], set())
        self.assertEqual(seen, dict.fromkeys(range(32), 32))


class Module(unittest.TestCase):
    def test_the_active_flag_starts_as_not_helper_invocation(self):
        text = support.translate_lines("ps_entry_exec", sized(["s_bcnt1_i32_b32 s4, exec_lo"] + EXP + END))
        self.assertIn("OpDecorate %helper_inv BuiltIn HelperInvocation", text)
        self.assertRegex(text, r"%live_at_entry = OpLogicalNot %bool %helper_at_entry\s+OpStore %active %live_at_entry")


if __name__ == "__main__":
    unittest.main()

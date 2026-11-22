"""A constant reaching a lane-mask operand through an SGPR (SHADER-304): lane L takes bit L of it.

`s_mov_b32 s20, 0x3ffffc00` leaves the raw constant in every invocation, while a lane-mask reader
(`Gen.mask_src`) takes an SGPR as this lane's 0 / ~0 bit, so until 2026-11-22 any non-zero constant was every
lane: `s_and_saveexec_b32 s6, s20` left 32 lanes on instead of 10..29. The immediate form was already right
(the "Constant lane masks" row of docs/design.md). Each case runs a translated pixel shader as one wave in
tests/semantic/spirv_wave_interp.py and checks which lanes end up active. Standard library only; no game files.

    python -I tests/semantic/test_sgpr_const_mask.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_salu64_bitwise_data import run_ps_wave  # noqa: E402  (also puts tools/ on sys.path)


def active(setup, op, wave=32):
    """The lanes still active after `setup` lines and `op`, starting from a full wave: v1 = 0x22 where active."""
    lines = (["v_mov_b32 v0, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "v_mov_b32 v1, 0x11"] + setup +
             [op, "v_mov_b32 v1, 0x22", f"s_mov_b{wave} exec, -1"])
    outs, _ = run_ps_wave(lines, wave=wave)
    return {lane for lane in range(wave) if outs[lane][1] == 0x22}


def bits(m, wave=32):
    return {lane for lane in range(wave) if m >> lane & 1}


class SgprConstantMask(unittest.TestCase):
    def test_saveexec_b32_reads_the_sgpr_constant(self):
        # @LouisRiverstone's repro: hardware gives lanes 10..29, the same as the immediate form.
        want = bits(0x3FFFFC00)
        self.assertEqual(active(["s_mov_b32 s20, 0x3ffffc00"], "s_and_saveexec_b32 s6, s20"), want)
        self.assertEqual(active([], "s_and_saveexec_b32 s6, 0x3ffffc00"), want)

    def test_other_readers_and_a_copy(self):
        cases = [
            (["s_mov_b32 s20, 0xff00"], "s_mov_b32 exec_lo, s20", bits(0xFF00)),
            (["s_mov_b32 s20, 5", "s_mov_b32 s21, s20"], "s_and_saveexec_b32 s6, s21", bits(5)),
            (["s_mov_b32 s20, -16"], "s_and_saveexec_b32 s6, s20", bits(-16 & 0xFFFFFFFF)),  # inline, sign-extended
            (["s_mov_b32 s20, 0xff00"], "s_andn2_wrexec_b32 s6, s20", set()),         # S0 & ~EXEC, EXEC all on
            (["s_mov_b32 exec_lo, 0xff", "s_mov_b32 s20, 0xff00"], "s_or_saveexec_b32 s6, s20", bits(0xFFFF)),
        ]
        for setup, op, want in cases:
            with self.subTest(op=op):
                self.assertEqual(active(setup, op), want)

    def test_cndmask_mask_operand(self):
        # v_cndmask_b32_e64 with an SGPR mask: lane L picks src1 when bit L of the constant is set.
        outs, _ = run_ps_wave(["v_mov_b32 v0, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "s_mov_b32 s20, 0xff00",
                               "v_cndmask_b32_e64 v1, 0x11, 0x22, s20"], wave=32)
        self.assertEqual({lane for lane in range(32) if outs[lane][1] == 0x22}, bits(0xFF00))

    def test_wave64_pair(self):
        # The pair's two registers are the mask's two halves: 0xf0 gives lanes 4..7, a high word of 1 lane 32.
        self.assertEqual(active(["s_mov_b32 s20, 0xf0", "s_mov_b32 s21, 1"],
                                "s_and_saveexec_b64 s[6:7], s[20:21]", wave=64), {4, 5, 6, 7, 32})
        self.assertEqual(active(["s_mov_b64 s[20:21], -16"], "s_and_saveexec_b64 s[6:7], s[20:21]", wave=64),
                         set(range(4, 64)))

    def test_not_a_constant_any_more(self):
        # Overwritten by something other than a constant move: read as before (a compare result, a mask).
        self.assertEqual(active(["s_mov_b32 s20, 0xff00", "v_mbcnt_lo_u32_b32 v4, -1, 0",
                                 "v_cmp_gt_u32_e64 s20, 3, v4"], "s_and_saveexec_b32 s6, s20"), {0, 1, 2})
        # A saved EXEC copied over the constant is the mask it holds.
        self.assertEqual(active(["s_mov_b32 s20, 0xff00", "v_mbcnt_lo_u32_b32 v4, -1, 0",
                                 "v_cmp_gt_u32_e64 s21, 2, v4", "s_mov_b32 s20, s21"],
                                "s_and_saveexec_b32 s6, s20"), {0, 1})


if __name__ == "__main__":
    unittest.main()

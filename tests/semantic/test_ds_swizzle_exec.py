"""ds_swizzle_b32 (SHADER-678): a source lane that is off in EXEC reads as 0, as ds_bpermute_b32 does on RDNA2.

Hand-written program, no game data. Data v2 = 7 * lane + 1; EXEC is set only around the swizzle; lane l stores v10.
"""
import unittest

import spirv_mini_interp as si
from test_ds_permute import OLD, VSHARP, run, translate_text


def program(pattern, exec_mask):
    return ["v_mad_u32_u24 v2, v0, 7, 1", f"v_mov_b32_e32 v10, {OLD}", f"s_mov_b32 exec_lo, {exec_mask:#x}",
            f"ds_swizzle_b32 v10, v2 offset:swizzle({pattern})", "s_waitcnt lgkmcnt(0)", "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v21, 2, v0", "buffer_store_dword v10, v21, s[0:3], 0 offen", "s_endpgm"]


class SwizzleExec(unittest.TestCase):
    def check(self, pattern, src, mask):
        got = run(translate_text("cs_swz_exec", program(pattern, mask), 32), 32, 32)
        for l in range(32):
            if not (mask >> l) & 1:
                want = OLD  # off lane keeps its VDST
            else:
                want = 7 * src(l) + 1 if (mask >> src(l)) & 1 else 0
            self.assertEqual(got[l], want, f"{pattern} lane {l}")

    def test_swap_even_lanes(self):
        self.check("SWAP,1", lambda l: l ^ 1, 0x55555555)  # every source is odd, so off: all 0

    def test_swap_all_lanes_unchanged(self):
        self.check("SWAP,1", lambda l: l ^ 1, 0xFFFFFFFF)

    def test_bitmask_half(self):
        self.check('BITMASK_PERM,"pppp1"', lambda l: l | 1, 0x0000FFFF)  # lanes 0-15 read odd lanes, on
        self.check('BITMASK_PERM,"ppppi"', lambda l: l ^ 1, 0x0000FFF7)  # lane 2 reads lane 3, which is off


if __name__ == "__main__":
    unittest.main()

"""The translator makes one pointer per register and basic block, not one per access.

`Gen.load_reg` and `store_reg` used to emit a new `OpAccessChain` on `%vgpr` or `%sgpr` for every read and write.
ScreenProbes_ActivateProbes had 244,087 of them (16.6% of its words), most repeating the previous one. The pointer is
now made once in a basic block and reused until the block ends (an OpLabel, a block boundary, an emulated segment cut).
Standard library only, no GPU, no game files. A straight-line program that touches v7, v8 and v9 many times has a
single chain for each, and still computes the right words. That no pointer is reused past a block end is covered by
the rest of tests/semantic: its branch, loop and wave64-emulation tests fail with an invalid or wrong module if one is.
"""
import re
import unittest

import support

TAIL = ["exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]


def translate(lines):
    text = "; ptr_cache\n" + "".join("\t" + ln + "\n" for ln in lines + TAIL)
    return support.translate_listing("ps_ptr_cache", text.encode("utf-8"))


def chains(asm, arr, idx):
    return len(re.findall(rf"OpAccessChain %_ptr_Function_uint {arr} %c{idx:x}\b", asm))


class OnePointerPerBlock(unittest.TestCase):
    STRAIGHT = ["v_mov_b32 v7, 5",
                "v_add_nc_u32 v8, v7, v7",
                "v_mul_lo_u32 v9, v7, v8",
                "v_add_nc_u32 v7, v7, v9",
                "v_xor_b32 v10, v7, v8",
                "v_mov_b32 v11, v7",
                "v_mov_b32 v12, v9",
                "v_mov_b32 v13, v8"]

    def test_straight_line_has_one_chain_per_register(self):
        asm = translate(self.STRAIGHT)
        self.assertEqual(chains(asm, "%vgpr", 7), 1)
        self.assertEqual(chains(asm, "%vgpr", 8), 1)
        self.assertEqual(chains(asm, "%vgpr", 9), 1)

    def test_words(self):
        # v7 = 5, v8 = 10, v9 = 50, v7 = 55, v10 = 55 ^ 10 = 61
        words = support.mrt0_words(translate(self.STRAIGHT))
        self.assertEqual(words, [61, 55, 50, 10])


def active_loads(asm):
    return len(re.findall(r"= OpLoad %bool %active\b", asm))


class OneActiveLoadPerExecState(unittest.TestCase):
    """Every VALU write needs this lane's EXEC bit. One load of %active serves the writes until EXEC changes."""
    WRITES = ["v_mov_b32 v7, 5", "v_add_nc_u32 v8, v7, v7", "v_mul_lo_u32 v9, v7, v8", "v_mov_b32 v10, v9",
              "v_mov_b32 v11, v8", "v_mov_b32 v12, v7", "v_mov_b32 v13, v7"]

    def test_straight_line_loads_active_once(self):
        asm = translate(self.WRITES)
        self.assertEqual(active_loads(asm), 1)
        self.assertEqual(support.mrt0_words(asm), [50, 10, 5, 5])

    def test_exec_write_in_between_loads_again(self):
        lines = self.WRITES[:3] + ["s_mov_b32 exec_lo, 1"] + self.WRITES[3:]
        asm = translate(lines)
        self.assertGreater(active_loads(asm), active_loads(translate(self.WRITES)))
        self.assertEqual(support.mrt0_words(asm), [50, 10, 5, 5])


if __name__ == "__main__":
    unittest.main()

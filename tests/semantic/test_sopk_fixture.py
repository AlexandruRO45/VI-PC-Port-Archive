"""Check signed scalar immediates through the translated wave32 shader."""

import struct
import unittest

import spirv_mini_interp as si
import support


class SopkFixture(unittest.TestCase):
    def test_result_and_scc(self):
        # Four user SGPRs for the output buffer descriptor, then the workgroup ID.
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x88, 32, 1, 1)
        spv = support.translate_fixture("cs_w32_sopk", fssm=bytes(fssm))
        memory = si.run_workgroup(
            spv, 1, 32,
            inputs_for=lambda lane: {
                "%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0), "%sg_lane": lane,
            },
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
            memory={},
        )
        self.assertEqual(memory, {
            0: (-32768) & 0xFFFFFFFF,           # s_movk sign-extends 0x8000
            4: 0,                              # -32768 does not equal unsigned 0x8000
            8: (7 * -3) & 0xFFFFFFFF,
            12: 0,                             # s_mulk leaves the clear SCC alone
            16: -32768 * -3,
            20: 1,                             # s_mulk keeps SCC from the signed equality
            24: 0x7FFFFFFF + 1,
            28: 1,                             # s_addk sets SCC on signed overflow
        })


if __name__ == "__main__":
    unittest.main()

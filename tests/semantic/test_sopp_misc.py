"""s_cbranch_cdbgsys / cdbguser / cdbgsys_or_user / cdbgsys_and_user (never taken without a shader debugger),
s_wait_idle (no effect) and s_endpgm_ordered_ps_done (an s_endpgm): runtime/test/shaders/cs_w32_sopp_misc.asm,
docs/design.md. Run through spirv_mini_interp on all 32 lanes. Before, all six raised Unsupported."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import rdna2spirv  # noqa: E402

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")


class SoppMisc(unittest.TestCase):
    def test_fixture_words(self):
        sys.path.insert(0, SHADERS)
        from compute_fixture import compute_header
        hdr = bytes(compute_header(os.path.join(SHADERS, "cs_w32_sopp_misc.asm")))
        spv = support.translate_fixture("cs_w32_sopp_misc", fssm=hdr)
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        # Plain reference: every cdbg branch falls through (its v_mov runs), nothing after the program end runs.
        want = [0x11, 0x22, 0x33, 0x44, 5, 6, 7, 8]
        self.assertEqual([memory[4 * i] for i in range(8)], want)

    def test_normalize(self):
        ins = [{"op": "s_cbranch_cdbgsys", "args": [{"k": "imm", "v": 1}], "mods": [], "pc": 0, "enc": "00000000"},
               {"op": "s_endpgm_ordered_ps_done", "args": [], "mods": [], "pc": 4, "enc": "00000000"}]
        out = rdna2spirv.normalize_sopp(ins)
        self.assertEqual([(i["op"], i["pc"]) for i in out], [("s_nop", 0), ("s_endpgm", 4)])
        self.assertIn("s_wait_idle", rdna2spirv.NOOPS)


if __name__ == "__main__":
    unittest.main()

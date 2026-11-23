"""image_sample with tfe/lwe writes the status VGPR (vdata[data_count]).

RDNA2: image_sample* with tfe (texel feedback) or lwe writes a status word into the VGPR
immediately after the last data component (vdata[data_count]). rdna2spirv must emit a write
to that status VGPR; a shader reading it must not get a stale value.

The test translates a compute shader with image_sample + tfe and checks that the SPIR-V
contains a store to the status VGPR (the VGPR right after the sampled data). Without the
fix the translator ignores tfe/lwe entirely and the emitted SPIR-V is byte-identical to the
form without them.

Standard library only; no GPU, no LLVM, no game files.
"""
import struct
import sys
import unittest

import support

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)

DESC = "s[8:15], s[16:19]"
PRO = [
    "v_mov_b32 v4, 1.0",
    "v_mov_b32 v5, 0.5",
    "v_mov_b32 v6, 0.25",
    "v_mov_b32 v7, 0.75",
    "v_mov_b32 v8, 0",
]


def translate(name, line, fssm=CS_FSSM):
    return support.translate_lines(
        name, PRO + [line, "s_endpgm"], fssm=fssm)


class ImageTfeStatus(unittest.TestCase):
    def status_stores(self, spv, base_reg: int, ncomp: int) -> list:
        """OpStore lines writing the data VGPRs (base_reg .. base_reg+ncomp-1) plus the status VGPR (base_reg+ncomp)."""
        regs = {f"%reg_v{base_reg + i}" for i in range(ncomp + 1)}
        return [ln for ln in spv.splitlines()
                if ln.strip().startswith("OpStore ") and any(r in ln.split() for r in regs)]

    def test_tfe_writes_status_vgpr(self):
        """image_sample with tfe must emit a store to the status VGPR after the data."""
        spv = translate("cs_sample_tfe",
                        "image_sample v9, v[4:6], " + DESC + " dmask:0x1 dim:SQ_RSRC_IMG_2D tfe")
        stores = self.status_stores(spv, 9, 1)
        self.assertEqual(len(stores), 2, f"expected 2 stores (data v9 + status v10), got {len(stores)}:\n{spv}")
        self.assertTrue(any("OpStore %reg_v10" in ln for ln in stores), "missing status store to v10:\n" + spv)

    def test_lwe_writes_status_vgpr(self):
        spv = translate("cs_sample_lwe",
                        "image_sample v9, v[4:6], " + DESC + " dmask:0x1 dim:SQ_RSRC_IMG_2D lwe")
        stores = self.status_stores(spv, 9, 1)
        self.assertEqual(len(stores), 2, f"expected 2 stores (data v9 + status v10), got {len(stores)}:\n{spv}")
        self.assertTrue(any("OpStore %reg_v10" in ln for ln in stores), "missing status store to v10:\n" + spv)

    def test_without_tfe_no_status_store(self):
        """Without tfe/lwe only the data component is stored."""
        spv = translate("cs_sample_plain",
                        "image_sample v9, v[4:6], " + DESC + " dmask:0x1 dim:SQ_RSRC_IMG_2D")
        stores = self.status_stores(spv, 9, 1)
        self.assertEqual(len(stores), 1, f"expected 1 store to v9 (data only), got {len(stores)}:\n{spv}")


if __name__ == "__main__":
    unittest.main()

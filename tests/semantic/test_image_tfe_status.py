"""SHADER-1266: an image read with tfe or lwe writes its status dword, vdata[data_count].

Standard library only; no GPU, no LLVM, no game files.

With tfe (or lwe) MIMG returns one more VGPR after the data: the fetch status, which is 0 when the fetch succeeds.
Nothing the translator binds is partially resident, so the status is always 0. Before this, image_op and image_misc
wrote only the data VGPRs, and a shader that read the status register got whatever it held before the instruction.
rdna2_decode sizes vdata the same way (_mimg_data_count): four data words for a gather, one per dmask bit otherwise,
two per VGPR under d16, then the status.

Each program puts a marker in the status register, runs the read, and exports that register; the last value the
translated code stores into it must be the constant 0, not the marker.
"""
import re
import struct
import unittest

import support

DESC = "s[8:15], s[16:19]"
D2 = "dim:SQ_RSRC_IMG_2D"
MARK = 0x12345678
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def last_store(name, status, line):
    """The constant the translated program last stores into v<status> after `line`, or None if not a constant."""
    body = ["v_mov_b32 v20, 0.5", "v_mov_b32 v21, 0.5", "v_mov_b32 v22, 0", "v_mov_b32 v23, 0",
            f"v_mov_b32 v{status}, 0x{MARK:x}", line]
    fssm = CS_FSSM if name.startswith("cs_") else None
    if name.startswith("ps_"):
        body.append(f"exp mrt0 v{status}, v{status}, v{status}, v{status} done vm")
    body.append("s_endpgm")
    spv = support.translate_lines(name, body, fssm=fssm)
    stores = re.findall(rf"OpStore %reg_v{status} (%\w+)", spv)
    if not stores:
        raise AssertionError(f"nothing stored into v{status}")
    m = re.search(rf"^\s*{re.escape(stores[-1])} = OpConstant %uint (\d+)$", spv, re.M)
    return int(m.group(1)) if m else None


class TfeStatus(unittest.TestCase):
    def check(self, name, status, line):
        self.assertEqual(last_store(name, status, line), 0, line)

    def test_sample_four_components(self):
        self.check("ps_s4", 7, f"image_sample v[3:7], v[20:21], {DESC} dmask:0xf {D2} tfe")

    def test_sample_one_component(self):
        self.check("ps_s1", 4, f"image_sample v[3:4], v[20:21], {DESC} dmask:0x1 {D2} tfe")

    def test_sample_lz_two_components(self):
        self.check("ps_lz", 5, f"image_sample_lz v[3:5], v[20:21], {DESC} dmask:0x5 {D2} tfe")

    def test_sample_lwe_alone(self):
        self.check("ps_lwe", 7, f"image_sample v[3:7], v[20:21], {DESC} dmask:0xf {D2} lwe")

    def test_sample_d16_packs_two_per_register(self):
        # four f16 results in two VGPRs, then the status
        self.check("ps_d16", 5, f"image_sample v[3:5], v[20:21], {DESC} dmask:0xf {D2} d16 tfe")

    def test_gather_has_four_data_words(self):
        # one dmask bit, but a gather returns four texels
        self.check("ps_g4", 7, f"image_gather4_lz v[3:7], v[20:21], {DESC} dmask:0x1 {D2} tfe")

    def test_load(self):
        self.check("ps_ld", 7, f"image_load v[3:7], v[22:23], s[8:15] dmask:0xf {D2} unorm tfe")

    def test_load_mip(self):
        self.check("ps_mip", 7, f"image_load_mip v[3:7], v[22:24], s[8:15] dmask:0xf {D2} unorm tfe")

    def test_compute_stage_too(self):
        self.check("cs_ld", 4, f"image_load v[3:4], v[22:23], s[8:15] dmask:0x1 {D2} unorm tfe")

    def test_without_tfe_the_next_register_is_untouched(self):
        self.assertEqual(last_store("ps_none", 7, f"image_sample v[3:6], v[20:21], {DESC} dmask:0xf {D2}"), MARK)


if __name__ == "__main__":
    unittest.main()

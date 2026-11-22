"""16-bit integer VOP3 ops with a 32-bit literal operand keep the high half of the destination.

The 16-bit integer ops (v_add_nc_u16, v_lshlrev_b16, v_max_u16, v_mul_lo_u16, ...) are VOP3-only on GFX10 and write
the low half of the destination only (docs/design.md, the 16-bit integer row; Mesa ACO's can_use_opsel). The
translator told VOP3 from VOP2 by the length of the encoding, 8 bytes. A VOP3 whose operand is a 32-bit literal is
12 bytes, so it was translated as a VOP2 write and the high half of the destination became 0: the same instruction
gave 0xAAAA1235 with the operand in a register and 0x00001235 with it as a literal.

Each case runs the same op twice, once with 0x1234 in v2 and once as a literal, into destinations preset to
0xAAAA0000 and 0x5555FFFF. The listings carry the encodings llvm-mc 18.1.3 (-mcpu=gfx1030) prints for them, since
the encoding is what decides; they are hand-written (no game data). Expected words are computed here from the op's
definition on the low halves.
"""
import unittest

from support import mrt0_words, translate_listing

HIGH_A, HIGH_B = 0xAAAA0000, 0x5555FFFF
K = 0x1234

# op -> (register form: line and encoding, literal form: line and encoding, the low-half result)
# v1 = 1 in every case; the register form has 0x1234 in v2. dst is v0 (register form) or v3 (literal form).
CASES = {
    "v_add_nc_u16": (("v_add_nc_u16 v0, v2, v1", "0x00,0x00,0x03,0xd7,0x02,0x03,0x02,0x00"),
                     ("v_add_nc_u16 v3, 0x1234, v1", "0x03,0x00,0x03,0xd7,0xff,0x02,0x02,0x00,0x34,0x12,0x00,0x00"),
                     (K + 1) & 0xFFFF),
    "v_lshlrev_b16": (("v_lshlrev_b16 v0, v1, v2", "0x00,0x00,0x14,0xd7,0x01,0x05,0x02,0x00"),
                      ("v_lshlrev_b16 v3, v1, 0x1234", "0x03,0x00,0x14,0xd7,0x01,0xff,0x01,0x00,0x34,0x12,0x00,0x00"),
                      (K << 1) & 0xFFFF),
    "v_max_u16": (("v_max_u16 v0, v1, v2", "0x00,0x00,0x09,0xd7,0x01,0x05,0x02,0x00"),
                  ("v_max_u16 v3, v1, 0x1234", "0x03,0x00,0x09,0xd7,0x01,0xff,0x01,0x00,0x34,0x12,0x00,0x00"),
                  max(1, K)),
    "v_mul_lo_u16": (("v_mul_lo_u16 v0, v2, v1", "0x00,0x00,0x05,0xd7,0x02,0x03,0x02,0x00"),
                     ("v_mul_lo_u16 v3, 0x1234, v1", "0x03,0x00,0x05,0xd7,0xff,0x02,0x02,0x00,0x34,0x12,0x00,0x00"),
                     (K * 1) & 0xFFFF),
}


def run_case(name, high):
    (reg, reg_enc), (lit, lit_enc), _ = CASES[name]
    lines = ["v_mov_b32_e32 v1, 1", f"v_mov_b32_e32 v2, {K:#x}", f"v_mov_b32_e32 v0, {high:#x}",
             f"v_mov_b32_e32 v3, {high:#x}", f"{reg} ; encoding: [{reg_enc}]", f"{lit} ; encoding: [{lit_enc}]",
             "v_mov_b32_e32 v4, 0", "exp mrt0 v0, v3, v4, v4 done vm", "s_endpgm"]
    text = "; ps_u16_literal  hand-written\n" + "".join("\t" + line + "\n" for line in lines)
    return mrt0_words(translate_listing("ps_u16_literal", text.encode()))[:2]


class LiteralOperand(unittest.TestCase):
    def test_the_literal_form_writes_what_the_register_form_writes(self):
        for name, (_, _, low) in CASES.items():
            for high in (HIGH_A, HIGH_B):
                with self.subTest(op=name, high=hex(high)):
                    want = (high & 0xFFFF0000) | low
                    self.assertEqual([hex(w) for w in run_case(name, high)], [hex(want), hex(want)])

    def test_the_encodings_are_vop3(self):
        # bits 31:26 of the first dword are 110101 (GFX10 VOP3); the literal forms are 12 bytes, the others 8
        for name, ((_, reg_enc), (_, lit_enc), _) in CASES.items():
            for enc, size in ((reg_enc, 8), (lit_enc, 12)):
                octets = [int(x, 16) for x in enc.split(",")]
                self.assertEqual((len(octets), octets[3] >> 2), (size, 0b110101), name)


if __name__ == "__main__":
    unittest.main()

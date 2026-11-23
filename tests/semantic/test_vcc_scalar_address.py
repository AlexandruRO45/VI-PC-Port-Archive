"""VCC as a 64-bit scalar address: s_getpc_b64 into vcc, then an SMEM load whose base is vcc.

Compiled code reads a constant table placed in its own code this way (s_getpc_b64 vcc; s_add_u32 vcc_lo, ...;
s_addc_u32 vcc_hi, ...; s_load_dwordx2 s[10:11], vcc, 0). The translator refused both operands: "s_getpc_b64
destination" and "s_load_dwordx2 operands sgpr vcc". The PC is the code base (push constants 34-35) plus the offset
of the next instruction, as for an SGPR pair; VCC holds it as data, its low word in vcc_lo and its high word in vcc_hi.

    python -I tests/semantic/test_vcc_scalar_address.py
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
CODE_BASE = 0x10000
OUT = 0x3000

# llvm-mc 21.1.8 -mcpu=gfx1013 -show-encoding; the s_getpc_b64 is at offset 0, so the PC it reads is 4.
LISTING = """\
\ts_getpc_b64 vcc                         ; encoding: [0x00,0x1f,0xea,0xbe]
\ts_add_u32 vcc_lo, vcc_lo, 0x100         ; encoding: [0x6a,0xff,0x6a,0x80,0x00,0x01,0x00,0x00]
\ts_addc_u32 vcc_hi, vcc_hi, 0            ; encoding: [0x6b,0x80,0x6b,0x82]
\ts_load_dwordx2 s[10:11], vcc, 0x0       ; encoding: [0xb5,0x02,0x04,0xf4,0x00,0x00,0x00,0xfa]
\ts_waitcnt lgkmcnt(0)                    ; encoding: [0x7f,0xc0,0x8c,0xbf]
\tv_mov_b32_e32 v10, s10                  ; encoding: [0x0a,0x02,0x14,0x7e]
\tv_mov_b32_e32 v11, s11                  ; encoding: [0x0b,0x02,0x16,0x7e]
\tv_mov_b32_e32 v6, 0x3000                ; encoding: [0xff,0x02,0x0c,0x7e,0x00,0x30,0x00,0x00]
\tglobal_store_dwordx2 v6, v[10:11], s[0:1] ; encoding: [0x00,0x80,0x74,0xdc,0x06,0x0a,0x00,0x00]
\ts_endpgm                                ; encoding: [0x00,0x00,0x81,0xbf]
"""


def translate(listing):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "cs_vcc_addr.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", "cs_vcc_addr.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_vcc_addr.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write("; cs_vcc_addr  hand-written test program\n" + listing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


class VccScalarAddressTests(unittest.TestCase):
    def test_table_read_through_vcc(self):
        spv = translate(LISTING)
        table = CODE_BASE + 4 + 0x100
        mem = {table: 0xCAFE0001, table + 4: 0xCAFE0002, table - 4: 0xBAD, table + 8: 0xBAD}
        push = [0] * 36
        push[34], push[35] = CODE_BASE, 0
        si.run_workgroup(spv, 1, 32,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=push, memory=mem)
        self.assertEqual([mem.get(OUT), mem.get(OUT + 4)], [0xCAFE0001, 0xCAFE0002])

    def test_carry_into_vcc_hi(self):
        # The code base's low word is near 2^32: the add carries into vcc_hi, and the load reads above 4 GiB.
        spv = translate(LISTING)
        base = 0xFFFFFF00
        table = (1 << 32) + (base + 4 + 0x100 - (1 << 32))
        mem = {table: 7, table + 4: 9}
        push = [0] * 36
        push[34], push[35] = base, 0
        si.run_workgroup(spv, 1, 32,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=push, memory=mem)
        self.assertEqual([mem.get(OUT), mem.get(OUT + 4)], [7, 9])


if __name__ == "__main__":
    unittest.main()

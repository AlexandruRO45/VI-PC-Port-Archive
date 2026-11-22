"""ds_wrap_rtn_b32 (docs/design.md "ds_wrap_rtn_b32"; runtime/test/shaders/cs_w32_ds_wrap.asm).

RDNA ISA: tmp = M; M = (tmp >= DATA) ? tmp - DATA : tmp + DATA2 (unsigned compare, 32-bit wrap); vD = tmp. The 16
cases below were measured natively on RDNA2 gfx1036 through tools/hw_oracle (one lane, one LDS dword each). Run
through spirv_mini_interp, one invocation, as tests/semantic/test_lds_atomics_cas.py does. Before, the op raised
Unsupported.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_lds_atomics_cas import OUT, run_cs  # noqa: E402

M32 = 0xFFFFFFFF
# (M, DATA, DATA2, returned, new M) measured on gfx1036
HW = [(0xA, 0x5, 0x64, 0xA, 0x5), (0x4, 0x5, 0x64, 0x4, 0x68), (0x5, 0x5, 0x64, 0x5, 0x0), (0x0, 0x0, 0x7, 0x0, 0x0),
      (0xFFFFFFF0, 0x20, 0x30, 0xFFFFFFF0, 0xFFFFFFD0), (0x80000000, 0x7FFFFFFF, 0x1, 0x80000000, 0x1),
      (0x3, 0xFFFFFFFF, 0xFFFFFFFE, 0x3, 0x1), (0x64, 0x1, 0x0, 0x64, 0x63), (0x0, 0x1, 0xFFFFFFFF, 0x0, 0xFFFFFFFF),
      (0x7, 0x8, 0x10, 0x7, 0x17), (0x40, 0x40, 0x40, 0x40, 0x0), (0xFFFFFFFF, 0xFFFFFFFF, 0x2, 0xFFFFFFFF, 0x0),
      (0x1, 0x2, 0x3, 0x1, 0x4), (0x2, 0x1, 0x3, 0x2, 0x1), (0x12345678, 0x12345677, 0x5, 0x12345678, 0x1),
      (0x12345676, 0x12345677, 0x5, 0x12345676, 0x1234567B)]


def run_wrap(m, d, d2, offset=0):
    off = f" offset:{offset}" if offset else ""
    body = ["v_mov_b32_e32 v1, 16", f"v_mov_b32_e32 v2, {m:#x}", "ds_write_b32 v1, v2", "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v1, {16 - offset}", f"v_mov_b32_e32 v3, {d:#x}", f"v_mov_b32_e32 v4, {d2:#x}",
            "v_mov_b32_e32 v5, 0xdead", f"ds_wrap_rtn_b32 v5, v1, v3, v4{off}", "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, 16", "ds_read_b32 v6, v1", "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v7, {OUT:#x}", "global_store_dword v7, v5, s[0:1]",
            f"v_mov_b32_e32 v7, {OUT + 4:#x}", "global_store_dword v7, v6, s[0:1]"]
    mem = run_cs(body)
    return mem.get(OUT), mem.get(OUT + 4)


class LdsWrap(unittest.TestCase):
    def test_hardware_words(self):
        for m, d, d2, ret, new in HW:
            with self.subTest(m=hex(m), d=hex(d), d2=hex(d2)):
                self.assertEqual(run_wrap(m, d, d2), (ret, new))

    def test_offset(self):
        self.assertEqual(run_wrap(10, 3, 1, offset=8), (10, 7))

    def test_reference_matches_hardware(self):
        # Plain-Python reference of the ISA text against the measured words.
        for m, d, d2, ret, new in HW:
            self.assertEqual((m, ((m - d) if m >= d else (m + d2)) & M32), (ret, new))


if __name__ == "__main__":
    unittest.main()

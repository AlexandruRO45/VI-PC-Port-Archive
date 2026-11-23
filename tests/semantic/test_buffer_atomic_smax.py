"""Value tests for RDNA2 buffer_atomic_smax, run through spirv_mini_interp on the CPU.

RDNA2 ISA (MUBUF):
  buffer_atomic_smax vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc] [dlc]

Performs a 32-bit signed atomic maximum at the buffer element address:
  old = memory[addr]
  memory[addr] = max(s32(old), s32(vdata))
  if glc: vdata = old

Semantics verified:
1. Signed comparison: s32(m) vs s32(d). Distinguishes signed max from unsigned max
   (e.g., max(-1, 0) == 0 in signed max, whereas 0xFFFFFFFF > 0 in unsigned max).
2. Returning forms (glc=True): pre-op memory value is returned to vdata.
3. Non-returning forms (glc=False): vdata is preserved.
4. MUBUF addressing: offen, inst_offset, soffset, and idxen with stride.
5. Out-of-bounds (OOB) suppression: accesses beyond num_records leave memory untouched
   and return 0 with glc.
6. Natural alignment: 32-bit atomics require 4-byte alignment; unaligned addresses
   leave memory untouched and return 0 with glc.
7. Inactive lanes (EXEC mask): disabled lanes do not modify memory or destination VGPRs.
8. Multi-lane contention: multiple concurrent lanes atomically reduce to the true signed maximum.
9. Unsupported forms: 64-bit buffer_atomic_smax_x2 is rejected as Unsupported.

Standard library only; no GPU, no LLVM, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, translate_lines  # noqa: E402

M32 = 0xFFFFFFFF
RAW_W3 = 0x31016FAC  # OOB_SELECT = 3 (RAW), format 62 (32-bit uint)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, workgroup id in s4, 32x1x1
FSSM = bytes(FSSM)

ELEM = 0x40
BOUT = 0x80
RECORDS = 0x100
CANARY = 0xCAFEF00D


def s32(x: int) -> int:
    x &= M32
    return x - (1 << 32) if (x & 0x80000000) else x


def u32(x: int) -> int:
    return x & M32


def model_smax(m: int, d: int) -> int:
    return u32(m if s32(m) >= s32(d) else d)


def run_program(lines: list[str], memory: dict[int, int], nlanes: int = 1) -> dict[int, int]:
    spv = translate_lines("cs_batomic_smax", lines + ["s_endpgm"], dxbc=STUB_DXBC, fssm=FSSM)
    mem = dict(memory)
    si.run_workgroup(
        spv,
        nlanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0] * 36,
        memory=mem,
    )
    return mem


def buf_desc_lines(base: int = 0, stride: int = 0, records: int = RECORDS, word3: int = RAW_W3) -> list[str]:
    w0 = base & M32
    w1 = ((base >> 32) & 0xFFFF) | ((stride & 0x3FFF) << 16)
    return [
        f"s_mov_b32 s8, {w0:#x}",
        f"s_mov_b32 s9, {w1:#x}",
        f"s_mov_b32 s10, {records:#x}",
        f"s_mov_b32 s11, {word3:#x}",
    ]


class TestBufferAtomicSmax(unittest.TestCase):
    def test_signed_max_returning_values(self):
        pairs = [
            (0, 0),
            (10, 20),
            (50, 20),
            (0, 10),
            # Signed vs unsigned divergence:
            # -1 is 0xFFFFFFFF unsigned. Signed max(-1, 0) == 0, whereas unsigned max is 0xFFFFFFFF.
            (u32(-1), 0),
            (0, u32(-1)),
            (u32(-10), u32(-5)),
            (u32(-50), 50),
            # INT32 boundaries:
            (0x80000000, 0),             # INT32_MIN vs 0 -> 0
            (0x80000000, 0x7FFFFFFF),    # INT32_MIN vs INT32_MAX -> INT32_MAX
            (0x7FFFFFFF, u32(-1)),       # INT32_MAX vs -1 -> INT32_MAX
            (0x12345678, 0x23456789),
        ]

        for m, d in pairs:
            with self.subTest(m=hex(m), d=hex(d), s_m=s32(m), s_d=s32(d)):
                body = (
                    buf_desc_lines(records=RECORDS)
                    + [
                        f"v_mov_b32_e32 v1, {ELEM:#x}",
                        f"v_mov_b32_e32 v2, {d:#x}",
                        "buffer_atomic_smax v2, v1, s[8:11], 0 offen glc",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v4, {BOUT:#x}",
                        "s_mov_b32 s10, 0x1000",
                        "buffer_store_dword v2, v4, s[8:11], 0 offen",
                    ]
                )
                mem = run_program(body, {ELEM: m})
                returned_val = mem.get(BOUT)
                final_mem = mem.get(ELEM)
                expected_mem = model_smax(m, d)
                self.assertEqual(returned_val, m, f"glc returned {returned_val:#x}, expected pre-op {m:#x}")
                self.assertEqual(final_mem, expected_mem, f"final mem {final_mem:#x}, expected {expected_mem:#x}")

    def test_without_glc_vdata_is_kept(self):
        m = u32(-50)
        d = 30
        body = (
            buf_desc_lines(records=RECORDS)
            + [
                f"v_mov_b32_e32 v1, {ELEM:#x}",
                f"v_mov_b32_e32 v2, {d:#x}",
                "buffer_atomic_smax v2, v1, s[8:11], 0 offen",
                "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v4, {BOUT:#x}",
                "s_mov_b32 s10, 0x1000",
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        mem = run_program(body, {ELEM: m})
        self.assertEqual(mem.get(BOUT), d, "without glc, vdata must retain its original input value")
        self.assertEqual(mem.get(ELEM), model_smax(m, d), "memory must be updated to signed max")

    def test_addressing_modes(self):
        # 1. Immediate instruction offset: offset:16
        body_inst_off = (
            buf_desc_lines(records=RECORDS)
            + [
                "v_mov_b32_e32 v1, 0x40",
                f"v_mov_b32_e32 v2, {50:#x}",
                "buffer_atomic_smax v2, v1, s[8:11], 0 offen offset:16 glc",
                "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v4, {BOUT:#x}",
                "s_mov_b32 s10, 0x1000",
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        target = 0x40 + 16  # 0x50
        mem = run_program(body_inst_off, {target: 10})
        self.assertEqual(mem.get(BOUT), 10)
        self.assertEqual(mem.get(target), 50)

        # 2. Scalar soffset in SGPR
        body_soff = (
            buf_desc_lines(records=RECORDS)
            + [
                "s_mov_b32 s4, 0x20",
                "v_mov_b32_e32 v1, 0x40",
                f"v_mov_b32_e32 v2, {u32(-5):#x}",
                "buffer_atomic_smax v2, v1, s[8:11], s4 offen glc",
                "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v4, {BOUT:#x}",
                "s_mov_b32 s10, 0x1000",
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        target = 0x40 + 0x20  # 0x60
        mem = run_program(body_soff, {target: u32(-20)})
        self.assertEqual(mem.get(BOUT), u32(-20))
        self.assertEqual(mem.get(target), u32(-5))

        # 3. Structured idxen addressing (stride = 16, index = 3 -> 0x30)
        body_idx = (
            buf_desc_lines(stride=16, records=RECORDS)
            + [
                "v_mov_b32_e32 v1, 3",
                f"v_mov_b32_e32 v2, {100:#x}",
                "buffer_atomic_smax v2, v1, s[8:11], 0 idxen glc",
                "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v4, {BOUT:#x}",
                "s_mov_b32 s10, 0x1000",
                "s_mov_b32 s9, 0",  # clear stride for store
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        target = 3 * 16  # 0x30
        mem = run_program(body_idx, {target: 25})
        self.assertEqual(mem.get(BOUT), 25)
        self.assertEqual(mem.get(target), 100)

    def test_out_of_bounds_suppression(self):
        m, d = 0x12345678, 0x76543210
        # Exactly in-bounds: offset = RECORDS - 4
        in_bounds_off = RECORDS - 4
        body_in = (
            buf_desc_lines(records=RECORDS)
            + [
                f"v_mov_b32_e32 v1, {in_bounds_off:#x}",
                f"v_mov_b32_e32 v2, {d:#x}",
                "buffer_atomic_smax v2, v1, s[8:11], 0 offen glc",
                "s_waitcnt vmcnt(0)",
                f"v_mov_b32_e32 v4, {BOUT:#x}",
                "s_mov_b32 s10, 0x1000",
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        mem = run_program(body_in, {in_bounds_off: m})
        self.assertEqual(mem.get(BOUT), m)
        self.assertEqual(mem.get(in_bounds_off), model_smax(m, d))

        # Out-of-bounds: offset = RECORDS and offset = RECORDS + 4
        for oob_off in (RECORDS, RECORDS + 4):
            with self.subTest(oob_off=hex(oob_off)):
                body_oob = (
                    buf_desc_lines(records=RECORDS)
                    + [
                        f"v_mov_b32_e32 v1, {oob_off:#x}",
                        f"v_mov_b32_e32 v2, {d:#x}",
                        "buffer_atomic_smax v2, v1, s[8:11], 0 offen glc",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v4, {BOUT:#x}",
                        "s_mov_b32 s10, 0x1000",
                        "buffer_store_dword v2, v4, s[8:11], 0 offen",
                    ]
                )
                mem = run_program(body_oob, {oob_off: m})
                self.assertEqual(mem.get(BOUT), 0, "OOB glc must return 0")
                self.assertEqual(mem.get(oob_off), m, "OOB access must leave memory untouched")

    def test_natural_alignment_enforcement(self):
        m, d = 0x11223344, 0x55667788
        # Unaligned offsets (addr % 4 != 0)
        for unaligned_off in (ELEM + 1, ELEM + 2, ELEM + 3):
            with self.subTest(unaligned_off=hex(unaligned_off)):
                body = (
                    buf_desc_lines(records=RECORDS)
                    + [
                        f"v_mov_b32_e32 v1, {unaligned_off:#x}",
                        f"v_mov_b32_e32 v2, {d:#x}",
                        "buffer_atomic_smax v2, v1, s[8:11], 0 offen glc",
                        "s_waitcnt vmcnt(0)",
                        f"v_mov_b32_e32 v4, {BOUT:#x}",
                        "s_mov_b32 s10, 0x1000",
                        "buffer_store_dword v2, v4, s[8:11], 0 offen",
                    ]
                )
                mem = run_program(body, {unaligned_off: m})
                self.assertEqual(mem.get(BOUT), 0, "unaligned glc must return 0")
                self.assertEqual(mem.get(unaligned_off), m, "unaligned atomic must not modify memory")

    def test_exec_mask_inactive_lanes(self):
        # Wave of 2 lanes: Lane 0 active, Lane 1 inactive
        addr0 = 0x40
        addr1 = 0x44
        m0, m1 = 10, 20
        d0, d1 = 50, 999

        body = (
            buf_desc_lines(records=RECORDS)
            + [
                # Set per-lane address: Lane 0 -> 0x40, Lane 1 -> 0x44
                "v_mul_lo_u32 v1, v0, 4",
                f"v_add_nc_u32_e32 v1, {addr0:#x}, v1",
                # Set per-lane vdata while both lanes active
                "v_cmp_eq_u32_e64 vcc, v0, 0",
                f"v_cndmask_b32_e64 v2, {d1:#x}, {d0:#x}, vcc",
                # Mask out Lane 1 so only Lane 0 executes the atomic
                "s_mov_b32 exec_lo, 1",
                "s_mov_b32 exec_hi, 0",
                "buffer_atomic_smax v2, v1, s[8:11], 0 offen glc",
                "s_waitcnt vmcnt(0)",
                # Re-enable exec to store results from both lanes
                "s_mov_b32 exec_lo, 3",
                "v_mul_lo_u32 v3, v0, 4",
                f"v_add_nc_u32_e32 v4, {BOUT:#x}, v3",
                "s_mov_b32 s10, 0x1000",
                "buffer_store_dword v2, v4, s[8:11], 0 offen",
            ]
        )
        mem = run_program(body, {addr0: m0, addr1: m1}, nlanes=2)
        # Lane 0 was active: memory updated to max(10, 50) = 50, returned 10
        self.assertEqual(mem.get(addr0), 50)
        self.assertEqual(mem.get(BOUT), 10)
        # Lane 1 was inactive: memory UNTOUCHED (still 20), vdata kept original value d1
        self.assertEqual(mem.get(addr1), m1)
        self.assertEqual(mem.get(BOUT + 4), d1)

    def test_multi_lane_concurrent_smax(self):
        # 4 active lanes atomically performing buffer_atomic_smax on the same address ELEM
        initial_mem = u32(-100)
        lane_values = [u32(-500), 42, u32(-10), 15]

        # Use v_cndmask chain or per-lane value selection based on v0
        body = (
            buf_desc_lines(records=RECORDS)
            + [
                f"v_mov_b32_e32 v1, {ELEM:#x}",
                # Select value based on lane index v0:
                "v_mov_b32_e32 v2, 0",
                f"v_cmp_eq_u32_e64 vcc, v0, 0",
                f"v_cndmask_b32_e64 v2, v2, {lane_values[0]:#x}, vcc",
                f"v_cmp_eq_u32_e64 vcc, v0, 1",
                f"v_cndmask_b32_e64 v2, v2, {lane_values[1]:#x}, vcc",
                f"v_cmp_eq_u32_e64 vcc, v0, 2",
                f"v_cndmask_b32_e64 v2, v2, {lane_values[2]:#x}, vcc",
                f"v_cmp_eq_u32_e64 vcc, v0, 3",
                f"v_cndmask_b32_e64 v2, v2, {lane_values[3]:#x}, vcc",
                "buffer_atomic_smax v2, v1, s[8:11], 0 offen",
                "s_waitcnt vmcnt(0)",
            ]
        )
        mem = run_program(body, {ELEM: initial_mem}, nlanes=4)
        all_vals = [s32(initial_mem)] + [s32(v) for v in lane_values]
        expected_max = u32(max(all_vals))
        self.assertEqual(mem.get(ELEM), expected_max, f"expected max {expected_max:#x}, got {mem.get(ELEM):#x}")

    def test_unsupported_rejections(self):
        # 64-bit form buffer_atomic_smax_x2 is not supported on RDNA2 hardware
        for bad_line in (
            "buffer_atomic_smax_x2 v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_smax_x4 v[2:5], v1, s[8:11], 0 offen glc",
        ):
            with self.subTest(bad_line=bad_line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("cs_bad_smax", [bad_line, "s_endpgm"], dxbc=STUB_DXBC, fssm=FSSM)


if __name__ == "__main__":
    unittest.main()

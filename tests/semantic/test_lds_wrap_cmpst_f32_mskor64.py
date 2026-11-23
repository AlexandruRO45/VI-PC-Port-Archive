"""Semantic, decoder, and spirv-val tests for 5 RDNA2 LDS atomic instructions:
  - ds_wrap_rtn_b32
  - ds_cmpst_f32, ds_cmpst_rtn_f32
  - ds_mskor_b64, ds_mskor_rtn_b64

Each lowers through Gen.lds_access() in tools/rdna2spirv.py (preserving EXEC active-lane masking and
SHADER-309 bounds checks) using Workgroup-scope OpAtomicCompareExchange CAS loops without requiring
Int64Atomics or float atomic extensions.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OUT = 0x3000

F32_PZERO = 0x00000000
F32_NZERO = 0x80000000
F32_ONE = 0x3F800000
F32_TWO = 0x40000000
F32_THREE = 0x40400000
F32_NEG_FIVE = 0xC0A00000
F32_QNAN = 0x7FC00000
F32_QNAN2 = 0x7FC01234


def make_fssm(lds_units=1, threads=32):
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x88 | (lds_units << 15), threads, 1, 1)
    return bytes(fssm)


def translate(body, stage="cs", lds_units=1, threads=32, emulate=False):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        name = f"{stage}_lds_ext"
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if stage == "cs":
            with open(os.path.join(tmp, "fssm", f"{name}.fssm"), "wb") as f:
                f.write(make_fssm(lds_units=lds_units, threads=threads))
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(body, nlanes=1, lds_units=1, threads=32):
    spv = translate(body, stage="cs", lds_units=lds_units, threads=threads)
    mem = {}
    si.run_workgroup(spv, nlanes, threads,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0] * 36, memory=mem)
    return mem


def wrap_model(old, d0, d1):
    return ((old - d0) & M32) if old >= d0 else ((old + d1) & M32)


def run_wrap(old_val, d0, d1, offset=0, alias_dst=None):
    addr = 32
    off = f" offset:{offset}" if offset else ""
    dst = alias_dst or "v5"
    body = [
        f"v_mov_b32_e32 v1, {addr}",
        f"v_mov_b32_e32 v2, {old_val:#x}",
        "ds_write_b32 v1, v2",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v1, {addr - offset}",
        f"v_mov_b32_e32 v3, {d0:#x}",
        f"v_mov_b32_e32 v4, {d1:#x}",
        "v_mov_b32_e32 v5, 0xdead",
        f"ds_wrap_rtn_b32 {dst}, v1, v3, v4{off}",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v8, {addr}",
        "ds_read_b32 v9, v8",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v6, {OUT:#x}",
        f"global_store_dword v6, {dst}, s[0:1]",
        f"v_mov_b32_e32 v7, {OUT + 4:#x}",
        "global_store_dword v7, v9, s[0:1]",
    ]
    mem = run_cs(body)
    return mem.get(OUT), mem.get(OUT + 4)


def run_cmpst_f32(old_bits, cmp_bits, new_bits, rtn=True, offset=0, alias_dst=None):
    addr = 32
    off = f" offset:{offset}" if offset else ""
    op = "ds_cmpst_rtn_f32" if rtn else "ds_cmpst_f32"
    dst = alias_dst or "v5"
    body = [
        f"v_mov_b32_e32 v1, {addr}",
        f"v_mov_b32_e32 v2, {old_bits:#x}",
        "ds_write_b32 v1, v2",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v1, {addr - offset}",
        f"v_mov_b32_e32 v3, {cmp_bits:#x}",
        f"v_mov_b32_e32 v4, {new_bits:#x}",
        "v_mov_b32_e32 v5, 0xdead",
        f"{op} {dst}, v1, v3, v4{off}" if rtn else f"{op} v1, v3, v4{off}",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v8, {addr}",
        "ds_read_b32 v9, v8",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v6, {OUT:#x}",
        f"global_store_dword v6, {dst if rtn else 'v5'}, s[0:1]",
        f"v_mov_b32_e32 v7, {OUT + 4:#x}",
        "global_store_dword v7, v9, s[0:1]",
    ]
    mem = run_cs(body)
    return mem.get(OUT), mem.get(OUT + 4)


def run_mskor_b64(old_lo, old_hi, msk_lo, msk_hi, dat_lo, dat_hi, rtn=True, offset=0, alias_dst=None):
    addr = 32
    off = f" offset:{offset}" if offset else ""
    op = "ds_mskor_rtn_b64" if rtn else "ds_mskor_b64"
    dst = alias_dst or "v[6:7]"
    dst_lo, dst_hi = (int(dst[2:].split(":")[0]), int(dst[2:].split(":")[1][:-1]))
    body = [
        f"v_mov_b32_e32 v1, {addr}",
        f"v_mov_b32_e32 v10, {old_lo:#x}",
        f"v_mov_b32_e32 v11, {old_hi:#x}",
        "ds_write_b64 v1, v[10:11]",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v1, {addr - offset}",
        f"v_mov_b32_e32 v2, {msk_lo:#x}",
        f"v_mov_b32_e32 v3, {msk_hi:#x}",
        f"v_mov_b32_e32 v4, {dat_lo:#x}",
        f"v_mov_b32_e32 v5, {dat_hi:#x}",
        "v_mov_b32_e32 v6, 0xdead0001",
        "v_mov_b32_e32 v7, 0xdead0002",
        f"{op} {dst}, v1, v[2:3], v[4:5]{off}" if rtn else f"{op} v1, v[2:3], v[4:5]{off}",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v8, {addr}",
        "ds_read_b64 v[12:13], v8",
        "s_waitcnt lgkmcnt(0)",
        f"v_mov_b32_e32 v14, {OUT:#x}",
        f"global_store_dword v14, v{dst_lo if rtn else 6}, s[0:1]",
        f"v_mov_b32_e32 v14, {OUT + 4:#x}",
        f"global_store_dword v14, v{dst_hi if rtn else 7}, s[0:1]",
        f"v_mov_b32_e32 v14, {OUT + 8:#x}",
        "global_store_dword v14, v12, s[0:1]",
        f"v_mov_b32_e32 v14, {OUT + 12:#x}",
        "global_store_dword v14, v13, s[0:1]",
    ]
    mem = run_cs(body)
    return (mem.get(OUT), mem.get(OUT + 4)), (mem.get(OUT + 8), mem.get(OUT + 12))


class LdsWrapCmpstF32Mskor64Tests(unittest.TestCase):
    def test_ds_wrap_rtn_b32_semantics_and_edges(self):
        cases = [
            # (old, d0, d1, label)
            (10, 7, 3, "wrap_gt"),
            (7, 7, 5, "wrap_eq_boundary"),
            (4, 7, 5, "nowrap_lt"),
            (0, 1, 9, "zero_nowrap"),
            (0, 0, 9, "zero_eq_zero"),
            (M32, 10, 5, "max_wrap"),
            (M32 - 2, M32, 5, "nowrap_overflow_u32"),
            (0x80000000, 0x7FFFFFFF, 1, "unsigned_gt_signbit"),
            (0x7FFFFFFF, 0x80000000, 1, "unsigned_lt_signbit"),
        ]
        for old_v, d0, d1, label in cases:
            with self.subTest(case=label):
                got_old, got_mem = run_wrap(old_v, d0, d1)
                self.assertEqual((got_old, got_mem), (old_v, wrap_model(old_v, d0, d1)))

    def test_ds_wrap_rtn_b32_offset_and_aliasing(self):
        self.assertEqual(run_wrap(12, 5, 3, offset=16), (12, 7))
        for dst in ("v1", "v3", "v4"):
            with self.subTest(alias_dst=dst):
                self.assertEqual(run_wrap(10, 15, 6, alias_dst=dst), (10, 16))

    def test_ds_cmpst_f32_ieee_semantics(self):
        cases = [
            # (old, cmp, new, should_swap, label)
            (F32_ONE, F32_ONE, F32_TWO, True, "exact_match"),
            (F32_ONE, F32_TWO, F32_THREE, False, "mismatch"),
            (F32_NZERO, F32_PZERO, F32_ONE, True, "neg_zero_eq_pos_zero"),
            (F32_PZERO, F32_NZERO, F32_NEG_FIVE, True, "pos_zero_eq_neg_zero"),
            (F32_QNAN, F32_QNAN, F32_ONE, False, "nan_ne_nan_same_bits"),
            (F32_QNAN, F32_QNAN2, F32_ONE, False, "nan_ne_nan_diff_payload"),
            (F32_QNAN, F32_ONE, F32_TWO, False, "nan_ne_num"),
            (F32_ONE, F32_QNAN, F32_TWO, False, "num_ne_nan"),
        ]
        for old_b, cmp_b, new_b, swap, label in cases:
            expected_mem = new_b if swap else old_b
            for rtn in (True, False):
                with self.subTest(case=label, rtn=rtn):
                    got_rtn, got_mem = run_cmpst_f32(old_b, cmp_b, new_b, rtn=rtn)
                    self.assertEqual(got_rtn, old_b if rtn else 0xDEAD)
                    self.assertEqual(got_mem, expected_mem)

    def test_ds_cmpst_f32_offset_and_aliasing(self):
        self.assertEqual(run_cmpst_f32(F32_NZERO, F32_PZERO, F32_THREE, rtn=True, offset=8),
                         (F32_NZERO, F32_THREE))
        self.assertEqual(run_cmpst_f32(F32_NZERO, F32_PZERO, F32_THREE, rtn=False, offset=8),
                         (0xDEAD, F32_THREE))
        for dst in ("v1", "v3", "v4"):
            with self.subTest(alias_dst=dst):
                self.assertEqual(run_cmpst_f32(F32_TWO, F32_TWO, F32_NEG_FIVE, rtn=True, alias_dst=dst),
                                 (F32_TWO, F32_NEG_FIVE))

    def test_ds_mskor_b64_semantics_offset_and_aliasing(self):
        cases = [
            (0x11223344, 0x55667788, 0x00FF00FF, 0xFF00FF00, 0x00AA00BB, 0xCC00DD00),
            (0xFFFFFFFF, 0x00000000, 0xFFFF0000, 0x00000000, 0x12340000, 0xCAFEBABE),
            (0xA5A5A5A5, 0x5A5A5A5A, 0x0F0F0F0F, 0xF0F0F0F0, 0x01020304, 0x10203040),
        ]
        for old_lo, old_hi, msk_lo, msk_hi, dat_lo, dat_hi in cases:
            exp_lo = (old_lo & (~msk_lo & M32)) | dat_lo
            exp_hi = (old_hi & (~msk_hi & M32)) | dat_hi
            for rtn in (True, False):
                with self.subTest(old=(hex(old_lo), hex(old_hi)), rtn=rtn):
                    got_rtn, got_mem = run_mskor_b64(old_lo, old_hi, msk_lo, msk_hi, dat_lo, dat_hi, rtn=rtn)
                    self.assertEqual(got_rtn, (old_lo, old_hi) if rtn else (0xDEAD0001, 0xDEAD0002))
                    self.assertEqual(got_mem, (exp_lo, exp_hi))

        # offset:16 and destination aliasing (v[2:3] == mask, v[4:5] == data, v[0:1] overlaps v1 addr)
        self.assertEqual(
            run_mskor_b64(0x12345678, 0x9ABCDEF0, 0x0000FFFF, 0xFFFF0000, 0x0000BEEF, 0xCAFE0000,
                          rtn=True, offset=16),
            ((0x12345678, 0x9ABCDEF0), (0x1234BEEF, 0xCAFEDEF0)),
        )
        for dst in ("v[2:3]", "v[4:5]", "v[0:1]"):
            with self.subTest(alias_dst=dst):
                self.assertEqual(
                    run_mskor_b64(0x12345678, 0x9ABCDEF0, 0x0000FFFF, 0xFFFF0000, 0x0000BEEF, 0xCAFE0000,
                                  rtn=True, alias_dst=dst),
                    ((0x12345678, 0x9ABCDEF0), (0x1234BEEF, 0xCAFEDEF0)),
                )

    def test_exec_inactive_lane_does_not_mutate_lds_or_vdst(self):
        # Lane 0 seeds LDS, then disables EXEC (exec = 0) before executing all 5 ops, then re-enables EXEC.
        body = [
            "v_mov_b32_e32 v1, 16",
            "v_mov_b32_e32 v2, 0x11112222",
            "v_mov_b32_e32 v3, 0x33334444",
            "ds_write_b64 v1, v[2:3]",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 5",
            "v_mov_b32_e32 v5, 9",
            "v_mov_b32_e32 v6, 0xaaaaaaaa",
            "v_mov_b32_e32 v7, 0xbbbbbbbb",
            "s_mov_b32 exec_lo, 0",
            "ds_wrap_rtn_b32 v6, v1, v4, v5",
            "ds_cmpst_rtn_f32 v7, v1, v2, v5",
            "ds_mskor_rtn_b64 v[6:7], v1, v[2:3], v[4:5]",
            "s_mov_b32 exec_lo, 1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b64 v[8:9], v1",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v10, {OUT:#x}",
            "global_store_dword v10, v6, s[0:1]",
            f"v_mov_b32_e32 v10, {OUT + 4:#x}",
            "global_store_dword v10, v7, s[0:1]",
            f"v_mov_b32_e32 v10, {OUT + 8:#x}",
            "global_store_dword v10, v8, s[0:1]",
            f"v_mov_b32_e32 v10, {OUT + 12:#x}",
            "global_store_dword v10, v9, s[0:1]",
        ]
        mem = run_cs(body)
        self.assertEqual(
            [mem.get(OUT + 4 * i) for i in range(4)],
            [0xAAAAAAAA, 0xBBBBBBBB, 0x11112222, 0x33334444],
        )

    def test_shader309_zero_lds_bounds_check(self):
        # LDS_SIZE=0 with offset:4 forces lds_bound=0: _rtn_ returns 0, non-_rtn_ preserves vdst.
        for op, ins, exp in (
            ("ds_wrap_rtn_b32", "ds_wrap_rtn_b32 v5, v1, v3, v4 offset:4", 0),
            ("ds_cmpst_rtn_f32", "ds_cmpst_rtn_f32 v5, v1, v3, v4 offset:4", 0),
            ("ds_cmpst_f32", "ds_cmpst_f32 v1, v3, v4 offset:4", 0xDEAD),
            ("ds_mskor_rtn_b64", "ds_mskor_rtn_b64 v[5:6], v1, v[2:3], v[4:5] offset:4", 0),
            ("ds_mskor_b64", "ds_mskor_b64 v1, v[2:3], v[4:5] offset:4", 0xDEAD),
        ):
            with self.subTest(op=op):
                body = [
                    "v_mov_b32_e32 v1, 0",
                    "v_mov_b32_e32 v2, 0x0f0f0f0f",
                    "v_mov_b32_e32 v3, 7",
                    "v_mov_b32_e32 v4, 3",
                    "v_mov_b32_e32 v5, 0xdead",
                    "v_mov_b32_e32 v6, 0xbeef",
                    ins,
                    "s_waitcnt lgkmcnt(0)",
                    f"v_mov_b32_e32 v7, {OUT:#x}",
                    "global_store_dword v7, v5, s[0:1]",
                ]
                mem = run_cs(body, lds_units=0)
                self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 0)
                self.assertEqual(mem.get(OUT), exp)

    def test_binary_decode_and_translate_roundtrip(self):
        # Exact gfx1030 encodings verified with `llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding`.
        encoded = [
            ("ds_wrap_rtn_b32", bytes([0x10, 0x00, 0xD0, 0xD8, 0x01, 0x03, 0x04, 0x05]),
             "ds_wrap_rtn_b32 v5, v1, v3, v4 offset:16"),
            ("ds_cmpst_f32", bytes([0x08, 0x00, 0x44, 0xD8, 0x01, 0x03, 0x04, 0x00]),
             "ds_cmpst_f32 v1, v3, v4 offset:8"),
            ("ds_cmpst_rtn_f32", bytes([0x08, 0x00, 0xC4, 0xD8, 0x01, 0x03, 0x04, 0x05]),
             "ds_cmpst_rtn_f32 v5, v1, v3, v4 offset:8"),
            ("ds_mskor_b64", bytes([0x10, 0x00, 0x30, 0xD9, 0x01, 0x02, 0x04, 0x00]),
             "ds_mskor_b64 v1, v[2:3], v[4:5] offset:16"),
            ("ds_mskor_rtn_b64", bytes([0x10, 0x00, 0xB0, 0xD9, 0x01, 0x02, 0x04, 0x06]),
             "ds_mskor_rtn_b64 v[6:7], v1, v[2:3], v[4:5] offset:16"),
        ]
        decoded_lines = []
        for expected_op, raw, _asm in encoded:
            ins, nbytes = rdna2_decode._decode_one(raw, 0, wave32=True, mcpu="gfx1030")
            self.assertEqual(nbytes, 8)
            self.assertEqual(ins["op"], expected_op)
            args_str = ", ".join(
                f"v[{a['r']}:{a['r'] + a['n'] - 1}]" if a["n"] > 1 else f"v{a['r']}"
                for a in ins["args"]
            )
            mods_str = (" " + " ".join(ins["mods"])) if ins["mods"] else ""
            decoded_lines.append(f"{ins['op']} {args_str}{mods_str}")
        spv = translate(["v_mov_b32_e32 v1, 0"] + decoded_lines)
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(struct.unpack_from("<I", binary, 0)[0], 0x07230203)

    def _validation_modules(self):
        return (
            ("unbounded_all5", translate([
                "v_mov_b32_e32 v1, 0",
                "v_mov_b32_e32 v2, 0x0f0f0f0f",
                "v_mov_b32_e32 v3, 0x3f800000",
                "v_mov_b32_e32 v4, 0x40000000",
                "v_mov_b32_e32 v5, 0x00ff00ff",
                "ds_wrap_rtn_b32 v6, v1, v3, v4",
                "ds_cmpst_f32 v1, v3, v4",
                "ds_cmpst_rtn_f32 v6, v1, v3, v4 offset:4",
                "ds_mskor_b64 v1, v[2:3], v[4:5] offset:8",
                "ds_mskor_rtn_b64 v[6:7], v1, v[2:3], v[4:5] offset:16",
            ])),
            ("bounded_all5", translate([
                "v_mov_b32_e32 v8, 0xfc00",
                "ds_read_b32 v9, v8 offset:1024",
                "v_mov_b32_e32 v1, 0",
                "v_mov_b32_e32 v2, 0x0f0f0f0f",
                "v_mov_b32_e32 v3, 0x3f800000",
                "v_mov_b32_e32 v4, 0x40000000",
                "v_mov_b32_e32 v5, 0x00ff00ff",
                "ds_wrap_rtn_b32 v6, v1, v3, v4",
                "ds_cmpst_f32 v1, v3, v4",
                "ds_cmpst_rtn_f32 v6, v1, v3, v4 offset:4",
                "ds_mskor_b64 v1, v[2:3], v[4:5] offset:8",
                "ds_mskor_rtn_b64 v[6:7], v1, v[2:3], v[4:5] offset:16",
            ], lds_units=1)),
            ("emulate_shrunk_all5", translate([
                "s_mov_b64 exec, -1",
                "v_mov_b32_e32 v1, 0",
                "v_readfirstlane_b32 s16, v1",
                "v_mov_b32_e32 v2, 0x0f0f0f0f",
                "v_mov_b32_e32 v3, 0x3f800000",
                "v_mov_b32_e32 v4, 0x40000000",
                "v_mov_b32_e32 v5, 0x00ff00ff",
                "ds_wrap_rtn_b32 v6, v1, v3, v4",
                "ds_cmpst_f32 v1, v3, v4",
                "ds_cmpst_rtn_f32 v6, v1, v3, v4 offset:4",
                "ds_mskor_b64 v1, v[2:3], v[4:5] offset:8",
                "ds_mskor_rtn_b64 v[6:7], v1, v[2:3], v[4:5] offset:16",
            ], lds_units=100, threads=64, emulate=True)),
        )

    def test_spirv_assembly(self):
        for label, spv in self._validation_modules():
            with self.subTest(case=label):
                binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
                self.assertEqual(struct.unpack_from("<I", binary, 0)[0], 0x07230203)

    @unittest.skipUnless(shutil.which("spirv-val"), "needs spirv-val")
    def test_spirv_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            for label, spv in self._validation_modules():
                with self.subTest(case=label):
                    spv_path = os.path.join(tmp, f"{label}.spv")
                    with open(spv_path, "wb") as f:
                        f.write(spirv_asm.assemble(spv, target_env="vulkan1.2"))
                    res = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv_path],
                                         capture_output=True, text=True)
                    if res.returncode != 0:
                        self.fail(f"spirv-val failed for {label}: {res.stderr}")


if __name__ == "__main__":
    unittest.main()

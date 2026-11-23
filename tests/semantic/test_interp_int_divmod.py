"""Self-tests for integer division and remainder in spirv_mini_interp (OpUDiv, OpUMod, OpSDiv, OpSRem).

    python -I tests/semantic/test_interp_int_divmod.py

Covers the SPIR-V unsigned and signed integer division/remainder instructions emitted by
tools/rdna2spirv.py (OpUMod for --wave64-emulate 3-slot exchange ring buffers, OpUDiv for
scratch-swizzle chunking, OpSRem for texture coordinate wrapping), plus OpSDiv:
  - 32-bit and 64-bit unsigned division/modulo (including values with the high bit set).
  - 32-bit and 64-bit signed truncating division and remainder (sign of OpSRem matches operand 1).
  - Undefined results on divisor == 0 and signed INT_MIN / -1 overflow (safe when unselected by
    OpSelect, raises UnsupportedConstruct when observed, and propagates UNDEF through _PURE).
  - End-to-end execution of a synthetic --wave64-emulate divergent loop shader whose translated
    SPIR-V emits OpUMod through spirv_mini_interp.run_workgroup.

Hand-written SPIR-V and synthetic RDNA2 only; standard library only; no game data.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

HEADER_32 = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%uint = OpTypeInt 32 0
%int = OpTypeInt 32 1
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%t = OpConstantTrue %bool
%f = OpConstantFalse %bool
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c3 = OpConstant %uint 3
%c4 = OpConstant %uint 4
%c7 = OpConstant %uint 7
%c42 = OpConstant %uint 42
%cmax = OpConstant %uint 4294967295
%chigh = OpConstant %uint 2147483649
%i0 = OpConstant %int 0
%im1 = OpConstant %int -1
%i3 = OpConstant %int 3
%im3 = OpConstant %int -3
%i4 = OpConstant %int 4
%im6 = OpConstant %int -6
%i7 = OpConstant %int 7
%im7 = OpConstant %int -7
%imin = OpConstant %int -2147483648
%main = OpFunction %void None %fn
%entry = OpLabel
"""

HEADER_64 = """\
OpCapability Shader
OpCapability Int64
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%ulong = OpTypeInt 64 0
%long = OpTypeInt 64 1
%_ptr_Output_v2uint = OpTypePointer Output %v2uint
%out = OpVariable %_ptr_Output_v2uint Output
%u0 = OpConstant %ulong 0
%u3 = OpConstant %ulong 3
%u10 = OpConstant %ulong 10
%u32 = OpConstant %ulong 32
%umax = OpConstant %ulong 18446744073709551615
%s3 = OpConstant %long 3
%sm7 = OpConstant %long -7
%sm1 = OpConstant %long -1
%smin = OpConstant %long -9223372036854775808
%main = OpFunction %void None %fn
%entry = OpLabel
"""

FOOTER = """\
OpReturn
OpFunctionEnd
"""


def run32(body: str) -> int:
    return si.run_fragment(HEADER_32 + body + FOOTER)["%out"][0]


def run64(body: str) -> int:
    epilogue = (
        "%lo = OpUConvert %uint %u\n"
        "%hi64 = OpShiftRightLogical %ulong %u %u32\n"
        "%hi = OpUConvert %uint %hi64\n"
        "%pair = OpCompositeConstruct %v2uint %lo %hi\n"
        "OpStore %out %pair\n"
    )
    lo, hi = si.run_fragment(HEADER_64 + body + epilogue + FOOTER)["%out"]
    return lo | (hi << 32)


class IntDivModInterpreterTests(unittest.TestCase):
    def test_udiv_and_umod_32bit_including_high_bit(self):
        self.assertEqual(run32("%r = OpUDiv %uint %c7 %c3\nOpStore %out %r\n"), 2)
        self.assertEqual(run32("%r = OpUMod %uint %c7 %c3\nOpStore %out %r\n"), 1)
        # Unsigned 0xFFFFFFFF / 3 = 0x55555555, rem 0; 0x80000001 / 3 = 0x2AAAAAAB, rem 0.
        self.assertEqual(run32("%r = OpUDiv %uint %cmax %c3\nOpStore %out %r\n"), 0x55555555)
        self.assertEqual(run32("%r = OpUMod %uint %cmax %c4\nOpStore %out %r\n"), 3)
        self.assertEqual(run32("%r = OpUDiv %uint %chigh %c3\nOpStore %out %r\n"), 0x2AAAAAAB)
        self.assertEqual(run32("%r = OpUMod %uint %chigh %c3\nOpStore %out %r\n"), 0)

    def test_sdiv_and_srem_32bit_sign_combinations(self):
        # OpSDiv truncates toward zero; OpSRem sign matches operand 1 (unlike Python %).
        cases = [
            ("OpSDiv", "%i7", "%i3", 2),
            ("OpSDiv", "%im7", "%i3", (-2) & 0xFFFFFFFF),
            ("OpSDiv", "%i7", "%im3", (-2) & 0xFFFFFFFF),
            ("OpSDiv", "%im7", "%im3", 2),
            ("OpSDiv", "%im6", "%i3", (-2) & 0xFFFFFFFF),
            ("OpSRem", "%i7", "%i3", 1),
            ("OpSRem", "%im7", "%i3", (-1) & 0xFFFFFFFF),
            ("OpSRem", "%i7", "%im3", 1),
            ("OpSRem", "%im7", "%im3", (-1) & 0xFFFFFFFF),
            ("OpSRem", "%im6", "%i3", 0),
            # Texture-wrap case relied on by rdna2spirv.py: OpSRem(-1, 4) == -1 (not +3).
            ("OpSRem", "%im1", "%i4", (-1) & 0xFFFFFFFF),
        ]
        for op, lhs, rhs, expected in cases:
            with self.subTest(op=op, lhs=lhs, rhs=rhs):
                got = run32(f"%r = {op} %int {lhs} {rhs}\n%u = OpBitcast %uint %r\nOpStore %out %u\n")
                self.assertEqual(got, expected)

    def test_64bit_unsigned_and_signed_divmod(self):
        self.assertEqual(run64("%u = OpUDiv %ulong %umax %u3\n"), 0x5555555555555555)
        self.assertEqual(run64("%u = OpUMod %ulong %umax %u10\n"), 5)
        self.assertEqual(run64("%r = OpSDiv %long %sm7 %s3\n%u = OpBitcast %ulong %r\n"),
                         (-2) & 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(run64("%r = OpSRem %long %sm7 %s3\n%u = OpBitcast %ulong %r\n"),
                         (-1) & 0xFFFFFFFFFFFFFFFF)

    def test_div_by_zero_and_signed_overflow_produce_undef(self):
        # Unselected OpSelect branch keeps exact value; observing UNDEF raises UnsupportedConstruct.
        undef_exprs = [
            ("%r = OpUDiv %uint %c7 %c0\n", "%uint"),
            ("%r = OpUMod %uint %c7 %c0\n", "%uint"),
            ("%r = OpSDiv %int %i7 %i0\n", "%int"),
            ("%r = OpSRem %int %i7 %i0\n", "%int"),
            ("%r = OpSDiv %int %imin %im1\n", "%int"),
            ("%r = OpSRem %int %imin %im1\n", "%int"),
        ]
        for expr, ty in undef_exprs:
            with self.subTest(expr=expr.strip()):
                if ty == "%int":
                    safe = expr + "%u = OpBitcast %uint %r\n%s = OpSelect %uint %t %c42 %u\nOpStore %out %s\n"
                    bad = expr + "%u = OpBitcast %uint %r\nOpStore %out %u\n"
                else:
                    safe = expr + "%s = OpSelect %uint %t %c42 %r\nOpStore %out %s\n"
                    bad = expr + "OpStore %out %r\n"
                self.assertEqual(run32(safe), 42)
                with self.assertRaisesRegex(si.UnsupportedConstruct, "undefined"):
                    run32(bad)

        # UNDEF input propagates cleanly through _PURE ops without raising TypeError.
        prop = (
            "%u0 = OpUDiv %uint %c7 %c0\n"
            "%u1 = OpUMod %uint %u0 %c3\n"
            "%s = OpSelect %uint %t %c42 %u1\n"
            "OpStore %out %s\n"
        )
        self.assertEqual(run32(prop), 42)

        # 64-bit INT64_MIN / -1 also produces UNDEF.
        with self.assertRaisesRegex(si.UnsupportedConstruct, "undefined"):
            run64("%r = OpSDiv %long %smin %sm1\n%u = OpBitcast %ulong %r\n")

    def test_wave64_emulate_loop_with_opumod_ring_buffer_runs_in_workgroup(self):
        """A wave64 divergent loop with a cross-half s_bcnt1 emits OpUMod under --wave64-emulate."""
        # Each lane i in 0..63 runs (i & 3) + 1 iterations, accumulating s_bcnt1(exec) per iteration.
        # Iteration k (0..3) has active lanes {i : (i & 3) >= k}: counts are 64, 48, 32, 16.
        # Expected sum for lane i with trips t = (i & 3) + 1:
        #   t=1 -> 64, t=2 -> 112, t=3 -> 144, t=4 -> 160.
        lines = [
            "s_mov_b64 s[30:31], exec",
            "v_and_b32_e32 v2, 3, v0",
            "v_mov_b32_e32 v3, 0",
            "v_mov_b32_e32 v4, 0",
            "s_mov_b64 s[32:33], 0",
            # Loop body starts at instruction index 5 (PC 0x14), s_cbranch_execnz is at index 11 (PC 0x2C):
            # next PC = 0x30, target = 0x14 -> delta = -28 bytes = -7 dwords (0xFFF9).
            "s_bcnt1_i32_b64 s10, exec",
            "v_add_nc_u32_e32 v4, s10, v4",
            "v_cmp_ge_u32_e64 s[34:35], v3, v2",
            "v_add_nc_u32_e32 v3, 1, v3",
            "s_or_b64 s[32:33], s[34:35], s[32:33]",
            "s_andn2_b64 exec, exec, s[32:33]",
            "s_cbranch_execnz -7  ; encoding: [0xf9,0xff,0x89,0xbf]",
            "s_or_b64 exec, exec, s[32:33]",
            "s_mov_b64 exec, s[30:31]",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v4, v5, s[0:3], 0 offen",
            "s_endpgm",
        ]
        stub_dxbc = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
        fssm = bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1)
        vsharp = [0, 0, 0xFFFFFFFF, 0x31016FAC]

        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", "cs_w64_loop_umod.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_w64_loop_umod synthetic\n")
                for ln in lines:
                    if "; encoding:" in ln:
                        f.write(f"\t{ln}\n")
                    else:
                        f.write(f"\t{ln}  ; encoding: [0x00,0x00,0x00,0x00]\n")
            with open(os.path.join(tmp, "cs_w64_loop_umod.dxbc"), "wb") as f:
                f.write(stub_dxbc)
            with open(os.path.join(tmp, "fssm", "cs_w64_loop_umod.fssm"), "wb") as f:
                f.write(fssm)
            old_emulate = rdna2spirv.WAVE64_EMULATE
            try:
                rdna2spirv.WAVE64_EMULATE = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    spv = rdna2spirv.translate(s_path)
                    gc.collect()
            finally:
                rdna2spirv.WAVE64_EMULATE = old_emulate

        self.assertIn("OpUMod", spv)
        mem = si.run_workgroup(
            spv, 64, 32,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32, "%sg_id": i // 32},
            push_constants=vsharp + [0] * 28,
            memory={},
        )
        expected_by_rem = (64, 112, 144, 160)
        for lane in range(64):
            self.assertEqual(mem.get(lane * 4), expected_by_rem[lane & 3], f"lane {lane}")


if __name__ == "__main__":
    unittest.main()

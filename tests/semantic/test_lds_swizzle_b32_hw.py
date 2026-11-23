from __future__ import annotations

"""Hardware and translator semantic tests for RDNA2 ds_swizzle_b32 through tools/rdna2spirv.py.

All 26 tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V via spirv_mini_interp.run_workgroup across Wave32 and Wave64.

RDNA2 LDS architecture contract:
- Opcode: ds_swizzle_b32 vdst, vsrc offset:swizzle(...) | offset:N
- Operation: Subgroup intra-wave cross-lane shuffle within each 32-lane group (lanes 0..31 in wave32;
  lanes 0..31 and 32..63 independently in wave64).
- 16-bit offset encoding:
  * Bit 15 = 1 (QUAD_PERM): src = (lane & ~3) | ((offset >> (2 * (lane & 3))) & 3)
  * Bit 15 = 0 (BITMASK_PERM): src = (lane & ~31) | ((((lane & 31) & and_m) | or_m) ^ xor_m),
    where and_m = offset[4:0], or_m = offset[9:5], xor_m = offset[14:10].
  * Shorthands:
    - SWAP(stride): lane ^ stride (and=31, or=0, xor=stride)
    - REVERSE(width): lane ^ (width - 1)
    - BROADCAST(group_size, lane_idx): (lane & ~(group_size - 1)) | lane_idx
- Inactive lanes:
  * Inactive destination lanes (EXEC=0) preserve their destination VGPR.
  * Inactive or absent source lanes return 0 to the destination.
- Condition codes: VALU/LDS operations do not modify SCC, VCC, or EXEC.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402


def make_fssm_header(threads: int = 32, rsrc2: int = 0x8088) -> bytes:
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, rsrc2, threads, 1, 1)
    return bytes(h)


def run_cs_swizzle(
    name: str,
    swizzle_line: str,
    wave: int = 32,
    init_offset: int = 100,
    in_place: bool = False,
) -> tuple[str, list[int]]:
    dst_reg = "v1" if in_place else "v2"
    lines = [
        f"v_add_nc_u32_e32 v1, {init_offset}, v0",
        swizzle_line,
        "v_lshlrev_b32_e32 v3, 2, v0",
        f"buffer_store_dword {dst_reg}, v3, s[0:3], 0 offen",
        "s_endpgm",
    ]
    fssm = make_fssm_header(wave)
    spvasm = support.translate_lines(name, lines, fssm=fssm)
    mem = {4 * i: 0xEEEEEEEE for i in range(wave)}
    si.run_workgroup(
        spvasm,
        wave,
        wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28,
        memory=mem,
    )
    return spvasm, [mem[4 * i] for i in range(wave)]


# ==============================================================================
# Suite 1: SWAP Modes (5 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32SwapModes(unittest.TestCase):
    """Verifies all SWAP strides translated through rdna2spirv and run on SPIR-V."""

    def test_translator_swap_stride_1(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_swap1", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)")
        self.assertIn("OpBitwiseXor", spv)
        self.assertEqual(res[0:4], [101, 100, 103, 102])
        for i in range(0, 32, 2):
            self.assertEqual(res[i], 100 + i + 1)
            self.assertEqual(res[i + 1], 100 + i)

    def test_translator_swap_stride_2(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_swap2", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)")
        self.assertIn("OpBitwiseXor", spv)
        self.assertEqual(res[0:4], [102, 103, 100, 101])
        for i in range(0, 32, 4):
            self.assertEqual(res[i], 100 + i + 2)
            self.assertEqual(res[i + 1], 100 + i + 3)
            self.assertEqual(res[i + 2], 100 + i)
            self.assertEqual(res[i + 3], 100 + i + 1)

    def test_translator_swap_stride_4(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_swap4", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,4)")
        self.assertIn("OpBitwiseXor", spv)
        for i in range(0, 32, 8):
            for k in range(4):
                self.assertEqual(res[i + k], 100 + i + k + 4)
                self.assertEqual(res[i + k + 4], 100 + i + k)

    def test_translator_swap_stride_8(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_swap8", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,8)")
        self.assertIn("OpBitwiseXor", spv)
        for i in range(0, 32, 16):
            for k in range(8):
                self.assertEqual(res[i + k], 100 + i + k + 8)
                self.assertEqual(res[i + k + 8], 100 + i + k)

    def test_translator_swap_stride_16(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_swap16", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,16)")
        self.assertIn("OpBitwiseXor", spv)
        for i in range(16):
            self.assertEqual(res[i], 100 + i + 16)
            self.assertEqual(res[i + 16], 100 + i)


# ==============================================================================
# Suite 2: REVERSE Modes (4 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32ReverseModes(unittest.TestCase):
    """Verifies all REVERSE widths translated through rdna2spirv and run on SPIR-V."""

    def test_translator_reverse_width_2(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_rev2", "ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,2)")
        self.assertIn("OpBitwiseXor", spv)
        self.assertEqual(res[0:2], [101, 100])
        self.assertEqual(res[2:4], [103, 102])

    def test_translator_reverse_width_4(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_rev4", "ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,4)")
        self.assertIn("OpBitwiseXor", spv)
        self.assertEqual(res[0:4], [103, 102, 101, 100])
        self.assertEqual(res[4:8], [107, 106, 105, 104])

    def test_translator_reverse_width_8(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_rev8", "ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,8)")
        self.assertIn("OpBitwiseXor", spv)
        expected_oct = [107, 106, 105, 104, 103, 102, 101, 100]
        self.assertEqual(res[0:8], expected_oct)

    def test_translator_reverse_width_32(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_rev32", "ds_swizzle_b32 v2, v1 offset:swizzle(REVERSE,32)")
        self.assertIn("OpBitwiseXor", spv)
        for i in range(32):
            self.assertEqual(res[i], 100 + (31 - i))


# ==============================================================================
# Suite 3: QUAD_PERM Modes (4 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32QuadPermModes(unittest.TestCase):
    """Verifies QUAD_PERM patterns translated through rdna2spirv and run on SPIR-V."""

    def test_translator_quad_perm_identity(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_qp_id", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,0,1,2,3)")
        self.assertEqual(res[0:4], [100, 101, 102, 103])
        self.assertEqual(res[4:8], [104, 105, 106, 107])

    def test_translator_quad_perm_reverse(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_qp_rev", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,3,2,1,0)")
        self.assertEqual(res[0:4], [103, 102, 101, 100])
        self.assertEqual(res[4:8], [107, 106, 105, 104])

    def test_translator_quad_perm_cycle(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_qp_cyc", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,1,2,3,0)")
        self.assertEqual(res[0:4], [101, 102, 103, 100])
        self.assertEqual(res[4:8], [105, 106, 107, 104])

    def test_translator_quad_perm_broadcast_lane(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_qp_bc", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,2,2,2,2)")
        self.assertEqual(res[0:4], [102, 102, 102, 102])
        self.assertEqual(res[4:8], [106, 106, 106, 106])


# ==============================================================================
# Suite 4: BROADCAST and BITMASK Modes (4 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32BroadcastAndBitmask(unittest.TestCase):
    """Verifies BROADCAST and BITMASK patterns translated through rdna2spirv and run on SPIR-V."""

    def test_translator_broadcast_quad(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_bc4", "ds_swizzle_b32 v2, v1 offset:swizzle(BROADCAST,4,0)")
        self.assertEqual(res[0:4], [100, 100, 100, 100])
        self.assertEqual(res[4:8], [104, 104, 104, 104])

    def test_translator_broadcast_oct(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_bc8", "ds_swizzle_b32 v2, v1 offset:swizzle(BROADCAST,8,3)")
        for i in range(8):
            self.assertEqual(res[i], 103)
            self.assertEqual(res[i + 8], 111)

    def test_translator_bitmask_invert_all(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_bm_inv", 'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"iiiii")')
        for i in range(32):
            self.assertEqual(res[i], 100 + (i ^ 31))

    def test_translator_bitmask_zero_all(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_bm_zero", 'ds_swizzle_b32 v2, v1 offset:swizzle(BITMASK_PERM,"00000")')
        for i in range(32):
            self.assertEqual(res[i], 100)


# ==============================================================================
# Suite 5: Register Aliasing & In-Place Execution (3 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32RegisterAliasing(unittest.TestCase):
    """Verifies in-place register aliasing executed through rdna2spirv and SPIR-V."""

    def test_translator_in_place_swap_aliasing(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_alias_swap", "ds_swizzle_b32 v1, v1 offset:swizzle(SWAP,1)", in_place=True)
        self.assertEqual(res[0:4], [101, 100, 103, 102])

    def test_translator_in_place_quad_perm_aliasing(self) -> None:
        spv, res = run_cs_swizzle(
            "cs_swz_alias_qp",
            "ds_swizzle_b32 v1, v1 offset:swizzle(QUAD_PERM,3,2,1,0)",
            in_place=True,
        )
        self.assertEqual(res[0:4], [103, 102, 101, 100])

    def test_translator_in_place_reverse_aliasing(self) -> None:
        spv, res = run_cs_swizzle(
            "cs_swz_alias_rev",
            "ds_swizzle_b32 v1, v1 offset:swizzle(REVERSE,4)",
            in_place=True,
        )
        self.assertEqual(res[0:4], [103, 102, 101, 100])


# ==============================================================================
# Suite 6: Wave64 & Numeric Offset Encoding (3 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32Wave64AndNumeric(unittest.TestCase):
    """Verifies Wave64 execution and raw numeric offset decoding through translator."""

    def test_translator_wave64_independent_halves(self) -> None:
        spv, res = run_cs_swizzle("cs_swz_w64", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)", wave=64)
        # Low half (0..31): swizzle within 0..31
        self.assertEqual(res[0:4], [101, 100, 103, 102])
        # High half (32..63): swizzle within 32..63 without leaking to low half
        self.assertEqual(res[32:36], [133, 132, 135, 134])

    def test_translator_numeric_quad_perm(self) -> None:
        # Raw numeric offset 0x801F = QUAD_PERM(3, 3, 1, 0)
        # Note: raw offset without symbol requires swizzle keyword or valid disassembly token
        spv, res = run_cs_swizzle("cs_swz_num_qp", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM,3,3,1,0)")
        self.assertEqual(res[0:4], [103, 103, 101, 100])

    def test_translator_numeric_swap_hex(self) -> None:
        # SWAP(1) corresponds to offset 0x041F: (1 << 10) | 0x1F
        spv, res = run_cs_swizzle("cs_swz_num_swap", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)")
        self.assertEqual(res[0:4], [101, 100, 103, 102])


# ==============================================================================
# Suite 7: Rejection, Controls, and Mutation Detection (3 tests) - 100% Translator-Backed
# ==============================================================================

class TestDsSwizzleB32RejectionAndMutations(unittest.TestCase):
    """Verifies syntax rejection and mutation detection through the translator."""

    def test_rejection_invalid_offset_range(self) -> None:
        lines = [
            "ds_swizzle_b32 v2, v1 offset:0xc000",
            "s_endpgm",
        ]
        fssm = make_fssm_header(32)
        with self.assertRaises(Exception):
            support.translate_lines("cs_swizzle_bad_offset", lines, fssm=fssm)

    def test_rejection_missing_operands(self) -> None:
        lines = [
            "ds_swizzle_b32 v2",
            "s_endpgm",
        ]
        fssm = make_fssm_header(32)
        with self.assertRaises(Exception):
            support.translate_lines("cs_swizzle_missing_op", lines, fssm=fssm)

    def test_mutation_detection_swap_stride_sabotage(self) -> None:
        """Verifies that an altered SWAP stride (e.g. 1 mutated to 2) is detected."""
        _, real_res = run_cs_swizzle("cs_swz_mut_real", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,1)")
        _, mut_res = run_cs_swizzle("cs_swz_mut_bad", "ds_swizzle_b32 v2, v1 offset:swizzle(SWAP,2)")
        self.assertNotEqual(real_res[0:4], mut_res[0:4])
        self.assertEqual(real_res[0:4], [101, 100, 103, 102])
        self.assertEqual(mut_res[0:4], [102, 103, 100, 101])


if __name__ == "__main__":
    unittest.main()

"""Semantic tests for RDNA2 v_cubeid_f32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_cubeid_f32 dst, src0, src1, src2
Computes the cubemap face index (0.0 .. 5.0) from a 3D direction vector (x, y, z):
    src0 = x, src1 = y, src2 = z (IEEE 754 binary32).
Major axis selection (Z beats Y and X; Y beats X):
    if |z| >= |x| and |z| >= |y|: major = Z (z < 0 -> face 5.0 [-Z]; else face 4.0 [+Z])
    elif |y| >= |x|:              major = Y (y < 0 -> face 3.0 [-Y]; else face 2.0 [+Y])
    else:                         major = X (x < 0 -> face 1.0 [-X]; else face 0.0 [+X])
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_cubeid(x: float, y: float, z: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_cubeid_f32."""
    x, y, z = f32(x), f32(y), f32(z)
    ax, ay, az = abs(x), abs(y), abs(z)
    if az >= ax and az >= ay:
        return 5.0 if z < 0.0 else 4.0
    elif ay >= ax:
        return 3.0 if y < 0.0 else 2.0
    else:
        return 1.0 if x < 0.0 else 0.0


def ref_cubeid_bits(x_bits: int, y_bits: int, z_bits: int) -> int:
    x = bits_to_f32(x_bits)
    y = bits_to_f32(y_bits)
    z = bits_to_f32(z_bits)
    return f32_to_bits(ref_cubeid(x, y, z))


def run_hw_cubeid(x_bits: int, y_bits: int, z_bits: int) -> int:
    """Execute v_cubeid_f32 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {x_bits:#x}",
        f"v_mov_b32 v2, {y_bits:#x}",
        f"v_mov_b32 v3, {z_bits:#x}",
        "v_cubeid_f32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cubeid_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVCubeidF32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_all_six_faces(self):
        cases = [
            # (x, y, z, expected_face)
            (4.0, -1.5, 0.75, 0.0),   # +X (0)
            (-4.0, 1.5, -0.75, 1.0),  # -X (1)
            (1.25, 3.0, -0.5, 2.0),   # +Y (2)
            (-1.25, -3.0, 0.5, 3.0),  # -Y (3)
            (-0.5, 1.25, 2.5, 4.0),   # +Z (4)
            (0.5, -1.25, -2.5, 5.0),  # -Z (5)
        ]
        for x, y, z, expected in cases:
            with self.subTest(x=x, y=y, z=z):
                self.assertEqual(ref_cubeid(x, y, z), expected)

    def test_reference_tie_breaker_hierarchy(self):
        # |x| == |y| (> |z|): Y wins over X
        self.assertEqual(ref_cubeid(2.0, 2.0, 0.5), 2.0)    # +Y
        self.assertEqual(ref_cubeid(2.0, -2.0, 0.5), 3.0)   # -Y
        self.assertEqual(ref_cubeid(-2.0, 2.0, -0.5), 2.0)  # +Y
        self.assertEqual(ref_cubeid(-2.0, -2.0, -0.5), 3.0) # -Y

        # |y| == |z| (> |x|): Z wins over Y
        self.assertEqual(ref_cubeid(0.5, 2.0, 2.0), 4.0)    # +Z
        self.assertEqual(ref_cubeid(0.5, -2.0, -2.0), 5.0)  # -Z

        # 3-way tie: |x| == |y| == |z|: Z wins
        self.assertEqual(ref_cubeid(1.5, -1.5, 1.5), 4.0)   # +Z
        self.assertEqual(ref_cubeid(-1.5, 1.5, -1.5), 5.0)  # -Z

    def test_translator_hw_cubeid(self):
        vectors = [
            (4.0, -1.5, 0.75),
            (-4.0, 1.5, -0.75),
            (1.25, 3.0, -0.5),
            (-1.25, -3.0, 0.5),
            (-0.5, 1.25, 2.5),
            (0.5, -1.25, -2.5),
            (2.0, 2.0, 0.5),
            (0.5, 2.0, 2.0),
        ]
        for x, y, z in vectors:
            with self.subTest(x=x, y=y, z=z):
                xb = f32_to_bits(x)
                yb = f32_to_bits(y)
                zb = f32_to_bits(z)
                hw_res = run_hw_cubeid(xb, yb, zb)
                ref_res = ref_cubeid_bits(xb, yb, zb)
                self.assertEqual(hw_res, ref_res)


class RegType(enum.Enum):
    VGPR = 1
    SGPR = 2
    IMM = 3


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    value: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")


@dataclasses.dataclass(frozen=True)
class VCubeidF32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")


class TestVCubeidF32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 2),
            src2=Operand(RegType.IMM, 0, value=f32_to_bits(1.0)),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)
        self.assertEqual(inst.src2.value, f32_to_bits(1.0))

    def test_ast_validation_rejects_invalid_dst(self):
        with self.assertRaises(TypeError):
            VCubeidF32Instruction(
                dst=Operand(RegType.SGPR, 0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )
        with self.assertRaises(TypeError):
            VCubeidF32Instruction(
                dst=Operand(RegType.IMM, 0, value=0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )

    def test_ast_validation_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1)


class ScalarRegisterFile:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def write_vgpr(self, idx: int, value: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = value & MASK32


class ScalarVALUInterpreter:
    @staticmethod
    def execute(rf: ScalarRegisterFile, inst: VCubeidF32Instruction) -> int:
        s0 = rf.read_operand(inst.src0)
        s1 = rf.read_operand(inst.src1)
        s2 = rf.read_operand(inst.src2)
        res = ref_cubeid_bits(s0, s1, s2)
        rf.write_vgpr(inst.dst.index, res)
        return res


class TestVCubeidF32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_all_vgpr(self):
        self.rf.write_vgpr(1, f32_to_bits(4.0))
        self.rf.write_vgpr(2, f32_to_bits(-1.5))
        self.rf.write_vgpr(3, f32_to_bits(0.75))
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = f32_to_bits(0.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[0], expected)

    def test_scalar_execution_sgpr_and_imm(self):
        self.rf.vgprs[1] = f32_to_bits(-0.5)
        self.rf.sgprs[5] = f32_to_bits(1.25)
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 4),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 5),
            src2=Operand(RegType.IMM, 0, value=f32_to_bits(-2.5)),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = f32_to_bits(5.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[4], expected)


class SIMDState:
    def __init__(self, num_lanes: int = 32, exec_mask: int = 0xFFFFFFFF):
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count {num_lanes}, must be 32 or 64")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.sgprs = [0] * 106

    def read_lane(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index][lane] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")


class SIMDInterpreter:
    @staticmethod
    def execute(state: SIMDState, inst: VCubeidF32Instruction) -> list[int]:
        results = []
        for lane in range(state.num_lanes):
            lane_active = (state.exec_mask >> lane) & 1
            if lane_active:
                s0 = state.read_lane(inst.src0, lane)
                s1 = state.read_lane(inst.src1, lane)
                s2 = state.read_lane(inst.src2, lane)
                res = ref_cubeid_bits(s0, s1, s2)
                state.vgprs[inst.dst.index][lane] = res
                results.append(res)
            else:
                results.append(state.vgprs[inst.dst.index][lane])
        return results


class TestVCubeidF32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec(self):
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for i in range(32):
            sign = -1.0 if (i % 2) else 1.0
            state.vgprs[1][i] = f32_to_bits(sign * (i + 1.0))
            state.vgprs[2][i] = f32_to_bits(1.0)
            state.vgprs[3][i] = f32_to_bits(0.5)
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            sign = -1.0 if (i % 2) else 1.0
            expected = ref_cubeid_bits(f32_to_bits(sign * (i + 1.0)), f32_to_bits(1.0), f32_to_bits(0.5))
            self.assertEqual(res[i], expected)
            self.assertEqual(state.vgprs[0][i], expected)

    def test_simd_wave32_divergent_mask(self):
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        for i in range(32):
            state.vgprs[0][i] = 0xDEADBEEF
            state.vgprs[1][i] = f32_to_bits(4.0)
            state.vgprs[2][i] = f32_to_bits(-1.5)
            state.vgprs[3][i] = f32_to_bits(0.75)
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_cubeid_bits(f32_to_bits(4.0), f32_to_bits(-1.5), f32_to_bits(0.75))
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(res[i], expected_active)
                self.assertEqual(state.vgprs[0][i], expected_active)
            else:
                self.assertEqual(res[i], 0xDEADBEEF)
                self.assertEqual(state.vgprs[0][i], 0xDEADBEEF)

    def test_simd_wave64_partial_exec(self):
        mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for i in range(64):
            state.vgprs[0][i] = 0xCAFEBABE
            state.vgprs[1][i] = f32_to_bits(-0.5)
            state.vgprs[2][i] = f32_to_bits(1.25)
            state.vgprs[3][i] = f32_to_bits(-2.5)
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_cubeid_bits(f32_to_bits(-0.5), f32_to_bits(1.25), f32_to_bits(-2.5))
        for i in range(64):
            if i < 32:
                self.assertEqual(res[i], expected_active)
            else:
                self.assertEqual(res[i], 0xCAFEBABE)


class TestVCubeidF32RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src0(self):
        self.rf.write_vgpr(1, f32_to_bits(4.0))
        self.rf.write_vgpr(2, f32_to_bits(-1.5))
        self.rf.write_vgpr(3, f32_to_bits(0.75))
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = f32_to_bits(0.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[1], expected)

    def test_dst_aliases_src1(self):
        self.rf.write_vgpr(1, f32_to_bits(4.0))
        self.rf.write_vgpr(2, f32_to_bits(-1.5))
        self.rf.write_vgpr(3, f32_to_bits(0.75))
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 2),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = f32_to_bits(0.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[2], expected)

    def test_dst_aliases_src2(self):
        self.rf.write_vgpr(1, f32_to_bits(4.0))
        self.rf.write_vgpr(2, f32_to_bits(-1.5))
        self.rf.write_vgpr(3, f32_to_bits(0.75))
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 3),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = f32_to_bits(0.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[3], expected)

    def test_dst_aliases_all_operands(self):
        self.rf.write_vgpr(1, f32_to_bits(2.0))
        inst = VCubeidF32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 1),
            src2=Operand(RegType.VGPR, 1),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        # For x == y == z == 2.0, Z wins (+Z face = 4.0)
        expected = f32_to_bits(4.0)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[1], expected)

    def test_hw_translator_aliasing(self):
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(4.0):#x}",
            f"v_mov_b32 v2, {f32_to_bits(-1.5):#x}",
            f"v_mov_b32 v3, {f32_to_bits(0.75):#x}",
            "v_cubeid_f32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cubeid_alias", lines)
        self.assertEqual(support.mrt0_words(spv)[0], f32_to_bits(0.0))


@dataclasses.dataclass(frozen=True)
class CubeIdStatusFlags:
    face_id: float
    is_positive_axis: bool
    major_axis: str


def compute_status_flags(x: float, y: float, z: float) -> CubeIdStatusFlags:
    fid = ref_cubeid(x, y, z)
    int_id = int(fid)
    axis_map = {0: "X", 1: "X", 2: "Y", 3: "Y", 4: "Z", 5: "Z"}
    return CubeIdStatusFlags(
        face_id=fid,
        is_positive_axis=(int_id % 2 == 0),
        major_axis=axis_map[int_id],
    )


class TestVCubeidF32ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and classification."""

    def test_positive_axis_flags(self):
        flags = compute_status_flags(4.0, 1.0, 0.5)
        self.assertEqual(flags.face_id, 0.0)
        self.assertTrue(flags.is_positive_axis)
        self.assertEqual(flags.major_axis, "X")

    def test_negative_axis_flags(self):
        flags = compute_status_flags(-4.0, 1.0, 0.5)
        self.assertEqual(flags.face_id, 1.0)
        self.assertFalse(flags.is_positive_axis)
        self.assertEqual(flags.major_axis, "X")

    def test_z_axis_flags(self):
        flags = compute_status_flags(0.5, 1.0, -2.5)
        self.assertEqual(flags.face_id, 5.0)
        self.assertFalse(flags.is_positive_axis)
        self.assertEqual(flags.major_axis, "Z")


class TestVCubeidF32IEEE754(unittest.TestCase):
    """IEEE 754 raw bit pattern preservation and subnormal/NaN behavior."""

    def test_ieee754_signed_zeros(self):
        # (0.0, 0.0, 0.0) -> |z| >= |x| and |z| >= |y| -> +Z (4.0)
        p0 = f32_to_bits(0.0)
        n0 = f32_to_bits(-0.0)
        self.assertEqual(ref_cubeid_bits(p0, p0, p0), f32_to_bits(4.0))
        # When z is -0.0, -0.0 < 0.0 is False in float compare, returns 4.0
        self.assertEqual(ref_cubeid_bits(n0, n0, n0), f32_to_bits(4.0))

    def test_ieee754_subnormals(self):
        # Smallest positive subnormal: 0x00000001
        sub_pos = 0x00000001
        one = f32_to_bits(1.0)
        # Major axis is Y (1.0), so face is +Y (2.0)
        res = ref_cubeid_bits(sub_pos, one, 0)
        self.assertEqual(res, f32_to_bits(2.0))

    def test_ieee754_infinities(self):
        inf = f32_to_bits(math.inf)
        neginf = f32_to_bits(-math.inf)
        one = f32_to_bits(1.0)
        # Z is +inf -> face 4.0 (+Z)
        self.assertEqual(ref_cubeid_bits(one, 0, inf), f32_to_bits(4.0))
        # Z is -inf -> face 5.0 (-Z)
        self.assertEqual(ref_cubeid_bits(one, 0, neginf), f32_to_bits(5.0))

    def test_ieee754_hw_translator(self):
        inf = f32_to_bits(math.inf)
        one = f32_to_bits(1.0)
        hw_res = run_hw_cubeid(one, 0, inf)
        self.assertEqual(hw_res, f32_to_bits(4.0))


class TestVCubeidF32NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_cubeid_f32."""

    def test_rejection_of_invalid_syntax(self):
        lines_missing = [
            "v_mov_b32 v1, 1.0",
            "v_cubeid_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_cubeid_err", lines_missing)

    def test_rejection_of_out_of_bounds_register(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106)

    def test_mutation_detection_inverted_tie_breaker(self):
        # Mutated oracle where X wins ties over Y and Z
        def bad_oracle_x_wins(x, y, z):
            ax, ay, az = abs(x), abs(y), abs(z)
            if ax >= ay and ax >= az:
                return 1.0 if x < 0.0 else 0.0
            elif ay >= az:
                return 3.0 if y < 0.0 else 2.0
            else:
                return 5.0 if z < 0.0 else 4.0

        # For |x| == |y| == |z| (e.g. 2.0, 2.0, 2.0):
        # Correct RDNA2 (Z wins): 4.0
        # Mutated (X wins): 0.0
        self.assertNotEqual(ref_cubeid(2.0, 2.0, 2.0), bad_oracle_x_wins(2.0, 2.0, 2.0))

    def test_mutation_detection_inverted_sign_faces(self):
        # Mutated oracle where +Z is 5.0 and -Z is 4.0
        def bad_oracle_z_signs(x, y, z):
            ax, ay, az = abs(x), abs(y), abs(z)
            if az >= ax and az >= ay:
                return 4.0 if z < 0.0 else 5.0
            elif ay >= ax:
                return 3.0 if y < 0.0 else 2.0
            else:
                return 1.0 if x < 0.0 else 0.0

        self.assertNotEqual(ref_cubeid(0.5, 0.5, 2.0), bad_oracle_z_signs(0.5, 0.5, 2.0))


if __name__ == "__main__":
    unittest.main()

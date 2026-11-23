from __future__ import annotations

"""Semantic interpreter tests and reference architectural model for RDNA2 LDS Sub-Dword Memory Suite.

Covers:
- ds_read_u8: 8-bit unsigned byte read from LDS, zero-extended to 32 bits.
- ds_read_i8: 8-bit signed byte read from LDS, sign-extended to 32 bits.
- ds_read_u16: 16-bit unsigned halfword read from LDS, zero-extended to 32 bits.
- ds_read_i16: 16-bit signed halfword read from LDS, sign-extended to 32 bits.
- ds_write_b8: 8-bit byte write to LDS, modifying only addressed byte within dword.
- ds_write_b16: 16-bit halfword write to LDS, modifying only addressed halfword within dword.

Standard library only (python -I, struct, math, unittest). Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
LDS_SIZE: int = 65536


# ------------------------------------------------------------------------------
# Architectural LDS Memory Reference Model
# ------------------------------------------------------------------------------

class LDSMemory:
    """Byte-addressable LDS memory reference model wrapping at LDS_SIZE."""

    def __init__(self, size: int = LDS_SIZE):
        self.size = size
        self.mem: bytearray = bytearray(size)

    def write_u8(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF

    def read_u8(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        return self.mem[eff]

    def read_i8(self, addr: int, offset: int = 0) -> int:
        b = self.read_u8(addr, offset)
        return (b - 256 if (b & 0x80) else b) & MASK32

    def write_u16(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF
        self.mem[(eff + 1) % self.size] = (val >> 8) & 0xFF

    def read_u16(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        lo = self.mem[eff]
        hi = self.mem[(eff + 1) % self.size]
        return (lo | (hi << 8)) & 0xFFFF

    def read_i16(self, addr: int, offset: int = 0) -> int:
        h = self.read_u16(addr, offset)
        return (h - 65536 if (h & 0x8000) else h) & MASK32


class TestDsReadU8Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_u8_zero_extension(self):
        self.lds.write_u8(0, 0xAB)
        self.assertEqual(self.lds.read_u8(0), 0xAB)
        self.lds.write_u8(1, 0xFF)
        self.assertEqual(self.lds.read_u8(1), 0xFF)

    def test_read_u8_with_offset(self):
        self.lds.write_u8(10, 0x42)
        self.assertEqual(self.lds.read_u8(4, offset=6), 0x42)


# ------------------------------------------------------------------------------
# ds_read_u8 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"v{self.index}"


@dataclass
class DSReadU8Instruction:
    dst: RegisterOperand
    addr: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR or self.addr.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_read_u8 must be VGPRs")


class ScalarMachineDS:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = LDSMemory()

    def step_read_u8(self, insn: DSReadU8Instruction):
        addr = self.vgprs[insn.addr.index]
        res = self.lds.read_u8(addr, insn.offset)
        self.vgprs[insn.dst.index] = res


class TestDsReadU8ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = DSReadU8Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            addr=RegisterOperand(OperandType.VGPR, 1),
            offset=4,
        )
        self.assertEqual(insn.dst.name, "v2")
        self.assertEqual(insn.offset, 4)

    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.lds.write_u8(20, 0x7E)
        machine.vgprs[1] = 16
        insn = DSReadU8Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            addr=RegisterOperand(OperandType.VGPR, 1),
            offset=4,
        )
        machine.step_read_u8(insn)
        self.assertEqual(machine.vgprs[2], 0x7E)


# ------------------------------------------------------------------------------
# ds_read_u8 SIMD Divergence & Addressing
# ------------------------------------------------------------------------------

class SIMDWaveDS:
    def __init__(self, wave_size: int = 64):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.exec_mask: int = (1 << wave_size) - 1
        self.lds: LDSMemory = LDSMemory()

    def step_read_u8(self, insn: DSReadU8Instruction):
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                addr = self.vgprs[insn.addr.index][lane]
                self.vgprs[insn.dst.index][lane] = self.lds.read_u8(addr, insn.offset)


class TestDsReadU8SIMD(unittest.TestCase):
    def test_simd_divergent_byte_addresses(self):
        wave = SIMDWaveDS(wave_size=32)
        for i in range(32):
            wave.lds.write_u8(i * 4, (i * 7) & 0xFF)
            wave.vgprs[1][i] = i * 4
            wave.vgprs[2][i] = 0

        insn = DSReadU8Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            addr=RegisterOperand(OperandType.VGPR, 1),
        )
        wave.step_read_u8(insn)

        for i in range(32):
            self.assertEqual(wave.vgprs[2][i], (i * 7) & 0xFF)


# ------------------------------------------------------------------------------
# ds_read_u8 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import lds_run as L
import support


class TestDsReadU8HardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 300)

    def test_hardware_translator_execution(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xef",
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xEF)


# ------------------------------------------------------------------------------
# ds_read_i8 Reference Oracle
# ------------------------------------------------------------------------------

class TestDsReadI8Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_i8_sign_extension(self):
        # 0x7F = +127
        self.lds.write_u8(0, 0x7F)
        self.assertEqual(self.lds.read_i8(0), 127)
        # 0x80 = -128 -> 0xFFFFFF80
        self.lds.write_u8(1, 0x80)
        self.assertEqual(self.lds.read_i8(1), 0xFFFF_FF80)
        # 0xFF = -1 -> 0xFFFFFFFF
        self.lds.write_u8(2, 0xFF)
        self.assertEqual(self.lds.read_i8(2), 0xFFFF_FFFF)


# ------------------------------------------------------------------------------
# ds_read_i8 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class DSReadI8Instruction:
    dst: RegisterOperand
    addr: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR or self.addr.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_read_i8 must be VGPRs")


class TestDsReadI8ASTAndScalar(unittest.TestCase):
    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.lds.write_u8(5, 0xFE)  # -2
        machine.vgprs[1] = 5
        insn = DSReadI8Instruction(dst=RegisterOperand(OperandType.VGPR, 3), addr=RegisterOperand(OperandType.VGPR, 1))
        addr = machine.vgprs[insn.addr.index]
        machine.vgprs[insn.dst.index] = machine.lds.read_i8(addr, insn.offset)
        self.assertEqual(machine.vgprs[3], 0xFFFF_FFFE)


# ------------------------------------------------------------------------------
# ds_read_i8 SIMD Divergence & Sign Extension
# ------------------------------------------------------------------------------

class TestDsReadI8SIMD(unittest.TestCase):
    def test_simd_mixed_positive_and_negative(self):
        wave = SIMDWaveDS(wave_size=32)
        for i in range(32):
            val = 0x80 + i  # negative signed values in [-128, -97]
            wave.lds.write_u8(i, val)
            wave.vgprs[1][i] = i

        for i in range(32):
            wave.vgprs[3][i] = wave.lds.read_i8(wave.vgprs[1][i])
            expected = (0x80 + i - 256) & MASK32
            self.assertEqual(wave.vgprs[3][i], expected)


# ------------------------------------------------------------------------------
# ds_read_i8 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestDsReadI8HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x85",  # -123 signed 8-bit
            "ds_write_b8 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i8 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFF_FF85)


# ------------------------------------------------------------------------------
# ds_read_u16 Reference Oracle
# ------------------------------------------------------------------------------

class TestDsReadU16Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_u16_zero_extension(self):
        self.lds.write_u16(0, 0xABCD)
        self.assertEqual(self.lds.read_u16(0), 0xABCD)
        self.lds.write_u16(2, 0xFFFF)
        self.assertEqual(self.lds.read_u16(2), 0xFFFF)


# ------------------------------------------------------------------------------
# ds_read_u16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class DSReadU16Instruction:
    dst: RegisterOperand
    addr: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR or self.addr.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_read_u16 must be VGPRs")


class TestDsReadU16ASTAndScalar(unittest.TestCase):
    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.lds.write_u16(8, 0x1234)
        machine.vgprs[1] = 8
        addr = machine.vgprs[1]
        machine.vgprs[2] = machine.lds.read_u16(addr)
        self.assertEqual(machine.vgprs[2], 0x1234)


# ------------------------------------------------------------------------------
# ds_read_u16 SIMD Divergence & Halfword Addressing
# ------------------------------------------------------------------------------

class TestDsReadU16SIMD(unittest.TestCase):
    def test_simd_halfword_reads(self):
        wave = SIMDWaveDS(wave_size=32)
        for i in range(32):
            wave.lds.write_u16(i * 2, i * 1000)
            wave.vgprs[1][i] = i * 2

        for i in range(32):
            wave.vgprs[2][i] = wave.lds.read_u16(wave.vgprs[1][i])
            self.assertEqual(wave.vgprs[2][i], (i * 1000) & 0xFFFF)


# ------------------------------------------------------------------------------
# ds_read_u16 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestDsReadU16HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xcdef",
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xCDEF)


# ------------------------------------------------------------------------------
# ds_read_i16 Reference Oracle
# ------------------------------------------------------------------------------

class TestDsReadI16Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_i16_sign_extension(self):
        # 0x7FFF = +32767
        self.lds.write_u16(0, 0x7FFF)
        self.assertEqual(self.lds.read_i16(0), 32767)
        # 0x8000 = -32768 -> 0xFFFF8000
        self.lds.write_u16(2, 0x8000)
        self.assertEqual(self.lds.read_i16(2), 0xFFFF_8000)
        # 0xFFFF = -1 -> 0xFFFFFFFF
        self.lds.write_u16(4, 0xFFFF)
        self.assertEqual(self.lds.read_i16(4), 0xFFFF_FFFF)


# ------------------------------------------------------------------------------
# ds_read_i16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class DSReadI16Instruction:
    dst: RegisterOperand
    addr: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR or self.addr.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_read_i16 must be VGPRs")


class TestDsReadI16ASTAndScalar(unittest.TestCase):
    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.lds.write_u16(0, 0xFFFE)  # -2
        machine.vgprs[1] = 0
        addr = machine.vgprs[1]
        machine.vgprs[2] = machine.lds.read_i16(addr)
        self.assertEqual(machine.vgprs[2], 0xFFFF_FFFE)


# ------------------------------------------------------------------------------
# ds_read_i16 SIMD Divergence & Sign Extension
# ------------------------------------------------------------------------------

class TestDsReadI16SIMD(unittest.TestCase):
    def test_simd_divergent_signed_halfwords(self):
        wave = SIMDWaveDS(wave_size=32)
        for i in range(32):
            val = 0x8000 + i * 10
            wave.lds.write_u16(i * 2, val)
            wave.vgprs[1][i] = i * 2

        for i in range(32):
            wave.vgprs[2][i] = wave.lds.read_i16(wave.vgprs[1][i])
            expected = (0x8000 + i * 10 - 65536) & MASK32
            self.assertEqual(wave.vgprs[2][i], expected)


# ------------------------------------------------------------------------------
# ds_read_i16 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestDsReadI16HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xff00",  # -256 signed 16-bit
            "ds_write_b16 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0xFFFF_FF00)


# ------------------------------------------------------------------------------
# ds_write_b8 Reference Oracle
# ------------------------------------------------------------------------------

class TestDsWriteB8Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_byte_write_preserves_adjacent_bytes(self):
        # Write 0x11223344 as dword
        self.lds.write_u8(0, 0x44)
        self.lds.write_u8(1, 0x33)
        self.lds.write_u8(2, 0x22)
        self.lds.write_u8(3, 0x11)

        # Overwrite only byte 1 with 0x99
        self.lds.write_u8(1, 0x99)
        self.assertEqual(self.lds.read_u8(0), 0x44)
        self.assertEqual(self.lds.read_u8(1), 0x99)
        self.assertEqual(self.lds.read_u8(2), 0x22)
        self.assertEqual(self.lds.read_u8(3), 0x11)


# ------------------------------------------------------------------------------
# ds_write_b8 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class DSWriteB8Instruction:
    addr: RegisterOperand
    data: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.addr.op_type != OperandType.VGPR or self.data.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_write_b8 must be VGPRs")


class TestDsWriteB8ASTAndScalar(unittest.TestCase):
    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.vgprs[1] = 10
        machine.vgprs[2] = 0xAA
        insn = DSWriteB8Instruction(addr=RegisterOperand(OperandType.VGPR, 1), data=RegisterOperand(OperandType.VGPR, 2))
        addr = machine.vgprs[insn.addr.index]
        data = machine.vgprs[insn.data.index]
        machine.lds.write_u8(addr, data, insn.offset)
        self.assertEqual(machine.lds.read_u8(10), 0xAA)


# ------------------------------------------------------------------------------
# ds_write_b8 SIMD Divergence & Byte Masking
# ------------------------------------------------------------------------------

class TestDsWriteB8SIMD(unittest.TestCase):
    def test_simd_independent_byte_stores(self):
        wave = SIMDWaveDS(wave_size=32)
        for i in range(32):
            wave.vgprs[1][i] = i * 4  # addr
            wave.vgprs[2][i] = 0x50 + i  # data

        for i in range(32):
            addr = wave.vgprs[1][i]
            val = wave.vgprs[2][i]
            wave.lds.write_u8(addr, val)

        for i in range(32):
            self.assertEqual(wave.lds.read_u8(i * 4), 0x50 + i)


# ------------------------------------------------------------------------------
# ds_write_b8 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestDsWriteB8HardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        body = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x5a",
            "ds_write_b8 v1, v2 offset:3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u8 v3, v1 offset:3",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 0x5A)


# ------------------------------------------------------------------------------
# ds_write_b16 Reference Oracle
# ------------------------------------------------------------------------------

class TestDsWriteB16Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_halfword_write_preserves_other_half(self):
        self.lds.write_u16(0, 0x1234)
        self.lds.write_u16(2, 0x5678)

        # Overwrite only low halfword
        self.lds.write_u16(0, 0x9ABC)
        self.assertEqual(self.lds.read_u16(0), 0x9ABC)
        self.assertEqual(self.lds.read_u16(2), 0x5678)


if __name__ == '__main__':
    unittest.main()

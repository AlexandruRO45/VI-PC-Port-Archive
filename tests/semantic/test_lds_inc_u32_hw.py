"""Semantic interpreter tests and translator verification for 32-bit unsigned LDS atomic increment with wrap (ds_inc_u32, ds_inc_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_inc_u32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_inc_u32_hw.py
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import lds_run as L
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
UINT32_MAX: int = 0xFFFFFFFF


def ref_ds_inc_u32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_inc_u32: M = (M >= DATA) ? 0 : M + 1 (unsigned)."""
    cur = curr_mem & MASK32
    data = data_val & MASK32
    return 0 if cur >= data else (cur + 1) & MASK32


def ref_ds_inc_rtn_u32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic increment with wrap: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_inc_u32(old, data_val)
    return old, new


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, size_bytes: int = 4096) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.storage = [0] * self.num_dwords

    def resolve_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        dword_idx = eff_addr >> 2
        return dword_idx & (self.num_dwords - 1)

    def read_dword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.resolve_dword_index(byte_addr, offset)
        return self.storage[idx]

    def write_dword(self, byte_addr: int, value: int, offset: int = 0) -> None:
        idx = self.resolve_dword_index(byte_addr, offset)
        self.storage[idx] = value & MASK32

    def atomic_inc_u32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_inc_u32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSIncU32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_reference_oracle_basic_increments(self) -> None:
        # Standard increment: cur < data
        self.assertEqual(ref_ds_inc_u32(0, 10), 1)
        self.assertEqual(ref_ds_inc_u32(1, 10), 2)
        self.assertEqual(ref_ds_inc_u32(8, 10), 9)

    def test_reference_oracle_wrap_resets(self) -> None:
        # Wrap when cur == data
        self.assertEqual(ref_ds_inc_u32(10, 10), 0)

        # Wrap when cur > data
        self.assertEqual(ref_ds_inc_u32(15, 10), 0)

        # When data == 0: cur >= 0 is always True (unsigned), so always resets to 0
        self.assertEqual(ref_ds_inc_u32(0, 0), 0)
        self.assertEqual(ref_ds_inc_u32(5, 0), 0)

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_inc_rtn_u32(3, 5)
        self.assertEqual(old, 3)
        self.assertEqual(new, 4)

        old, new = ref_ds_inc_rtn_u32(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, 4)

        # Non-returning atomic inc with data=10: 4 < 10 -> 5
        old, new = lds.atomic_inc_u32(addr, 10, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, 5)
        self.assertEqual(lds.read_dword(addr), 5)

        # Returning atomic inc with data=5: 5 >= 5 -> resets to 0
        old, new = lds.atomic_inc_u32(addr, 5, rtn=True)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)
        self.assertEqual(lds.read_dword(addr), 0)

class Register:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index

    def __repr__(self) -> str:
        return f"{self.name}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return self.name == other.name and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.name, self.index))


class VGPR(Register):
    def __init__(self, index: int) -> None:
        if not (0 <= index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {index}")
        super().__init__("v", index)


class DSIncU32Instruction:
    """Symbolic AST node representing ds_inc_u32 and ds_inc_rtn_u32 instructions."""

    def __init__(
        self,
        mnemonic: str,
        vaddr: VGPR,
        vdata0: VGPR,
        vdst: Optional[VGPR] = None,
        offset: int = 0,
        gds: bool = False,
    ) -> None:
        if mnemonic not in ("ds_inc_u32", "ds_inc_rtn_u32"):
            raise ValueError(f"Invalid mnemonic for ds_inc_u32 AST: {mnemonic}")
        if mnemonic == "ds_inc_rtn_u32" and vdst is None:
            raise ValueError("ds_inc_rtn_u32 requires vdst operand")
        if mnemonic == "ds_inc_u32" and vdst is not None:
            raise ValueError("ds_inc_u32 does not accept vdst operand")
        if not (0 <= offset <= 65535):
            raise ValueError(f"Offset out of range [0, 65535]: {offset}")

        self.mnemonic = mnemonic
        self.vaddr = vaddr
        self.vdata0 = vdata0
        self.vdst = vdst
        self.offset = offset
        self.gds = gds

    @property
    def is_return(self) -> bool:
        return self.mnemonic == "ds_inc_rtn_u32"

    def __repr__(self) -> str:
        parts = [self.mnemonic]
        if self.vdst is not None:
            parts.append(str(self.vdst))
        parts.append(str(self.vaddr))
        parts.append(str(self.vdata0))
        text = f"{parts[0]} {', '.join(parts[1:])}"
        if self.offset:
            text += f" offset:{self.offset}"
        if self.gds:
            text += " gds"
        return text

    @classmethod
    def parse(cls, asm: str) -> "DSIncU32Instruction":
        line = asm.strip()
        if not line:
            raise ValueError("Empty assembly string")
        tokens = line.split()
        mnem = tokens[0]
        if mnem not in ("ds_inc_u32", "ds_inc_rtn_u32"):
            raise ValueError(f"Unsupported mnemonic: {mnem}")

        rest = " ".join(tokens[1:])
        mod_tokens = rest.split()
        clean_operands: list[str] = []
        offset = 0
        gds = False

        for tok in mod_tokens:
            tok_clean = tok.rstrip(",")
            if tok_clean.startswith("offset:"):
                try:
                    offset = int(tok_clean.split(":", 1)[1], 0)
                except ValueError:
                    raise ValueError(f"Invalid offset format: {tok_clean}")
            elif tok_clean == "gds":
                gds = True
            elif tok_clean:
                clean_operands.append(tok_clean)

        def parse_vgpr(reg_str: str) -> VGPR:
            m = re.fullmatch(r"v(\d+)", reg_str)
            if not m:
                raise ValueError(f"Expected VGPR (e.g. v1), got: {reg_str}")
            return VGPR(int(m.group(1)))

        if mnem == "ds_inc_u32":
            if len(clean_operands) != 2:
                raise ValueError(f"ds_inc_u32 requires 2 operands (vaddr, vdata0), got {len(clean_operands)}")
            vaddr = parse_vgpr(clean_operands[0])
            vdata0 = parse_vgpr(clean_operands[1])
            return cls(mnem, vaddr, vdata0, vdst=None, offset=offset, gds=gds)
        else:
            if len(clean_operands) != 3:
                raise ValueError(f"ds_inc_rtn_u32 requires 3 operands (vdst, vaddr, vdata0), got {len(clean_operands)}")
            vdst = parse_vgpr(clean_operands[0])
            vaddr = parse_vgpr(clean_operands[1])
            vdata0 = parse_vgpr(clean_operands[2])
            return cls(mnem, vaddr, vdata0, vdst=vdst, offset=offset, gds=gds)


class TestDSIncU32ASTAndOpcodeDefinitions(unittest.TestCase):
    """Part 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_properties(self) -> None:
        inst_non_rtn = DSIncU32Instruction("ds_inc_u32", VGPR(1), VGPR(2), offset=16)
        self.assertFalse(inst_non_rtn.is_return)
        self.assertEqual(inst_non_rtn.vaddr, VGPR(1))
        self.assertEqual(inst_non_rtn.vdata0, VGPR(2))
        self.assertIsNone(inst_non_rtn.vdst)
        self.assertEqual(inst_non_rtn.offset, 16)
        self.assertEqual(repr(inst_non_rtn), "ds_inc_u32 v1, v2 offset:16")

        inst_rtn = DSIncU32Instruction("ds_inc_rtn_u32", VGPR(3), VGPR(4), vdst=VGPR(5), offset=32, gds=True)
        self.assertTrue(inst_rtn.is_return)
        self.assertEqual(inst_rtn.vdst, VGPR(5))
        self.assertEqual(repr(inst_rtn), "ds_inc_rtn_u32 v5, v3, v4 offset:32 gds")

    def test_ast_parsing_valid(self) -> None:
        inst1 = DSIncU32Instruction.parse("ds_inc_u32 v10, v20")
        self.assertEqual(inst1.mnemonic, "ds_inc_u32")
        self.assertEqual(inst1.vaddr, VGPR(10))
        self.assertEqual(inst1.vdata0, VGPR(20))

        inst2 = DSIncU32Instruction.parse("ds_inc_rtn_u32 v5, v1, v2 offset:0x40")
        self.assertEqual(inst2.mnemonic, "ds_inc_rtn_u32")
        self.assertEqual(inst2.vdst, VGPR(5))
        self.assertEqual(inst2.offset, 64)

    def test_ast_parsing_invalid(self) -> None:
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_u32 v1")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_rtn_u32 v1, v2")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_u32 s1, v2")


if __name__ == '__main__':
    unittest.main()

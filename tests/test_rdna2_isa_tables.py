"""Schema and decoder-contract invariant tests for tools/rdna2_isa_tables.py.

Pins the structural contract between the 20 RDNA2 ISA opcode tables in tools/rdna2_isa_tables.py and their consumer
tools/rdna2_decode.py without hardcoding table lengths or specific legacy/GFX9 opcode membership (so additive PRs
and dead-opcode cleanups stay green):
  - Exact 20-table bijection with every T.<TABLE> reference in tools/rdna2_decode.py.
  - Zero duplicate literal dict keys in the AST of tools/rdna2_isa_tables.py (which Python would silently overwrite).
  - Every opcode key is a plain int within its RDNA2 hardware encoding bit-width.
  - Every entry is (mnemonic, operands) with a valid lowercase identifier, per-encoding mnemonic prefix, no duplicate
    mnemonics within a table, and well-typed (name, width, kind) operand 3-tuples.
  - VOP3 and VOP3B are strictly disjoint, and every VOP3B entry declares an SDST operand.
  - Shared e32/VOP3 opcodes (VOPC < 0x100, VOP2 + 0x100, VOP1 + 0x180) agree on their mnemonic.

Standard library only; no GPU, no LLVM, no game files.
"""
import ast
import os
import re
import sys
import unittest

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2_isa_tables as T  # noqa: E402

TABLES_PATH = os.path.join(TOOLS_DIR, "rdna2_isa_tables.py")
DECODE_PATH = os.path.join(TOOLS_DIR, "rdna2_decode.py")

# RDNA2 hardware opcode bitfield width per encoding table.
ENCODING_OPCODE_BITS = {
    "VINTRP": 2,
    "MTBUF": 4,
    "SOPK": 5,
    "VOP2": 6,
    "SOP2": 7,
    "SOPC": 7,
    "SOPP": 7,
    "VOP3P": 7,
    "FLAT": 7,
    "GLOBAL": 7,
    "SCRATCH": 7,
    "SOP1": 8,
    "VOP1": 8,
    "VOPC": 8,
    "DS": 8,
    "MIMG": 8,
    "MUBUF": 8,
    "SMEM": 8,
    "VOP3": 10,
    "VOP3B": 10,
}

ENCODING_MNEMONIC_PREFIX = {
    "DS": "ds_",
    "FLAT": "flat_",
    "GLOBAL": "global_",
    "SCRATCH": "scratch_",
    "MIMG": "image_",
    "MTBUF": "tbuffer_",
    "MUBUF": "buffer_",
    "SMEM": "s_",
    "SOP1": "s_",
    "SOP2": "s_",
    "SOPC": "s_",
    "SOPK": "s_",
    "SOPP": "s_",
    "VINTRP": "v_",
    "VOP1": "v_",
    "VOP2": "v_",
    "VOP3": "v_",
    "VOP3B": "v_",
    "VOP3P": "v_",
    "VOPC": "v_",
}

VALID_OPERAND_NAMES = frozenset({
    "ADDR", "ATTR", "DATA", "DATA0", "DATA1", "LITERAL", "OPR_EXEC",
    "SADDR", "SBASE", "SDATA", "SDST", "SIMM16", "SIMM32", "SOFFSET",
    "SRC0", "SRC1", "SRC2", "SRSRC", "SSAMP", "SSRC0", "SSRC1",
    "VADDR", "VCC", "VDATA", "VDST", "VSRC", "VSRC1",
})

VALID_OPERAND_KINDS = frozenset({"i", "f16", "f32", "f64"})
MNEMONIC_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_plain_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


class Rdna2IsaTablesTests(unittest.TestCase):
    def test_exported_tables_match_decoder_references(self):
        exported = {k for k in vars(T) if not k.startswith("_")}
        self.assertEqual(exported, set(ENCODING_OPCODE_BITS))
        with open(DECODE_PATH, encoding="utf-8") as f:
            decode_src = f.read()
        referenced = set(re.findall(r"\bT\.([A-Za-z0-9_]+)\b", decode_src))
        self.assertEqual(referenced, exported)
        for name in sorted(exported):
            tbl = getattr(T, name)
            self.assertIsInstance(tbl, dict, name)
            self.assertGreater(len(tbl), 0, f"{name} is empty")

    def test_source_ast_has_no_duplicate_literal_dict_keys(self):
        with open(TABLES_PATH, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=TABLES_PATH)
        found_tables = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                tname = node.targets[0].id
                if isinstance(node.value, ast.Dict):
                    found_tables.add(tname)
                    seen = set()
                    for key_node in node.value.keys:
                        self.assertIsNotNone(key_node, f"{tname} uses dict unpacking")
                        key_val = ast.literal_eval(key_node)
                        self.assertNotIn(
                            key_val,
                            seen,
                            f"{tname} has duplicate literal opcode key {key_val:#x}",
                        )
                        seen.add(key_val)
        self.assertEqual(found_tables, set(ENCODING_OPCODE_BITS))

    def test_opcode_keys_within_hardware_encoding_bit_widths(self):
        for tname, bits in sorted(ENCODING_OPCODE_BITS.items()):
            max_opc = (1 << bits) - 1
            tbl = getattr(T, tname)
            with self.subTest(table=tname, bits=bits):
                for opc in tbl:
                    self.assertTrue(is_plain_int(opc), f"{tname}: non-int key {opc!r}")
                    self.assertTrue(
                        0 <= opc <= max_opc,
                        f"{tname}: opcode {opc:#x} exceeds {bits}-bit range 0..{max_opc:#x}",
                    )

    def test_entry_schemas_prefixes_and_unique_mnemonics(self):
        for tname, prefix in sorted(ENCODING_MNEMONIC_PREFIX.items()):
            tbl = getattr(T, tname)
            seen_mnemonics = {}
            with self.subTest(table=tname):
                for opc, entry in tbl.items():
                    self.assertIsInstance(entry, tuple, f"{tname}[{opc:#x}] is not a tuple")
                    self.assertEqual(len(entry), 2, f"{tname}[{opc:#x}] expected 2-tuple")
                    mnem, operands = entry

                    self.assertIsInstance(mnem, str, f"{tname}[{opc:#x}] mnemonic is not str")
                    self.assertRegex(mnem, MNEMONIC_RE, f"{tname}[{opc:#x}] invalid mnemonic {mnem!r}")
                    self.assertTrue(
                        mnem.startswith(prefix),
                        f"{tname}[{opc:#x}] mnemonic {mnem!r} does not start with {prefix!r}",
                    )
                    self.assertNotIn(
                        mnem,
                        seen_mnemonics,
                        f"{tname}: duplicate mnemonic {mnem!r} at {seen_mnemonics.get(mnem)} and {opc:#x}",
                    )
                    seen_mnemonics[mnem] = hex(opc)

                    self.assertIsInstance(operands, tuple, f"{tname}[{opc:#x}] operands is not a tuple")
                    op_counts = {}
                    for idx, spec in enumerate(operands):
                        self.assertIsInstance(spec, tuple, f"{tname}[{opc:#x}] operand {idx} is not a tuple")
                        self.assertEqual(len(spec), 3, f"{tname}[{opc:#x}] operand {idx} expected 3-tuple")
                        op_name, width, kind = spec
                        self.assertIn(op_name, VALID_OPERAND_NAMES, f"{tname}[{opc:#x}] unknown operand {op_name!r}")
                        self.assertTrue(
                            is_plain_int(width) and (width == -1 or 0 <= width <= 16),
                            f"{tname}[{opc:#x}] operand {op_name} invalid width {width!r}",
                        )
                        self.assertIn(kind, VALID_OPERAND_KINDS, f"{tname}[{opc:#x}] unknown kind {kind!r}")
                        op_counts[op_name] = op_counts.get(op_name, 0) + 1
                    # Only implicit VCC may appear twice (carry-out + carry-in on VOP2 *_co_ci_*).
                    for op_name, count in op_counts.items():
                        if count > 1:
                            self.assertTrue(
                                op_name == "VCC" and count == 2 and tname == "VOP2" and "_co_ci_" in mnem,
                                f"{tname}[{opc:#x}] ({mnem}) repeats operand {op_name!r} {count} times",
                            )

    def test_vop3_and_vop3b_disjoint_and_vop3b_has_sdst(self):
        overlap = set(T.VOP3) & set(T.VOP3B)
        self.assertEqual(overlap, set(), f"VOP3/VOP3B overlap: {[hex(o) for o in sorted(overlap)]}")
        for opc, (mnem, operands) in T.VOP3B.items():
            op_names = {spec[0] for spec in operands}
            self.assertIn("SDST", op_names, f"VOP3B[{opc:#x}] ({mnem}) missing SDST operand")

    def test_vop3_e32_promotion_mnemonics_agree(self):
        # tools/rdna2_decode.py maps VOPC (0x000..0x0FF), VOP2 (0x100..0x13F), and VOP1 (0x180..0x1FF)
        # into VOP3/VOP3B for _e32/_e64 suffix formatting; shared opcodes must agree on the mnemonic.
        for opc, (mnem, _) in T.VOPC.items():
            if opc in T.VOP3:
                self.assertEqual(
                    T.VOP3[opc][0],
                    mnem,
                    f"VOPC[{opc:#x}] ({mnem}) != VOP3[{opc:#x}] ({T.VOP3[opc][0]})",
                )
        for opc, (mnem, _) in T.VOP2.items():
            v3_opc = opc + 0x100
            v3 = T.VOP3B.get(v3_opc) or T.VOP3.get(v3_opc)
            if v3 is not None:
                self.assertEqual(
                    v3[0],
                    mnem,
                    f"VOP2[{opc:#x}] ({mnem}) != VOP3/B[{v3_opc:#x}] ({v3[0]})",
                )
        for opc, (mnem, _) in T.VOP1.items():
            v3_opc = opc + 0x180
            if v3_opc in T.VOP3:
                self.assertEqual(
                    T.VOP3[v3_opc][0],
                    mnem,
                    f"VOP1[{opc:#x}] ({mnem}) != VOP3[{v3_opc:#x}] ({T.VOP3[v3_opc][0]})",
                )


if __name__ == "__main__":
    unittest.main()

"""tools/rdna2spirv.py lower_addr64: 64-bit buffer addresses as 32-bit pairs, without the Int64 capability.

Every PhysicalStorageBuffer address the translator builds is one idiom (widen, shift the high word by 32, OR, add,
convert to a pointer). A shader whose only 64-bit integers are that idiom is rewritten to (low, high) word pairs
with OpIAddCarry carrying into the high word and a uvec2 bitcast to the pointer, and drops `OpCapability Int64`
(Vulkan shaderInt64, which Intel's drivers lack: #313). A shader with any other 64-bit integer use keeps Int64.

Each program is translated twice, with the lowering (the default) and with --int64-addresses, and both run in
tests/semantic/spirv_mini_interp.py on the same memory: the loaded word must land at the same address, including
when the low words carry and when the 64-bit sum wraps. A negative control drops the carry from the lowered module
and checks that the carry case then reads the wrong word, so the test is sensitive to the lowering it checks.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
THREADS = 32
# Push constants: s[0:3] = the output V# (base 0, NUM_RECORDS all ones), s4..s31 = 0, dwords 32..33 = heap base.
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
MAGIC = 0x5EED1234


def translate(name, body_lines, lowered=True):
    """Translate a hand-written wave32 compute program; `lowered` False keeps Int64 (--int64-addresses)."""
    saved = rdna2spirv.ADDR_PAIR
    rdna2spirv.ADDR_PAIR = lowered
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body_lines) + "\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2 0x0088: 4 user SGPRs (s[0:3] = the output V#), TGID_X; 32x1x1 threads, wave32.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x0088, THREADS, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)
    finally:
        rdna2spirv.ADDR_PAIR = saved


def run(spv, heap, memory, descriptor=VSHARP):
    """Run every lane once; the lanes all store the same word at heap + 0. Returns the memory afterwards."""
    pc = descriptor + [0] * 28 + [heap & M32, (heap >> 32) & M32]
    return si.run_workgroup(spv, THREADS, THREADS,
                            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                            push_constants=pc, memory=dict(memory))


def load_program(base):
    """s_load_dword from the 64-bit pointer `base`, stored through the output V# at offset 0."""
    return [f"s_mov_b32 s6, {base & M32:#x}", f"s_mov_b32 s7, {(base >> 32) & M32:#x}",
            "s_load_dword s8, s[6:7], 0x0", "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v1, s8", "buffer_store_dword v1, off, s[0:3], 0", "s_waitcnt vmcnt(0)"]


# (heap base, pointer): plain; the low words carry into the high word; the 64-bit sum wraps to zero.
CASES = [(0x1000, 0x2000), (0x1_FFFF_FFF0, 0x20), (0xFFFF_FFFF_FFFF_FFF0, 0x10)]


def global_program(base, offset):
    """global_load_dword from the 64-bit VGPR address `base` with the signed instruction offset, stored at heap + 0.
    The offset is sign-extended through a 64-bit signed integer (Gen.global_address), the one signed form the
    lowering handles."""
    return [f"v_mov_b32 v2, {base & M32:#x}", f"v_mov_b32 v3, {(base >> 32) & M32:#x}",
            f"global_load_dword v4, v[2:3], off offset:{offset}", "s_waitcnt vmcnt(0)",
            "buffer_store_dword v4, off, s[0:3], 0", "s_waitcnt vmcnt(0)"]


# (heap base, VGPR address, signed offset): a negative offset across the low-word boundary, and a positive one.
GLOBAL_CASES = [(0x1000, 0x1_0000_0000, -4), (0x1000, 0x2000, -2048), (0x1_FFFF_FFF0, 0x20, 2044)]


class AddressLoweringTests(unittest.TestCase):
    def test_lowered_module_has_no_int64(self):
        spv = translate("cs_a64_plain", load_program(0x2000))
        self.assertNotIn("OpCapability Int64", spv)
        self.assertNotIn("%ulong", spv)
        self.assertNotIn("%long", spv)
        self.assertNotIn("OpConvertUToPtr", spv)
        self.assertIn("OpIAddCarry %umul", spv)
        self.assertIn("OpBitcast %_ptr_PSB_arr", spv)
        self.assertIn("OpCapability PhysicalStorageBufferAddresses", spv)
        self.assertIn("OpMemoryModel PhysicalStorageBuffer64", spv)

    def test_int64_addresses_flag_keeps_the_old_form(self):
        spv = translate("cs_a64_flag", load_program(0x2000), lowered=False)
        self.assertIn("OpCapability Int64", spv)
        self.assertIn("%ulong = OpTypeInt 64 0", spv)
        self.assertIn("OpConvertUToPtr", spv)
        self.assertNotIn("OpIAddCarry %umul", spv)

    def test_same_address_as_the_int64_form(self):
        for heap, base in CASES:
            with self.subTest(heap=hex(heap), base=hex(base)):
                addr = (heap + base) & M64
                memory = {addr: MAGIC, (addr + 4) & M64: 0x0BAD0004, (addr - 4) & M64: 0x0BAD0000}
                want = run(translate("cs_a64_ref", load_program(base), lowered=False), heap, memory)
                got = run(translate("cs_a64_low", load_program(base)), heap, memory)
                self.assertEqual(want[heap & M64], MAGIC)     # the reference read the word at heap + base
                self.assertEqual(got, want)

    def test_signed_global_offset_same_address_as_the_int64_form(self):
        for heap, base, offset in GLOBAL_CASES:
            with self.subTest(heap=hex(heap), base=hex(base), offset=offset):
                addr = (heap + base + offset) & M64
                memory = {addr: MAGIC, (addr + 4) & M64: 0x0BAD0004, (addr - 4) & M64: 0x0BAD0000}
                lowered = translate("cs_a64_glo", global_program(base, offset))
                self.assertNotIn("OpCapability Int64", lowered)       # the sign-extension is lowered too
                self.assertNotIn("%long", lowered)
                want = run(translate("cs_a64_glo_ref", global_program(base, offset), lowered=False), heap, memory)
                got = run(lowered, heap, memory)
                self.assertEqual(want[heap & M64], MAGIC)
                self.assertEqual(got, want)

    def test_dynamic_scalar_buffer_bounds(self):
        # Offset 8 ends at byte 12. The large structured limit is 2**32 + 8:
        # discarding its high word would incorrectly reject the load.
        body = ["s_buffer_load_dword s8, s[0:3], 8", "s_waitcnt lgkmcnt(0)",
                "s_mov_b32 s1, 0", "s_mov_b32 s2, -1",
                "v_mov_b32 v1, s8", "buffer_store_dword v1, off, s[0:3], 0"]
        native = translate("cs_a64_bounds", body, lowered=False)
        lowered = rdna2spirv.lower_addr64(native)
        self.assertNotIn("OpCapability Int64", lowered)
        self.assertIn("OpUMulExtended %umul", lowered)
        for stride, records, expected in [(0, 11, 0), (0, 12, MAGIC), (4, 2, 0), (4, 3, MAGIC),
                                          (4, 0x40000000, MAGIC), (4, 0x40000002, MAGIC)]:
            desc = [0, stride << 16, records, VSHARP[3]]
            for module in (native, lowered):
                with self.subTest(stride=stride, records=records, lowered=module == lowered):
                    self.assertEqual(run(module, 0x1000, {0x1008: MAGIC}, desc)[0x1000], expected)

        # With the structured limit on the true arm, its high word must survive there too.
        select = next(line for line in native.splitlines() if "OpSelect %ulong" in line)
        prefix, cond, raw, structured = select.rsplit(" ", 3)
        compare = next(line for line in native.splitlines() if line.strip().startswith(cond + " ="))
        reversed_arms = native.replace(compare, compare.replace("OpIEqual", "OpINotEqual"))
        reversed_arms = reversed_arms.replace(select, f"{prefix} {cond} {structured} {raw}")
        reversed_lowered = rdna2spirv.lower_addr64(reversed_arms)
        self.assertNotIn("OpCapability Int64", reversed_lowered)
        self.assertEqual(run(reversed_lowered, 0x1000, {0x1008: MAGIC}, [0, 4 << 16, 0x40000002, VSHARP[3]])[0x1000],
                         MAGIC)

        # A full-width operand requires cross products. Keep the entire module intact.
        product = next(line for line in native.splitlines() if "OpIMul %ulong" in line)
        address = next(line.split()[0] for line in native.splitlines() if "OpBitwiseOr %ulong" in line)
        unsupported = native.replace(product, product.rsplit(" ", 1)[0] + " " + address)
        self.assertEqual(rdna2spirv.lower_addr64(unsupported), unsupported)

    def test_dropping_the_carry_is_caught(self):
        heap, base = CASES[1]
        addr = (heap + base) & M64
        spv = translate("cs_a64_nocarry", load_program(base))
        carries = [ln for ln in spv.split("\n") if ln.strip().endswith("s 1") and "OpCompositeExtract %uint %a64_" in ln]
        self.assertTrue(carries, "no carry extract in the lowered module")
        broken = spv
        for ln in carries:
            res = ln.split()[0]
            broken = broken.replace(ln, f"         {res} = OpCopyObject %uint %a64c_0")
        got = run(broken, heap, {addr: MAGIC})
        self.assertNotEqual(got.get(heap & M64), MAGIC)

    def test_a_64bit_data_op_keeps_int64(self):
        prog = ["s_mov_b32 s6, 0x2000", "s_mov_b32 s7, 0", "s_lshl_b64 s[6:7], s[6:7], 4"] + load_program(0)[2:]
        spv = translate("cs_a64_data", prog)
        self.assertIn("OpCapability Int64", spv)
        self.assertIn("%ulong = OpTypeInt 64 0", spv)
        self.assertNotIn("%a64", spv)
        # and it still runs: the shift makes the pointer 0x20000
        memory = {0x1000 + 0x20000: MAGIC}
        got = run(spv, 0x1000, memory)
        self.assertEqual(got[0x1000], MAGIC)

    def test_lowering_is_idempotent_and_valid_without_psb(self):
        spv = translate("cs_a64_idem", load_program(0x2000))
        self.assertEqual(rdna2spirv.lower_addr64(spv), spv)
        plain = ["v_mov_b32 v1, 7", "v_add_nc_u32 v1, v1, v1"]
        text = "\n".join(["OpCapability Shader", "%uint = OpTypeInt 32 0"])
        self.assertEqual(rdna2spirv.lower_addr64(text), text)       # no 64-bit types: untouched
        self.assertNotIn("%ulong", translate("cs_a64_noload", plain + load_program(0)[4:]))


if __name__ == "__main__":
    unittest.main()

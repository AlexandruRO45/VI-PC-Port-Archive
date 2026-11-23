"""Translator tests for the tfe/lwe status VGPR of image_atomic_* and image_get_resinfo (the wavehs finding,
https://github.com/ItsNotRexed/Leonida/issues/8683#issuecomment-7111622270).

With `tfe` or `lwe`, an MIMG atomic or resinfo carries one extra 32-bit status dword after its data words, at
vdata[data_count]: the RDNA2 ISA names the bits "Texture Fail Enable (for partially resident textures)" and
"LOD Warning Enable (for partially resident textures)" (AMD's machine-readable RDNA2 ISA, TFE/LWE fields of
ENC_MIMG), AMD's ISA guides place the word at "DST+1 (first GPR after all fetch-dest GPRs)", and LLVM's
image-intrinsic lowering adds the same dword for every image op, atomics included ("NumVDataDwords += 1" with
texfail set, SIISelLowering.cpp). The translator writes 0: the status is non-zero only on a failed fetch of a
partially resident texture or a clamped LOD, and everything the runtime binds is a whole image or the index-0
fallback, so no fetch fails (resinfo reads no memory at all).

Every case puts a marker (0xdeadbeef) in the would-be status register, runs the instruction, and checks that
the register's last written value is the shared zero constant %c0 — and that the compare/sentinel operands of
cmpswap/fcmpswap and the high dword of the two-VGPR dmask:0x3 form are left intact. The no-tfe/no-lwe controls
assert the marker survives (no new store), so the plain forms' output is unchanged. Representative modules are
checked with spirv-as/spirv-val when those tools are on PATH. Standard library only; no GPU, no LLVM, no game
files."""
import os
import re
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, 32x1x1
FSSM = bytes(FSSM)
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")
MARKER = 0xDEADBEEF

STANDARD_ATOMICS = ("image_atomic_add", "image_atomic_sub", "image_atomic_and", "image_atomic_or",
                    "image_atomic_xor", "image_atomic_smin", "image_atomic_smax",
                    "image_atomic_umin", "image_atomic_umax")
# hardcoded (not read from the translator's table) so a mutant table entry fails the test
SPV_ATOMIC_OPS = {"image_atomic_add": "OpAtomicIAdd", "image_atomic_sub": "OpAtomicISub",
                  "image_atomic_and": "OpAtomicAnd", "image_atomic_or": "OpAtomicOr",
                  "image_atomic_xor": "OpAtomicXor", "image_atomic_smin": "OpAtomicSMin",
                  "image_atomic_smax": "OpAtomicSMax", "image_atomic_umin": "OpAtomicUMin",
                  "image_atomic_umax": "OpAtomicUMax"}
EXTRA_ATOMICS = ("image_atomic_swap", "image_atomic_inc", "image_atomic_dec",
                 "image_atomic_fmin", "image_atomic_fmax")


def translate_cs(body, name="cs_img_status"):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"{name}.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def check_cfg(spv):
    """Parse the SPIR-V function body and validate its structured CFG with spirv_mini_interp.check_structure."""
    body = []
    in_body = False
    for raw in spv.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = si._LINE.match(line)
        assert m, f"unparseable SPIR-V line: {line!r}"
        res, op, rest = m.groups()
        if in_body:
            body.append((res, op, rest.split(), line))
        elif op == "OpFunction":
            in_body = True
    assert body, "empty SPIR-V function body"
    si.check_structure(body)


def writes_to(spv, vgpr):
    """The ordered ids stored into VGPR `vgpr` (both the %reg_vN pointer and a fresh %vgpr AccessChain)."""
    ids, ptr = [], None
    for raw in spv.splitlines():
        s = raw.strip()
        if re.match(rf"OpStore %reg_v{vgpr}\b", s):
            ids.append(s.split()[2])
            continue
        m = re.match(r"(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c" + str(vgpr) + r"\b", s)
        if m:
            ptr = m.group(1)
        elif ptr and s.startswith(f"OpStore {ptr} "):
            ids.append(s.split()[2])
            ptr = None
    return ids


def validate_spirv(spv):
    if not SPIRV_AS or not SPIRV_VAL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv_bin = os.path.join(tmp, "img_status.spv")
        p1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", spv_bin],
                            input=spv.encode("utf-8"), capture_output=True)
        assert p1.returncode == 0, p1.stderr.decode("utf-8", "replace")
        p2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_bin], capture_output=True)
        assert p2.returncode == 0, p2.stderr.decode("utf-8", "replace")


def atomic_body(op, status_reg, glc, extra_mods=(), dim="SQ_RSRC_IMG_2D", addr="v[4:5]"):
    """A compute program whose only real work is one image atomic. The marker sits in the status VGPR."""
    lines = [
        "v_mov_b32_e32 v0, 7",                       # the value operand
        f"v_mov_b32_e32 v{status_reg}, {MARKER:#x}",
        "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
        f"{op} v[0:{status_reg}], {addr}, s[4:11] dmask:0x1 dim:{dim} unorm{' glc' if glc else ''}"
        f"{' ' + ' '.join(extra_mods) if extra_mods else ''}",
        "s_waitcnt vmcnt(0)",
    ]
    return lines


class ImageStatusVGPRTests(unittest.TestCase):
    def assert_status_written(self, spv, status_reg, atomic_op):
        """The last value stored into the status VGPR is the shared zero constant, after the atomic."""
        ids = writes_to(spv, status_reg)
        self.assertEqual(ids[-1], "%c0", f"status VGPR v{status_reg} not last written with %c0: {ids}")
        after = spv[spv.index(atomic_op):]
        self.assertIn(f"OpStore %reg_v{status_reg} %c0", after)

    def test_standard_atomics_tfe_glc_and_non_glc(self):
        for op in STANDARD_ATOMICS:
            for glc in (False, True):
                with self.subTest(op=op, glc=glc):
                    spv = translate_cs(atomic_body(op, 1, glc, ("tfe",)))
                    check_cfg(spv)
                    self.assertIn(SPV_ATOMIC_OPS[op], spv)   # the exact SPIR-V atomic opcode
                    self.assert_status_written(spv, 1, "OpAtomic")
                    # the value operand register: with glc its last write is the atomic's returned old value,
                    # without glc the marker setup survives (the compute prologue's local-id seed aside)
                    v0 = writes_to(spv, 0)
                    atomic_res = re.search(r"(%\w+) = OpAtomic\w+ ", spv).group(1)
                    self.assertEqual(v0[-1] == atomic_res, glc)
                    validate_spirv(spv)

    def test_standard_atomics_lwe(self):
        for op in ("image_atomic_add", "image_atomic_umax"):
            with self.subTest(op=op):
                spv = translate_cs(atomic_body(op, 1, False, ("lwe",)))
                check_cfg(spv)
                self.assert_status_written(spv, 1, "OpAtomic")

    def test_standard_atomics_no_status_control(self):
        for op in ("image_atomic_add", "image_atomic_umax"):
            for mod in ("", "glc"):
                with self.subTest(op=op, mod=mod):
                    body = atomic_body(op, 1, bool(mod))
                    body[4] = body[4].replace(" tfe", "").replace(" glc", f" {mod}" if mod else "")
                    spv = translate_cs(body)
                    check_cfg(spv)
                    # the marker survives: nothing stored into the would-be status register
                    self.assertEqual(writes_to(spv, 1)[-1], f"%c{MARKER:x}")

    def test_atomic_tfe_3d(self):
        spv = translate_cs(atomic_body("image_atomic_add", 1, True, ("tfe",),
                                       dim="SQ_RSRC_IMG_3D", addr="v[4:6]"))
        check_cfg(spv)
        self.assertIn("OpCompositeConstruct %v3int", spv)
        self.assert_status_written(spv, 1, "OpAtomic")

    def test_atomic_dmask03_two_vgpr_form_status_after_the_high_dword(self):
        # dmask:0x3 sizes the data at two dwords (the form kept low-dword 32-bit): the status is the third VGPR,
        # and the high dword (v1) is neither read as an operand here nor written.
        spv = translate_cs([
            "v_mov_b32_e32 v0, 7",
            "v_mov_b32_e32 v1, 0x11111111",
            "v_mov_b32_e32 v2, 0x22222222",
            "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
            "image_atomic_add v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc tfe",
        ])
        check_cfg(spv)
        self.assertEqual(writes_to(spv, 2)[-1], "%c0")
        self.assertIn("OpStore %reg_v2 %c0", spv[spv.index("OpAtomic"):])
        # the high dword keeps its marker
        self.assertEqual(writes_to(spv, 1)[-1], "%c11111111")

    def test_cmpswap_tfe_status_and_operands_intact(self):
        spv = translate_cs([
            "v_mov_b32_e32 v0, 7",                      # new value
            "v_mov_b32_e32 v1, 9",                      # comparand (sentinel)
            "v_mov_b32_e32 v2, 0x33333333",             # the status dword's slot
            "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
            "image_atomic_cmpswap v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm tfe",
        ])
        check_cfg(spv)
        self.assertIn("OpAtomicCompareExchange %uint", spv)
        # comparand register: the setup store only, and it is the OpAtomicCompareExchange comparand operand
        self.assertEqual(writes_to(spv, 1), ["%c9"])
        ce = re.search(r"OpAtomicCompareExchange %uint (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)", spv)
        self.assertIsNotNone(ce)
        # new value loaded from v0, comparand loaded from the sentinel register v1
        self.assertIsNotNone(re.search(rf"{ce.group(5)} = OpLoad %uint %reg_v0\b", spv))
        self.assertIsNotNone(re.search(rf"{ce.group(6)} = OpLoad %uint %reg_v1\b", spv))
        self.assertEqual(writes_to(spv, 2)[-1], "%c0")
        validate_spirv(spv)

    def test_cmpswap_glc_tfe_returns_old_value_to_the_data_slot(self):
        spv = translate_cs([
            "v_mov_b32_e32 v0, 7", "v_mov_b32_e32 v1, 9",
            "v_mov_b32_e32 v2, 0x33333333",
            "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
            "image_atomic_cmpswap v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc tfe",
        ])
        check_cfg(spv)
        # glc writes the old value into v0 (the first data word), the status is still v2
        self.assertIn("OpStore %reg_v0 ", spv[spv.index("OpAtomicCompareExchange"):])
        self.assertEqual(writes_to(spv, 1), ["%c9"])
        self.assertEqual(writes_to(spv, 2)[-1], "%c0")

    def test_extra_atomics_tfe_status(self):
        for op in EXTRA_ATOMICS:
            with self.subTest(op=op):
                spv = translate_cs(atomic_body(op, 1, True, ("tfe",)))
                check_cfg(spv)
                self.assert_status_written(spv, 1, "OpAtomic")
        for op in ("image_atomic_swap", "image_atomic_inc", "image_atomic_dec",
                   "image_atomic_fmin", "image_atomic_fmax"):
            with self.subTest(op=op):
                spv = translate_cs(atomic_body(op, 1, False, ("tfe",)))
                check_cfg(spv)
                self.assert_status_written(spv, 1, "OpAtomic")

    def test_fcmpswap_tfe_status_and_operands_intact(self):
        spv = translate_cs([
            "v_mov_b32_e32 v0, 0x3f800000",             # new value
            "v_mov_b32_e32 v1, 0x40000000",            # comparand (sentinel)
            "v_mov_b32_e32 v2, 0x33333333",            # the status dword's slot
            "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4",
            "image_atomic_fcmpswap v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm tfe",
        ])
        check_cfg(spv)
        self.assertEqual(writes_to(spv, 1), ["%c40000000"])   # comparand register: the setup store only
        self.assertEqual(writes_to(spv, 2)[-1], "%c0")
        self.assertIn("OpStore %reg_v2 %c0", spv[spv.index("OpAtomicCompareExchange"):])

    def test_resinfo_tfe_and_lwe_status(self):
        for mod in ("tfe", "lwe"):
            with self.subTest(mod=mod):
                spv = translate_cs([
                    "v_mov_b32_e32 v5, 0",                     # the mip level
                    f"image_get_resinfo v[0:4], v5, s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D {mod}",
                ])
                check_cfg(spv)
                self.assertIn("OpImageQueryLevels", spv)
                ids = writes_to(spv, 4)
                self.assertEqual(ids[-1], "%c0", f"{mod}: the status dword v4 must be written 0")
                self.assertIn("OpStore %reg_v4 %c0", spv[spv.index("OpImageQueryLevels"):])
                validate_spirv(spv)

    def test_resinfo_dmask_packs_data_then_status(self):
        # dmask:0xa keeps components 1 and 3 -> data in v0..v1, the status dword is v2
        spv = translate_cs([
            "v_mov_b32_e32 v5, 0",
            "image_get_resinfo v[0:2], v5, s[4:11] dmask:0xa dim:SQ_RSRC_IMG_2D tfe",
        ])
        check_cfg(spv)
        # both data words are written (width and level count), the status dword follows them
        self.assertGreater(len(writes_to(spv, 0)), 0)
        self.assertGreater(len(writes_to(spv, 1)), 0)
        self.assertEqual(writes_to(spv, 2)[-1], "%c0")
        self.assertIn("OpStore %reg_v2 %c0", spv[spv.index("OpImageQueryLevels"):])

    def test_resinfo_2d_array_tfe(self):
        spv = translate_cs([
            "v_mov_b32_e32 v5, 0",
            "image_get_resinfo v[0:4], v5, s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY tfe",
        ])
        check_cfg(spv)
        self.assertIn("OpImageQuerySizeLod %v3int", spv)
        self.assertEqual(writes_to(spv, 4)[-1], "%c0")

    def test_resinfo_no_status_control(self):
        spv = translate_cs([
            "v_mov_b32_e32 v5, 0",
            "image_get_resinfo v[0:3], v5, s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
        ])
        check_cfg(spv)
        self.assertEqual(writes_to(spv, 4), [])           # nothing stored into the would-be status VGPR

    def test_unestablished_and_unchanged_rejections(self):
        for line in (
            # cmpswap + tfe with a dmask no reference settles: only 0x3 (new value + comparand) is
            "image_atomic_cmpswap v[0:2], v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm tfe",
            # the 64-bit cmpswap (dmask 0xf, four data words) stays refused with tfe as without it
            "image_atomic_cmpswap v[0:4], v[4:5], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm tfe",
            # a range wider than data + one status dword
            "image_atomic_cmpswap v[0:3], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm tfe",
            "image_atomic_fcmpswap v[0:3], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm tfe",
            "image_atomic_swap v[0:2], v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm tfe",
            # without tfe/lwe these widths keep their old rejections
            "image_atomic_cmpswap v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
            "image_atomic_fcmpswap v[0:2], v[4:5], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm",
            "image_atomic_swap v[0:1], v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            # d16 stays refused for the atomics (with tfe too)
            "image_atomic_add v[0:1], v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16 tfe",
        ):
            with self.subTest(line=line):
                setup = ["v_mov_b32_e32 v0, 7", "v_mov_b32_e32 v1, 9", "v_mov_b32_e32 v2, 0",
                         "v_mov_b32_e32 v3, 0", "v_mov_b32_e32 v4, 3", "v_mov_b32_e32 v5, 4"]
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_cs(setup + [line])

    def test_status_write_sits_inside_the_active_guard(self):
        # the status store belongs to the guarded block (inactive lanes preserve their status VGPR)
        spv = translate_cs(atomic_body("image_atomic_add", 1, False, ("tfe",)))
        m = re.search(r"(%\w+_do) = OpLabel(.*?OpBranch %\w+_end)", spv, re.S)
        self.assertIsNotNone(m)
        seg = m.group(2)
        self.assertIn("OpAtomic", seg)
        self.assertIn("OpStore %reg_v1 %c0", seg)


if __name__ == "__main__":
    unittest.main()

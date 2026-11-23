"""Semantic + SPIR-V validation tests for image_atomic_* on 1D, 2D_ARRAY, and CUBE storage images.

Covers all 16 RDNA2 32-bit image atomic ops:
  add, sub, umin, umax, smin, smax, and, or, xor, swap, cmpswap, inc, dec, fmin, fmax, fcmpswap
across SQ_RSRC_IMG_1D (binding 18, Image1D capability, scalar int coord),
SQ_RSRC_IMG_2D_ARRAY and SQ_RSRC_IMG_CUBE (binding 12, 6-layer 2D array, v3int coord),
with and without glc, with and without a16, in both CS and PS stages, plus co-existence with
image_store aliasing bindings 12/18, and llvm-mc -mcpu=gfx1030 binary decode round-trips.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.dont_write_bytecode = True
for p in (TOOLS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2_decode  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

rdna2spirv = support.rdna2spirv

CS_FSSM = bytes(0x138) + struct.pack("<4I", 0x88 | (12 << 1), 32, 1, 1)
PS_FSSM = bytes(0x138) + struct.pack("<4I", 0x88 | (12 << 1), 0, 0, 0)

ALL_OPS = [
    ("image_atomic_add", "v2", "0x1", "OpAtomicIAdd %uint"),
    ("image_atomic_sub", "v2", "0x1", "OpAtomicISub %uint"),
    ("image_atomic_umin", "v2", "0x1", "OpAtomicUMin %uint"),
    ("image_atomic_umax", "v2", "0x1", "OpAtomicUMax %uint"),
    ("image_atomic_smin", "v2", "0x1", "OpAtomicSMin %uint"),
    ("image_atomic_smax", "v2", "0x1", "OpAtomicSMax %uint"),
    ("image_atomic_and", "v2", "0x1", "OpAtomicAnd %uint"),
    ("image_atomic_or", "v2", "0x1", "OpAtomicOr %uint"),
    ("image_atomic_xor", "v2", "0x1", "OpAtomicXor %uint"),
    ("image_atomic_swap", "v2", "0x1", "OpAtomicExchange %uint"),
    ("image_atomic_cmpswap", "v[2:3]", "0x3", "OpAtomicCompareExchange %uint"),
    ("image_atomic_inc", "v2", "0x1", "OpUGreaterThanEqual %bool"),
    ("image_atomic_dec", "v2", "0x1", "OpUGreaterThan %bool"),
    ("image_atomic_fmin", "v2", "0x1", "OpExtInst %float %glsl NMin"),
    ("image_atomic_fmax", "v2", "0x1", "OpExtInst %float %glsl NMax"),
    ("image_atomic_fcmpswap", "v[2:3]", "0x3", "OpAtomicCompareExchange %uint"),
]

DIMS = [
    ("SQ_RSRC_IMG_1D", "v4", "v4", "%simages1d", 18, "OpTypeImage %uint 1D 0 0 0 2 R32ui", None, True),
    ("SQ_RSRC_IMG_2D_ARRAY", "v[4:6]", "v[4:5]", "%simages2da", 12, "OpTypeImage %uint 2D 0 1 0 2 R32ui", "%v3int", False),
    ("SQ_RSRC_IMG_CUBE", "v[4:6]", "v[4:5]", "%simages2da", 12, "OpTypeImage %uint 2D 0 1 0 2 R32ui", "%v3int", False),
]


def _find_tool(env_name, exe_name):
    val = os.environ.get(env_name)
    if val:
        return val
    found = shutil.which(exe_name)
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        cand = os.path.join(sdk, "Bin", exe_name + (".exe" if os.name == "nt" else ""))
        if os.path.isfile(cand):
            return cand
    return None


SPIRV_AS = _find_tool("SPIRV_AS", "spirv-as")
SPIRV_VAL = _find_tool("SPIRV_VAL", "spirv-val")


def _llvm_mc_cmd():
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    found = shutil.which("llvm-mc")
    return [found] if found else None


def translate_stage(stage, body):
    if stage == "cs":
        return support.translate_lines("cs_img_atomic_dims", body + ["s_endpgm"], fssm=CS_FSSM)
    ps_body = body + [
        "v_mov_b32_e32 v0, 0",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    return support.translate_lines("ps_img_atomic_dims", ps_body, fssm=PS_FSSM)


def check_cfg(spv):
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


def validate_spirv(spv):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm = os.path.join(tmp, "s.spvasm")
        bin_ = os.path.join(tmp, "s.spv")
        with open(asm, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", asm, "-o", bin_],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", bin_],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def assemble_gfx1030(lines):
    cmd = _llvm_mc_cmd()
    if not cmd:
        return None
    proc = subprocess.run(
        cmd + ["-mcpu=gfx1030", "-triple=amdgcn--", "-show-encoding"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{proc.stderr}")
    out = bytearray()
    for line in proc.stdout.splitlines():
        m = re.search(r"encoding:\s*\[([^\]]+)\]", line)
        if m:
            out.extend(int(tok.strip(), 0) for tok in m.group(1).split(","))
    return bytes(out)


class ImageAtomicDimsTests(unittest.TestCase):
    def test_all_16_ops_across_1d_2d_array_cube_cs_and_ps(self):
        for stage in ("cs", "ps"):
            for op, vdata, dmask, expected_tok in ALL_OPS:
                for dim, addr_32, addr_16, img_var, binding, img_type, vec_ty, need_img1d in DIMS:
                    for a16 in (False, True):
                        for glc in (False, True):
                            # Validate full op x dim x glc x a16 matrix in CS; in PS check glc=True + a16 toggle
                            # to keep runtime snappy while still covering every op and dim in PS.
                            if stage == "ps" and not glc:
                                continue
                            with self.subTest(stage=stage, op=op, dim=dim, a16=a16, glc=glc):
                                addr = addr_16 if a16 else addr_32
                                mods = f"dmask:{dmask} dim:{dim} unorm"
                                if a16:
                                    mods += " a16"
                                if glc:
                                    mods += " glc"
                                spv = translate_stage(stage, [
                                    "v_mov_b32_e32 v2, 0x3f800000",
                                    "v_mov_b32_e32 v3, 0x40000000",
                                    "v_mov_b32_e32 v4, 0x00020001",
                                    "v_mov_b32_e32 v5, 0x00000003",
                                    "v_mov_b32_e32 v6, 4",
                                    f"{op} {vdata}, {addr}, s[4:11] {mods}",
                                    "s_waitcnt vmcnt(0)",
                                ])
                                check_cfg(spv)
                                validate_spirv(spv)
                                self.assertIn(img_type, spv)
                                self.assertIn(f"OpDecorate {img_var} DescriptorSet 0", spv)
                                self.assertIn(f"OpDecorate {img_var} Binding {binding}", spv)
                                self.assertIn("OpImageTexelPointer %_ptr_Image_uint", spv)
                                self.assertIn("OpCapability StorageImageArrayDynamicIndexing", spv)
                                self.assertIn(expected_tok, spv)
                                if need_img1d:
                                    self.assertIn("OpCapability Image1D", spv)
                                else:
                                    self.assertIn(f"OpCompositeConstruct {vec_ty}", spv)
                                if a16:
                                    self.assertIn("OpBitwiseAnd %uint", spv)
                                tp_m = re.search(r"(%\w+) = OpImageTexelPointer %_ptr_Image_uint", spv)
                                self.assertIsNotNone(tp_m)
                                after_tp = spv[tp_m.start():]
                                v2_stores = re.findall(
                                    r"OpAccessChain %_ptr_Function_uint %vgpr %c2\b|OpStore %reg_v2\b",
                                    after_tp,
                                )
                                if glc:
                                    self.assertGreaterEqual(len(v2_stores), 1)
                                else:
                                    self.assertEqual(len(v2_stores), 0)

    def test_coexistence_with_image_store_aliasing_bindings_12_and_18(self):
        # Mixing image_store (Unknown format) and image_atomic_* (R32ui format) on the same
        # 2D_ARRAY/CUBE (binding 12) and 1D (binding 18) dimensions must produce valid SPIR-V.
        spv = translate_stage("cs", [
            "v_mov_b32_e32 v2, 7",
            "v_mov_b32_e32 v4, 1",
            "v_mov_b32_e32 v5, 2",
            "v_mov_b32_e32 v6, 3",
            "image_store v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm",
            "image_atomic_add v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm glc",
            "image_store v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm",
            "image_atomic_umax v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm glc",
            "image_store v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm",
            "image_atomic_xor v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm glc",
            "s_waitcnt vmcnt(0)",
        ])
        check_cfg(spv)
        validate_spirv(spv)
        self.assertIn("OpDecorate %stv1_1 Binding 12", spv)
        self.assertIn("OpDecorate %simages2da Binding 12", spv)
        self.assertIn("OpDecorate %stv1_3 Binding 18", spv)
        self.assertIn("OpDecorate %simages1d Binding 18", spv)
        self.assertEqual([ln.strip() for ln in spv.splitlines()].count("OpCapability Image1D"), 1)

    def test_llvm_mc_gfx1030_binary_decode_roundtrip(self):
        if not _llvm_mc_cmd():
            self.skipTest("llvm-mc not available")
        cases = [
            "image_atomic_add v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm",
            "image_atomic_add v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm glc a16",
            "image_atomic_cmpswap v[2:3], v4, s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm glc",
            "image_atomic_fmin v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm",
            "image_atomic_fmax v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm glc a16",
            "image_atomic_fcmpswap v[2:3], v[4:6], s[4:11] dmask:0x3 dim:SQ_RSRC_IMG_CUBE unorm glc",
            "image_atomic_swap v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm glc a16",
            "image_atomic_inc v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm glc",
            "image_atomic_dec v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm a16",
        ]
        for insn in cases:
            with self.subTest(insn=insn):
                code = assemble_gfx1030([insn])
                decoded = rdna2_decode.disasm_bytes(code, wave32=True)
                if isinstance(decoded, tuple):
                    decoded = decoded[0]
                decoded_lines = [ln.strip() for ln in decoded.splitlines() if ln.strip() and not ln.strip().startswith(";")]
                self.assertEqual(len(decoded_lines), 1)
                spv = translate_stage("cs", [
                    "v_mov_b32_e32 v2, 1",
                    "v_mov_b32_e32 v3, 2",
                    "v_mov_b32_e32 v4, 0x00010002",
                    "v_mov_b32_e32 v5, 0x00000003",
                    "v_mov_b32_e32 v6, 4",
                    decoded_lines[0],
                    "s_waitcnt vmcnt(0)",
                ])
                check_cfg(spv)
                validate_spirv(spv)

    def test_rejected_forms(self):
        for line in (
            "image_atomic_add v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm",     # too few 2D_ARRAY coords
            "image_atomic_add v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY unorm a16",     # too few a16 2D_ARRAY coords
            "image_atomic_add v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm",         # too few CUBE coords
            "image_atomic_add v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE unorm a16",         # too few a16 CUBE coords
            "image_atomic_add v2, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D_ARRAY unorm",     # unsupported 1D_ARRAY
            "image_atomic_add v2, v[4:6], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA unorm",      # unsupported MSAA
            "image_atomic_add v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm d16",           # unsupported d16
            "image_atomic_cmpswap v2, v4, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_1D unorm",           # 1-VGPR cmpswap
        ):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_stage("cs", [
                        "v_mov_b32_e32 v2, 0",
                        "v_mov_b32_e32 v4, 0",
                        "v_mov_b32_e32 v5, 0",
                        "v_mov_b32_e32 v6, 0",
                        line,
                    ])


if __name__ == "__main__":
    unittest.main()

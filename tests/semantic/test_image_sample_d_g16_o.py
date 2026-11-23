"""Semantic, spirv-val, and llvm-mc decode tests for the 6 explicit-gradient
image_sample_d_{g16,cl_g16,o,cl_o,o_g16,cl_o_g16} RDNA2 instructions.

Covers:
  - All 6 opcodes across 1D, 2D, 2D_ARRAY, CUBE, and 3D
  - Normal and a16-packed coordinates / min-LOD clamp
  - Normal and d16-packed output results
  - Strict SPIR-V Image Operands mask ordering: Grad -> Offset -> MinLod
  - --texel-offset-fallback folding for the _o variants (including _cl_o / _o_g16 / _cl_o_g16)
  - spirv-val --target-env vulkan1.2 validation of every generated SPIR-V module
  - rdna2_decode._mimg_addr_count and llvm-mc (gfx1030) binary decode round-trips
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support  # noqa: E402

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402


OPS = (
    "image_sample_d_g16",
    "image_sample_d_cl_g16",
    "image_sample_d_o",
    "image_sample_d_cl_o",
    "image_sample_d_o_g16",
    "image_sample_d_cl_o_g16",
)

DIMS = (
    ("1D", 0, 1, 1),
    ("2D", 1, 2, 2),
    ("2D_ARRAY", 5, 3, 2),
    ("CUBE", 3, 3, 2),
    ("3D", 2, 3, 3),
)


def _expected_addr_vgprs(op: str, ncoord: int, grad_width: int, a16: bool) -> int:
    toks = set(op.split("_")[2:])
    fixed = 0
    if "o" in toks:
        fixed += 1
    if "g16" in toks:
        fixed += 2 * ((grad_width + 1) // 2)
    else:
        fixed += 2 * grad_width
    packable = ncoord + (1 if "cl" in toks else 0)
    return fixed + ((packable + 1) // 2 if a16 else packable)


def _find_spirv_val() -> str | None:
    env_sdk = os.environ.get("VULKAN_SDK")
    candidates = []
    if env_sdk:
        candidates.append(os.path.join(env_sdk, "Bin", "spirv-val.exe"))
        candidates.append(os.path.join(env_sdk, "bin", "spirv-val"))
    candidates.append(r"C:\VulkanSDK\1.4.363.0\Bin\spirv-val.exe")
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which("spirv-val")


def _find_llvm_mc_cmd() -> list[str] | None:
    direct = shutil.which("llvm-mc")
    if direct:
        return [direct]
    if os.name == "nt":
        wsl = shutil.which("wsl")
        if wsl:
            probe = subprocess.run(
                [wsl, "-d", "Ubuntu-24.04", "-e", "test", "-x", "/usr/lib/llvm-21/bin/llvm-mc"],
                capture_output=True,
            )
            if probe.returncode == 0:
                return [wsl, "-d", "Ubuntu-24.04", "-e", "/usr/lib/llvm-21/bin/llvm-mc"]
    return None


def _build_shader(op: str, dim_name: str, n_addr: int, a16: bool = False, d16: bool = False,
                  dmask: str = "0xf") -> list[str]:
    vdata = "v[4:5]" if d16 else "v[4:7]"
    vaddr = "v8" if n_addr == 1 else f"v[8:{8 + n_addr - 1}]"
    mods = [f"dmask:{dmask}", f"dim:SQ_RSRC_IMG_{dim_name}"]
    if a16:
        mods.append("a16")
    if d16:
        mods.append("d16")
    lines = [
        "s_load_dwordx8 s[8:15], s[0:1], 0x0",
        "s_waitcnt lgkmcnt(0)",
        "s_mov_b32 s16, 0x92",
        "s_mov_b32 s17, 0xc00000",
        "s_mov_b32 s18, 0x14500000",
        "s_mov_b32 s19, 0",
    ]
    for i in range(n_addr):
        lines.append(f"v_mov_b32_e32 v{8 + i}, 0x3c003c00")
    lines.append(f"{op} {vdata}, {vaddr}, s[8:15], s[16:19] {' '.join(mods)}")
    lines.append("s_waitcnt vmcnt(0)")
    if d16:
        lines.append("exp mrt0 v4, v5, v4, v5 done vm")
    else:
        lines.append("exp mrt0 v4, v5, v6, v7 done vm")
    lines.append("s_endpgm")
    return lines


def _validate_spvasm(spvasm: str, spirv_val: str | None, allow_offset: bool = False) -> None:
    words = spirv_asm.assemble(spvasm)
    if not spirv_val:
        return
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as tf:
        tf.write(words)
        path = tf.name
    try:
        cmd = [spirv_val, "--target-env", "vulkan1.2"]
        if allow_offset:
            cmd.append("--allow-offset-texture-operand")
        cmd.append(path)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{p.stdout}\n{p.stderr}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class TestImageSampleDG16O(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spirv_val = _find_spirv_val()
        cls.llvm_mc = _find_llvm_mc_cmd()

    def test_mimg_addr_count_all_variants(self):
        for op in OPS:
            for dim_name, dim_id, ncoord, grad_width in DIMS:
                for a16 in (False, True):
                    want = _expected_addr_vgprs(op, ncoord, grad_width, a16)
                    got = rdna2_decode._mimg_addr_count(op, dim_id, a16)
                    self.assertEqual(
                        got, want,
                        f"_mimg_addr_count({op}, {dim_name}, a16={a16}): got {got}, want {want}",
                    )

    def test_translate_and_validate_all_dims_and_a16(self):
        for op in OPS:
            toks = set(op.split("_")[2:])
            has_o = "o" in toks
            has_cl = "cl" in toks
            has_g16 = "g16" in toks
            expected_mask = "|".join(
                m for m, cond in (("Grad", True), ("Offset", has_o), ("MinLod", has_cl)) if cond
            )
            for dim_name, _dim_id, ncoord, grad_width in DIMS:
                for a16 in (False, True):
                    with self.subTest(op=op, dim=dim_name, a16=a16):
                        n_addr = _expected_addr_vgprs(op, ncoord, grad_width, a16)
                        lines = _build_shader(op, dim_name, n_addr, a16=a16)
                        spvasm = support.translate_lines(f"ps_{op}_{dim_name}", lines)
                        self.assertIn(
                            f"OpImageSampleExplicitLod %v4float", spvasm
                        )
                        self.assertRegex(
                            spvasm,
                            rf"OpImageSampleExplicitLod %v4float %\S+ %\S+ {re.escape(expected_mask)} ",
                        )
                        if has_g16:
                            self.assertIn("UnpackHalf2x16", spvasm)
                        if has_o:
                            self.assertIn("OpBitFieldSExtract", spvasm)
                        if has_cl:
                            self.assertIn("OpCapability MinLod", spvasm)
                        _validate_spvasm(spvasm, self.spirv_val, allow_offset=has_o)

    def test_translate_and_validate_d16(self):
        for op in OPS:
            has_o = "o" in set(op.split("_")[2:])
            for dim_name, _dim_id, ncoord, grad_width in (("2D", 1, 2, 2), ("3D", 2, 3, 3)):
                with self.subTest(op=op, dim=dim_name):
                    n_addr = _expected_addr_vgprs(op, ncoord, grad_width, a16=True)
                    lines = _build_shader(op, dim_name, n_addr, a16=True, d16=True)
                    spvasm = support.translate_lines(f"ps_{op}_{dim_name}_d16", lines)
                    self.assertIn("OpImageSampleExplicitLod %v4float", spvasm)
                    _validate_spvasm(spvasm, self.spirv_val, allow_offset=has_o)

    def test_texel_offset_fallback_all_o_variants(self):
        o_ops = (
            "image_sample_d_o",
            "image_sample_d_cl_o",
            "image_sample_d_o_g16",
            "image_sample_d_cl_o_g16",
        )
        old_fb = rdna2spirv.TEXEL_OFFSET_FALLBACK
        try:
            rdna2spirv.TEXEL_OFFSET_FALLBACK = True
            for op in o_ops:
                has_cl = "_cl_" in op
                expected_mask = "Grad|MinLod" if has_cl else "Grad"
                for dim_name, _dim_id, ncoord, grad_width in DIMS:
                    with self.subTest(op=op, dim=dim_name):
                        n_addr = _expected_addr_vgprs(op, ncoord, grad_width, a16=False)
                        lines = _build_shader(op, dim_name, n_addr, a16=False)
                        spvasm = support.translate_lines(f"ps_{op}_{dim_name}_fb", lines)
                        self.assertNotIn("Grad|Offset", spvasm)
                        self.assertNotIn("ImageGatherExtended", spvasm)
                        self.assertIn("OpImageQuerySizeLod", spvasm)
                        self.assertIn("Log2", spvasm)
                        self.assertRegex(
                            spvasm,
                            rf"OpImageSampleExplicitLod %v4float %\S+ %\S+ {re.escape(expected_mask)} ",
                        )
                        _validate_spvasm(spvasm, self.spirv_val, allow_offset=False)
        finally:
            rdna2spirv.TEXEL_OFFSET_FALLBACK = old_fb

    def test_llvm_mc_binary_decode_roundtrip(self):
        if not self.llvm_mc:
            self.skipTest("llvm-mc not available")
        asm_lines = []
        meta = []
        for op in OPS:
            for dim_name, _dim_id, ncoord, grad_width in DIMS:
                for a16 in (False, True):
                    n_addr = _expected_addr_vgprs(op, ncoord, grad_width, a16)
                    vaddr = "v8" if n_addr == 1 else f"v[8:{8 + n_addr - 1}]"
                    a16s = " a16" if a16 else ""
                    asm_lines.append(
                        f"{op} v[0:3], {vaddr}, s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_{dim_name}{a16s}"
                    )
                    meta.append((op, dim_name, n_addr, a16))

        p = subprocess.run(
            self.llvm_mc + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
            input="\n".join(asm_lines) + "\n",
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, f"llvm-mc failed:\n{p.stderr}")
        encodings = re.findall(r"encoding:\s*\[([^\]]+)\]", p.stdout)
        self.assertEqual(len(encodings), len(meta))

        for (op, dim_name, n_addr, a16), enc in zip(meta, encodings):
            raw = bytes(int(tok.strip(), 16) for tok in enc.split(","))
            with self.subTest(op=op, dim=dim_name, a16=a16):
                ins, length = rdna2_decode._decode_one(raw, 0, wave32=True, mcpu="gfx1030")
                self.assertIsNotNone(ins)
                self.assertEqual(length, len(raw))
                self.assertEqual(ins["op"], op)
                vaddr_op = ins["args"][1]
                self.assertEqual(vaddr_op["k"], "vgpr")
                self.assertEqual(vaddr_op["r"], 8)
                self.assertEqual(vaddr_op["n"], n_addr)


if __name__ == "__main__":
    unittest.main()

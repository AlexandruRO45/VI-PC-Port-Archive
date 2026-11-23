"""Semantic and spirv-val tests for image_get_resinfo across all supported image dimensions.

Covers 1D, 2D, 2D_ARRAY, 3D, CUBE, 2D_MSAA, and 2D_MSAA_ARRAY:
  - exact bindless set-0 bindings and OpTypeImage declarations (matching runtime/src/vulkan_bindless.h),
  - SPIR-V query opcodes (OpImageQuerySizeLod / OpImageQueryLevels for mipmapped dims,
    OpImageQuerySize / OpImageQuerySamples for MSAA dims, scalar OpImageQuerySizeLod %int for 1D),
  - semantic evaluation of resinfo_words (in-range mips, OOB mip zeroing, MSAA level != 0 zeroing,
    unused axis zeroing, 3D depth, CUBE face count, 2D_ARRAY minification),
  - dmask VGPR compaction subsets and a16 16-bit mip masking,
  - llvm-mc assembly round-trip and spirv-as + spirv-val --target-env vulkan1.2 validation.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support  # noqa: E402

ROOT = support.ROOT
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mc_command  # noqa: E402
import rdna2spirv  # noqa: E402


def _find_spirv_tool(name):
    found = shutil.which(name)
    if found:
        return found
    for root in (os.environ.get("VULKAN_SDK"), r"C:\VulkanSDK\1.4.363.0"):
        if not root:
            continue
        for sub in ("Bin", "bin"):
            cand = os.path.join(root, sub, name + (".exe" if os.name == "nt" else ""))
            if os.path.isfile(cand):
                return cand
    return None


SPIRV_AS = _find_spirv_tool("spirv-as")
SPIRV_VAL = _find_spirv_tool("spirv-val")


def _find_llvm_mc():
    cmd = mc_command.mc_command()
    if cmd == ["llvm-mc"] and not shutil.which("llvm-mc"):
        return None
    try:
        r = subprocess.run(
            cmd + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"],
            input="s_endpgm\n", text=True, capture_output=True, timeout=15,
        )
        return cmd if r.returncode == 0 else None
    except OSError:
        return None


LLVM_MC_CMD = _find_llvm_mc()


def validate_spirv(spvasm):
    """Assemble and validate SPIR-V text with spirv-as + spirv-val --target-env vulkan1.2."""
    if not SPIRV_AS or not SPIRV_VAL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r.stderr}\n{r.stdout}")
        r = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r.stderr}\n{r.stdout}")


def assemble_with_llvm_mc(name, lines):
    """Assemble RDNA2 lines with llvm-mc -show-encoding into rdna2spirv's .s listing format."""
    src = "".join(f"\t{ln.strip()}\n" for ln in lines)
    r = subprocess.run(
        LLVM_MC_CMD + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-show-encoding"],
        input=src, text=True, capture_output=True, timeout=30,
    )
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed for {name}:\n{r.stderr}\n{src}")
    out = [f"; {name}  llvm-mc assembled\n"]
    for ln in r.stdout.splitlines():
        if ln.strip().startswith(".") or not ln.strip():
            continue
        out.append(ln + "\n")
    return "".join(out).encode("utf-8")


def parse_defs(spvasm):
    defs = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
        if m:
            rest = m.group(3).split() if m.group(3) else []
            defs[m.group(1)] = (m.group(2), rest)
    return defs


def vgpr_stores(spvasm):
    ptr_to_idx = {}
    const_u = {}
    stores = []
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = OpConstant %uint (\d+)", line)
        if m:
            const_u[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\s*(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr (%\w+)", line)
        if m and m.group(2) in const_u:
            ptr_to_idx[m.group(1)] = const_u[m.group(2)]
            continue
        m = re.match(r"\s*OpStore (%\w+) (%\w+)", line)
        if m and m.group(1) in ptr_to_idx:
            stores.append((ptr_to_idx[m.group(1)], m.group(2)))
    return stores


class _ResinfoStub:
    """Minimal Gen stub for executing Gen.resinfo_words directly."""

    def __init__(self):
        self.n = 0
        self.lines = []

    def tmp(self):
        self.n += 1
        return f"%t{self.n}"

    def cu(self, v):
        return f"%c{v & 0xFFFFFFFF:x}"

    def emit(self, s):
        self.lines.append(s)

    def as_int(self, u):
        i = self.tmp()
        self.emit(f"{i} = OpBitcast %int {u}")
        return i


def eval_resinfo(lines, dim, view, level, word3=0):
    """Evaluate Gen.resinfo_words SPIR-V lines against a Vulkan view description.

    view:
      - 1D:             (width, levels)
      - 2D:             (width, height, levels)
      - 2D_ARRAY / CUBE:(width, height, layers, levels)
      - 3D:             (width, height, depth, levels)
      - 2D_MSAA:        (width, height, samples)
      - 2D_MSAA_ARRAY:  (width, height, layers, samples)
    """
    env = {"%img": None, "%level": level & 0xFFFFFFFF, "%word3": word3 & 0xFFFFFFFF}

    def val(x):
        return int(x[2:], 16) if x.startswith("%c") else env[x]

    for line in lines:
        dst, _, op, rtype, *args = line.split()
        if op == "OpImageQueryLevels":
            assert dim not in ("dim:SQ_RSRC_IMG_2D_MSAA", "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY")
            env[dst] = view[-1]
        elif op == "OpImageQuerySamples":
            assert dim in ("dim:SQ_RSRC_IMG_2D_MSAA", "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY")
            env[dst] = view[-1]
        elif op == "OpImageQuerySizeLod":
            assert dim not in ("dim:SQ_RSRC_IMG_2D_MSAA", "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY")
            lod = val(args[1])
            levels = view[-1]
            assert 0 <= lod < levels, f"OOB OpImageQuerySizeLod at lod={lod} (levels={levels})"
            if dim == "dim:SQ_RSRC_IMG_1D":
                assert rtype == "%int"
                env[dst] = max(1, view[0] >> lod)
            elif dim == "dim:SQ_RSRC_IMG_2D":
                assert rtype == "%v2int"
                env[dst] = (max(1, view[0] >> lod), max(1, view[1] >> lod))
            elif dim == "dim:SQ_RSRC_IMG_3D":
                assert rtype == "%v3int"
                env[dst] = (max(1, view[0] >> lod), max(1, view[1] >> lod), max(1, view[2] >> lod))
            else:
                assert rtype == "%v3int"
                env[dst] = (max(1, view[0] >> lod), max(1, view[1] >> lod), view[2])
        elif op == "OpImageQuerySize":
            assert dim in ("dim:SQ_RSRC_IMG_2D_MSAA", "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY")
            if dim == "dim:SQ_RSRC_IMG_2D_MSAA":
                assert rtype == "%v2int"
                env[dst] = (view[0], view[1])
            else:
                assert rtype == "%v3int"
                env[dst] = (view[0], view[1], view[2])
        elif op == "OpCompositeExtract":
            env[dst] = env[args[0]][int(args[1])]
        elif op == "OpExtInst":
            assert args[1] == "UMax"
            env[dst] = max(val(args[2]), val(args[3]))
        elif op == "OpBitFieldUExtract":
            env[dst] = (val(args[0]) >> val(args[1])) & ((1 << val(args[2])) - 1)
        elif op == "OpBitcast":
            env[dst] = val(args[0])
        elif op == "OpULessThan":
            env[dst] = val(args[0]) < val(args[1])
        elif op == "OpIEqual":
            env[dst] = val(args[0]) == val(args[1])
        elif op == "OpSelect":
            env[dst] = val(args[1]) if val(args[0]) else val(args[2])
        elif op == "OpIAdd":
            env[dst] = (val(args[0]) + val(args[1])) & 0xFFFFFFFF
        elif op == "OpShiftRightLogical":
            sh = val(args[1])
            env[dst] = (val(args[0]) >> sh) if sh < 32 else 0
        else:
            raise AssertionError(f"unexpected op in resinfo_words: {line}")
    return env


class ImageGetResinfoDimsTests(unittest.TestCase):
    DIM_SPECS = [
        ("SQ_RSRC_IMG_1D", 5, "%images_1d", "%img1d", "OpImageQuerySizeLod %int", "OpImageQueryLevels %int"),
        ("SQ_RSRC_IMG_2D", 0, "%images", "%img2d", "OpImageQuerySizeLod %v2int", "OpImageQueryLevels %int"),
        ("SQ_RSRC_IMG_2D_ARRAY", 3, "%images_arr", "%img2da", "OpImageQuerySizeLod %v3int", "OpImageQueryLevels %int"),
        ("SQ_RSRC_IMG_3D", 6, "%images_3d", "%img3d", "OpImageQuerySizeLod %v3int", "OpImageQueryLevels %int"),
        ("SQ_RSRC_IMG_CUBE", 3, "%images_arr", "%img2da", "OpImageQuerySizeLod %v3int", "OpImageQueryLevels %int"),
        ("SQ_RSRC_IMG_2D_MSAA", 20, "%images_2dms", "%img2dms", "OpImageQuerySize %v2int", "OpImageQuerySamples %int"),
        ("SQ_RSRC_IMG_2D_MSAA_ARRAY", 21, "%images_2dmsa", "%img2dmsa", "OpImageQuerySize %v3int", "OpImageQuerySamples %int"),
    ]

    def test_all_dims_translate_and_validate_ps_and_cs(self):
        cs_fssm = bytearray(0x148)
        struct.pack_into("<4I", cs_fssm, 0x138, 0x8088, 64, 1, 1)
        for dim, binding, var_name, img_ty, size_op, w_op in self.DIM_SPECS:
            for stage, epilogue, fssm in (
                ("ps", ["exp mrt0 v4, v5, v6, v7 done vm", "s_endpgm"], None),
                ("cs", ["s_endpgm"], bytes(cs_fssm)),
            ):
                with self.subTest(dim=dim, stage=stage):
                    spv = support.translate_lines(f"{stage}_resinfo_{dim.lower()}", [
                        "v_mov_b32 v0, 0",
                        f"image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:{dim}",
                        "s_waitcnt vmcnt(0)",
                        *epilogue,
                    ], fssm=fssm)
                    validate_spirv(spv)
                    self.assertIn("OpCapability ImageQuery", spv)
                    if dim == "SQ_RSRC_IMG_1D":
                        self.assertIn("OpCapability Sampled1D", spv)
                    self.assertIn(f"OpDecorate {var_name} Binding {binding}", spv)
                    self.assertIn(f"OpLoad {img_ty} ", spv)
                    self.assertIn(size_op, spv)
                    self.assertIn(w_op, spv)
                    if "MSAA" in dim:
                        self.assertNotIn("OpImageQuerySizeLod", spv)
                        self.assertNotIn("OpImageQueryLevels", spv)

    def test_dmask_subsets_and_vgpr_compaction_all_dims(self):
        for dim, _, _, _, _, _ in self.DIM_SPECS:
            for dmask, expected_count in ((0x1, 1), (0x3, 2), (0x4, 1), (0x7, 3), (0x8, 1), (0x9, 2), (0xa, 2), (0xf, 4)):
                dst = "v4" if expected_count == 1 else f"v[4:{4 + expected_count - 1}]"
                with self.subTest(dim=dim, dmask=hex(dmask)):
                    spv = support.translate_lines(f"ps_resinfo_dm_{dim.lower()}_{dmask:x}", [
                        "v_mov_b32 v0, 1",
                        f"image_get_resinfo {dst}, v0, s[8:15] dmask:{dmask:#x} dim:{dim}",
                        "s_waitcnt vmcnt(0)",
                        "exp mrt0 v4, v5, v6, v7 done vm",
                        "s_endpgm",
                    ])
                    validate_spirv(spv)
                    written = [idx for idx, _ in vgpr_stores(spv) if idx >= 4]
                    self.assertEqual(written, list(range(4, 4 + expected_count)))

    def test_a16_masks_mip_level_and_validates_all_dims(self):
        for dim, _, _, _, _, _ in self.DIM_SPECS:
            with self.subTest(dim=dim):
                spv = support.translate_lines(f"ps_resinfo_a16_{dim.lower()}", [
                    "v_mov_b32 v0, 0x00020001",
                    f"image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:{dim} a16",
                    "s_waitcnt vmcnt(0)",
                    "exp mrt0 v4, v5, v6, v7 done vm",
                    "s_endpgm",
                ])
                validate_spirv(spv)
                self.assertIn("OpConstant %uint 65535", spv)
                self.assertRegex(spv, r"%\w+ = OpBitwiseAnd %uint %\w+ %\w+")

    def test_semantic_resinfo_words_3d(self):
        dim = "dim:SQ_RSRC_IMG_3D"
        view = (32, 16, 8, 6)  # 32x16x8, 6 mips (0..5)
        cases = [
            (0, [32, 16, 8, 6]),
            (1, [16, 8, 4, 6]),
            (3, [4, 2, 1, 6]),
            (5, [1, 1, 1, 6]),
            (6, [0, 0, 0, 6]),
            (0x80000000, [0, 0, 0, 6]),
        ]
        for level, want in cases:
            with self.subTest(level=level):
                stub = _ResinfoStub()
                ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", dim, "%word3")
                env = eval_resinfo(stub.lines, dim, view, level)
                self.assertEqual([env[i] & 0xFFFFFFFF for i in ids], want)

    def test_semantic_resinfo_words_cube(self):
        dim = "dim:SQ_RSRC_IMG_CUBE"
        # Single cube (6 faces) and cube array of 2 cubes (12 faces)
        for faces in (6, 12):
            view = (64, 64, faces, 7)
            for level, want in (
                (0, [64, 64, faces, 7]),
                (2, [16, 16, faces, 7]),
                (6, [1, 1, faces, 7]),
                (7, [0, 0, 0, 7]),
                (0x80000000, [0, 0, 0, 7]),
            ):
                with self.subTest(faces=faces, level=level):
                    stub = _ResinfoStub()
                    ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", dim, "%word3")
                    env = eval_resinfo(stub.lines, dim, view, level)
                    self.assertEqual([env[i] & 0xFFFFFFFF for i in ids], want)

    def test_semantic_resinfo_words_1d(self):
        dim = "dim:SQ_RSRC_IMG_1D"
        view = (128, 8)
        for level, want in (
            (0, [128, 0, 0, 8]),
            (3, [16, 0, 0, 8]),
            (7, [1, 0, 0, 8]),
            (8, [0, 0, 0, 8]),
            (0x80000000, [0, 0, 0, 8]),
        ):
            with self.subTest(level=level):
                stub = _ResinfoStub()
                ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", dim, "%word3")
                env = eval_resinfo(stub.lines, dim, view, level)
                self.assertEqual([env[i] & 0xFFFFFFFF for i in ids], want)

    def test_semantic_resinfo_words_2d_msaa(self):
        dim = "dim:SQ_RSRC_IMG_2D_MSAA"
        view = (1920, 1080, 4)
        for level, want in (
            (0, [1920, 1080, 0, 4]),
            (1, [0, 0, 0, 4]),
            (0x80000000, [0, 0, 0, 4]),
        ):
            with self.subTest(level=level):
                stub = _ResinfoStub()
                ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", dim, "%word3")
                env = eval_resinfo(stub.lines, dim, view, level)
                self.assertEqual([env[i] & 0xFFFFFFFF for i in ids], want)

    def test_semantic_resinfo_words_2d_msaa_array(self):
        dim = "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY"
        view = (512, 256, 6, 8)
        for level, want in (
            (0, [512, 256, 6, 8]),
            (1, [0, 0, 0, 8]),
            (0x80000000, [0, 0, 0, 8]),
        ):
            with self.subTest(level=level):
                stub = _ResinfoStub()
                ids = rdna2spirv.Gen.resinfo_words(stub, "%img", "%level", dim, "%word3")
                env = eval_resinfo(stub.lines, dim, view, level)
                self.assertEqual([env[i] & 0xFFFFFFFF for i in ids], want)

    @unittest.skipUnless(LLVM_MC_CMD, "requires llvm-mc with amdgcn target")
    def test_llvm_mc_assembled_all_dims_and_a16(self):
        for dim, binding, var_name, img_ty, size_op, w_op in self.DIM_SPECS:
            for extra in ("", " a16"):
                tag = f"ps_mc_resinfo_{dim.lower()}{'_a16' if extra else ''}"
                with self.subTest(dim=dim, a16=bool(extra)):
                    listing = assemble_with_llvm_mc(tag, [
                        "s_load_dwordx8 s[8:15], s[0:1], 0x0",
                        "s_waitcnt lgkmcnt(0)",
                        "v_mov_b32_e32 v0, 0",
                        f"image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:{dim}{extra}",
                        "s_waitcnt vmcnt(0)",
                        "exp mrt0 v4, v5, v6, v7 done vm",
                        "s_endpgm",
                    ])
                    spv = support.translate_listing(tag, listing)
                    validate_spirv(spv)
                    self.assertIn(f"OpDecorate {var_name} Binding {binding}", spv)
                    self.assertIn(f"OpLoad {img_ty} ", spv)
                    self.assertIn(size_op, spv)
                    self.assertIn(w_op, spv)

    def test_unsupported_1d_array_and_d16(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_resinfo_1da", [
                "v_mov_b32 v0, 0",
                "image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY",
                "exp mrt0 v4, v5, v6, v7 done vm",
                "s_endpgm",
            ])
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_resinfo_d16", [
                "v_mov_b32 v0, 0",
                "image_get_resinfo v[4:5], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D d16",
                "exp mrt0 v4, v5, v4, v5 done vm",
                "s_endpgm",
            ])


if __name__ == "__main__":
    unittest.main()

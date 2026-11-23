"""Translation, structure, llvm-mc round-trip, and spirv-val tests for image_get_lod dimensions and a16.

Covers 1D, 2D, 2D_ARRAY, 3D, and CUBE with and without a16, dmask compaction, cube_coords lowering,
short-vaddr / modifier / stage guards, and full spirv-as + spirv-val --target-env vulkan1.2 validation.
"""
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from support import rdna2spirv, translate_lines  # noqa: E402

_FSSM_BUF = bytearray(0x148)
struct.pack_into("<4I", _FSSM_BUF, 0x138, 0x88, 32, 1, 1)
STUB_FSSM = bytes(_FSSM_BUF)


def _find_spirv_tool(name):
    found = shutil.which(name)
    if found:
        return found
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for ext in ("", ".exe"):
            cand = os.path.join(sdk, "Bin", name + ext)
            if os.path.isfile(cand):
                return cand
    return None


SPIRV_AS = _find_spirv_tool("spirv-as")
SPIRV_VAL = _find_spirv_tool("spirv-val")


def _find_llvm_mc():
    env = os.environ.get("LLVM_MC")
    if env:
        return shlex.split(env)
    found = shutil.which("llvm-mc")
    if found:
        return [found]
    return None


LLVM_MC = _find_llvm_mc()


def validate_spirv(spvasm):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "mod.spvasm")
        spv_path = os.path.join(tmp, "mod.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r_as = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
                              capture_output=True, text=True)
        if r_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r_as.stderr}")
        r_val = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
                               capture_output=True, text=True)
        if r_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r_val.stderr}")


def parse_defs(spvasm):
    defs = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split() if m.group(3) else [])
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


def unwrap_dst(defs, sid):
    op, args = defs[sid]
    return args[2] if op == "OpSelect" else sid


def find_query_lod(defs):
    return [(rid, args) for rid, (op, args) in defs.items() if op == "OpImageQueryLod"]


class ImageGetLodDimsTests(unittest.TestCase):
    def test_1d_non_a16_and_a16(self):
        for a16 in (False, True):
            with self.subTest(a16=a16):
                mod = " a16" if a16 else ""
                spv = translate_lines("ps_lod_1d", [
                    "v_mov_b32 v0, 0x38003400" if a16 else "v_mov_b32 v0, 0.25",
                    f"image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_1D{mod}",
                    "exp mrt0 v4, v5, v4, v5 done vm",
                    "s_endpgm",
                ])
                validate_spirv(spv)
                self.assertIn("OpCapability Sampled1D", spv)
                self.assertIn("OpCapability ImageQuery", spv)
                self.assertIn("OpAccessChain %_ptr_UC_img1 %images_1d ", spv)
                self.assertIn("OpLoad %img1d ", spv)
                self.assertIn("OpSampledImage %simg1d ", spv)
                defs = parse_defs(spv)
                queries = find_query_lod(defs)
                self.assertEqual(len(queries), 1)
                _, (ty, si, coord) = queries[0]
                self.assertEqual(ty, "%v2float")
                self.assertEqual(defs[si][1][0], "%simg1d")
                if a16:
                    self.assertEqual(defs[coord][0], "OpCompositeExtract")
                    self.assertEqual(defs[coord][1][0], "%float")
                    self.assertEqual(defs[coord][1][2], "0")
                    self.assertEqual(defs[defs[coord][1][1]][0], "OpExtInst")
                    self.assertEqual(defs[defs[coord][1][1]][1][2], "UnpackHalf2x16")
                else:
                    self.assertEqual(defs[coord][0], "OpBitcast")
                    self.assertEqual(defs[coord][1][0], "%float")

    def test_2d_and_2d_array_non_a16_and_a16(self):
        cases = [
            ("SQ_RSRC_IMG_2D", False, "v[0:1]", "%_ptr_UC_img", "%images", "%img2d", "%simg2d"),
            ("SQ_RSRC_IMG_2D", True, "v0", "%_ptr_UC_img", "%images", "%img2d", "%simg2d"),
            ("SQ_RSRC_IMG_2D_ARRAY", False, "v[0:2]", "%_ptr_UC_imga", "%images_arr", "%img2da", "%simg2da"),
            ("SQ_RSRC_IMG_2D_ARRAY", True, "v[0:1]", "%_ptr_UC_imga", "%images_arr", "%img2da", "%simg2da"),
        ]
        for dim, a16, vaddr, ptr_ty, arr_name, img_ty, simg_ty in cases:
            with self.subTest(dim=dim, a16=a16):
                mod = " a16" if a16 else ""
                spv = translate_lines("ps_lod_2d", [
                    "v_mov_b32 v0, 0x38003400" if a16 else "v_mov_b32 v0, 0.25",
                    "v_mov_b32 v1, 0x3c00" if a16 else "v_mov_b32 v1, 0.5",
                    "v_mov_b32 v2, 1.0",
                    f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:{dim}{mod}",
                    "exp mrt0 v4, v5, v4, v5 done vm",
                    "s_endpgm",
                ])
                validate_spirv(spv)
                self.assertIn("OpCapability ImageQuery", spv)
                self.assertIn(f"OpAccessChain {ptr_ty} {arr_name} ", spv)
                self.assertIn(f"OpLoad {img_ty} ", spv)
                self.assertIn(f"OpSampledImage {simg_ty} ", spv)
                defs = parse_defs(spv)
                queries = find_query_lod(defs)
                self.assertEqual(len(queries), 1)
                _, (ty, si, coord) = queries[0]
                self.assertEqual(ty, "%v2float")
                self.assertEqual(defs[si][1][0], simg_ty)
                if a16:
                    self.assertEqual(defs[coord][0], "OpExtInst")
                    self.assertEqual(defs[coord][1][:3], ["%v2float", "%glsl", "UnpackHalf2x16"])
                else:
                    self.assertEqual(defs[coord][0], "OpCompositeConstruct")
                    self.assertEqual(defs[coord][1][0], "%v2float")

    def test_3d_non_a16_and_a16(self):
        for a16, vaddr in ((False, "v[0:2]"), (True, "v[0:1]")):
            with self.subTest(a16=a16):
                mod = " a16" if a16 else ""
                spv = translate_lines("ps_lod_3d", [
                    "v_mov_b32 v0, 0x38003400" if a16 else "v_mov_b32 v0, 0.25",
                    "v_mov_b32 v1, 0x3c00" if a16 else "v_mov_b32 v1, 0.5",
                    "v_mov_b32 v2, 0.75",
                    f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D{mod}",
                    "exp mrt0 v4, v5, v4, v5 done vm",
                    "s_endpgm",
                ])
                validate_spirv(spv)
                self.assertIn("OpCapability ImageQuery", spv)
                self.assertIn("OpAccessChain %_ptr_UC_img3 %images_3d ", spv)
                self.assertIn("OpLoad %img3d ", spv)
                self.assertIn("OpSampledImage %simg3d ", spv)
                defs = parse_defs(spv)
                queries = find_query_lod(defs)
                self.assertEqual(len(queries), 1)
                _, (ty, si, coord) = queries[0]
                self.assertEqual(ty, "%v2float")
                self.assertEqual(defs[si][1][0], "%simg3d")
                self.assertEqual(defs[coord][0], "OpCompositeConstruct")
                self.assertEqual(defs[coord][1][0], "%v3float")
                self.assertEqual(len(defs[coord][1]), 4)  # %v3float u v w
                if a16:
                    # u and v extracted from addr[0] UnpackHalf2x16, w from addr[1] low half.
                    u_id, v_id, w_id = defs[coord][1][1:]
                    self.assertEqual((defs[u_id][0], defs[u_id][1][2]), ("OpCompositeExtract", "0"))
                    self.assertEqual((defs[v_id][0], defs[v_id][1][2]), ("OpCompositeExtract", "1"))
                    self.assertEqual((defs[w_id][0], defs[w_id][1][2]), ("OpCompositeExtract", "0"))
                    self.assertEqual(defs[u_id][1][1], defs[v_id][1][1])
                    self.assertNotEqual(defs[u_id][1][1], defs[w_id][1][1])

    def test_cube_non_a16_a16_and_vcube_sequence(self):
        for a16, vaddr in ((False, "v[0:2]"), (True, "v[0:1]")):
            with self.subTest(a16=a16):
                mod = " a16" if a16 else ""
                spv = translate_lines("ps_lod_cube", [
                    "v_mov_b32 v0, 0x3e003e00" if a16 else "v_mov_b32 v0, 1.5",
                    "v_mov_b32 v1, 0x3c00" if a16 else "v_mov_b32 v1, 1.5",
                    "v_mov_b32 v2, 1.0",
                    f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE{mod}",
                    "exp mrt0 v4, v5, v4, v5 done vm",
                    "s_endpgm",
                ])
                validate_spirv(spv)
                self.assertIn("OpCapability ImageQuery", spv)
                self.assertIn("OpAccessChain %_ptr_UC_imga %images_arr ", spv)
                self.assertIn("OpLoad %img2da ", spv)
                self.assertIn("OpSampledImage %simg2da ", spv)
                defs = parse_defs(spv)
                queries = find_query_lod(defs)
                self.assertEqual(len(queries), 1)
                _, (ty, si, coord) = queries[0]
                self.assertEqual(ty, "%v2float")
                self.assertEqual(defs[si][1][0], "%simg2da")
                self.assertEqual(defs[coord][0], "OpCompositeConstruct")
                self.assertEqual(defs[coord][1][0], "%v2float")
                # cube_coords subtracts %f1 from s and t before passing (u, v) to OpImageQueryLod.
                u_id, v_id = defs[coord][1][1:3]
                self.assertEqual(defs[u_id][0], "OpFSub")
                self.assertEqual(defs[u_id][1][2], "%f1")
                self.assertEqual(defs[v_id][0], "OpFSub")
                self.assertEqual(defs[v_id][1][2], "%f1")

        # Full v_cube* -> image_get_lod dim:SQ_RSRC_IMG_CUBE sequence.
        spv_vcube = translate_lines("ps_lod_vcube", [
            "v_mov_b32 v10, 0.5",
            "v_mov_b32 v11, -0.25",
            "v_mov_b32 v12, 1.0",
            "v_cubesc_f32 v0, v10, v11, v12",
            "v_cubetc_f32 v1, v10, v11, v12",
            "v_cubema_f32 v3, v10, v11, v12",
            "v_cubeid_f32 v2, v10, v11, v12",
            "v_rcp_f32 v3, v3",
            "v_fma_f32 v0, v0, v3, 1.5",
            "v_fma_f32 v1, v1, v3, 1.5",
            "image_get_lod v[4:5], v[0:2], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ])
        validate_spirv(spv_vcube)
        self.assertIn("OpImageQueryLod %v2float ", spv_vcube)

    def test_dmask_compaction_across_new_dims(self):
        for dim, vaddr in (
            ("SQ_RSRC_IMG_1D", "v0"),
            ("SQ_RSRC_IMG_3D", "v[0:2]"),
            ("SQ_RSRC_IMG_CUBE", "v[0:2]"),
        ):
            for dmask, expected_comps in (("0x1", [0]), ("0x2", [1]), ("0x3", [0, 1])):
                with self.subTest(dim=dim, dmask=dmask):
                    dst = "v[4:5]" if len(expected_comps) == 2 else "v4"
                    spv = translate_lines("ps_lod_dmask", [
                        "v_mov_b32 v0, 0.25",
                        "v_mov_b32 v1, 0.5",
                        "v_mov_b32 v2, 0.75",
                        f"image_get_lod {dst}, {vaddr}, s[8:15], s[16:19] dmask:{dmask} dim:{dim}",
                        "exp mrt0 v4, v5, v4, v5 done vm" if len(expected_comps) == 2 else "exp mrt0 v4, v4, v4, v4 done vm",
                        "s_endpgm",
                    ])
                    validate_spirv(spv)
                    defs = parse_defs(spv)
                    written = [(idx, unwrap_dst(defs, sid)) for idx, sid in vgpr_stores(spv) if idx >= 4]
                    self.assertEqual([idx for idx, _ in written], list(range(4, 4 + len(expected_comps))))
                    extracted = [int(defs[defs[sid][1][1]][1][2]) for _, sid in written]
                    self.assertEqual(extracted, expected_comps)

    def test_negative_guards(self):
        # Non-PS stages must be rejected for all dims.
        for prefix in ("vs_lod", "cs_lod"):
            for dim, vaddr in (("SQ_RSRC_IMG_1D", "v0"), ("SQ_RSRC_IMG_3D", "v[0:2]"), ("SQ_RSRC_IMG_CUBE", "v[0:2]")):
                with self.subTest(stage=prefix, dim=dim):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_lines(
                            prefix,
                            [
                                "v_mov_b32 v0, 0.25",
                                "v_mov_b32 v1, 0.5",
                                "v_mov_b32 v2, 0.75",
                                f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:{dim}",
                                "s_endpgm",
                            ],
                            fssm=STUB_FSSM if prefix.startswith("cs_") else None,
                        )

        # tfe, lwe, d16 rejected with and without a16.
        for flag in ("tfe", "lwe", "d16"):
            for a16, vaddr in ((False, "v[0:2]"), (True, "v[0:1]")):
                with self.subTest(flag=flag, a16=a16):
                    mod = " a16" if a16 else ""
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_lines("ps_lod_bad_flag", [
                            "v_mov_b32 v0, 0.25",
                            "v_mov_b32 v1, 0.5",
                            "v_mov_b32 v2, 0.75",
                            f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D{mod} {flag}",
                            "exp mrt0 v4, v5, v4, v5 done vm",
                            "s_endpgm",
                        ])

        # Short vaddr must raise Unsupported("image_get_lod coords") instead of IndexError.
        short_cases = [
            ("SQ_RSRC_IMG_2D", False, "v0"),
            ("SQ_RSRC_IMG_2D_ARRAY", False, "v[0:1]"),
            ("SQ_RSRC_IMG_2D_ARRAY", True, "v0"),
            ("SQ_RSRC_IMG_3D", False, "v[0:1]"),
            ("SQ_RSRC_IMG_3D", True, "v0"),
            ("SQ_RSRC_IMG_CUBE", False, "v[0:1]"),
            ("SQ_RSRC_IMG_CUBE", True, "v0"),
        ]
        for dim, a16, vaddr in short_cases:
            with self.subTest(dim=dim, a16=a16, vaddr=vaddr):
                mod = " a16" if a16 else ""
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_lines("ps_lod_short_addr", [
                        "v_mov_b32 v0, 0.25",
                        "v_mov_b32 v1, 0.5",
                        f"image_get_lod v[4:5], {vaddr}, s[8:15], s[16:19] dmask:0x3 dim:{dim}{mod}",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                        "s_endpgm",
                    ])
                self.assertIn("coords", str(ctx.exception))

        # Non-VGPR destination rejected cleanly.
        for dst in ("off", "s4"):
            with self.subTest(dst=dst):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_lod_bad_dst", [
                        "v_mov_b32 v0, 0.25",
                        f"image_get_lod {dst}, v0, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_1D",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ])

    @unittest.skipUnless(LLVM_MC, "needs llvm-mc (or LLVM_MC env var)")
    def test_llvm_mc_binary_round_trip(self):
        insns = [
            "image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_1D",
            "image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_1D a16",
            "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D",
            "image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D a16",
            "image_get_lod v[4:5], v[0:2], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY a16",
            "image_get_lod v[4:5], v[0:2], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D",
            "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_3D a16",
            "image_get_lod v[4:5], v[0:2], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE",
            "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_CUBE a16",
        ]
        asm_src = "\n".join(insns) + "\n"
        r_enc = subprocess.run(
            LLVM_MC + ["-arch=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
            input=asm_src, capture_output=True, text=True,
        )
        self.assertEqual(r_enc.returncode, 0, f"llvm-mc assemble failed:\n{r_enc.stderr}")
        hex_lines = []
        for line in r_enc.stdout.splitlines():
            m = re.search(r"encoding:\s*\[([^\]]+)\]", line)
            if m:
                hex_lines.append(" ".join(tok.strip() for tok in m.group(1).split(",")))
        self.assertEqual(len(hex_lines), len(insns))

        r_dis = subprocess.run(
            LLVM_MC + ["-disassemble", "-arch=amdgcn", "-mcpu=gfx1030"],
            input="\n".join(hex_lines) + "\n", capture_output=True, text=True,
        )
        self.assertEqual(r_dis.returncode, 0, f"llvm-mc disassemble failed:\n{r_dis.stderr}")
        dis_insns = [ln.strip() for ln in r_dis.stdout.splitlines() if "image_get_lod" in ln]
        self.assertEqual(len(dis_insns), len(insns))

        # Translate every disassembled instruction and validate with spirv-val.
        spv = translate_lines("ps_lod_roundtrip", [
            "v_mov_b32 v0, 0.25",
            "v_mov_b32 v1, 0.5",
            "v_mov_b32 v2, 0.75",
            *dis_insns,
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ])
        validate_spirv(spv)
        defs = parse_defs(spv)
        self.assertEqual(len(find_query_lod(defs)), len(insns))


if __name__ == "__main__":
    unittest.main()

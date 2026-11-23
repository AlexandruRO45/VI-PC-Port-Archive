"""Semantic and validation tests for RDNA2 image_store on GFX10_FORMAT_5_9_9_9_FLOAT (R9G9B9E5_SHAREDEXP).

Verifies:
- GFX10_FORMAT_5_9_9_9_FLOAT (132) routes dynamically to the %stv1 (uint) storage view with a packed E5B9G9R9 uint32
  word (packed, 0, 0, 0), while %stv1 holds the mutable R32_UINT view on GPUs lacking native E5B9G9R9 storage.
- Packing matches RDNA2 round-toward-zero (RTZ) hardware semantics (bit-for-bit verified against local gfx1036 GPU):
  negative / -0.0 / NaN clamp to 0 (without polluting shared exponent), +Inf / >= 65408.0 clamp to 65408.0 (E=31, m=511),
  and mantissas truncate without rounding up or bumping E at 512.
- Non-132 T# formats keep their existing paths: RGBA8_UNORM (56) -> %stv0, R32_UINT (20) -> %stv1 unpacked,
  R32_SINT (21) -> %stv2.
- Supports sparse dmask, d16, a16, glc, all STORE_DIMS (1D, 2D, 2D_ARRAY, CUBE, 3D), inactive EXEC masking, single
  shared %rgb9e5_pack_fn helper per module, spirv-val (native, --wave64-emulate, --opt, --dispatch-loop,
  --no-float-controls), and llvm-mc gfx1030 round-trip.
"""
import contextlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mc_command  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


@contextlib.contextmanager
def _emulate_r9g9b9e5(enabled: bool = True):
    old = rdna2spirv.R9G9B9E5_UAV_EMULATE
    rdna2spirv.R9G9B9E5_UAV_EMULATE = enabled
    try:
        yield
    finally:
        rdna2spirv.R9G9B9E5_UAV_EMULATE = old


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _f16_bits(v: float) -> int:
    return struct.unpack("<H", struct.pack("<e", v))[0]


def _pack_u16(lo: int, hi: int) -> int:
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def pack_rgb9e5_rtz(r_bits: int, g_bits: int, b_bits: int) -> int:
    """Reference E5B9G9R9 uint32 packer matching RDNA2 round-toward-zero semantics."""
    def clamp_bits(u: int) -> int:
        if u <= 0x7F800000:
            return min(u, 0x477F8000)
        return 0

    rc, gc, bc = clamp_bits(r_bits), clamp_bits(g_bits), clamp_bits(b_bits)
    m = max(rc, gc, bc)
    e = ((m >> 23) - 111) if m >= 0x38000000 else 0

    def quant(c: int) -> int:
        ce = c >> 23
        csig = (c & 0x7FFFFF) | 0x800000
        sh = min(126 + e - ce, 24)
        return csig >> sh

    return (e << 27) | (quant(bc) << 18) | (quant(gc) << 9) | quant(rc)


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _llvm_mc_cmd():
    if (os.environ.get("LLVM_MC") or "").strip():
        return mc_command.mc_command()
    exe = shutil.which("llvm-mc")
    return [exe] if exe else None


def _eval_pack_fn_in_mini_interp(spvasm: str, r_bits: int, g_bits: int, b_bits: int) -> int:
    """Run the translated module's %rgb9e5_pack_fn via OpFunctionCall inside spirv_mini_interp."""
    lines = spvasm.splitlines()
    pack_idx = next(i for i, ln in enumerate(lines) if ln.startswith("%rgb9e5_pack_fn = OpFunction "))
    end_idx = next(i for i in range(pack_idx + 1, len(lines)) if lines[i].strip() == "OpFunctionEnd")
    pack_fn_lines = lines[pack_idx:end_idx + 1]

    hdr = []
    skip_hdr = (
        "OpTypeImage", "OpTypeSampledImage", "OpTypeSampler", "OpTypeRuntimeArray", "UniformConstant",
        "StorageBuffer", "Private", "OpConstantComposite", "%stv", "%stimg", "%desc", "%t_desc",
        "%images", "%samplers", "%itx", "DescriptorSet", "Binding", "RuntimeDescriptorArray",
        "StorageImage", "NonUniform", "%user_data", "%UserData", "%arr_ud", "%_ptr_PC_",
        "%_ptr_Output_v4", "%out_mrt", "OpTypeFunction",
    )
    fn_idx = next(i for i, ln in enumerate(lines) if " = OpFunction " in ln)
    for ln in lines[:fn_idx]:
        if re.search(r"OpEntryPoint ", ln):
            hdr.append('OpEntryPoint Fragment %main "main" %cap_pack')
        elif any(tok in ln for tok in skip_hdr):
            continue
        else:
            hdr.append(ln)
    hdr += [
        "%fn_void = OpTypeFunction %void",
        "%fn_rgb9e5_pack = OpTypeFunction %uint %float %float %float",
        "%_ptr_Out_u = OpTypePointer Output %uint",
        "%cap_pack = OpVariable %_ptr_Out_u Output",
        "%in_r = OpConstant %uint " + str(r_bits),
        "%in_g = OpConstant %uint " + str(g_bits),
        "%in_b = OpConstant %uint " + str(b_bits),
    ]
    main_fn = [
        "%main = OpFunction %void None %fn_void",
        "%main_entry = OpLabel",
        "%fr = OpBitcast %float %in_r",
        "%fg = OpBitcast %float %in_g",
        "%fb = OpBitcast %float %in_b",
        "%res = OpFunctionCall %uint %rgb9e5_pack_fn %fr %fg %fb",
        "OpStore %cap_pack %res",
        "OpReturn",
        "OpFunctionEnd",
    ]
    mini_spv = "\n".join(hdr + pack_fn_lines + main_fn) + "\n"
    out = si.run_fragment(mini_spv)
    return out["%cap_pack"][0] & 0xFFFFFFFF


def _eval_image_store_dispatch(setup_lines: list[str], store_line: str, emulate: bool = True) -> dict[str, object]:
    """Execute the full translated image_store format-selection and packing path in spirv_mini_interp."""
    with _emulate_r9g9b9e5(emulate):
        spv = support.translate_lines(
            "ps_imgstore_r9g9b9e5_eval",
            setup_lines + [store_line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"],
        )
    _validate(spv)

    lines = spv.splitlines()
    fn_idx = next(i for i, ln in enumerate(lines) if " = OpFunction " in ln)
    main_idx = next(i for i, ln in enumerate(lines) if ln.startswith("%main = OpFunction "))
    desc_idx = next(i for i, ln in enumerate(lines) if ln.startswith("%desc_lookup_fn = OpFunction "))
    desc_end = next(i for i in range(desc_idx + 1, len(lines)) if lines[i].strip() == "OpFunctionEnd")

    skip_hdr = (
        "OpTypeImage", "OpTypeSampledImage", "OpTypeSampler", "OpTypeRuntimeArray", "UniformConstant",
        "StorageBuffer", "%stv", "%stimg", "%desc", "%t_desc", "%images", "%samplers", "%itx",
        "DescriptorSet", "Binding", "RuntimeDescriptorArray", "StorageImage", "NonUniform",
        "%fn_desc_lookup", "%user_data", "%UserData", "%arr_ud", "%_ptr_PC_",
    )
    hdr = []
    for ln in lines[:fn_idx]:
        if re.search(r"OpEntryPoint ", ln):
            hdr.append('OpEntryPoint Fragment %main "main" %cap_wrote %cap_cls %cap_texel')
        elif any(tok in ln for tok in skip_hdr):
            continue
        else:
            hdr.append(ln)
    hdr += [
        "%_ptr_Out_u = OpTypePointer Output %uint",
        "%_ptr_Out_vu = OpTypePointer Output %v4uint_st",
        "%cap_wrote = OpVariable %_ptr_Out_u Output",
        "%cap_cls = OpVariable %_ptr_Out_u Output",
        "%cap_texel = OpVariable %_ptr_Out_vu Output",
        "%cap_one = OpConstant %uint 1",
        "%cap_two = OpConstant %uint 2",
    ]

    # Helpers between desc_fn and %main (keep %rgb9e5_pack_fn, drop %desc_lookup_fn).
    mid_helpers = lines[desc_end + 1:main_idx]
    body_lines = list(lines[main_idx:])

    # Rewrite descriptor lookup call + broadcast to %c0, and capture OpImageWrite class + texel words.
    rewritten = []
    skip_next_bcast = False
    img_cls = {}
    i = 0
    while i < len(body_lines):
        ln = body_lines[i]
        s = ln.strip()
        if "OpFunctionCall %uint %desc_lookup_fn" in s:
            lhs = s.split("=")[0].strip()
            rewritten.append(f"{lhs} = OpCopyObject %uint %c0")
            skip_next_bcast = True
            i += 1
            continue
        if skip_next_bcast and "OpGroupNonUniformBroadcastFirst" in s:
            lhs = s.split("=")[0].strip()
            rewritten.append(f"{lhs} = OpCopyObject %uint %c0")
            skip_next_bcast = False
            i += 1
            continue
        if "OpAccessChain %_ptr_UC_stv" in s or "OpGroupNonUniformBroadcastFirst" in s:
            i += 1
            continue
        m_load = re.match(r"\s*(%\w+)\s*=\s*OpLoad\s+%stimg(\d+)_\d+\s+", ln)
        if m_load:
            img_cls[m_load.group(1)] = int(m_load.group(2))
            i += 1
            continue
        m_write = re.match(r"\s*OpImageWrite\s+(%\w+)\s+(%\w+)\s+(%\w+)$", s)
        if m_write:
            cls = img_cls[m_write.group(1)]
            texel = m_write.group(3)
            cls_c = {0: "%c0", 1: "%cap_one", 2: "%cap_two"}[cls]
            rewritten.append("OpStore %cap_wrote %cap_one")
            rewritten.append(f"OpStore %cap_cls {cls_c}")
            if cls == 1:
                rewritten.append(f"OpStore %cap_texel {texel}")
            else:
                bc = f"%cap_bc_{i}"
                rewritten.append(f"{bc} = OpBitcast %v4uint_st {texel}")
                rewritten.append(f"OpStore %cap_texel {bc}")
            i += 1
            continue
        if "OpAccessChain %_ptr_PC_uint %user_data" in s or "OpStore %sg" in s and "%ud" in s:
            i += 1
            continue
        if re.search(r"%\w+\s*=\s*OpLoad\s+%uint\s+%ud", s):
            lhs = s.split("=")[0].strip()
            rewritten.append(f"{lhs} = OpCopyObject %uint %c0")
            i += 1
            continue
        if s.startswith("OpKill"):
            rewritten.append("OpReturn")
            i += 1
            continue
        rewritten.append(ln)
        i += 1

    assert len(img_cls) == 3, f"expected 3 OpImageWrite class paths, got {img_cls}"
    mini_spv = "\n".join(hdr + mid_helpers + rewritten) + "\n"
    _, inv_outs = si.run_workgroup(
        mini_spv,
        invocations=1,
        subgroup_size=32,
        inputs_for=lambda _i: {"%sg_lane": 0},
        return_outputs=True,
    )
    out = inv_outs[0]
    return {
        "wrote": out["%cap_wrote"][0] & 0xFFFFFFFF,
        "cls": out["%cap_cls"][0] & 0xFFFFFFFF,
        "texel": tuple(x & 0xFFFFFFFF for x in out["%cap_texel"]),
    }


def _tsharp_setup(reg_base: int, fmt_code: int) -> list[str]:
    w1 = (fmt_code & 0x1FF) << 20
    lines = []
    for i in range(8):
        val = w1 if i == 1 else 0
        lines.append(f"s_mov_b32 s{reg_base + i}, 0x{val:08x}")
    return lines


class ImageStoreR9G9B9E5Tests(unittest.TestCase):
    # Measured native AMD RDNA2 (gfx1036) hardware words from VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 compute imageStore:
    GFX1036_MEASURED = [
        ((1.0, 2.0, 4.0), 0x94010040),
        ((1.0 + 0.25 / 256.0, 1.0 + 0.5 / 256.0, 1.0 + 1.5 / 256.0), 0x84060100),
        ((1.0 + 0.75 / 256.0, 1.0 + 1.75 / 256.0, 0.0), 0x80020300),
        ((511.0 / 512.0, 511.25 / 512.0, 511.5 / 512.0), 0x7FFFFFFF),
        ((511.75 / 512.0, 0x3F7FFFFF, 1.0), 0x8401FEFF),
        ((0.5 * 2**-24, 1.5 * 2**-24, 511.5 * 2**-24), 0x07FC0200),
        ((511.75 * 2**-24, 512.0 * 2**-24, 2**-30), 0x080200FF),
        ((-1.0, 0x80000000, 0.5), 0x7C000000),
        ((0x7F800000, 0xFF800000, 0x7FC00001), 0xF80001FF),
        ((0xFFC00001, 0.0, 0.0), 0x00000000),
        ((65408.0, 65409.0, 65536.0), 0xFFFFFFFF),
        ((65472.0, 1e20, 0.0), 0xF803FFFF),
        ((0.0, 0.0, 0.0), 0x00000000),
        ((2**-10, 1.0, 65408.0), 0xFFFC0000),
        ((65344.0, 65407.0, 65408.0), 0xFFFFFDFE),
        ((256.5 / 512.0, 257.5 / 512.0, 510.5 / 512.0), 0x7FFA0300),
        ((0x7FC00001, 1.0, 2.0), 0x8C010000),
        ((0x7F800001, 0.5, 0.25), 0x7A020000),
        ((0x00000001, 0x007FFFFF, 0x00800000), 0x00000000),
        ((0x00000001, 1.0, 0.0), 0x80020000),
        ((65407.996, 65407.0, 65344.0), 0xFFFBFDFE),
        ((-1e30, 0.25, 0.5), 0x7C010000),
        ((32768.0, 65280.0, 65408.0), 0xFFFFFD00),
        ((65504.0, 65408.0, 65472.0), 0xFFFFFFFF),
        ((1.0 * 2**-24, 1.999 * 2**-24, 511.99 * 2**-24), 0x07FC0201),
        ((255.99 * 2**-24, 256.0 * 2**-24, 256.99 * 2**-24), 0x040200FF),
        ((0x3F7FFFFF, 0x3F800000, 0x3F800001), 0x840200FF),
        ((0x80000001, 0x33800000, 0x33FFFFFF), 0x00040200),
    ]

    def test_pack_helper_matches_rtz_and_gfx1036_across_edge_cases(self):
        with _emulate_r9g9b9e5(True):
            spv = support.translate_lines(
                "ps_rgb9e5_base",
                [
                    "v_mov_b32_e32 v0, 0",
                    "v_mov_b32_e32 v1, 0",
                    "v_mov_b32_e32 v2, 0",
                    "v_mov_b32_e32 v3, 0",
                    "v_mov_b32_e32 v4, 0",
                    "v_mov_b32_e32 v5, 0",
                    "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
            )
        _validate(spv)
        for vals, want in self.GFX1036_MEASURED:
            bits = tuple(v if isinstance(v, int) else _f32_bits(v) for v in vals)
            with self.subTest(bits=tuple(hex(b) for b in bits)):
                self.assertEqual(pack_rgb9e5_rtz(*bits), want)
                self.assertEqual(_eval_pack_fn_in_mini_interp(spv, *bits), want)

        # Sweep all E = 0..31 plus signaling/quiet negative/positive NaNs and subnormals.
        extra = [
            (0x00000000, 0x80000000, 0x80000001),
            (0xFF800000, 0xFFC00000, 0xFF800001),
            (0x7F800000, 0x7F800001, 0x7FFFFFFF),
        ]
        for exp_unbiased in range(-24, 17):
            x = math.ldexp(1.25, exp_unbiased)
            extra.append((_f32_bits(x), _f32_bits(0.5 * x), _f32_bits(0.0)))
        for rb, gb, bb in extra:
            with self.subTest(rb=hex(rb), gb=hex(gb), bb=hex(bb)):
                self.assertEqual(
                    _eval_pack_fn_in_mini_interp(spv, rb, gb, bb),
                    pack_rgb9e5_rtz(rb, gb, bb),
                )

    def test_runtime_tsharp_dispatch_routes_132_vs_norm_uint_sint(self):
        r, g, b, a = 1.0, 2.0, 4.0, 99.0
        want_e5 = pack_rgb9e5_rtz(_f32_bits(r), _f32_bits(g), _f32_bits(b))
        setup_regs = [
            f"v_mov_b32_e32 v0, 0x{_f32_bits(r):08x}",
            f"v_mov_b32_e32 v1, 0x{_f32_bits(g):08x}",
            f"v_mov_b32_e32 v2, 0x{_f32_bits(b):08x}",
            f"v_mov_b32_e32 v3, 0x{_f32_bits(a):08x}",
            "v_mov_b32_e32 v4, 3",
            "v_mov_b32_e32 v5, 5",
        ]
        store = "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"

        # FORMAT 132 (5_9_9_9_FLOAT) -> stv1 (cls=1) with (packed_e5, 0, 0, 0).
        res_132 = _eval_image_store_dispatch(_tsharp_setup(8, 132) + setup_regs, store)
        self.assertEqual((res_132["wrote"], res_132["cls"], res_132["texel"]), (1, 1, (want_e5, 0, 0, 0)))

        # FORMAT 56 (8_8_8_8_UNORM) -> stv0 (cls=0) with original floats.
        res_56 = _eval_image_store_dispatch(_tsharp_setup(8, 56) + setup_regs, store)
        self.assertEqual(
            (res_56["wrote"], res_56["cls"], res_56["texel"]),
            (1, 0, (_f32_bits(r), _f32_bits(g), _f32_bits(b), _f32_bits(a))),
        )

        # FORMAT 20 (32_UINT) -> stv1 (cls=1) WITHOUT RGB9E5 packing.
        res_20 = _eval_image_store_dispatch(_tsharp_setup(8, 20) + setup_regs, store)
        self.assertEqual(
            (res_20["wrote"], res_20["cls"], res_20["texel"]),
            (1, 1, (_f32_bits(r), _f32_bits(g), _f32_bits(b), _f32_bits(a))),
        )

        # FORMAT 21 (32_SINT) -> stv2 (cls=2).
        res_21 = _eval_image_store_dispatch(_tsharp_setup(8, 21) + setup_regs, store)
        self.assertEqual(
            (res_21["wrote"], res_21["cls"], res_21["texel"]),
            (1, 2, (_f32_bits(r), _f32_bits(g), _f32_bits(b), _f32_bits(a))),
        )

        # Inactive lane (EXEC = 0) skips store and leaves outputs untouched.
        res_inactive = _eval_image_store_dispatch(["s_mov_b64 exec, 0"] + _tsharp_setup(8, 132) + setup_regs, store)
        self.assertEqual(res_inactive["wrote"], 0)

        # Native mode (R9G9B9E5_UAV_EMULATE=False, default): FORMAT 132 routes to %stv0 (cls=0) with zero pack helper,
        # so GPUs with native VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 storage never write uint bits into float storage views.
        res_native = _eval_image_store_dispatch(_tsharp_setup(8, 132) + setup_regs, store, emulate=False)
        self.assertEqual(
            (res_native["wrote"], res_native["cls"], res_native["texel"]),
            (1, 0, (_f32_bits(r), _f32_bits(g), _f32_bits(b), _f32_bits(a))),
        )
        native_spv = support.translate_lines("ps_rgb9e5_native", setup_regs + [store, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertNotIn("%rgb9e5_pack_fn", native_spv)

    def test_dmask_and_d16_interactions_for_r9g9b9e5(self):
        # Sparse dmask:0x5 (R and B only; G defaults to 0.0).
        res_rb = _eval_image_store_dispatch(
            _tsharp_setup(8, 132) + [
                f"v_mov_b32_e32 v0, 0x{_f32_bits(1.0):08x}",
                f"v_mov_b32_e32 v1, 0x{_f32_bits(4.0):08x}",
                "v_mov_b32_e32 v4, 0",
                "v_mov_b32_e32 v5, 0",
            ],
            "image_store v[0:1], v[4:5], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm",
        )
        self.assertEqual(res_rb["cls"], 1)
        self.assertEqual(res_rb["texel"], (pack_rgb9e5_rtz(_f32_bits(1.0), 0, _f32_bits(4.0)), 0, 0, 0))

        # dmask:0x8 (Alpha only; RGB are all missing = 0.0 -> packed 0x00000000).
        res_a = _eval_image_store_dispatch(
            _tsharp_setup(8, 132) + [
                f"v_mov_b32_e32 v0, 0x{_f32_bits(5.0):08x}",
                "v_mov_b32_e32 v4, 0",
                "v_mov_b32_e32 v5, 0",
            ],
            "image_store v0, v[4:5], s[8:15] dmask:0x8 dim:SQ_RSRC_IMG_2D unorm",
        )
        self.assertEqual(res_a["cls"], 1)
        self.assertEqual(res_a["texel"], (0, 0, 0, 0))

        # d16 + a16 + dmask:0x7: f16 (1.5, -2.25, 4.0) widened via UnpackHalf2x16 -> (1.5, 0, 4.0).
        w0 = _pack_u16(_f16_bits(1.5), _f16_bits(-2.25))
        w1 = _pack_u16(_f16_bits(4.0), 0x7FFF)
        res_d16 = _eval_image_store_dispatch(
            _tsharp_setup(8, 132) + [
                f"v_mov_b32_e32 v0, 0x{w0:08x}",
                f"v_mov_b32_e32 v1, 0x{w1:08x}",
                f"v_mov_b32_e32 v4, 0x{_pack_u16(2, 7):08x}",
            ],
            "image_store v[0:1], v4, s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm a16 d16",
        )
        self.assertEqual(res_d16["cls"], 1)
        self.assertEqual(
            res_d16["texel"],
            (pack_rgb9e5_rtz(_f32_bits(1.5), _f32_bits(-2.25), _f32_bits(4.0)), 0, 0, 0),
        )

    def test_single_pack_helper_and_no_bloat_when_unused(self):
        with _emulate_r9g9b9e5(True):
            no_store = support.translate_lines("ps_no_store", ["v_mov_b32_e32 v0, 0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertNotIn("%rgb9e5_pack_fn", no_store)
        self.assertNotIn("%fn_rgb9e5_pack", no_store)

        with _emulate_r9g9b9e5(True):
            two_stores = support.translate_lines(
                "ps_two_stores",
                [
                    "v_mov_b32_e32 v0, 0",
                    "v_mov_b32_e32 v4, 0",
                    "v_mov_b32_e32 v5, 0",
                    "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                    "image_store v[0:3], v[4:5], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                    "exp mrt0 v0, v0, v0, v0 done vm",
                    "s_endpgm",
                ],
            )
        _validate(two_stores)
        self.assertEqual(two_stores.count("%rgb9e5_pack_fn = OpFunction "), 1)
        self.assertEqual(two_stores.count("OpFunctionCall %uint %rgb9e5_pack_fn"), 2)

    def test_all_dims_and_translator_modes_validate(self):
        dims = (
            ("1D", "v4"),
            ("2D", "v[4:5]"),
            ("2D_ARRAY", "v[4:6]"),
            ("CUBE", "v[4:6]"),
            ("3D", "v[4:6]"),
        )
        with _emulate_r9g9b9e5(True):
            for dim, addr in dims:
                for extra in ("", "glc", "a16 d16", "glc a16 d16"):
                    addr_use = "v4" if ("a16" in extra and dim != "1D" and dim == "2D") else ("v[4:5]" if ("a16" in extra and dim != "1D") else addr)
                    with self.subTest(dim=dim, extra=extra):
                        spv = support.translate_lines(
                            f"ps_e5_{dim.lower()}",
                            [
                                "v_mov_b32_e32 v0, 0",
                                "v_mov_b32_e32 v1, 0",
                                "v_mov_b32_e32 v4, 0",
                                "v_mov_b32_e32 v5, 0",
                                "v_mov_b32_e32 v6, 0",
                                f"image_store v[0:1], {addr_use}, s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_{dim} unorm {extra}",
                                "exp mrt0 v0, v0, v0, v0 done vm",
                                "s_endpgm",
                            ],
                        )
                        _validate(spv)

        # Validate --wave64-emulate, --opt, --dispatch-loop, --no-float-controls on compute shader.
        for opt_flag, dl_flag, nfc_flag in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ):
            with self.subTest(opt=opt_flag, dispatch_loop=dl_flag, no_float_controls=nfc_flag):
                with tempfile.TemporaryDirectory() as d:
                    os.makedirs(os.path.join(d, "disasm"))
                    os.makedirs(os.path.join(d, "fspx"))
                    s_path = os.path.join(d, "disasm", "cs_e5.s")
                    with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write("; cs_e5  synthetic\n")
                        for ln in (
                            "s_mov_b64 s[16:17], exec",
                            "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                            "s_endpgm",
                        ):
                            f.write(f"\t{ln} ; encoding: [0x00,0x00,0x00,0x00]\n")
                    import json
                    with open(os.path.join(d, "fspx", "cs_e5.json"), "w", encoding="utf-8", newline="\n") as f:
                        json.dump({"rsrc2": (16 << 1), "threads": [64, 1, 1]}, f)
                    old = (rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV,
                           rdna2spirv.STRUCTURED, rdna2spirv.FLOAT_CONTROLS,
                           rdna2spirv.R9G9B9E5_UAV_EMULATE)
                    try:
                        rdna2spirv.WAVE64_EMULATE = True
                        rdna2spirv.OPT_SPIRV = opt_flag
                        rdna2spirv.STRUCTURED = not dl_flag
                        rdna2spirv.FLOAT_CONTROLS = not nfc_flag
                        rdna2spirv.R9G9B9E5_UAV_EMULATE = True
                        spv = rdna2spirv.translate(s_path)
                    finally:
                        (rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV,
                         rdna2spirv.STRUCTURED, rdna2spirv.FLOAT_CONTROLS,
                         rdna2spirv.R9G9B9E5_UAV_EMULATE) = old
                _validate(spv)

    def test_cli_flag_toggles_emulation(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = os.path.join(d, "disasm")
            out_off = os.path.join(d, "out_off")
            out_on = os.path.join(d, "out_on")
            os.makedirs(src_dir)
            s_path = os.path.join(src_dir, "ps_cli_e5.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; ps_cli_e5\n\tv_mov_b32_e32 v0, 0\n\timage_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
            with open(os.path.join(d, "ps_cli_e5.dxbc"), "wb") as f:
                f.write(support.STUB_DXBC)
            translator = os.path.join(TOOLS, "rdna2spirv.py")
            for extra, out_dir, want_pack in (([], out_off, False), (["--r9g9b9e5-uav-emulate"], out_on, True)):
                r = subprocess.run([sys.executable, "-I", translator, src_dir, "--out", out_dir, *extra],
                                   capture_output=True, text=True, check=False)
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
                with open(os.path.join(out_dir, "ps_cli_e5.spvasm"), encoding="utf-8") as f:
                    spvasm = f.read()
                if want_pack:
                    self.assertIn("%rgb9e5_pack_fn", spvasm)
                else:
                    self.assertNotIn("%rgb9e5_pack_fn", spvasm)

    @unittest.skipUnless(_llvm_mc_cmd() is not None, "needs llvm-mc or LLVM_MC")
    def test_llvm_mc_roundtrip(self):
        mc = _llvm_mc_cmd()
        insns = [
            "image_store v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "image_store v[0:1], v4, s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm glc a16 d16",
        ]
        enc = subprocess.run(mc + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                             input="\n".join(insns) + "\n", capture_output=True, text=True, check=False)
        self.assertEqual(enc.returncode, 0, enc.stderr + enc.stdout)
        byte_lines = re.findall(r"encoding:\s*\[([^\]]+)\]", enc.stdout)
        hex_in = "\n".join(b.replace(",", " ") for b in byte_lines) + "\n"
        dec = subprocess.run(mc + ["-triple=amdgcn", "-mcpu=gfx1030", "--disassemble"],
                             input=hex_in, capture_output=True, text=True, check=False)
        self.assertEqual(dec.returncode, 0, dec.stderr + dec.stdout)
        decoded = [ln.strip() for ln in dec.stdout.splitlines() if ln.strip() and not ln.strip().startswith(".")]
        self.assertEqual(len(decoded), len(insns))
        with _emulate_r9g9b9e5(True):
            for insn in decoded:
                spv = support.translate_lines("ps_e5_mc", [insn, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
                _validate(spv)

    def test_llvm_mc_command_preserves_windows_backslashes(self):
        win_exe = r"C:\Program Files\LLVM\bin\llvm-mc.exe"
        with mock.patch.dict(os.environ, {"LLVM_MC": win_exe}, clear=False), \
                mock.patch("mc_command.os.path.isfile", return_value=True):
            self.assertEqual(_llvm_mc_cmd(), [win_exe])
        quoted_cmd = r'"C:\LLVM\bin\llvm-mc.exe" -show-encoding'
        self.assertEqual(mc_command.split_windows_command(quoted_cmd),
                         [r"C:\LLVM\bin\llvm-mc.exe", "-show-encoding"])


if __name__ == "__main__":
    unittest.main()

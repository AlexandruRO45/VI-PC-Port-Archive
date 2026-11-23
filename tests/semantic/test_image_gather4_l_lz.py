"""Semantic and structural translator tests for image_gather4_l, image_gather4_l_o,
image_gather4_lz, and image_gather4_lz_o.

Covers:
  - Signed 6-bit texel offset extraction (x in [5:0], y in [13:8]) for image_gather4_l_o
    and image_gather4_lz_o, including negative offsets, bit-4 (+16) values, and ignored bits >= 6.
  - Explicit-LOD 2x2 footprint evaluation for image_gather4_l and image_gather4_l_o:
    S# word 1 MIN_LOD/MAX_LOD clamping (literal and runtime), ties-up LOD rounding
    (floor + (frac >= 0.5)), mip clamping to [0, levels - 1], 2x2 gather tap order
    (u0,v1), (u1,v1), (u1,v0), (u0,v0), negative pre-floor coordinates (st - 0.5 < 0),
    S# address modes (wrap, mirror, clamp, mirror-once, border modes 6/7), and literal
    plus runtime S# word 3 border colors.
  - Zero-LOD hardware gather translation for image_gather4_lz and image_gather4_lz_o across
    2D (typed float/uint/sint wrapper), 2D_ARRAY, and CUBE, plus a16 and d16 modifiers.
  - Rejection (Unsupported) of invalid dmask, 3D dimensions, a16 with tfe/lwe, and literal S#
    FORCE_UNNORMALIZED, half-border address modes (4, 5), and border-color table (type 3).

Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest

SEMANTIC_DIR = os.path.dirname(os.path.abspath(__file__))
if SEMANTIC_DIR not in sys.path:
    sys.path.insert(0, SEMANTIC_DIR)

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _u32_to_f32(w: int) -> float:
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def _s32(w: int) -> int:
    w &= 0xFFFFFFFF
    return w - 0x100000000 if w & 0x80000000 else w


def _pack_offset(dx: int, dy: int, extra_bits: int = 0) -> int:
    return ((dx & 0x3F) | ((dy & 0x3F) << 8) | extra_bits) & 0xFFFFFFFF


def _ps_program(
    inst: str,
    *,
    v_inits: dict[int, int],
    s0_lit: int | None = 0x00000012,
    s1_lit: int | None = 0x00FFF000,
    s3_lit: int | None = 0x00000000,
) -> list[str]:
    lines: list[str] = []
    if s0_lit is not None:
        lines.append(f"s_mov_b32 s16, {hex(s0_lit)}")
    if s1_lit is not None:
        lines.append(f"s_mov_b32 s17, {hex(s1_lit)}")
    lines.append("s_mov_b32 s18, 0")
    if s3_lit is not None:
        lines.append(f"s_mov_b32 s19, {hex(s3_lit)}")
    for reg, val in sorted(v_inits.items()):
        lines.append(f"v_mov_b32 v{reg}, {hex(val & 0xFFFFFFFF)}")
    lines += [
        inst,
        "s_waitcnt vmcnt(0)",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    return lines


class GatherSpirvEvaluator:
    """Evaluates the SSA DAG emitted for image_gather4_l[_o] and image_gather4_lz[_o]."""

    def __init__(
        self,
        spvasm: str,
        *,
        levels: int = 4,
        mip_sizes: dict[int, tuple[int, int]] | None = None,
        texel_fn=None,
        sgpr_words: dict[int, int] | None = None,
    ) -> None:
        self.spvasm = spvasm
        self.levels = levels
        self.mip_sizes = mip_sizes or {0: (8, 8), 1: (4, 4), 2: (2, 2), 3: (1, 1)}
        self.texel_fn = texel_fn or (lambda mip, x, y: (
            float(mip * 100 + x * 10 + y + 1),
            float(mip * 100 + x * 10 + y + 2),
            float(mip * 100 + x * 10 + y + 3),
            float(mip * 100 + x * 10 + y + 4),
        ))
        self.sgpr_words = sgpr_words or {}
        self.defs: dict[str, tuple[str, str, list[str]]] = {}
        self.types: dict[str, str] = {}
        self.vgpr_ptrs: dict[str, int] = {}
        self.sgpr_ptrs: dict[str, int] = {}
        self.vgprs: dict[int, int] = {}
        self.fetches: list[tuple[int, int, int]] = []
        self.gathers: list[dict[str, object]] = []
        self._parse()

    def _parse(self) -> None:
        # First pass: collect constants and types
        consts: dict[str, int | float] = {}
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            m = re.match(r"^(%\w+)\s*=\s*(\w+)\s+(.+)$", line)
            if not m:
                continue
            lhs, op, rest = m.groups()
            tokens = rest.split()
            if op == "OpConstant":
                ty, lit = tokens[0], tokens[1]
                self.types[lhs] = ty
                if ty == "%float":
                    consts[lhs] = _f32(float(lit))
                elif ty == "%int":
                    consts[lhs] = _s32(int(lit, 0))
                else:
                    consts[lhs] = int(lit, 0) & 0xFFFFFFFF
            elif op == "OpConstantFalse":
                self.types[lhs] = "%bool"
                consts[lhs] = False
            elif op == "OpConstantTrue":
                self.types[lhs] = "%bool"
                consts[lhs] = True
            self.defs[lhs] = (op, tokens[0] if tokens else "", tokens[1:] if len(tokens) > 1 else [])

        # Execute in program order inside %main (after %entry = OpLabel) and inside the float
        # branch (skipping the _s SINT and _u UINT branches) so VGPR stores and loads work sequentially.
        self.env: dict[str, object] = dict(consts)
        in_main = False
        in_uint_or_sint_branch = False
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if re.match(r"^%\w+\s*=\s*OpLabel$", line):
                lbl = line.split()[0]
                if lbl == "%entry":
                    in_main = True
                elif lbl.endswith(("_s", "_u")):
                    in_uint_or_sint_branch = True
                elif lbl.endswith(("_f", "_e3", "_e2")):
                    in_uint_or_sint_branch = False
                continue
            if not in_main or in_uint_or_sint_branch:
                continue
            m_store = re.match(r"^OpStore\s+(%\w+)\s+(%\w+)$", line)
            if m_store:
                ptr, val_id = m_store.groups()
                if ptr in self.vgpr_ptrs and val_id in self.env:
                    v = self.env[val_id]
                    if isinstance(v, float):
                        v = _f32_bits(v)
                    self.vgprs[self.vgpr_ptrs[ptr]] = int(v) & 0xFFFFFFFF
                elif ptr in self.sgpr_ptrs and not ptr.startswith("%ud") and val_id in self.env:
                    v = self.env[val_id]
                    self.sgpr_words[self.sgpr_ptrs[ptr]] = int(v) & 0xFFFFFFFF
                continue
            m = re.match(r"^(%\w+)\s*=\s*(\w+)\s+(.+)$", line)
            if not m:
                continue
            lhs, op, rest = m.groups()
            if lhs in self.env:
                continue
            tokens = rest.split()
            ty = tokens[0]
            args = tokens[1:]
            self._exec_inst(lhs, op, ty, args)

    def _require_type(self, val_id: str, expected: str | tuple[str, ...], context: str) -> None:
        actual = self.types.get(val_id)
        allowed = (expected,) if isinstance(expected, str) else expected
        if actual not in allowed:
            raise AssertionError(f"{context}: operand {val_id} has type {actual!r}, expected {allowed!r}")

    def _exec_inst(self, lhs: str, op: str, ty: str, args: list[str]) -> None:
        if op == "OpAccessChain":
            base = args[0]
            idx_id = args[1] if len(args) > 1 else None
            idx = self.env.get(idx_id) if idx_id else None
            if base == "%vgpr" and isinstance(idx, int):
                self.vgpr_ptrs[lhs] = idx
            elif base == "%sgpr" and isinstance(idx, int):
                self.sgpr_ptrs[lhs] = idx
            self.types[lhs] = ty
            return
        if op == "OpLoad":
            ptr = args[0]
            if ptr in self.vgpr_ptrs:
                self.env[lhs] = self.vgprs.get(self.vgpr_ptrs[ptr], 0)
            elif ptr in self.sgpr_ptrs:
                self.env[lhs] = self.sgpr_words.get(self.sgpr_ptrs[ptr], 0) & 0xFFFFFFFF
            elif ty.startswith(("%img", "%itximg")) or ty == "%smp":
                self.env[lhs] = f"image:{ptr}"
            elif ty.startswith("%v4"):
                self.env[lhs] = (0.0, 0.0, 0.0, 0.0)
            else:
                self.env[lhs] = 0
            self.types[lhs] = ty
            return
        if op == "OpFunctionCall":
            self.env[lhs] = 1
            self.types[lhs] = ty
            return
        if op == "OpGroupNonUniformBallot":
            self.env[lhs] = (1, 0, 0, 0)
            self.types[lhs] = ty
            return
        if op == "OpGroupNonUniformShuffle":
            self.env[lhs] = self.env.get(args[1], 1)
            self.types[lhs] = ty
            return
        if op in ("OpBitwiseAnd", "OpBitwiseOr", "OpBitwiseXor"):
            a = int(self.env.get(args[0], 0)) & 0xFFFFFFFF
            b = int(self.env.get(args[1], 0)) & 0xFFFFFFFF
            self.env[lhs] = (a & b) if op == "OpBitwiseAnd" else ((a | b) if op == "OpBitwiseOr" else (a ^ b))
            self.types[lhs] = ty
            return
        if op == "OpVectorShuffle":
            v0 = self.env[args[0]]
            v1 = self.env[args[1]]
            comb = tuple(v0) + tuple(v1)
            self.env[lhs] = tuple(comb[int(i)] for i in args[2:])
            self.types[lhs] = ty
            return
        if op == "OpSampledImage":
            self.env[lhs] = ("sampled_image", args[0], args[1])
            self.types[lhs] = ty
            return
        if op == "OpBitcast":
            src = args[0]
            if src not in self.env:
                return
            v = self.env[src]
            self.types[lhs] = ty
            if ty == "%float":
                self.env[lhs] = _u32_to_f32(int(v))
            elif ty == "%int":
                if isinstance(v, float):
                    self.env[lhs] = _s32(_f32_bits(v))
                else:
                    self.env[lhs] = _s32(int(v))
            elif ty == "%uint":
                if isinstance(v, float):
                    self.env[lhs] = _f32_bits(v)
                else:
                    self.env[lhs] = int(v) & 0xFFFFFFFF
            elif ty in ("%v4uint", "%v4uint_st") and isinstance(v, tuple):
                self.env[lhs] = tuple(_f32_bits(x) if isinstance(x, float) else (int(x) & 0xFFFFFFFF) for x in v)
            return
        if op == "OpBitFieldSExtract":
            self._require_type(args[0], ("%uint", "%int"), op)
            base = int(self.env[args[0]]) & 0xFFFFFFFF
            off = int(self.env[args[1]])
            cnt = int(self.env[args[2]])
            mask = (1 << cnt) - 1
            raw = (base >> off) & mask
            if cnt > 0 and (raw & (1 << (cnt - 1))):
                raw -= 1 << cnt
            self.env[lhs] = raw & 0xFFFFFFFF if ty == "%uint" else raw
            self.types[lhs] = ty
            return
        if op == "OpBitFieldUExtract":
            self._require_type(args[0], ("%uint", "%int"), op)
            base = int(self.env[args[0]]) & 0xFFFFFFFF
            off = int(self.env[args[1]])
            cnt = int(self.env[args[2]])
            mask = (1 << cnt) - 1
            self.env[lhs] = (base >> off) & mask
            self.types[lhs] = ty
            return
        if op == "OpCompositeConstruct":
            for a in args:
                if a not in self.env:
                    return
            self.env[lhs] = tuple(self.env[a] for a in args)
            self.types[lhs] = ty
            return
        if op == "OpCompositeExtract":
            src = args[0]
            if src not in self.env or not isinstance(self.env[src], tuple):
                raise AssertionError(f"OpCompositeExtract source {src} invalid or not a tuple in {args}")
            if not re.match(r"^\d+$", args[1]):
                raise AssertionError(f"OpCompositeExtract index must be an integer literal, got {args[1]!r}")
            idx = int(args[1])
            self.env[lhs] = self.env[src][idx]
            self.types[lhs] = ty
            return
        if op == "OpImageQueryLevels":
            self.env[lhs] = self.levels
            self.types[lhs] = ty
            return
        if op == "OpImageQuerySizeLod":
            self._require_type(args[1], "%int", op)
            lod = int(self.env[args[1]])
            w, h = self.mip_sizes.get(lod, (1, 1))
            self.env[lhs] = (w, h) if ty == "%v2int" else (w, h, 1)
            self.types[lhs] = ty
            return
        if op == "OpConvertFToS":
            if ty not in ("%int", "%v2int"):
                raise AssertionError(f"OpConvertFToS requires signed int result type, got {ty}")
            self._require_type(args[0], ("%float", "%v2float"), op)
            v = self.env[args[0]]
            self.env[lhs] = tuple(int(x) for x in v) if isinstance(v, tuple) else int(v)
            self.types[lhs] = ty
            return
        if op == "OpConvertFToU":
            if ty not in ("%uint", "%v2uint"):
                raise AssertionError(f"OpConvertFToU requires unsigned int result type, got {ty}")
            self._require_type(args[0], ("%float", "%v2float"), op)
            v = self.env[args[0]]
            self.env[lhs] = tuple(int(x) & 0xFFFFFFFF for x in v) if isinstance(v, tuple) else (int(v) & 0xFFFFFFFF)
            self.types[lhs] = ty
            return
        if op == "OpConvertSToF":
            self._require_type(args[0], ("%int", "%v2int"), op)
            v = self.env[args[0]]
            self.env[lhs] = tuple(_f32(float(x)) for x in v) if isinstance(v, tuple) else _f32(float(v))
            self.types[lhs] = ty
            return
        if op == "OpConvertUToF":
            self._require_type(args[0], ("%uint", "%v2uint"), op)
            v = self.env[args[0]]
            self.env[lhs] = tuple(_f32(float(x & 0xFFFFFFFF)) for x in v) if isinstance(v, tuple) else _f32(float(int(v) & 0xFFFFFFFF))
            self.types[lhs] = ty
            return
        if op in ("OpFAdd", "OpFSub", "OpFMul"):
            self._require_type(args[0], ("%float", "%v2float"), op)
            self._require_type(args[1], ("%float", "%v2float"), op)
            a, b = self.env[args[0]], self.env[args[1]]
            fn = {
                "OpFAdd": lambda x, y: _f32(x + y),
                "OpFSub": lambda x, y: _f32(x - y),
                "OpFMul": lambda x, y: _f32(x * y),
            }[op]
            if isinstance(a, tuple):
                self.env[lhs] = tuple(fn(x, y) for x, y in zip(a, b))
            else:
                self.env[lhs] = fn(a, b)
            self.types[lhs] = ty
            return
        if op in ("OpIAdd", "OpISub", "OpSRem"):
            self._require_type(args[0], ("%int", "%uint", "%v2int"), op)
            self._require_type(args[1], ("%int", "%uint", "%v2int"), op)
            a, b = self.env[args[0]], self.env[args[1]]

            def _srem(x: int, y: int) -> int:
                q = int(math.trunc(x / y))
                return x - q * y

            fn = {
                "OpIAdd": lambda x, y: x + y,
                "OpISub": lambda x, y: x - y,
                "OpSRem": _srem,
            }[op]
            if isinstance(a, tuple):
                self.env[lhs] = tuple(fn(int(x), int(y)) for x, y in zip(a, b))
            else:
                self.env[lhs] = fn(int(a), int(b))
            self.types[lhs] = ty
            return
        if op == "OpShiftRightLogical":
            a, b = int(self.env[args[0]]) & 0xFFFFFFFF, int(self.env[args[1]]) & 0xFFFFFFFF
            self.env[lhs] = (a >> b) & 0xFFFFFFFF
            self.types[lhs] = ty
            return
        if op in ("OpFOrdGreaterThanEqual", "OpSLessThan", "OpSGreaterThan", "OpSGreaterThanEqual", "OpIEqual", "OpINotEqual"):
            a, b = self.env[args[0]], self.env[args[1]]
            self.env[lhs] = {
                "OpFOrdGreaterThanEqual": lambda: float(a) >= float(b),
                "OpSLessThan": lambda: _s32(int(a)) < _s32(int(b)),
                "OpSGreaterThan": lambda: _s32(int(a)) > _s32(int(b)),
                "OpSGreaterThanEqual": lambda: _s32(int(a)) >= _s32(int(b)),
                "OpIEqual": lambda: int(a) == int(b),
                "OpINotEqual": lambda: int(a) != int(b),
            }[op]()
            self.types[lhs] = ty
            return
        if op in ("OpLogicalOr", "OpLogicalAnd"):
            self._require_type(args[0], "%bool", op)
            self._require_type(args[1], "%bool", op)
            a, b = bool(self.env[args[0]]), bool(self.env[args[1]])
            self.env[lhs] = (a or b) if op == "OpLogicalOr" else (a and b)
            self.types[lhs] = ty
            return
        if op == "OpSelect":
            self._require_type(args[0], "%bool", op)
            self._require_type(args[1], ty, op)
            self._require_type(args[2], ty, op)
            cond = bool(self.env[args[0]])
            self.env[lhs] = self.env[args[1]] if cond else self.env[args[2]]
            self.types[lhs] = ty
            return
        if op == "OpExtInst":
            fn = args[1]
            fargs = args[2:]
            for fa in fargs:
                if fa not in self.env:
                    return
            vals = [self.env[fa] for fa in fargs]
            if fn in ("SMax", "SMin", "SClamp"):
                for fa in fargs:
                    self._require_type(fa, ("%int", "%v2int"), f"OpExtInst {fn}")
            elif fn in ("UMax", "UMin", "UClamp"):
                for fa in fargs:
                    self._require_type(fa, ("%uint", "%v2uint"), f"OpExtInst {fn}")
            if fn == "Floor":
                v = vals[0]
                self.env[lhs] = tuple(_f32(math.floor(x)) for x in v) if isinstance(v, tuple) else _f32(math.floor(v))
            elif fn == "Ceil":
                v = vals[0]
                self.env[lhs] = tuple(_f32(math.ceil(x)) for x in v) if isinstance(v, tuple) else _f32(math.ceil(v))
            elif fn == "Trunc":
                v = vals[0]
                self.env[lhs] = tuple(_f32(math.trunc(x)) for x in v) if isinstance(v, tuple) else _f32(math.trunc(v))
            elif fn in ("FClamp", "SClamp", "UClamp"):
                x, lo, hi = vals
                self.env[lhs] = min(max(x, lo), hi)
            elif fn in ("FMax", "SMax", "UMax"):
                self.env[lhs] = max(vals[0], vals[1])
            elif fn in ("FMin", "SMin", "UMin"):
                self.env[lhs] = min(vals[0], vals[1])
            elif fn == "UnpackHalf2x16":
                w = int(vals[0]) & 0xFFFFFFFF
                lo = struct.unpack("<e", struct.pack("<H", w & 0xFFFF))[0]
                hi = struct.unpack("<e", struct.pack("<H", (w >> 16) & 0xFFFF))[0]
                self.env[lhs] = (_f32(lo), _f32(hi))
            elif fn == "PackHalf2x16":
                v = vals[0]
                lo = struct.unpack("<H", struct.pack("<e", float(v[0])))[0]
                hi = struct.unpack("<H", struct.pack("<e", float(v[1])))[0]
                self.env[lhs] = (lo | (hi << 16)) & 0xFFFFFFFF
            elif fn == "FindILsb":
                self.env[lhs] = 0
            else:
                return
            self.types[lhs] = ty
            return
        if op == "OpImageFetch":
            self._require_type(args[1], "%v2int", op)
            if len(args) < 4 or args[2] != "Lod":
                raise AssertionError(f"Expected OpImageFetch with Lod, got {args}")
            self._require_type(args[3], "%int", op)
            cx, cy = self.env[args[1]]
            mip = int(self.env[args[3]])
            self.fetches.append((mip, int(cx), int(cy)))
            self.env[lhs] = self.texel_fn(mip, int(cx), int(cy))
            self.types[lhs] = ty
            return
        if op == "OpImageGather":
            uv = self.env[args[1]]
            comp = int(self.env[args[2]])
            off = None
            if len(args) >= 5 and args[3] == "Offset":
                self._require_type(args[4], "%v2int", op)
                off = self.env[args[4]]
            self.gathers.append({"uv": uv, "component": comp, "offset": off, "type": ty})
            self.env[lhs] = (
                _f32(10.0 + comp),
                _f32(20.0 + comp),
                _f32(30.0 + comp),
                _f32(40.0 + comp),
            )
            self.types[lhs] = ty
            return


class TestImageGather4LLz(unittest.TestCase):
    def test_offset_extraction_signed_6bit_l_o_and_lz_o(self) -> None:
        # Test signed 6-bit offset extraction x=[5:0], y=[13:8]:
        # - negative values (-3, -32) catch OpBitFieldSExtract -> OpBitFieldUExtract
        # - +16 (0x10, bit 4 set) catches width 6 -> 5 (where 0x10 sign-extends as -16)
        # - garbage in bit 6 (0x40) and bit 14 (0x4000) catches width 6 -> 7
        # - distinct dx != dy catches last-operand duplication in OpCompositeConstruct
        cases = (
            (16, -3, 0x00404040),
            (-32, 31, 0xFFFF8040),
            (1, -1, 0x00004040),
        )
        for op, is_l in (("image_gather4_l_o", True), ("image_gather4_lz_o", False)):
            for dx, dy, extra in cases:
                with self.subTest(op=op, dx=dx, dy=dy):
                    packed = _pack_offset(dx, dy, extra)
                    v_inits = {4: packed, 5: _f32_bits(0.5), 6: _f32_bits(0.5)}
                    addr = "v[4:6]"
                    if is_l:
                        v_inits[7] = _f32_bits(0.0)
                        addr = "v[4:7]"
                    inst = f"{op} v[0:3], {addr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                    spv = support.translate_lines(f"ps_{op}_{dx}_{dy}", _ps_program(inst, v_inits=v_inits))
                    self.assertIn("OpCapability ImageGatherExtended", spv)
                    ev = GatherSpirvEvaluator(spv)
                    if is_l:
                        # At mip 0 (8x8), u=0.5, v=0.5 -> base i0 = floor(4.0 - 0.5) = (3, 3)
                        # After adding goff = (dx, dy) and clamping or wrapping, verify i0 + goff before clamp:
                        # With wrap mode (s0_lit = 0x0, wrap on X and Y):
                        spv_wrap = support.translate_lines(
                            f"ps_{op}_wrap",
                            _ps_program(inst, v_inits=v_inits, s0_lit=0x0),
                        )
                        ev_w = GatherSpirvEvaluator(spv_wrap)
                        x0 = (3 + dx) % 8
                        y0 = (3 + dy) % 8
                        x1 = (3 + dx + 1) % 8
                        y1 = (3 + dy + 1) % 8
                        self.assertEqual(
                            ev_w.fetches,
                            [(0, x0, y1), (0, x1, y1), (0, x1, y0), (0, x0, y0)],
                        )
                    else:
                        self.assertEqual(len(ev.gathers), 1)
                        self.assertEqual(ev.gathers[0]["offset"], (dx, dy))

    def test_image_gather4_l_lod_rounding_and_clamping(self) -> None:
        # Must use Floor(lod), ties-up (frac >= 0.5 -> +1), and clamp to [0, levels - 1].
        # Cases cover:
        # - lod = 0.0 -> fl = 0, inc = 0, mi = 0 (catches SMax 0 -> 1 and OpSelect arm swap)
        # - lod = 0.5 -> fl = 0, inc = 1, mi = 1 (catches OpIAdd fl + inc -> fl + fl, and Floor -> Ceil)
        # - lod = 1.25 -> fl = 1, inc = 0, mi = 1 (catches OpIAdd -> OpISub when inc=1 vs inc=0, and inc 0 -> 1)
        # - lod = 1.75 -> fl = 1, inc = 1, mi = 2 (catches self.cu(1) -> self.cu(2) in inc)
        # - lod = 3.0 with levels = 4 -> mc = 3 (catches lvm = lv - 1 -> lv - 2)
        # - lod = 6.0 with levels = 4 -> mc = 3 (catches SMin -> SMax, lvm -> mz, and lv - 1 -> lv - 0 or lv + 1)
        lod_cases = (
            (0.0, 0),
            (0.25, 0),
            (0.5, 1),
            (0.49999997, 0),
            (1.25, 1),
            (1.5, 2),
            (1.75, 2),
            (3.0, 3),
            (6.0, 3),
        )
        for lod, expected_mip in lod_cases:
            with self.subTest(lod=lod):
                v_inits = {
                    4: _f32_bits(0.5),
                    5: _f32_bits(0.5),
                    6: _f32_bits(lod),
                }
                inst = "image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                spv = support.translate_lines("ps_gather4_l_lod", _ps_program(inst, v_inits=v_inits))
                # Verify scalar Floor is used for LOD (not Trunc or Ceil)
                self.assertRegex(spv, r"= OpExtInst %float %glsl Floor %\w+")
                ev = GatherSpirvEvaluator(spv, levels=4)
                self.assertEqual(len(ev.fetches), 4)
                for mip, _, _ in ev.fetches:
                    self.assertEqual(mip, expected_mip)

    def test_image_gather4_l_footprint_coordinates_and_negative_sub_floor(self) -> None:
        # Pick u = 0.05, v = 0.625 at mip 1 (size 4x4):
        # st = (0.2, 2.5), sub = st - 0.5 = (-0.3, 2.0)
        # Floor(-0.3) = -1 (whereas Trunc(-0.3) = 0 and Ceil(-0.3) = 0!).
        # With wrap mode (s0_lit = 0):
        # i0 = (-1, 2), i1 = (0, 3) -> wrapped on 4x4: x0 = 3, x1 = 0, y0 = 2, y1 = 3.
        # Gather tap order: (x0, y1), (x1, y1), (x1, y0), (x0, y0)
        # = (3, 3), (0, 3), (0, 2), (3, 2).
        # Also choose dmask:0x4 (channel 2, B) so ch = 2 != 0.
        v_inits = {
            4: _f32_bits(0.05),
            5: _f32_bits(0.625),
            6: _f32_bits(1.0),
        }
        inst = "image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x4 dim:SQ_RSRC_IMG_2D"
        spv = support.translate_lines("ps_gather4_l_footprint", _ps_program(inst, v_inits=v_inits, s0_lit=0x0))
        ev = GatherSpirvEvaluator(spv, levels=4)
        expected_taps = [(1, 3, 3), (1, 0, 3), (1, 0, 2), (1, 3, 2)]
        self.assertEqual(ev.fetches, expected_taps)
        # Channel 2 (dmask 0x4) of each tap is stored into v0..v3
        for idx, (mip, x, y) in enumerate(expected_taps):
            expected_val = ev.texel_fn(mip, x, y)[2]
            self.assertEqual(_u32_to_f32(ev.vgprs[idx]), expected_val)

    def test_image_gather4_l_border_modes_and_runtime_border_color(self) -> None:
        # Mode 6 (clamp to border) on both X and Y: s0_lit = 6 | (6 << 3) = 0x36.
        # At mip 1 (size 4x4), u = 0.05 -> x0 = -1 (out of bounds: ox[0] = True), x1 = 0 (in bounds: ox[1] = False).
        # Set v = 0.95 -> st_y = 3.8, sub_y = 3.3, y0 = 3 (in bounds: oy[0] = False), y1 = 4 (out of bounds: oy[1] = True).
        # Then the 4 taps (ax[0], ay[1]), (ax[1], ay[1]), (ax[1], ay[0]), (ax[0], ay[0]) have:
        #   tap 0: (x0=-1, y1=4) -> ox=True,  oy=True  -> out=True  (border)
        #   tap 1: (x1=0,  y1=4) -> ox=False, oy=True  -> out=True  (border) -- catches OpLogicalOr -> OpLogicalAnd & oy -> ox!
        #   tap 2: (x1=0,  y0=3) -> ox=False, oy=False -> out=False (fetched texel!) -- catches comp -> border_v!
        #   tap 3: (x0=-1, y0=3) -> ox=True,  oy=False -> out=True  (border) -- catches ox/oy asymmetry!
        # Also use a runtime S# word 3 (s3_lit=None) and evaluate with both opaque white (word 3 = 2 << 30 -> border=1.0)
        # and transparent black (word 3 = 0 -> border=0.0) to catch zero_v -> one_v on line 12671!
        v_inits = {
            4: _f32_bits(0.05),
            5: _f32_bits(0.95),
            6: _f32_bits(1.0),
        }
        inst = "image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D"
        spv = support.translate_lines(
            "ps_gather4_l_border_rt",
            _ps_program(inst, v_inits=v_inits, s0_lit=0x36, s3_lit=None),
        )
        # Evaluate with runtime S# word 3 = opaque white (2 << 30) -> border = 1.0
        ev_white = GatherSpirvEvaluator(spv, levels=4, sgpr_words={19: 2 << 30})
        in_bounds_val = ev_white.texel_fn(1, 0, 3)[1]
        self.assertEqual(
            [_u32_to_f32(ev_white.vgprs[i]) for i in range(4)],
            [1.0, 1.0, in_bounds_val, 1.0],
        )
        # Evaluate with runtime S# word 3 = transparent black (0) -> border = 0.0
        ev_black = GatherSpirvEvaluator(spv, levels=4, sgpr_words={19: 0})
        self.assertEqual(
            [_u32_to_f32(ev_black.vgprs[i]) for i in range(4)],
            [0.0, 0.0, in_bounds_val, 0.0],
        )

    def test_image_gather4_lz_and_lz_o_dimensions_and_modifiers(self) -> None:
        # 2D: uses typed image_read_typed wrapper and OpImageGather with %v2float uv
        v_inits_2d = {
            4: _f32_bits(0.25),
            5: _f32_bits(0.75),
        }
        spv_2d = support.translate_lines(
            "ps_gather4_lz_2d",
            _ps_program("image_gather4_lz v[0:3], v[4:5], s[8:15], s[16:19] dmask:0x8 dim:SQ_RSRC_IMG_2D", v_inits=v_inits_2d),
        )
        ev_2d = GatherSpirvEvaluator(spv_2d)
        self.assertEqual(len(ev_2d.gathers), 1)
        self.assertEqual(ev_2d.gathers[0]["uv"], (_f32(0.25), _f32(0.75)))
        self.assertEqual(ev_2d.gathers[0]["component"], 3)
        self.assertIsNone(ev_2d.gathers[0]["offset"])

        # 2D_ARRAY with image_gather4_lz_o: 4 address regs (offset, u, v, slice), %v3float uv,
        # and direct 4-component extraction into v0..v3 (lines 12716-12722).
        v_inits_arr = {
            4: _pack_offset(-2, 5),
            5: _f32_bits(0.125),
            6: _f32_bits(0.375),
            7: _f32_bits(2.0),
        }
        spv_arr = support.translate_lines(
            "ps_gather4_lz_o_arr",
            _ps_program(
                "image_gather4_lz_o v[0:3], v[4:7], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY",
                v_inits=v_inits_arr,
            ),
        )
        ev_arr = GatherSpirvEvaluator(spv_arr)
        self.assertEqual(len(ev_arr.gathers), 1)
        self.assertEqual(ev_arr.gathers[0]["uv"], (_f32(0.125), _f32(0.375), _f32(2.0)))
        self.assertEqual(ev_arr.gathers[0]["component"], 1)
        self.assertEqual(ev_arr.gathers[0]["offset"], (-2, 5))
        self.assertEqual(
            [_u32_to_f32(ev_arr.vgprs[i]) for i in range(4)],
            [11.0, 21.0, 31.0, 41.0],
        )

        # CUBE with image_gather4_lz: 3 address regs (x, y, z) -> cube_coords -> %v3float
        v_inits_cube = {
            4: _f32_bits(1.0),
            5: _f32_bits(0.25),
            6: _f32_bits(-0.5),
        }
        spv_cube = support.translate_lines(
            "ps_gather4_lz_cube",
            _ps_program(
                "image_gather4_lz v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
                v_inits=v_inits_cube,
            ),
        )
        self.assertIn("OpImageGather %v4float", spv_cube)

        # a16 and d16 modifiers on image_gather4_l_o and image_gather4_lz_o
        for op, vaddr in (("image_gather4_l_o", "v[4:6]"), ("image_gather4_lz_o", "v[4:5]")):
            with self.subTest(op=op, mods="a16 d16"):
                inst = f"{op} v[0:1], {vaddr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 d16"
                spv = support.translate_lines(
                    f"ps_{op}_a16_d16",
                    _ps_program(inst, v_inits={4: 0x0101, 5: 0x38003800, 6: 0x00003800}),
                )
                self.assertIn("UnpackHalf2x16", spv)
                self.assertIn("OpStore %reg_v0", spv)
                self.assertIn("OpStore %reg_v1", spv)
                self.assertNotIn("OpStore %reg_v2", spv)

    def test_unsupported_forms_rejected(self) -> None:
        bad_cases = (
            # Multi-bit dmask on gather4_l and gather4_lz
            ("image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D", 0x12, 0x0),
            ("image_gather4_lz v[0:3], v[4:5], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D", 0x12, 0x0),
            # 3D dimension on gather4_l and gather4_lz
            ("image_gather4_l v[0:3], v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_3D", 0x12, 0x0),
            ("image_gather4_lz v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_3D", 0x12, 0x0),
            # a16 with tfe / lwe on _l_o and _lz_o
            ("image_gather4_l_o v[0:4], v[5:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 tfe", 0x12, 0x0),
            ("image_gather4_lz_o v[0:4], v[5:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16 lwe", 0x12, 0x0),
            # Literal S# FORCE_UNNORMALIZED (bit 15), half-border mode (4), and table border color (type 3)
            ("image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D", 0x8012, 0x0),
            ("image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D", 0x0004, 0x0),
            ("image_gather4_l v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D", 0x0036, 3 << 30),
        )
        for inst, s0, s3 in bad_cases:
            with self.subTest(inst=inst, s0=hex(s0), s3=hex(s3)):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines(
                        "ps_bad_gather4",
                        _ps_program(inst, v_inits={4: 0, 5: 0, 6: 0, 7: 0}, s0_lit=s0, s3_lit=s3),
                    )


if __name__ == "__main__":
    unittest.main()

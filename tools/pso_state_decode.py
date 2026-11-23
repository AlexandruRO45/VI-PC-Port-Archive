"""Decode the graphics state blobs of the pipeline library in shaders.rpf (docs/shader_library.md).

Each graphics pipeline record names a state blob of 380 to 520 bytes. The blob is a block of GPU context register
values in two fixed layouts:
  * word 1 == 4: a VS + PS (or VS-only) pipeline, 380 to 420 bytes;
  * word 1 == 6: an MS (+ PS) pipeline, 464 to 520 bytes; its colour and depth registers sit 0x54 bytes later.
The registers below were identified from the values alone, by their RDNA2 field layouts (AMD's register
reference as used by Mesa's ac/sid.h) and by consistency checks over all 5,680 distinct blobs of 1.0.2.1:
  * CB_BLENDn_CONTROL x 8: blending enabled (bit 30) on render target n exactly when SX_MRTn_BLEND_OPT is not the
    "blend disabled" word 0x06000600, in all 45,440 (blob, target) pairs; every enabled word decodes to a D3D12 blend
    (SRC_ALPHA / INV_SRC_ALPHA, dual-source SRC1_*, MAX, ...).
  * SPI_SHADER_COL_FORMAT, CB_SHADER_MASK, CB_TARGET_MASK, SX_PS_DOWNCONVERT: 4 bits per render target, non-zero
    from target 0 upwards in 97-100% of the blobs.
  * DB_DEPTH_CONTROL: no blob writes depth with the depth test off; mesh pipelines mostly test GEQUAL (reversed Z).
  * DB_STENCIL_CONTROL and DB_STENCILREFMASK(_BF) next to it.
The meaning of the other words is not established. Nothing here is measured on a console: it is a reading of the
data, to be checked against a title's own pipeline descriptions.

Usage:
    python -I tools/pso_state_decode.py <library file or shaders.rpf> --out <file.json> [--summary]
The output is game data: keep it in your work folder.
"""
import argparse
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shader_library as sl  # noqa: E402

VS_LAYOUT, MS_LAYOUT = 4, 6
SHIFT = {VS_LAYOUT: 0, MS_LAYOUT: 0x54}   # the MS layout's colour/depth registers sit 0x54 bytes later
# Byte offsets in the VS layout (add SHIFT for the MS layout).
COL_FORMAT, SHADER_MASK, TARGET_MASK, BLEND = 0x070, 0x078, 0x0CC, 0x0D0
DEPTH_CONTROL, STENCIL_CONTROL, STENCIL_REF_MASK, STENCIL_REF_MASK_BF = 0x0B8, 0x0BC, 0x0C0, 0x0C4
DOWNCONVERT, BLEND_OPT_EPSILON, BLEND_OPT_CONTROL, MRT_BLEND_OPT = 0x0F4, 0x0F8, 0x0FC, 0x100
BLEND_OPT_DISABLED = 0x06000600

# RDNA2 BLEND_* and COMB_* values with their D3D12_BLEND / D3D12_BLEND_OP names.
BLEND_FACTOR = ["ZERO", "ONE", "SRC_COLOR", "INV_SRC_COLOR", "SRC_ALPHA", "INV_SRC_ALPHA", "DEST_ALPHA",
                "INV_DEST_ALPHA", "DEST_COLOR", "INV_DEST_COLOR", "SRC_ALPHA_SAT", "BOTH_SRC_ALPHA",
                "BOTH_INV_SRC_ALPHA", "BLEND_FACTOR", "INV_BLEND_FACTOR", "SRC1_COLOR", "INV_SRC1_COLOR",
                "SRC1_ALPHA", "INV_SRC1_ALPHA", "ALPHA_FACTOR", "INV_ALPHA_FACTOR"]
BLEND_OP = ["ADD", "SUBTRACT", "MIN", "MAX", "REV_SUBTRACT"]
COMPARE = ["NEVER", "LESS", "EQUAL", "LESS_EQUAL", "GREATER", "NOT_EQUAL", "GREATER_EQUAL", "ALWAYS"]
STENCIL_OP = ["KEEP", "ZERO", "ONES", "REPLACE_TEST", "REPLACE_OP", "ADD_CLAMP", "SUB_CLAMP", "INVERT",
              "ADD_WRAP", "SUB_WRAP", "AND", "OR", "XOR", "NAND", "NOR", "XNOR"]
COL_FORMAT_NAMES = ["ZERO", "32_R", "32_GR", "32_AR", "FP16_ABGR", "UNORM16_ABGR", "SNORM16_ABGR", "UINT16_ABGR",
                    "SINT16_ABGR", "32_ABGR"]
DOWNCONVERT_NAMES = ["NONE", "32_R", "32_A", "10_11_11", "2_10_10_10", "8_8_8_8", "5_6_5", "1_5_5_5", "4_4_4_4",
                     "16_16_GR", "16_16_AR", "9_9_9_E5"]


def name(table, i):
    return table[i] if i < len(table) else str(i)


def nibbles(v):
    return [(v >> (4 * i)) & 0xF for i in range(8)]


def blend(v):
    """CB_BLENDn_CONTROL as a D3D12_RENDER_TARGET_BLEND_DESC-like dict (None when blending is off)."""
    if not (v >> 30) & 1:
        return None
    color = (name(BLEND_FACTOR, v & 0x1F), name(BLEND_OP, (v >> 5) & 7), name(BLEND_FACTOR, (v >> 8) & 0x1F))
    alpha = (name(BLEND_FACTOR, (v >> 16) & 0x1F), name(BLEND_OP, (v >> 21) & 7),
             name(BLEND_FACTOR, (v >> 24) & 0x1F)) if (v >> 29) & 1 else color
    return {"src": color[0], "op": color[1], "dest": color[2],
            "src_alpha": alpha[0], "op_alpha": alpha[1], "dest_alpha": alpha[2]}


def decode(state: bytes) -> dict:
    """The decoded fields of one state blob."""
    if len(state) < 0x170 or len(state) % 4:
        raise ValueError(f"state blob of {len(state)} bytes is not a known layout")
    w = struct.unpack_from(f"<{len(state) // 4}I", state)
    layout = w[1]
    if layout not in SHIFT:
        raise ValueError(f"state blob layout word {layout} is not 4 (VS) or 6 (MS)")
    s = SHIFT[layout]
    if len(state) < MRT_BLEND_OPT + s + 32:
        raise ValueError(f"state blob of {len(state)} bytes is too short for layout {layout}")
    r = lambda off: w[(off + s) // 4]  # noqa: E731
    col = nibbles(r(COL_FORMAT))
    targets = 0  # render targets: the leading non-zero export formats
    while targets < 8 and col[targets]:
        targets += 1
    depth = r(DEPTH_CONTROL)
    sc = r(STENCIL_CONTROL)
    ref, ref_bf = r(STENCIL_REF_MASK), r(STENCIL_REF_MASK_BF)

    def ops(v, shift):
        return {"fail": name(STENCIL_OP, (v >> shift) & 0xF), "pass": name(STENCIL_OP, (v >> (shift + 4)) & 0xF),
                "depth_fail": name(STENCIL_OP, (v >> (shift + 8)) & 0xF)}
    return {
        "layout": "ms" if layout == MS_LAYOUT else "vs",
        "render_targets": targets,
        "export_format": [name(COL_FORMAT_NAMES, n) for n in col[:targets]],
        "downconvert": [name(DOWNCONVERT_NAMES, n) for n in nibbles(r(DOWNCONVERT))[:targets]],
        "write_mask": nibbles(r(TARGET_MASK))[:max(targets, 1)],
        "shader_mask": nibbles(r(SHADER_MASK))[:max(targets, 1)],
        "blend": [blend(w[(BLEND + s) // 4 + i]) for i in range(max(targets, 1))],
        "blend_opt_consistent": all(((w[(BLEND + s) // 4 + i] >> 30) & 1) == (w[(MRT_BLEND_OPT + s) // 4 + i] !=
                                                                             BLEND_OPT_DISABLED) for i in range(8)),
        "depth": {"enable": bool((depth >> 1) & 1), "write": bool((depth >> 2) & 1),
                  "func": COMPARE[(depth >> 4) & 7], "bounds": bool((depth >> 3) & 1)},
        "stencil": {"enable": bool(depth & 1), "func": COMPARE[(depth >> 8) & 7],
                    "func_back": COMPARE[(depth >> 20) & 7] if (depth >> 7) & 1 else COMPARE[(depth >> 8) & 7],
                    "front": ops(sc, 0), "back": ops(sc, 12),
                    "ref": ref & 0xFF, "read_mask": (ref >> 8) & 0xFF, "write_mask": (ref >> 16) & 0xFF,
                    "read_mask_back": (ref_bf >> 8) & 0xFF, "write_mask_back": (ref_bf >> 16) & 0xFF},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="the library (shader_library.py --save) or shaders.rpf")
    ap.add_argument("--out", required=True, help="JSON: per graphics record, the decoded state")
    ap.add_argument("--summary", action="store_true", help="print counts of the decoded fields")
    a = ap.parse_args(argv)
    with open(a.src, "rb") as f:
        data = f.read()
    b = data if data[:4] == sl.MAGIC else sl.find_library(data)
    lib = sl.parse(b)
    cache, records, failed = {}, [], 0
    for i, g in enumerate(lib["graphics"]):
        st = g["state"]
        if st is None:
            records.append(None)
            continue
        if st not in cache:
            try:
                cache[st] = decode(sl.blob(b, lib, st))
            except ValueError as e:
                cache[st] = {"error": str(e)}
                failed += 1
        records.append({"record": i, "state": st, **cache[st]})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(records, f, indent=1)
    print(f"{len(records)} graphics records, {len(cache)} distinct state blobs decoded ({failed} not) -> {a.out}")
    if a.summary:
        dec = [d for d in cache.values() if "error" not in d]
        print("blend/opt consistent:", sum(d["blend_opt_consistent"] for d in dec), "of", len(dec))
        print("render targets:", sorted(Counter(d["render_targets"] for d in dec).items()))
        print("depth:", Counter((d["depth"]["enable"], d["depth"]["write"], d["depth"]["func"])
                                for d in dec).most_common(6))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

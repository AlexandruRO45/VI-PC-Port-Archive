r"""FUZZ-114: differential fuzzer for VS/PS export shapes (standard library only).

Usage:
    python -I tools/export_diff.py --programs N [--seed S]
    python -I tools/export_diff.py --self-test
    python -I tools/export_diff.py --list-ops
"""
import os
import struct
import sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

# D3D_NAME system values the export rules read (rdna2spirv.output_signature_full / pos_export_layout).
D3D_NAME = {"clip": 2, "cull": 3, "rtai": 4, "vai": 5, "shading_rate": 23, "target": 64, "depth": 65}

# One 32-byte OSG1 signature element: system value, semantic index, register, component mask, read/write mask.
Osg1Element = namedtuple("Osg1Element", "sysv index register wmask_lo wmask_hi")


def osg1_dxbc(elements):
    """A synthetic DXBC container holding one OSG1 (output signature) chunk with `elements`.

    The layout is the one rdna2spirv.output_signature_full() reads: a <II (count, offset) header, then 32-byte
    elements of <6I (stream, name, semantic index, system value, type, register), the component mask at byte 24 and
    the read/write mask at byte 25. Name 0 and type 3 (float), as in the reference builder of
    tests/semantic/test_sparse_sv_target.py.
    """
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for e in elements:
        elem = bytearray(32)
        struct.pack_into("<6I", elem, 0, 0, 0, e.index, e.sysv, 3, e.register)
        elem[24], elem[25] = e.wmask_lo, e.wmask_hi
        payload += elem
    part = b"OSG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


# ---- the model: pixel-shader exports ---------------------------------------------------------------------------------
PsShape = namedtuple("PsShape", "targets has_osg1")             # declared SV_Target registers in order; an OSG1 at all
PsExport = namedtuple("PsExport", "kind index depth_x compr")   # kind "mrt" | "mrtz"
PsPrediction = namedtuple("PsPrediction", "locations refusals")  # %out_* -> Location | "FragDepth"; expected refusals


def ps_model(shape, exports):
    """Colour and depth exports of a pixel shader.

    With SV_Target outputs declared, `exp mrtK` writes the K-th declared one and the output variable carries that
    target's register (SHADER-210); an export past the declared count raises Unsupported and ends the program.
    Without a declared target the export index is the target. `exp mrtz` exports FragDepth only when x is on in an
    uncompressed export: the 16-bit compr form is the stencil and sample-mask format and never holds Z.
    """
    locations, refusals = {}, []
    for e in exports:
        if e.kind == "mrtz":
            if e.depth_x and not e.compr:
                locations["%out_depth"] = "FragDepth"
            continue
        if shape.targets and e.index >= len(shape.targets):
            refusals.append(f"mrt{e.index}")
            break
        rt = shape.targets[e.index] if shape.targets else e.index
        locations[f"%out_mrt{rt}"] = rt
    return PsPrediction(locations, refusals)


def structure_lines(locations):
    """Output decorations as canonical strings, sorted: "Location %var n" or "BuiltIn %var Name"."""
    return sorted(f"BuiltIn {var} {val}" if isinstance(val, str) else f"Location {var} {val}"
                  for var, val in locations.items())


def ps_structure(prediction):
    """The pixel-shader module's output decorations the translated module must carry."""
    return structure_lines(prediction.locations)


# ---- the model: vertex-shader exports ---------------------------------------------------------------------------------
VsShape = namedtuple("VsShape", "svs has_osg1")             # OSG1 elements (clip, cull, misc); whether an OSG1 exists
VsExport = namedtuple("VsExport", "kind index comps")       # kind "pos" | "param"; comps: four 0/1, 1 = a source is exported
VsPrediction = namedtuple("VsPrediction", "position distances misc clip_count cull_count layer viewport params "
                                          "discarded refusals")

MISC_SVS = (D3D_NAME["rtai"], D3D_NAME["vai"], D3D_NAME["shading_rate"])   # the pos1 misc vector's system values


def pos_layout(svs):
    """The pos1..pos3 packing of the declared clip and cull distances: (pos_map, misc system values, counts).

    Clip distances precede cull distances, each group in (semantic index, register, mask) order with the set
    components of a mask lowest first, packed contiguously into the 4-component pos exports: up to 8 floats, the
    rest dropped, the cull distances getting what is left of the 8 after the clip ones. They start at pos2 when a
    misc system value (RenderTargetArrayIndex, ViewportArrayIndex, ShadingRate) is declared, since pos1 then holds
    the misc vector, and at pos1 otherwise. pos_map is {pos index: {component: ("clip" | "cull", distance index)}}.
    """
    def declared(sysv):
        """How many distance components the elements with this system value declare, in signature order."""
        n = 0
        for _, _, mask in sorted((e.index, e.register, e.wmask_lo) for e in svs if e.sysv == sysv):
            n += bin(mask & 0xF).count("1")
        return n

    clip, cull = declared(D3D_NAME["clip"]), declared(D3D_NAME["cull"])
    clip_count = min(clip, 8)
    cull_count = min(cull, max(0, 8 - clip_count))
    misc = sorted({e.sysv for e in svs if e.sysv in MISC_SVS})
    base = 2 if misc else 1
    pos_map = {}
    for k, target in enumerate([("clip", i) for i in range(clip)] + [("cull", i) for i in range(cull)]):
        if k < 8:
            pos_map.setdefault(base + k // 4, {})[k % 4] = target
    return pos_map, misc, clip_count, cull_count


def vs_model(shape, exports):
    """Position, distance, misc and parameter exports of a vertex shader.

    `exp pos0` is Position. `exp posN` (N >= 1) with every source off does nothing; pos1 carries the misc vector
    when one is declared (its z component: Layer for RenderTargetArrayIndex, ViewportIndex for ViewportArrayIndex)
    and never distances; otherwise posN carries the distances pos_layout() puts there, a component with an off
    source left unwritten. Without a declared distance or misc system value, pos1..pos3 are discarded.
    `exp paramN` is output location N. Nothing here is refused.
    """
    pos_map, misc, clip_count, cull_count = pos_layout(shape.svs)
    position, layer, viewport, used = False, False, False, set()
    distances, params, discarded = [], {}, []
    for e in exports:
        if e.kind == "param":
            if any(e.comps):
                params[e.index] = e.index
            continue
        if not any(e.comps):
            if e.index:
                discarded.append(e.index)
            continue
        if e.index == 0:
            position = True
        elif e.index == 1 and misc:
            if e.comps[2]:
                layer = D3D_NAME["rtai"] in misc
                viewport = D3D_NAME["vai"] in misc
        else:
            mapping = pos_map.get(e.index) or {}
            hits = [(e.index, comp, kind, idx) for comp, (kind, idx) in sorted(mapping.items())
                    if comp < 4 and e.comps[comp]]
            if hits:
                distances += hits
                used.update(kind for _, _, kind, _ in hits)
            else:
                discarded.append(e.index)
    return VsPrediction(position, distances, "pos1" if misc else None,
                        clip_count if "clip" in used else 0, cull_count if "cull" in used else 0,
                        layer, viewport, params, discarded, [])


def vs_structure(prediction):
    """The vertex-shader module's output decorations and distance array lengths the translated module must carry."""
    lines = []
    if prediction.position:
        lines.append("BuiltIn %out_pos Position")
    if prediction.clip_count:
        lines += ["BuiltIn %out_clip ClipDistance", f"Array %out_clip {prediction.clip_count}"]
    if prediction.cull_count:
        lines += ["BuiltIn %out_cull CullDistance", f"Array %out_cull {prediction.cull_count}"]
    if prediction.layer:
        lines.append("BuiltIn %out_layer Layer")
    if prediction.viewport:
        lines.append("BuiltIn %out_viewport ViewportIndex")
    return sorted(lines + structure_lines({f"%out_param{n}": n for n in prediction.params}))

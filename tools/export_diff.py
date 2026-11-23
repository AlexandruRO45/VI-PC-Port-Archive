r"""FUZZ-114: differential fuzzer for VS/PS export shapes (standard library only).

Generates seeded vertex- and pixel-shader programs from the export shapes compilers emit and compares the structure
of the translated module with a model of the packing rules in docs/design.md ("Inputs and outputs"):

  * the model, here: output locations and built-ins, ClipDistance/CullDistance arrays and stores, expected refusals
  * tools/rdna2spirv.py through tests/semantic/support.py: the translated SPIR-V's decorations, arrays and stores

Shapes: a pixel shader's declared SV_Target set (consecutive, sparse or none, with or without an OSG1), `exp mrtK`
in any order (plain, with gated components, or compressed) and an optional `exp mrtz` (x on or off, compressed or
not); a vertex shader's declared SV_ClipDistance/SV_CullDistance components (up to 9 + 4, in elements of 1 to 4 components), its misc
system values (RenderTargetArrayIndex, ViewportArrayIndex, ShadingRate), `exp pos0..pos3` with any on/off component
pattern and `exp paramN` for a few N.

Usage:
    python -I tools/export_diff.py [--programs N] [--seed S]
    python -I tools/export_diff.py --seed S --program K --show     # print one program, its prediction and the module
    python -I tools/export_diff.py --self-test
    python -I tools/export_diff.py --list-ops

Exit code 0 when every program matches, 1 on the first mismatch (case, seed, program, the listing, and the expected
against the actual structure), 2 on a harness error, a bare invocation or when no program was compared. `--show`
follows the same codes for its one program (a predicted refusal is not a mismatch); `--self-test` exits 0 when
every canary is caught and 2 when the comparator has gone blind. Skips are counted with reasons: a refusal the
model predicts is a skip, one it does not is a mismatch.

A mismatch is a finding to post in the coordination issue before anything is fixed: the model is wrong until the
contract row proves otherwise, and then the translator is.
"""
import argparse
import os
import random
import re
import struct
import sys
from collections import Counter, namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SEM_DIR = os.path.join(os.path.dirname(HERE), "tests", "semantic")
if SEM_DIR not in sys.path:
    sys.path.insert(0, SEM_DIR)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

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
PsExport = namedtuple("PsExport", "kind index depth_x compr comps", defaults=[()])   # kind "mrt" | "mrtz"; comps: EN gates
PsPrediction = namedtuple("PsPrediction", "locations refusals")  # %out_* -> Location | "FragDepth"; expected refusals


def ps_model(shape, exports):
    """Colour and depth exports of a pixel shader.

    With SV_Target outputs declared, `exp mrtK` writes the K-th declared one and the output variable carries that
    target's register (SHADER-210); an export past the declared count raises Unsupported and ends the program.
    Without a declared target the export index is the target. `exp mrtz` exports FragDepth only when x is on in an
    uncompressed export: the 16-bit compr form is the stencil and sample-mask format and never holds Z.
    """
    locations, refusals = {}, []
    targets = shape.targets if shape.has_osg1 else []     # the translator reads the target set from the OSG1 alone
    for e in exports:
        if e.kind == "mrtz":
            if e.depth_x and not e.compr:
                locations["%out_depth"] = "FragDepth"
            continue
        if targets and e.index >= len(targets):
            refusals.append(f"mrt{e.index}")
            break
        rt = targets[e.index] if targets else e.index
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
            params[e.index] = e.index      # the output is declared for any `exp paramN`, off components included
            continue
        if e.index == 0:
            position = True                 # likewise for any `exp pos0`
            continue
        if not any(e.comps):
            discarded.append(e.index)
            continue
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
    """Vertex outputs, including the source component and destination of each distance store."""
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
    lines += [f"Distance pos{pos}.{comp} -> {kind}[{idx}]"
              for pos, comp, kind, idx in prediction.distances]
    return sorted(lines + structure_lines({f"%out_param{n}": n for n in prediction.params}))


# ---- programs --------------------------------------------------------------------------------------------------------
Case = namedtuple("Case", "name stage shape exports listing dxbc expected refusals")
ON = (1, 1, 1, 1)
MISC_REGISTERS = {"rtai": 1, "vai": 2, "shading_rate": 3}


def export_line(e):
    """The `exp` instruction for one modelled export; the caller adds `done vm` to the last of a program."""
    if isinstance(e, PsExport):
        if e.kind == "mrtz":
            x = "v0" if e.depth_x else "off"
            return f"exp mrtz {x}, off, off, off" + (" compr" if e.compr else "")
        # Compressed exports list each of the two registers twice, one slot per 16-bit half; an uncompressed export
        # may gate its components (EN bits): an off slot keeps the previous export.
        if e.compr:
            operands = "v0, v0, v1, v1 compr"
        elif e.comps:
            operands = ", ".join(f"v{c}" if on else "off" for c, on in enumerate(e.comps))
        else:
            operands = "v0, v1, v2, v3"
        return f"exp mrt{e.index} {operands}"
    # Distinct registers identify which pos export and component produced a distance store.
    offset = 4 * e.index if e.kind == "pos" else 0
    operands = ", ".join(f"v{offset + c}" if on else "off" for c, on in enumerate(e.comps))
    return f"exp {'pos' if e.kind == 'pos' else 'param'}{e.index} {operands}"


def make_case(name, stage, shape, exports):
    """A program from a shape and its exports: the listing, its DXBC container, and what the model predicts."""
    if stage == "ps":
        prediction = ps_model(shape, exports)
        expected = ps_structure(prediction)
        dxbc = osg1_dxbc([Osg1Element(D3D_NAME["target"], n, n, 0xF, 0xF) for n in shape.targets]) if shape.has_osg1 \
            else support.STUB_DXBC
    else:
        prediction = vs_model(shape, exports)
        expected = vs_structure(prediction)
        dxbc = osg1_dxbc(shape.svs) if shape.has_osg1 else support.STUB_DXBC
    registers = set(range(4))
    if stage == "vs":
        registers.update(4 * e.index + c for e in exports if e.kind == "pos"
                         for c, on in enumerate(e.comps) if on)
    lines = [f"v_mov_b32_e32 v{c}, {float(1 << c)!r}" for c in sorted(registers)]
    instructions = [export_line(e) for e in exports]
    if instructions:
        instructions[-1] += " done vm"
    return Case(name, stage, shape, exports, lines + instructions + ["s_endpgm"], dxbc, expected, prediction.refusals)


def _distance_elements(rng, kind, count):
    """SV_ClipDistance/SV_CullDistance declarations: `count` components in elements of 1 to 4, four per register."""
    svs, first = [], 0
    while first < count:
        width = min(rng.randint(1, 4), count - first, 4 - first % 4)
        svs.append(Osg1Element(D3D_NAME[kind], first, first // 4, ((1 << width) - 1) << (first % 4), 0))
        first += width
    return svs


def _components(rng):
    return tuple(rng.getrandbits(1) for _ in range(4))


def generate(seed, index):
    """Program `index` of `seed`: the same (seed, index) always gives the same program.

    The name starts with the stage: rdna2spirv.stage_of() takes it from there when no carve index is at hand."""
    rng = random.Random(f"export-diff:{seed}:{index}")
    name = f"ed_{seed}_{index}"
    if rng.random() < 0.5:
        has_osg1 = rng.random() > 0.1
        svs = []
        if has_osg1:
            svs = _distance_elements(rng, "clip", rng.choice((0, 1, 2, 2, 3, 4, 6, 8, 9)))
            svs += _distance_elements(rng, "cull", rng.choice((0, 0, 1, 2, 4)))
            svs += [Osg1Element(D3D_NAME[m], 0, MISC_REGISTERS[m], 0x1, 0) for m in MISC_REGISTERS if rng.random() < 0.3]
        exports = [VsExport("pos", 0, _components(rng))]
        exports += [VsExport("pos", p, _components(rng)) for p in (1, 2, 3) if rng.random() < 0.6]
        exports += [VsExport("param", n, _components(rng)) for n in range(8) if rng.random() < 0.25]
        rng.shuffle(exports)
        return make_case("vs_" + name, "vs", VsShape(svs, has_osg1), exports)
    has_osg1 = rng.random() > 0.15
    targets = sorted(rng.sample(range(8), rng.randint(1, 8))) if has_osg1 and rng.random() > 0.1 else []
    indices = list(range(len(targets))) if targets else sorted(rng.sample(range(8), rng.randint(1, 8)))
    if targets and len(targets) < 8 and rng.random() < 0.2:
        indices.append(len(targets))     # a legal mrt index, but past the declared SV_Target count
    exports = []
    for k in indices:
        if rng.random() < 0.3:
            exports.append(PsExport("mrt", k, False, True))
        elif rng.random() < 0.3:
            exports.append(PsExport("mrt", k, False, False, _components(rng)))
        else:
            exports.append(PsExport("mrt", k, False, False))
    if rng.random() < 0.5:
        exports.append(PsExport("mrtz", 0, rng.random() < 0.5, rng.random() < 0.5))
    rng.shuffle(exports)
    return make_case("ps_" + name, "ps", PsShape(targets, has_osg1), exports)


# ---- the comparator --------------------------------------------------------------------------------------------------
def module_structure(spvasm):
    """Output declarations and distance stores, in the models' canonical form."""
    lines = [f"Location {var} {n}" for var, n in re.findall(r"OpDecorate (%out_\S+) Location (\d+)", spvasm)]
    lines += [f"BuiltIn {var} {built_in}" for var, built_in in re.findall(r"OpDecorate (%out_\S+) BuiltIn (\S+)", spvasm)]
    for array, var in (("arr_clip", "%out_clip"), ("arr_cull", "%out_cull")):
        found = re.search(rf"%{array} = OpTypeArray %float %c(\d+)", spvasm)
        if found:
            lines.append(f"Array {var} {found.group(1)}")
    # Follow each output access chain through its store's OpSelect to the register read for that export.
    # The generated posN source registers are disjoint, so the register itself identifies (N, component).
    sources, pointers = {}, {}
    for line in spvasm.splitlines():
        line = line.strip()
        read = re.fullmatch(r"(%\w+) = OpLoad %uint %reg_v(\d+)", line)
        cast = re.fullmatch(r"(%\w+) = OpBitcast %float (%\w+)", line)
        select = re.fullmatch(r"(%\w+) = OpSelect %float %\w+ (%\w+) %\w+", line)
        chain = re.fullmatch(r"(%\w+) = OpAccessChain %_ptr_Output_float %out_(clip|cull) %c([0-9a-f]+)", line)
        store = re.fullmatch(r"OpStore (%\w+) (%\w+)", line)
        if read:
            reg = int(read[2])
            sources[read[1]] = f"pos{reg // 4}.{reg % 4}"
        elif cast:
            sources[cast[1]] = sources.get(cast[2], "unknown")
        elif select:
            sources[select[1]] = sources.get(select[2], "unknown")
        elif chain:
            pointers[chain[1]] = (chain[2], int(chain[3], 16))
        elif store and store[1] in pointers:
            kind, idx = pointers[store[1]]
            lines.append(f"Distance {sources.get(store[2], 'unknown')} -> {kind}[{idx}]")
    return sorted(lines)


def compare(spvasm, expected):
    """None when the module's output structure is exactly `expected`; otherwise what is missing and what is extra."""
    actual = module_structure(spvasm)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if not missing and not extra:
        return None
    return "; ".join(part for part in ("missing: " + ", ".join(missing) if missing else "",
                                       "unexpected: " + ", ".join(extra) if extra else "") if part)


def run_case(case):
    """('match' | 'skip' | 'mismatch', detail) for one program."""
    try:
        spvasm = support.translate_lines(case.name, case.listing, dxbc=case.dxbc)
    except rdna2spirv.Unsupported as refusal:
        # The translator's message: "export mrtK past the N declared SV_Target outputs".
        if case.refusals and any(f"export {r} past the" in str(refusal) for r in case.refusals):
            return "skip", f"expected refusal: {refusal}"
        return "mismatch", f"the translator refused what the model accepts: Unsupported: {refusal}"
    message = compare(spvasm, case.expected)
    if message:
        return "mismatch", message
    if case.refusals:
        return "mismatch", f"the model expected a refusal of {', '.join(case.refusals)}; the translator accepted it"
    return "match", ""


# ---- canaries --------------------------------------------------------------------------------------------------------
CANARIES = ("clip/cull swap", "sparse location", "compr depth", "misc shift", "distance index")


def self_test():
    """The canaries compare() failed to flag: each translates a real program and predicts its structure wrongly in
    one named way, so an empty list means the comparator still sees what it is meant to see."""
    missed = []

    def canary(name, case, wrong):
        status, detail = run_case(case)
        if status != "match":
            missed.append(f"{name} (the model itself: {detail})")
            return
        spvasm = support.translate_lines(case.name, case.listing, dxbc=case.dxbc)
        if compare(spvasm, wrong) is None:
            missed.append(name)

    clip_cull = [Osg1Element(D3D_NAME["clip"], 0, 0, 0x1, 0), Osg1Element(D3D_NAME["cull"], 0, 1, 0x1, 0)]
    both = [VsExport("pos", 0, ON), VsExport("pos", 1, ON)]
    canary(CANARIES[0], make_case("vs_canary_swap", "vs", VsShape(clip_cull, True), both),
           ["BuiltIn %out_pos Position", "BuiltIn %out_cull CullDistance", "Array %out_cull 2"])
    sparse = make_case("ps_canary_sparse", "ps", PsShape([0, 2], True), [PsExport("mrt", 0, False, False),
                                                                         PsExport("mrt", 1, False, False)])
    canary(CANARIES[1], sparse, ["Location %out_mrt0 0", "Location %out_mrt1 1"])
    depth = make_case("ps_canary_depth", "ps", PsShape([], False), [PsExport("mrtz", 0, True, True)])
    canary(CANARIES[2], depth, ["BuiltIn %out_depth FragDepth"])
    shifted = make_case("vs_canary_shift", "vs", VsShape(clip_cull[:1] + [Osg1Element(D3D_NAME["rtai"], 0, 1, 0x1, 0)],
                                                         True), both)
    canary(CANARIES[3], shifted, ["BuiltIn %out_pos Position", "BuiltIn %out_layer Layer",
                                   "BuiltIn %out_clip ClipDistance", "Array %out_clip 1"])
    clip_pair = make_case("vs_canary_indices", "vs", VsShape([
        Osg1Element(D3D_NAME["clip"], 0, 0, 0x3, 0)], True),
        [VsExport("pos", 1, (1, 1, 0, 0))])
    canary(CANARIES[4], clip_pair, [line.replace("Distance pos1.1 -> clip[1]", "Distance pos1.1 -> clip[0]")
                                    for line in clip_pair.expected])
    return missed


# ---- the runner ------------------------------------------------------------------------------------------------------
def describe(case):
    return "\n".join([f"{case.name} ({case.stage})"] + ["    " + ln for ln in case.listing] +
                     ["  model: " + (", ".join(case.expected) or "no outputs") +
                      ("  refused: " + ", ".join(case.refusals) if case.refusals else "")])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Differential fuzzer for VS/PS export shapes.")
    parser.add_argument("--programs", type=int, help="how many programs of the seed to run")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--program", type=int, metavar="K", help="with --show: the program to print")
    parser.add_argument("--show", action="store_true", help="print one program, its prediction and the module")
    parser.add_argument("--self-test", action="store_true", help="run the comparator canaries")
    parser.add_argument("--list-ops", action="store_true", help="list the export forms the generator emits")
    args = parser.parse_args(argv)
    if args.programs is None and not (args.self_test or args.list_ops or args.show):
        parser.print_usage(sys.stderr)      # a bare invocation is a usage error, never the start of a long run
        return 2
    if args.programs is not None and args.programs < 1:
        print("--programs must be at least 1", file=sys.stderr)
        return 2
    if args.self_test:
        missed = self_test()
        for name in missed:
            print(f"CANARY MISSED: {name}")
        print(f"self-test: {len(CANARIES) - len(missed)} of {len(CANARIES)} canaries caught")
        return 2 if missed else 0
    if args.list_ops:
        for line in ("exp mrtK v0, v1, v2, v3 [done vm]", "exp mrtK v0, v0, v1, v1 compr [done vm]",
                     "exp mrtz {v0 | off}, off, off, off [compr] [done vm]",
                     "exp pos0..3 {vN | off} x4 [done vm]", "exp paramN {vN | off} x4 [done vm]"):
            print(line)
        return 0
    if args.show:
        if args.program is None:
            parser.print_usage(sys.stderr)
            print("export_diff: --show requires --program", file=sys.stderr)
            return 2
        try:
            case = generate(args.seed, args.program)
            print(describe(case))
            status, detail = run_case(case)
        except Exception as e:  # noqa: BLE001 - a harness bug must be loud, and never pass for a mismatch
            print(f"harness error {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        print("  translator: " + ("matches the model" if status == "match" else detail))
        return 1 if status == "mismatch" else 0
    matched, skipped, reasons = 0, 0, Counter()
    for index in range(args.programs):
        try:
            case = generate(args.seed, index)
            status, detail = run_case(case)
        except Exception as e:  # noqa: BLE001 - a harness bug must be loud, and never pass for a mismatch
            print(f"seed {args.seed}, program {index}: harness error {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        if status == "match":
            matched += 1
        elif status == "skip":
            skipped += 1
            reasons[detail] += 1
        else:
            print("MISMATCH\n" + describe(case) + f"\n  {detail}", file=sys.stderr)
            print(f"seed {args.seed}, program {index}: mismatch", file=sys.stderr)
            return 1
    if not matched:
        print("no program ran: every one was skipped", file=sys.stderr)
        return 2
    summary = f"seed {args.seed}: {matched} programs matched the model, {skipped} skipped"
    print(summary + (": " + "; ".join(f"{n} x {why}" for why, n in reasons.items()) if skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

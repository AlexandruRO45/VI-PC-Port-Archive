"""Check user-defined varying interfaces between translated graphics stages.

Run `python -I tools/shader_link_check.py --help` for standalone pairs, batch checks,
and reflection of a single SPIR-V module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402
import fssm_header as fssm  # noqa: E402
import spirv_interface as spv  # noqa: E402
from spirv_interface import (Address, Declaration, InterfaceError, NumericType,  # noqa: E402
                             StageInterface, reflect_module)


STAGES = ("vs", "gs", "ms", "ps")
CHAINS = {("vs", "ps"), ("vs", "gs", "ps"), ("ms", "ps"), ("vs", "gs")}
# A manifest names a shader by its file stem in --shader-dir. The stem must stay one path component inside that
# folder: no separators or drive colon, not "." or "..", and no character a Windows file name can't hold. Anything
# else is allowed: make_game's corpus names use "~", "!", "&", "(" and a leading ".".
BAD_STEM = re.compile(r'[\\/:<>"|?*\x00-\x1f]')


def is_shader_stem(reference):
    return isinstance(reference, str) and reference not in ("", ".", "..") \
        and not BAD_STEM.search(reference) and reference == reference.strip()


@dataclass(frozen=True)
class ModulePath:
    path: Path
    stage: str
    entry: str = "main"


@dataclass(frozen=True)
class Pipeline:
    name: str
    modules: tuple[ModulePath, ...]


@dataclass(frozen=True)
class Problem:
    kind: str
    consumer: str
    message: str
    producer: str | None = None
    location: Address | None = None

    def display(self):
        stages = f"{self.producer} -> {self.consumer}" if self.producer else self.consumer
        where = f" at {self.location.describe()}" if self.location is not None else ""
        return f"{stages}{where} [{self.kind}] {self.message}"


@dataclass
class PipelineResult:
    name: str
    status: str
    links_checked: int
    problems: list[Problem]


def wider_vector_compatible(producer, consumer, allow):
    if not allow or producer.structure or consumer.structure:
        return False
    if producer.indexed_by_vertex_or_primitive or consumer.indexed_by_vertex_or_primitive:
        return False
    p, c = producer.varyings[0].type, consumer.varyings[0].type
    # maintenance4 relaxes the length of vector variables. Scalar vs vector, nested types,
    # and arrays which were unwrapped for a stage's invocation index still need exact types.
    return (isinstance(p, NumericType) and isinstance(c, NumericType)
            and p.kind == c.kind and p.width == c.width and p.components > c.components >= 2)


def compare_stages(producer: StageInterface, consumer: StageInterface, maintenance4=False):
    """Check consumer declarations against the preceding stage's output declarations."""
    output_slots = {slot: declaration
                    for declaration in producer.outputs
                    for varying in declaration.varyings
                    for slot in varying.occupied()}
    problems = []
    for expected in consumer.inputs:
        anchor = expected.varyings[0].address
        actual = output_slots.get(anchor)

        def problem(kind, message):
            return Problem(kind, consumer.stage, message, producer.stage, anchor)

        if actual is None:
            problems.append(problem("missing-output", f"consumer expects {expected.describe()}"))
            continue
        if actual.structure != expected.structure or actual.block != expected.block:
            problems.append(problem("structure-mismatch", f"{actual.describe()} cannot match {expected.describe()}"))
            continue
        if tuple(v.address for v in actual.varyings) != tuple(v.address for v in expected.varyings):
            problems.append(problem("location-mismatch",
                                    f"{actual.describe()} starts or places its members differently from "
                                    f"{expected.describe()}"))
            continue
        if tuple(v.per_primitive for v in actual.varyings) != tuple(v.per_primitive for v in expected.varyings):
            problems.append(problem("per-primitive-mismatch",
                                    f"producer declares {actual.describe()}, consumer expects {expected.describe()}"))
            continue
        if tuple(v.type for v in actual.varyings) != tuple(v.type for v in expected.varyings):
            if wider_vector_compatible(actual, expected, maintenance4):
                continue
            hint = (" (a wider output vector requires maintenance4)"
                    if wider_vector_compatible(actual, expected, True) else "")
            problems.append(problem("type-mismatch",
                                    f"producer declares {actual.describe()}, consumer expects {expected.describe()}"
                                    f"{hint}"))
    return problems


def stage_chain(refs):
    stages = tuple(stage for stage in STAGES if stage in refs)
    if stages not in CHAINS:
        raise ValueError("expected vs+ps, vs+gs, vs+gs+ps, or ms+ps")
    return stages


def reject_duplicate_keys(pairs):
    doc = {}
    for key, value in pairs:
        if key in doc:
            raise ValueError(f"duplicate key {key!r}")
        doc[key] = value
    return doc


def load_pipeline_manifest(path: Path, shader_dir: Path):
    try:
        with path.open(encoding="utf-8") as source:
            doc = json.load(source, object_pairs_hook=reject_duplicate_keys)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if (not isinstance(doc, dict) or set(doc) - {"pipelines", "version"}
            or type(doc.get("version", 1)) is not int or doc.get("version", 1) != 1):
        raise ValueError("expected a version 1 object with a pipelines list")
    if not isinstance(doc.get("pipelines"), list) or not doc["pipelines"]:
        raise ValueError("pipelines must be a nonempty list")
    names = set()
    pipelines = []
    for number, row in enumerate(doc["pipelines"], 1):
        try:
            if not isinstance(row, dict) or set(row) - {"name", *STAGES}:
                raise ValueError("unexpected pipeline field")
            name = row.get("name")
            if not isinstance(name, str) or not name.strip() or any(ord(ch) < 32 for ch in name):
                raise ValueError("expected one printable, nonempty pipeline name")
            if name in names:
                raise ValueError(f"duplicate pipeline name {name!r}")
            stages = stage_chain(row)
            modules = tuple(manifest_module(row[stage], shader_dir, stage) for stage in stages)
            names.add(name)
            pipelines.append(Pipeline(name, modules))
        except ValueError as exc:
            raise ValueError(f"pipelines[{number - 1}]: {exc}") from exc
    return pipelines


def manifest_module(reference, shader_dir, stage):
    entry = "main"
    if isinstance(reference, dict):
        if set(reference) - {"shader", "entry"}:
            raise ValueError(f"{stage}: unexpected shader reference field")
        entry, reference = reference.get("entry", entry), reference.get("shader")
    if not is_shader_stem(reference):
        raise ValueError(f"{stage}: expected a shader name without directories")
    if not isinstance(entry, str) or not entry or "\x00" in entry:
        raise ValueError(f"{stage}: expected a nonempty entry-point name")
    return ModulePath(shader_dir / (reference + ".spv"), stage, entry)


class InterfaceCache:
    def __init__(self):
        self.modules = {}

    def load(self, module):
        if module not in self.modules:
            try:
                self.modules[module] = reflect_module(module.path.read_bytes(), module.stage, module.entry)
            except (OSError, InterfaceError) as exc:
                self.modules[module] = InterfaceError(f"{module.path}: {exc}")
        result = self.modules[module]
        if isinstance(result, InterfaceError):
            raise result
        return result


def check_pipeline(pipeline, cache, maintenance4=False):
    interfaces = {}
    problems = []
    for module in pipeline.modules:
        try:
            interfaces[module.stage] = cache.load(module)
        except (OSError, InterfaceError) as exc:
            problems.append(Problem("module-error", module.stage, str(exc)))
    links = 0
    invalid = bool(problems)
    for producer, consumer in zip(pipeline.modules, pipeline.modules[1:]):
        if producer.stage in interfaces and consumer.stage in interfaces:
            links += 1
            problems.extend(compare_stages(interfaces[producer.stage], interfaces[consumer.stage], maintenance4))
    status = "invalid" if invalid else ("incompatible" if problems else "compatible")
    return PipelineResult(pipeline.name, status, links, problems)


def write_report(path: Path, doc):
    path = path.absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", delete=False,
                                         dir=path.parent, prefix=".shader-link-", suffix=".tmp") as target:
            temporary = Path(target.name)
            json.dump(doc, target, indent=2, ensure_ascii=False)
            target.write("\n")
        os.chmod(temporary, published_mode(path))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_report_is_separate(output, inputs, label="report"):
    if output is None:
        return
    out = output.resolve()
    try:
        out_stat = out.stat()
    except (FileNotFoundError, NotADirectoryError):
        out_stat = None
    for source in dict.fromkeys(inputs):
        if out_stat is None:
            aliases = out == source.resolve()
        else:
            try:
                # Existing targets can be compared by file identity, including hard links.
                aliases = os.path.samestat(out_stat, source.stat())
            except (FileNotFoundError, NotADirectoryError):
                aliases = out == source.resolve()
        if aliases:
            raise ValueError(f"{label} would overwrite an input file: {source}")


def inspect(args):
    ensure_report_is_separate(args.report, [args.module])
    interface = reflect_module(args.module.read_bytes(), args.stage, args.entry)
    doc = {
        "stage": interface.stage,
        "entry": interface.entry,
        "inputs": [declaration_row(item) for item in interface.inputs],
        "outputs": [declaration_row(item) for item in interface.outputs],
    }
    if args.report:
        write_report(args.report, doc)
    else:
        print(json.dumps(doc, indent=2))
    return 0


def declaration_row(declaration):
    return {
        "id": declaration.symbol,
        "structure": declaration.structure,
        "block": declaration.block,
        "indexed_by_vertex_or_primitive": declaration.indexed_by_vertex_or_primitive,
        "varyings": [{**asdict(varying.address), "type": varying.type.describe(),
                      "per_primitive": varying.per_primitive} for varying in declaration.varyings],
    }


def parameter_ids(declarations):
    """Plain one-location per-vertex variables emitted by the native mesh/fragment translation."""
    by_location = {}
    for decl in declarations:
        if decl.structure or len(decl.varyings) != 1:
            raise ValueError("expected standalone shader parameters")
        varying, = decl.varyings
        addr = varying.address
        if (addr.index or addr.component or varying.per_primitive or varying.type != NumericType("f", 32, 4)
                or addr.location in by_location):
            raise ValueError("expected one float vec4 per parameter location")
        by_location[addr.location] = decl.symbol
    return by_location


def link_mesh_fragment(mesh_spv, pixel_spv, mesh_fssm, pixel_fssm):
    export_slots, input_slots = fssm.mesh_pixel_slots(mesh_fssm, pixel_fssm)
    producer = reflect_module(mesh_spv, "ms")
    module = spv.Module(pixel_spv)
    if len(module.entry_points) != 1:
        raise ValueError("expected a single fragment entry point")
    consumer = module.reflect("ps")
    inputs = parameter_ids(consumer.inputs)
    if set(parameter_ids(producer.outputs)) != export_slots or set(inputs) != set(range(len(input_slots))):
        raise ValueError("native tables do not match the mesh exports and dense translated pixel inputs")
    if set(input_slots) == set(inputs) and input_slots != list(range(len(inputs))):
        raise ValueError("a dense permutation cannot distinguish original from already linked locations")

    # Edit only direct Location literals on the original fragment input IDs. The same numeric
    # location can also occur on a render target, or be an operand of an unrelated instruction.
    edits = {inputs[i]: slot for i, slot in enumerate(input_slots) if i != slot}
    words = spv.parse_words(pixel_spv)
    linked = bytearray(pixel_spv)
    endian = "<" if pixel_spv[:4] == struct.pack("<I", spv.SPIRV_MAGIC) else ">"
    for opcode, operands, offset in spv.instructions(words):
        if opcode == spv.OP_DECORATE and operands[1] == spv.DEC_LOCATION and operands[0] in edits:
            struct.pack_into(endian + "I", linked, 4 * (offset + 3), edits.pop(operands[0]))
    if edits:
        raise ValueError("expected direct Location decorations on translated inputs")
    linked_interface = reflect_module(linked, "ps")
    problems = compare_stages(producer, linked_interface)
    if problems:
        raise ValueError(problems[0].display())
    return bytes(linked), input_slots


def link_mesh(args):
    ensure_report_is_separate(args.out, [args.ms, args.ps, args.ms_fssm, args.ps_fssm], "linked module")
    with args.ms_fssm.open("rb", buffering=0) as f:
        mesh_header = f.read(fssm.MESH_EXPORT_COUNT + 4)
    with args.ps_fssm.open("rb", buffering=0) as f:
        # Include one more record so an oversized table still fails validation.
        pixel_header = f.read(fssm.TABLE_OFFSET + (fssm.MAX_MESH_PARAMS + 1) * fssm.RECORD_SIZE)
    linked, slots = link_mesh_fragment(args.ms.read_bytes(), args.ps.read_bytes(), mesh_header, pixel_header)
    # Publish only after checking the complete pair. Keep the translated input available for
    # other pipelines that use the same fragment shader with a different producer.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.out.parent, prefix=".mesh-link-", delete=False) as f:
            temporary = Path(f.name)
            f.write(linked)
        os.chmod(temporary, published_mode(args.out.absolute()))
        os.replace(temporary, args.out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("Fragment input ordinals map to mesh exports:", ", ".join(map(str, slots)))
    return 0


def run_checks(args):
    extra_inputs = []
    if args.command == "batch":
        pipelines = load_pipeline_manifest(args.manifest, args.shader_dir)
        extra_inputs = [args.manifest]
    else:
        refs = {stage: getattr(args, stage) for stage in STAGES if getattr(args, stage) is not None}
        pipelines = [Pipeline("pipeline", tuple(ModulePath(refs[stage], stage, args.entry)
                                                for stage in stage_chain(refs)))]
    ensure_report_is_separate(args.report, [*extra_inputs,
                                            *(module.path for pipeline in pipelines for module in pipeline.modules)])
    cache = InterfaceCache()
    results = [check_pipeline(pipeline, cache, args.maintenance4) for pipeline in pipelines]
    summary = {state: sum(result.status == state for result in results)
               for state in ("compatible", "incompatible", "invalid")}
    if args.report:
        write_report(args.report, {"version": 1, "maintenance4": args.maintenance4, "summary": summary,
                                   "pipelines": [asdict(result) for result in results]})
    for result in results:
        if args.quiet and not result.problems:
            continue
        print(f"{result.name}: {result.status} ({result.links_checked} links checked)")
        for problem in result.problems:
            print(f"  {problem.display()}")
    print(", ".join(f"{count} {state}" for state, count in summary.items()))
    if summary["invalid"]:
        return 2
    return int(bool(summary["incompatible"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    direct = commands.add_parser("check", help="check one graphics pipeline from SPIR-V file paths")
    for stage in STAGES:
        direct.add_argument("--" + stage, type=Path, metavar="SPV")
    direct.add_argument("--entry", default="main", help="entry-point name used by all direct paths")
    batch = commands.add_parser("batch", help="check a JSON list of pipelines sharing a shader directory")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--shader-dir", type=Path, required=True, help="directory holding the named .spv files")
    for command in (direct, batch):
        command.add_argument("--maintenance4", action="store_true",
                             help="allow wider output vectors when the pipeline enables maintenance4")
        command.add_argument("--quiet", action="store_true", help="display diagnostics and the summary")
        command.add_argument("--report", type=Path, help="write structured results to JSON")
    single = commands.add_parser("inspect", help="show the reflected user inputs and outputs of one entry point")
    single.add_argument("module", type=Path)
    single.add_argument("--stage", choices=STAGES, help="select a stage when multiple entry points share a name")
    single.add_argument("--entry", default="main")
    single.add_argument("--report", type=Path, help="write the reflection to JSON")
    mesh = commands.add_parser("link-mesh", help="link a translated native MeshEXT/fragment pair using its FSSM headers")
    mesh.add_argument("--ms", type=Path, required=True, help="translated mesh module")
    mesh.add_argument("--ps", type=Path, required=True, help="original dense translated fragment module")
    mesh.add_argument("--ms-fssm", type=Path, required=True, help="native header for --ms")
    mesh.add_argument("--ps-fssm", type=Path, required=True, help="native header for --ps")
    mesh.add_argument("--out", type=Path, required=True, help="separate linked fragment module")
    args = parser.parse_args(argv)
    try:
        if args.command == "link-mesh":
            return link_mesh(args)
        return inspect(args) if args.command == "inspect" else run_checks(args)
    except (OSError, ValueError) as exc:
        print(f"shader_link_check: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())

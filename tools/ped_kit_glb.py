"""Export the selected skinned components and textures of one ped appearance kit."""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dds_png
import glb_writer
import ped_catalogues
import ped_kit_parts
import skinned_glb
from report_paths import same_report_path


def _protect_component(output: Path, sidecar: Path) -> None:
    """Allow a new GLB beside a component, but never replace its source geometry or buffers."""
    model = skinned_glb._load_json(str(sidecar))
    buffers = model.get("buffer_files", [])
    if not isinstance(buffers, list):
        raise ValueError(f"{sidecar}: buffer_files must be a list")
    names = [model.get("glb", "model.glb"), *buffers]
    if same_report_path(output, sidecar):
        raise ValueError(f"output would replace the component sidecar: {sidecar}")
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"{sidecar}: invalid component filename")
        if same_report_path(output, sidecar.parent / name):
            raise ValueError(f"output would replace a component input: {name}")


def export_kit(catalogue: Path, kit_id: int, index: Path, skeleton: Path, output: Path) -> dict:
    """Resolve one catalogue selection against a read-only parts index and write a standalone GLB."""
    if output.suffix.lower() != ".glb":
        raise ValueError("output must have a .glb extension")
    _, skeleton_json = skinned_glb.model_json_path(str(skeleton))
    if any(same_report_path(output, path) for path in (catalogue, index, skeleton_json)):
        raise ValueError("output would replace a catalogue, index or skeleton input")
    if output.exists():
        _protect_component(output, Path(skeleton_json))

    members, warnings = ped_catalogues.resolve_kit(ped_catalogues._read_catalogue_file(str(catalogue)), kit_id)
    if warnings:
        raise ValueError("catalogue reports an incomplete or uncertain selection: " + "; ".join(warnings))

    # A kit supplies no placement transform for a second copy of the same drawable.
    # Identical appearances need one instance. Conflicting overrides need a slot assignment.
    selections: dict[int, tuple[int, int, int]] = {}
    for member in members:
        hashes = (member.albedo_hash, member.normal_hash, member.extra_hash)
        if not member.drawable_hash:
            if any(hashes):
                raise ValueError(f"preset 0x{member.preset_id:08x} has material references but no drawable")
            continue
        existing = selections.setdefault(member.drawable_hash, hashes)
        if existing != hashes:
            raise ValueError(f"ambiguous selections for drawable {member.drawable_hash:016X} within one kit")
    if not selections:
        raise ValueError(f"kit 0x{kit_id:08x} has no drawable to export")

    components = []
    appearances = []
    images: dict[int, bytes] = {}
    with closing(sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("BEGIN")  # use one snapshot for geometry and texture lookups
        if output.exists():
            ped_kit_parts.protect_indexed_glbs(db, output)
        for drawable, (albedo, normal, _extra) in selections.items():
            sidecar = ped_kit_parts.component_path(db, drawable, textured=bool(albedo or normal))
            if output.exists():
                _protect_component(output, sidecar)
            components.append(str(sidecar))
            for key in (albedo, normal):
                if key and key not in images:
                    path = ped_kit_parts.texture_path(db, key)
                    if same_report_path(output, path):
                        raise ValueError(f"output would replace texture {key:016X}")
                    images[key] = dds_png.dds_file_to_png_bytes(str(path))
            appearances.append((images.get(albedo), images.get(normal)))

    payload, stats = skinned_glb.build_skinned_glb(
        skeleton_json, components, component_textures=appearances, generator="Leonida ped_kit_glb"
    )
    glb_writer._write_bytes_atomic(str(output), payload)
    return {"drawables": len(components), "textures": len(images), "triangles": stats["triangles"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path, help="standalone PSIN or 0DRP ped catalogue")
    parser.add_argument("--id", type=ped_catalogues._kit_id, required=True, help="decimal or 0x-prefixed kit ID")
    parser.add_argument("--index", type=Path, required=True, help="SQLite database produced by viewer_parts_index.py")
    parser.add_argument("--skeleton", type=Path, required=True, help="decoded master skeleton directory or model.json")
    parser.add_argument("--out", type=Path, required=True, help="output skinned GLB")
    args = parser.parse_args(argv)
    try:
        stats = export_kit(args.catalogue, args.id, args.index, args.skeleton, args.out)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"ped_kit_glb: {exc}", file=sys.stderr)
        return 2
    print(f"{args.out}: {stats['drawables']} drawable(s), {stats['textures']} texture key(s), "
          f"{stats['triangles']} triangle(s)")
    return 0


if __name__ == "__main__":
    try:  # A redirected Windows console may not encode the output path.
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())

"""SGD5 shader texture parameter hash registry and post-pass parameter table reader.

Maps 32-bit joaat (Jenkins one-at-a-time, lowercased) shader texture parameter hashes used in
drawable material slot tables and SGD5 effect resource tables to semantic texture roles and
normalized renderer binding categories.

Every hash is computed programmatically via ``hashdict.joaat``. The 102 built-in entries cover
12 aliased (primary != secondary) texture parameters and 90 unaliased (primary == secondary)
parameters verified against both SGD5 databases in Xbox 1.0.2.1 ``shaders.rpf``:
  - 102 / 102 exact (primary_hash, secondary_hash) matches in type-2 (texture) resource slots
  - 12 / 12 aliased type-2 texture pairs (1,140 / 1,140 slots, 100%)
  - 3,345 / 4,916 total type-2 texture slots across all 932 effects (68.0%)

Usage:
    python -I tools/sgd5_params.py --list
    python -I tools/sgd5_params.py --lookup HASH [HASH ...]
    python -I tools/sgd5_params.py --pair PRIMARY_HASH SECONDARY_HASH
    python -I tools/sgd5_params.py --hash NAME [NAME ...]
    python -I tools/sgd5_params.py --archive PATH/TO/shaders.rpf
"""
import argparse
import json
from pathlib import Path
import re
import struct
import sys
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hashdict import joaat
import sgd5_effects

HEX32 = re.compile(r"(?:0[xX])?[0-9a-fA-F]{1,8}")
TEXTURE_RESOURCE_TYPE = 2
SAMPLER_RESOURCE_TYPE = 35
CBUFFER_RESOURCE_TYPE = 48

CATEGORIES = (
    "base_color",
    "normal",
    "specular",
    "material_map",
    "emissive",
    "tint",
    "mask",
    "detail",
)


class TextureParam(NamedTuple):
    role: str
    category: str
    secondary: str
    primary: str
    primary_hash: int
    secondary_hash: int
    aliased: bool

    def to_dict(self):
        return {
            "role": self.role,
            "category": self.category,
            "secondary": self.secondary,
            "primary": self.primary,
            "primary_hash": f"{self.primary_hash:08x}",
            "secondary_hash": f"{self.secondary_hash:08x}",
            "aliased": self.aliased,
            "evidence": "paired SGD5 hashes" if self.aliased else "single SGD5 hash",
        }


def _layer_param(layer, role, category, suffix=None):
    spelling = f"lyr{layer}{suffix or role}tex"
    return (f"Layer {layer} {role}", category, spelling, spelling)


def _uber_param(layer, role, category, prefix):
    spelling = f"{prefix}_layer{layer}"
    return (f"Uber layer {layer} {role}", category, spelling, spelling)


# (role, category, secondary/material spelling, primary/SGD5 spelling)
_RAW_PARAMETERS = (
    # Paired (aliased) SGD5 texture parameters (primary_hash != secondary_hash)
    ("Base color", "base_color", "diffusetex", "diffusetexture"),
    ("Bump / normal", "normal", "bumptex", "bumptexture"),
    ("Specular", "specular", "speculartex", "spectexture"),
    ("Secondary specular", "specular", "speculartex2", "spectexture2"),
    ("Emissive", "emissive", "emissivetex", "emissivetexture"),
    ("Clearcoat", "specular", "clearcoattex", "clearcoattexture"),
    ("Anisotropy", "specular", "anisotex", "anisotexture"),
    ("Iridescence", "specular", "iridescencetex", "iridescencetexture"),
    ("Diffraction", "specular", "diffractiontex", "diffractiontexture"),
    ("Noise / fur noise", "detail", "noisetexture", "furnoisetexture"),
    ("Translucency", "detail", "translucencytex", "translucencytexture"),
    ("Specular color", "specular", "specularcolortex", "specularcolortexture"),
    # Single-hash SGD5 texture parameters (primary_hash == secondary_hash)
    ("Secondary diffuse", "base_color", "diffusetex2", "diffusetex2"),
    ("Tint palette", "tint", "tintpalettetex", "tintpalettetex"),
    ("Height", "detail", "heighttexture", "heighttexture"),
    ("Retroreflection", "specular", "retroreflectiontexture", "retroreflectiontexture"),
    ("Retroreflection (short)", "specular", "retroreflectiontex", "retroreflectiontex"),
    ("Flow", "detail", "flowtexture", "flowtexture"),
    ("Roughness", "specular", "roughnessmap", "roughnessmap"),
    # Multi-layer 2/3/4-layer surface parameters
    *(_layer_param(i, "diffuse", "base_color") for i in range(4)),
    *(_layer_param(i, "normal", "normal") for i in range(4)),
    *(_layer_param(i, "material A", "material_map", "materiala") for i in range(4)),
    *(_layer_param(i, "material B", "material_map", "materialb") for i in range(4)),
    *(_layer_param(i, "control", "mask") for i in (2, 3)),
    ("Dirt diffuse", "base_color", "dirtdiffusetex", "dirtdiffusetex"),
    ("Layer blend control", "mask", "controltexturetex", "controltexturetex"),
    # Terrain multi-layer parameters
    ("Uber terrain lookup", "detail", "lookuptexture", "lookuptexture"),
    ("Uber terrain grass mask", "mask", "grassmasktexture", "grassmasktexture"),
    ("Uber terrain tint", "tint", "tinttexture", "tinttexture"),
    ("Uber terrain tint bump", "normal", "tintbumptexture", "tintbumptexture"),
    *(_uber_param(i, "diffuse", "base_color", "diffusetexture") for i in range(4)),
    *(_uber_param(i, "bump", "normal", "bumptexture") for i in range(4)),
    *(_uber_param(i, "material A", "material_map", "materialmapatexture") for i in range(4)),
    # Terrain LOD / secondary normal & material parameters
    ("Bump / normal (unaliased)", "normal", "bumptexture", "bumptexture"),
    ("Specular (unaliased)", "specular", "speculartexture", "speculartexture"),
    ("Material B", "material_map", "materialbtexture", "materialbtexture"),
    ("Secondary bump / normal", "normal", "bumptexture2", "bumptexture2"),
    ("Secondary specular (unaliased)", "specular", "speculartexture2", "speculartexture2"),
    ("Secondary material B", "material_map", "materialbtexture2", "materialbtexture2"),
    # Vehicle weathering & alternate glass/surface sets
    ("Dirt noise", "detail", "dirtnoisetex", "dirtnoisetex"),
    ("Dust", "detail", "dusttexture", "dusttexture"),
    ("Scratch", "detail", "scratchtexture", "scratchtexture"),
    ("Alt base color", "base_color", "altdiffusetex", "altdiffusetex"),
    ("Alt bump / normal", "normal", "altbumptexture", "altbumptexture"),
    ("Alt specular", "specular", "altspectexture", "altspectexture"),
    ("Alt secondary specular", "specular", "altspectexture2", "altspectexture2"),
    # Interior diffuse & mask maps
    ("Interior diffuse", "base_color", "interiordiffusemap", "interiordiffusemap"),
    ("Interior layer diffuse", "base_color", "interiorlayerdiffusemap", "interiorlayerdiffusemap"),
    ("Interior diffuse mask", "mask", "interiordiffusemaskmap", "interiordiffusemaskmap"),
    # Alpha / glass / refraction / liquid
    ("Refraction bump", "normal", "refractionbumptexture", "refractionbumptexture"),
    ("Extinction LUT", "detail", "extinctionlut", "extinctionlut"),
    ("Looking glass blur", "detail", "lookingglassblurtexture", "lookingglassblurtexture"),
    ("Liquid extinction LUT", "detail", "liquidextinctionlut", "liquidextinctionlut"),
    ("Liquid surface tint mask", "tint", "liquidsurfacetintmask", "liquidsurfacetintmask"),
    ("Liquid tint mask", "tint", "liquidtintmask", "liquidtintmask"),
    ("Liquid tint palette", "tint", "liquidtintpalette", "liquidtintpalette"),
    ("Liquid surface bump", "normal", "liquidsurfacebumptex", "liquidsurfacebumptex"),
    # Water detail & height maps
    ("Water surface detail", "detail", "watersurfacedetailtexture", "watersurfacedetailtexture"),
    ("Water surface map", "detail", "watersurfacemaptexture", "watersurfacemaptexture"),
    ("Water height", "detail", "waterheighttexture", "waterheighttexture"),
    *((f"Surface detail {i}", "detail", f"surfacedetailtexture{i}", f"surfacedetailtexture{i}") for i in range(3)),
    *((f"Surface detail normal {i}", "normal", f"surfacedetailnormal{i}", f"surfacedetailnormal{i}") for i in range(3)),
    # Character / weapon / wrinkle maps
    ("Material ID", "material_map", "materialid", "materialid"),
    ("Material C", "material_map", "materialctexture", "materialctexture"),
    ("Blood flow mask", "mask", "bloodflowmask", "bloodflowmask"),
    ("Wrinkle occlusion mask", "mask", "wrinkleocclusionmask", "wrinkleocclusionmask"),
    ("Standard wrinkle", "normal", "standardwrinkletexture", "standardwrinkletexture"),
    ("Standard wrinkle mask", "mask", "standardwrinklemasktexture", "standardwrinklemasktexture"),
    # Decal text / signs / animated emissive / lightmap
    ("Font diffuse", "base_color", "fontdiffusemap", "fontdiffusemap"),
    ("Font normal", "normal", "fontnormalmap", "fontnormalmap"),
    ("Distance map", "mask", "distancemap", "distancemap"),
    ("Sign icon", "detail", "signicon", "signicon"),
    ("Animated emissive", "emissive", "emissiveanimatedtexture", "emissiveanimatedtexture"),
    ("Light vector", "emissive", "lightvectortexture", "lightvectortexture"),
    ("Scrolling emissive", "emissive", "emissivescrollingtexture", "emissivescrollingtexture"),
    ("Emissive distortion", "emissive", "emissivedistortiontexture", "emissivedistortiontexture"),
)

PARAMETERS = tuple(
    TextureParam(
        role=role,
        category=category,
        secondary=secondary,
        primary=primary,
        primary_hash=joaat(primary),
        secondary_hash=joaat(secondary),
        aliased=(primary != secondary),
    )
    for role, category, secondary, primary in _RAW_PARAMETERS
)

BY_SECONDARY_HASH = {param.secondary_hash: param for param in PARAMETERS}
BY_PAIR = {(param.primary_hash, param.secondary_hash): param for param in PARAMETERS}
_PRIMARY_GROUPS = {}
for _param in PARAMETERS:
    _PRIMARY_GROUPS.setdefault(_param.primary_hash, []).append(_param)
BY_PRIMARY_HASH = {key: tuple(items) for key, items in _PRIMARY_GROUPS.items()}


def parse_hash(value):
    """Parse an integer or 1..8 hex digit string into a 32-bit unsigned integer."""
    if isinstance(value, bool):
        raise ValueError("hash must be an integer or hexadecimal string")
    if isinstance(value, int):
        if 0 <= value <= 0xFFFFFFFF:
            return value
        raise ValueError(f"hash {value:#x} outside 32-bit unsigned range")
    if isinstance(value, str):
        text = value.strip()
        if HEX32.fullmatch(text):
            return int(text, 16)
    raise ValueError(f"invalid 32-bit hexadecimal hash: {value!r}")


def lookup_hash(value):
    """Look up a drawable material / secondary SGD5 texture parameter hash."""
    return BY_SECONDARY_HASH.get(parse_hash(value))


def lookup_pair(primary, secondary):
    """Look up an SGD5 resource entry's (primary_hash, secondary_hash) pair."""
    return BY_PAIR.get((parse_hash(primary), parse_hash(secondary)))


def classify_role(value):
    """Return the normalized binding category for a secondary texture parameter hash, or None."""
    entry = lookup_hash(value)
    return entry.category if entry is not None else None


def parse_parameter_table(data, offset, limit=None):
    """Parse one length-prefixed SGD5 post-pass parameter table at ``offset``.

    Format:
        +0   u32  size (byte length of table body)
        +4   u16  resource_count
        +6   u16  parameter_count
        +8   u16  attribute_count
        +10  u16  size16 (matches size when size >= 9)
        +12  (resource_count + parameter_count) * 24-byte entries:
             <BBH5I -> (type_code, flags, slot, word1, word2, word3, primary_hash, secondary_hash)
    Returns a dict with ``resources``, ``parameters``, ``attributes``, ``size``, and ``end``.
    """
    stop = len(data) if limit is None else min(limit, len(data))
    if offset < 0 or offset + 4 > stop:
        raise ValueError(f"truncated SGD5 parameter table header at {offset:#x}")
    size = struct.unpack_from("<I", data, offset)[0]
    body = offset + 4
    end = body + size
    if end > stop:
        raise ValueError(f"SGD5 parameter table at {offset:#x} exceeds boundary")
    if size == 0:
        return {"resources": [], "parameters": [], "attributes": 0, "size": 0, "end": end}
    if size < 9:
        raise ValueError(f"invalid SGD5 parameter table size {size} at {offset:#x}")
    res_count, param_count, attr_count, size16 = struct.unpack_from("<4H", data, body)
    first = body + 8
    aux = first + 24 * (res_count + param_count)
    tail = aux + 8 * attr_count
    if size != size16 or tail > end:
        raise ValueError(f"invalid SGD5 parameter table counts at {offset:#x}")
    resources = []
    parameters = []
    for idx in range(res_count + param_count):
        pos = first + 24 * idx
        type_code, flags, slot, w1, w2, w3, h1, h2 = struct.unpack_from("<BBH5I", data, pos)
        entry = {
            "type_code": type_code,
            "flags": flags,
            "slot": slot,
            "primary_hash": h1,
            "secondary_hash": h2,
        }
        if idx < res_count:
            entry["binding_slot"] = slot & 0xFF
            entry["attribute_count"] = slot >> 8
            if type_code == TEXTURE_RESOURCE_TYPE:
                known = BY_PAIR.get((h1, h2))
                if known is not None:
                    entry["param"] = known
            resources.append(entry)
        else:
            entry["byte_offset"] = w1
            parameters.append(entry)
    return {
        "resources": resources,
        "parameters": parameters,
        "attributes": attr_count,
        "size": size,
        "end": end,
    }


def effect_parameter_tables(data, pass_end, limit=None):
    """Parse the two post-pass parameter tables immediately following an effect's pass table."""
    stop = len(data) if limit is None else min(limit, len(data))
    if pass_end + 4 > stop:
        raise ValueError("truncated post-pass word count")
    word_count = struct.unpack_from("<I", data, pass_end)[0]
    marker_at = pass_end + 4 + 4 * word_count
    if marker_at + 4 > stop or struct.unpack_from("<I", data, marker_at)[0] != 0x80000000:
        raise ValueError("invalid post-pass marker before parameter tables")
    cursor = marker_at + 4
    tables = []
    for _ in range(2):
        table = parse_parameter_table(data, cursor, stop)
        tables.append(table)
        cursor = table["end"]
    return tables


def scan_sgd5_archive(archive_path, databases=None):
    """Scan both SGD5 databases in ``shaders.rpf`` and return aggregate texture-parameter counts only."""
    layouts = sgd5_effects.DATABASES if databases is None else databases
    distinct_pairs = set()
    matched_pairs = set()
    aliased_pairs = set()
    matched_aliased_pairs = set()
    category_slots = {cat: 0 for cat in CATEGORIES}
    effects = 0
    texture_slots = 0
    matched_slots = 0
    aliased_slots = 0
    matched_aliased_slots = 0
    per_table_slots = [0, 0]

    with Path(archive_path).open("rb") as stream:
        for database, first_group, end in layouts:
            stream.seek(database)
            if stream.read(4) != b"SGD5":
                raise ValueError(f"expected SGD5 database at {database:#x}")
            stream.seek(first_group)
            groups = sgd5_effects.program_groups(stream, end)
            count = sgd5_effects.u32(stream.read(4))
            base = stream.tell()
            if base >= end:
                raise ValueError("missing effect region")
            data = stream.read(end - base)
            if len(data) != end - base:
                raise ValueError("truncated effect region")
            for record in sgd5_effects.effect_records(data, groups, count):
                effects += 1
                for table_idx, table in enumerate(effect_parameter_tables(data, record.pass_end, record.end)):
                    for res in table["resources"]:
                        if res["type_code"] != TEXTURE_RESOURCE_TYPE:
                            continue
                        pair = (res["primary_hash"], res["secondary_hash"])
                        per_table_slots[table_idx] += 1
                        texture_slots += 1
                        distinct_pairs.add(pair)
                        is_aliased = pair[0] != pair[1]
                        if is_aliased:
                            aliased_pairs.add(pair)
                            aliased_slots += 1
                        known = res.get("param")
                        if known is not None:
                            matched_pairs.add(pair)
                            matched_slots += 1
                            category_slots[known.category] += 1
                            if is_aliased:
                                matched_aliased_pairs.add(pair)
                                matched_aliased_slots += 1

    return {
        "registry_entries": len(PARAMETERS),
        "effects": effects,
        "texture_slots": texture_slots,
        "table_slots": per_table_slots,
        "distinct_texture_pairs": len(distinct_pairs),
        "matched_texture_pairs": len(matched_pairs),
        "matched_texture_slots": matched_slots,
        "aliased_texture_pairs": len(aliased_pairs),
        "matched_aliased_pairs": len(matched_aliased_pairs),
        "aliased_texture_slots": aliased_slots,
        "matched_aliased_slots": matched_aliased_slots,
        "category_slots": category_slots,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="print the built-in shader texture parameter registry as JSON")
    mode.add_argument("--lookup", nargs="+", metavar="HASH", help="look up one or more 32-bit hex parameter hashes")
    mode.add_argument("--pair", nargs=2, metavar=("PRIMARY", "SECONDARY"),
                      help="look up an SGD5 (primary_hash, secondary_hash) pair")
    mode.add_argument("--hash", nargs="+", metavar="NAME", help="compute joaat hash(es) and show known registry match")
    mode.add_argument("--archive", type=Path, help="scan shaders.rpf SGD5 databases for aggregate texture-slot coverage")
    args = parser.parse_args(argv)

    try:
        if args.list:
            print(json.dumps([p.to_dict() for p in PARAMETERS], indent=2))
            return 0
        if args.lookup:
            rows = []
            for raw in args.lookup:
                h = parse_hash(raw)
                match = lookup_hash(h)
                rows.append({"hash": f"{h:08x}", "match": match.to_dict() if match else None})
            print(json.dumps(rows, indent=2))
            return 0 if all(r["match"] is not None for r in rows) else 1
        if args.pair:
            h1, h2 = parse_hash(args.pair[0]), parse_hash(args.pair[1])
            match = lookup_pair(h1, h2)
            print(json.dumps({"primary_hash": f"{h1:08x}", "secondary_hash": f"{h2:08x}",
                              "match": match.to_dict() if match else None}, indent=2))
            return 0 if match is not None else 1
        if args.hash:
            rows = []
            for name in args.hash:
                h = joaat(name)
                match = lookup_hash(h)
                rows.append({"name": name, "hash": f"{h:08x}", "match": match.to_dict() if match else None})
            print(json.dumps(rows, indent=2))
            return 0
        if args.archive:
            print(json.dumps(scan_sgd5_archive(args.archive), indent=2))
            return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

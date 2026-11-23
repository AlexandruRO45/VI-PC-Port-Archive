"""Inspect keyed style records in a locally decoded PSIN garment bank.

python -I tools/garment_style.py BANK --list-styles
python -I tools/garment_style.py BANK --style 0x123

Listed keys include the 0x prefix and can be passed directly to --style.
Decimal input is also accepted.
The tile lookup uses the style slot's nonzero override, then the preset tile.
The report preserves source references and typed parameter rows separately.
It describes the stored inputs to the material.
Parameter names are candidate spellings whose 32-bit JOAAT matches using
the lowercase byte convention in tools/world_names.py.
"""
import argparse
import json
import math
import mmap
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psin_data import block_table
from world_names import joaat


ROOT_TYPE = 0x8419AB01
# Root map order and key/value pair strides. Each value starts eight bytes into its pair.
MAPS = (("tile", 64), ("preset", 88), ("extra", 56), ("style", 176))
# Offsets of the array descriptors within each LSCH value struct, in declaration order.
ARRAYS = {
    "tile": ((8, "float"), (24, "texture"), (40, "uint32")),
    "preset": ((8, "float"), (24, "rgba8"), (40, "uint32"), (56, "texture")),
    "extra": ((0, "float"), (16, "rgba8"), (32, "uint32")),
    "style": ((104, "float"), (120, "rgba8"), (136, "uint32"), (152, "texture")),
}
ROWS = {
    "float": (0x38F05860, struct.Struct("<IfBB2x")),
    "rgba8": (0xE9D99E0A, struct.Struct("<I4sBB2x")),
    "uint32": (0x4B2625BB, struct.Struct("<IIBB2x")),
    "texture": (0xF1CB6289, struct.Struct("<I4xQBB6x")),
}
PARAMETER_NAMES = {joaat(name): name for name in (
    "tiletint", "tilerotation", "tilescale", "tilematerialmap", "tilenormalmap",
    "tintmaskmap", "tintmaskopacity", "tintmaskrotation", "tintmaskscale", "tintmasktint",
)}
DESCRIPTOR = struct.Struct("<IIQ")
U32 = struct.Struct("<I")


def ref(key):
    return f"{key:08x}" if key else None


class StyleBank:
    def __init__(self, blob):
        if len(blob) < 8 or blob[:4] != b"PSIN":
            raise ValueError("expected a PSIN garment style bank")
        first, = struct.unpack_from(">I", blob, 4)
        if not 8 <= first <= len(blob):
            raise ValueError("PSIN data region exceeds the input")
        self.blob = blob
        self.blocks, filled = block_table(blob, first)
        if not filled:
            raise ValueError("PSIN block table does not fill the declared data region")
        roots = [(at, size) for kind, at, size in self.blocks if kind == ROOT_TYPE]
        if len(roots) != 1 or roots[0][1] != 64:
            raise ValueError("expected one 64-byte garment style root")
        root = roots[0][0]
        self.maps = {kind: self.read_map(root + i * DESCRIPTOR.size, stride, kind)
                     for i, (kind, stride) in enumerate(MAPS)}

    def resolve(self, pointer, length, label):
        number, offset = pointer & 0xFFFFF, pointer >> 20
        if not 1 <= number <= len(self.blocks):
            raise ValueError(f"{label}: invalid PSIN block reference")
        kind, begin, size = self.blocks[number - 1]
        if offset + length > size:
            raise ValueError(f"{label}: range exceeds its PSIN block")
        return kind, begin + offset

    def read_map(self, at, stride, label):
        groups, count, pointer = DESCRIPTOR.unpack_from(self.blob, at)
        if count > 16 * groups:
            raise ValueError(f"{label}: map count exceeds its slot capacity")
        if not groups:
            return {}
        group_size = 16 + 16 * stride
        kind, start = self.resolve(pointer, groups * group_size, label)
        if not 0x100 <= kind <= 0x1FF:
            raise ValueError(f"{label}: expected an anonymous map bucket block")
        found = {}
        for group in range(start, start + groups * group_size, group_size):
            for slot, control in enumerate(self.blob[group:group + 16]):
                if control & 0x80:
                    continue
                pair = group + 16 + slot * stride
                key, = U32.unpack_from(self.blob, pair)
                if key in found:
                    raise ValueError(f"{label}: duplicate map key {key:08x}")
                found[key] = pair + 8
        if len(found) != count:
            raise ValueError(f"{label}: occupied slots differ from declared count")
        return found

    def parameters(self, at, kind):
        count, capacity, pointer = DESCRIPTOR.unpack_from(self.blob, at)
        if count > capacity:
            raise ValueError(f"{kind}: array count exceeds capacity")
        if not count:
            return
        expected, row = ROWS[kind]
        actual, begin = self.resolve(pointer, count * row.size, kind)
        if actual != expected:
            raise ValueError(f"{kind}: parameter array has the wrong element type")
        for pos in range(begin, begin + count * row.size, row.size):
            key, value, index, slot = row.unpack_from(self.blob, pos)
            param = {"hash": f"{key:08x}", "type": kind, "index": index, "slot": slot}
            if key in PARAMETER_NAMES:
                param["name"] = PARAMETER_NAMES[key]
            if kind == "rgba8":
                value = list(value)
            elif kind == "texture":
                value = f"{value:016x}"
            elif kind == "float" and not math.isfinite(value):
                # JSON has no NaN or infinity. Keep the IEEE-754 payload for a lossless report.
                bits, = U32.unpack_from(self.blob, pos + 4)
                param["bits"] = f"{bits:08x}"
                value = None
            param["value"] = value
            yield param

    def style(self, key):
        records = {kind: {} for kind, _ in MAPS}

        def collect(kind, record_id):
            at = self.maps[kind].get(record_id)
            if at is None:
                raise ValueError(f"{kind} {record_id:08x} is missing from its map")
            name = f"{record_id:08x}"
            if name not in records[kind]:
                params = [p for offset, value_type in ARRAYS[kind]
                          for p in self.parameters(at + offset, value_type)]
                records[kind][name] = {"params": params}
            return at

        begin = collect("style", key)
        presets, tiles, extras = (struct.unpack_from("<8I", self.blob, begin + off) for off in (8, 40, 72))
        slots = []
        for i, (preset, tile, extra) in enumerate(zip(presets, tiles, extras)):
            preset_tile = preset_extra = 0
            if preset:
                at = collect("preset", preset)
                preset_tile, = U32.unpack_from(self.blob, at + 4)
                preset_extra, = U32.unpack_from(self.blob, at + 72)
            for record_type, record_id in (("tile", tile), ("tile", preset_tile),
                                            ("extra", extra), ("extra", preset_extra)):
                if record_id:
                    collect(record_type, record_id)
            slots.append({"slot": i, "preset": ref(preset), "tile_override": ref(tile),
                          "preset_tile": ref(preset_tile), "tile": ref(tile or preset_tile),
                          "extra_override": ref(extra), "preset_extra": ref(preset_extra)})
        return {"style": f"{key:08x}", "slots": slots, "records": records}


def decode_style(blob, style_id):
    if type(style_id) is not int or not 0 <= style_id <= 0xFFFFFFFF:
        raise ValueError("style ID must be an unsigned 32-bit integer")
    return StyleBank(blob).style(style_id)


def style_id(text):
    # Decimal by default, including leading zeros. Hex accepts a 0x prefix or a-f digits.
    base = 16 if text.lower().startswith("0x") or any(c in "abcdef" for c in text.lower()) else 10
    value = int(text, base)
    if not 0 <= value <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("style ID must fit an unsigned 32-bit integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path, help="standalone PSIN garment style bank")
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--style", type=style_id, help="decimal or hexadecimal style ID")
    query.add_argument("--list-styles", action="store_true", help="list available style IDs")
    args = parser.parse_args(argv)
    try:
        with args.bank.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as blob:
            bank = StyleBank(blob)
            if args.list_styles:
                result = {"styles": [f"0x{key:08x}" for key in sorted(bank.maps["style"])]}
            else:
                result = bank.style(args.style)
    except (OSError, ValueError, struct.error) as exc:
        print(f"garment_style: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2, allow_nan=False)
    print()
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

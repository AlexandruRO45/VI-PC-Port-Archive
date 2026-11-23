"""gamedata_sps1.parse_sps1 must give exactly what the record-by-record reference below gives.

    python -I tests/test_gamedata_sps1_parse_parity.py

The reference is the straightforward decoder (one _read_c_string per string, one set or list per statistic) kept
inline, so nothing here depends on a platform, a stored hash or game data. Every blob is synthetic and seeded.
"""
from collections import Counter, defaultdict
import os
import posixpath
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gamedata_sps1  # noqa: E402

HEADER = 8


def ref_c_string(blob, pos, label):
    if pos >= len(blob):
        raise ValueError(f"truncated {label} length at {pos:#x}")
    length = blob[pos]
    pos += 1
    if length < 2:
        raise ValueError(f"invalid {label} string length at {pos - 1:#x}")
    if pos + length > len(blob):
        raise ValueError(f"truncated {label} string at {pos - 1:#x}")
    raw = blob[pos:pos + length]
    if raw[-1] != 0 or 0 in raw[:-1]:
        raise ValueError(f"{label} string at {pos - 1:#x} must end with a single NUL")
    body = raw[:-1]
    if any(b < 0x20 or b > 0x7E for b in body):
        raise ValueError(f"non-printable byte in {label} string at {pos - 1:#x}")
    return body.decode("ascii"), pos + length


def ref_stem(preset_path):
    base = posixpath.basename(preset_path.replace("\\", "/"))
    return base[:-4] if base.lower().endswith(".sps") else base


def ref_parse(blob):
    if len(blob) < HEADER or blob[:4] != b"SPS1":
        raise ValueError("not an SPS1 payload")
    entry_count = struct.unpack_from("<I", blob, 4)[0]
    pos = HEADER
    entries = []
    for index in range(entry_count):
        preset, pos = ref_c_string(blob, pos, f"preset[{index}]")
        effect, pos = ref_c_string(blob, pos, f"effect[{index}]")
        if pos >= len(blob):
            raise ValueError(f"truncated render_bucket[{index}] at {pos:#x}")
        entries.append((preset, effect, blob[pos]))
        pos += 1
    if pos != len(blob):
        raise ValueError(f"SPS1 trailing bytes: consumed {pos} of {len(blob)} bytes")

    buckets = Counter()
    by_effect = defaultdict(set)
    presets, stems, root_stems, sub_stems = set(), set(), set(), []
    plen, elen = [], []
    sps_ext = root = sub = identity = 0
    for preset, effect, bucket in entries:
        low_preset, low_effect = preset.lower(), effect.lower()
        stem = ref_stem(preset).lower()
        presets.add(low_preset)
        stems.add(stem)
        plen.append(len(preset))
        elen.append(len(effect))
        buckets[bucket] += 1
        by_effect[low_effect].add(bucket)
        if low_preset.endswith(".sps"):
            sps_ext += 1
        if "/" in preset or "\\" in preset:
            sub += 1
            sub_stems.append(stem)
        else:
            root += 1
            root_stems.add(stem)
        if stem == low_effect:
            identity += 1
    combos = Counter(",".join(str(b) for b in sorted(s)) for s in by_effect.values() if len(s) > 1)
    low_byte = entry_count & 0xFF
    return {
        "type": "SPS1",
        "header": {"magic": "SPS1", "entry_count": entry_count,
                   "low_count_byte_ascii": chr(low_byte) if 0x20 <= low_byte <= 0x7E else None},
        "table": {
            "records": len(entries), "data_bytes": len(blob) - HEADER,
            "distinct_presets": len(presets), "distinct_preset_stems": len(stems),
            "distinct_effects": len(by_effect), "sps_extension_count": sps_ext,
            "root_presets": root, "subfolder_presets": sub,
            "subfolder_stem_collisions": sum(1 for s in sub_stems if s in root_stems),
            "identity_presets": identity, "aliased_presets": len(entries) - identity,
            "preset_length_min": min(plen) if plen else 0, "preset_length_max": max(plen) if plen else 0,
            "effect_length_min": min(elen) if elen else 0, "effect_length_max": max(elen) if elen else 0,
            "render_buckets": {str(k): buckets[k] for k in sorted(buckets)},
            "multi_bucket_effects": sum(1 for s in by_effect.values() if len(s) > 1),
            "multi_bucket_combinations": {k: combos[k] for k in sorted(combos, key=lambda s: (-combos[s], s))},
        },
        "_entries": entries,
    }


def build(entries, count=None):
    n = len(entries) if count is None else count
    buf = bytearray(b"SPS1" + struct.pack("<I", n))
    for preset, effect, bucket in entries:
        p = preset.encode("ascii") + b"\x00"
        e = effect.encode("ascii") + b"\x00"
        buf += bytes([len(p)]) + p + bytes([len(e)]) + e + bytes([bucket])
    return bytes(buf)


def random_entries(rng, n):
    words = ["wall", "Road", "GLASS", "cloth", "terrain", "water", "a", "x.y"]
    out = []
    for _ in range(n):
        folder = rng.choice(["", "", "env/", "Veh\\", "ped/skin\\", "a/b/"])
        stem = rng.choice(words) + str(rng.randrange(6))
        suffix = rng.choice([".sps", ".SPS", ".Sps", "", ".spsx"])
        effect = rng.choice([stem, stem.upper(), rng.choice(words), "fx" + str(rng.randrange(4))])
        out.append((folder + stem + suffix, effect, rng.choice([0, 1, 2, 3, 7, 7, 200, 255])))
    return out


def outcome(fn, blob):
    try:
        return ("ok", fn(blob))
    except ValueError as err:
        return ("error", str(err))


class ParseSps1ParityTest(unittest.TestCase):
    def check(self, blob):
        self.assertEqual(outcome(gamedata_sps1.parse_sps1, blob), outcome(ref_parse, blob))

    def test_random_valid_tables(self):
        rng = random.Random(1234)
        for n in (0, 1, 2, 3, 7, 40, 300):
            for _ in range(5):
                self.check(build(random_entries(rng, n)))

    def test_edge_names(self):
        self.check(build([(".sps", ".SPS", 0), ("a\b/C.Sps", "c", 255), ("x" * 253, "y" * 253, 7),
                          ("/", "z", 1), ("dir/", "e", 2), ("A.sps.SPS", "a.sps", 3)]))

    def test_every_bucket_value(self):
        self.check(build([(f"p{i}", f"e{i % 5}", i) for i in range(256)]))

    def test_corrupted_blobs_raise_the_same_error(self):
        rng = random.Random(99)
        for _ in range(1500):
            blob = bytearray(build(random_entries(rng, rng.choice([0, 1, 2, 5, 30]))))
            kind = rng.random()
            if kind < 0.55 and len(blob) > HEADER:
                for _ in range(rng.randrange(1, 4)):
                    blob[rng.randrange(len(blob))] = rng.choice([0, 1, 2, 0x1F, 0x20, 0x7E, 0x7F, 0xFF, rng.randrange(256)])
            elif kind < 0.7:
                del blob[rng.randrange(len(blob) + 1):]
            elif kind < 0.8:
                blob += bytes([rng.randrange(256)])
            elif kind < 0.95:
                blob[4:8] = struct.pack("<I", rng.choice([0, 1, 4, 9, 2 ** 32 - 1]))
            self.check(bytes(blob))

    def test_huge_count_with_little_data(self):
        self.check(b"SPS1" + struct.pack("<I", 0xFFFFFFFF))
        self.check(build([("a", "b", 1)], count=0xFFFFFFFF))

    def test_not_sps1(self):
        for blob in (b"", b"SPS1", b"SPS1\x00\x00\x00", b"XPS1\x00\x00\x00\x00"):
            self.check(blob)


if __name__ == "__main__":
    unittest.main()

"""Synthetic unit tests for tools/sgd5_params.py (hash registry, parameter table parser, CLI).

Run with: python -I tests/test_sgd5_params.py
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from hashdict import joaat
import sgd5_effects
import sgd5_params
import test_sgd5_effects as fx_builder


def _make_param_table(resources=(), parameters=(), attr_count=0):
    """Build a synthetic length-prefixed SGD5 post-pass parameter table body (without outer u32 size)."""
    entries = bytearray()
    for type_code, flags, slot, w1, w2, w3, h1, h2 in resources:
        entries.extend(struct.pack("<BBH5I", type_code, flags, slot, w1, w2, w3, h1, h2))
    for type_code, flags, slot, w1, w2, w3, h1, h2 in parameters:
        entries.extend(struct.pack("<BBH5I", type_code, flags, slot, w1, w2, w3, h1, h2))
    aux = bytes(8 * attr_count)
    body_size = 8 + len(entries) + len(aux)
    header = struct.pack("<4H", len(resources), len(parameters), attr_count, body_size)
    return header + bytes(entries) + aux


class RegistryInvariantTests(unittest.TestCase):
    def test_programmatic_joaat_hashes_and_uniqueness(self):
        self.assertEqual(len(sgd5_params.PARAMETERS), 102)
        self.assertEqual(len(sgd5_params.BY_SECONDARY_HASH), len(sgd5_params.PARAMETERS))
        self.assertEqual(len(sgd5_params.BY_PAIR), len(sgd5_params.PARAMETERS))
        aliased = [p for p in sgd5_params.PARAMETERS if p.aliased]
        unaliased = [p for p in sgd5_params.PARAMETERS if not p.aliased]
        self.assertEqual(len(aliased), 12)
        self.assertEqual(len(unaliased), 90)
        for p in sgd5_params.PARAMETERS:
            self.assertEqual(p.primary_hash, joaat(p.primary))
            self.assertEqual(p.secondary_hash, joaat(p.secondary))
            self.assertEqual(p.aliased, p.primary != p.secondary)
            self.assertIn(p.category, sgd5_params.CATEGORIES)
            self.assertEqual(p.primary, p.primary.lower())
            self.assertEqual(p.secondary, p.secondary.lower())

    def test_lookup_hash_pair_and_category_classification(self):
        # Aliased base color: diffusetexture -> diffusetex
        h_prim = joaat("diffusetexture")
        h_sec = joaat("diffusetex")
        self.assertNotEqual(h_prim, h_sec)
        p = sgd5_params.lookup_hash(h_sec)
        self.assertIsNotNone(p)
        self.assertEqual(p.category, "base_color")
        self.assertTrue(p.aliased)
        self.assertEqual(sgd5_params.lookup_hash(f"{h_sec:08x}"), p)
        self.assertEqual(sgd5_params.lookup_hash(f"0x{h_sec:08X}"), p)
        self.assertEqual(sgd5_params.lookup_pair(h_prim, h_sec), p)
        self.assertIsNone(sgd5_params.lookup_pair(h_sec, h_prim))
        self.assertEqual(sgd5_params.classify_role(h_sec), "base_color")

        # Disambiguating shared primary hash ("bumptexture") between aliased and unaliased forms
        h_bump_prim = joaat("bumptexture")
        h_bump_sec = joaat("bumptex")
        aliased_bump = sgd5_params.lookup_pair(h_bump_prim, h_bump_sec)
        unaliased_bump = sgd5_params.lookup_pair(h_bump_prim, h_bump_prim)
        self.assertIsNotNone(aliased_bump)
        self.assertIsNotNone(unaliased_bump)
        self.assertTrue(aliased_bump.aliased)
        self.assertFalse(unaliased_bump.aliased)
        self.assertEqual(len(sgd5_params.BY_PRIMARY_HASH[h_bump_prim]), 2)

        # Unknown hash returns None
        self.assertIsNone(sgd5_params.lookup_hash(0x00000001))
        self.assertIsNone(sgd5_params.classify_role("00000001"))

    def test_parse_hash_validation(self):
        self.assertEqual(sgd5_params.parse_hash(0), 0)
        self.assertEqual(sgd5_params.parse_hash(0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(sgd5_params.parse_hash("0x0a1b2c3d"), 0x0A1B2C3D)
        self.assertEqual(sgd5_params.parse_hash("0A1B2C3D"), 0x0A1B2C3D)
        for bad in (True, False, -1, 0x100000000, "", "zzzz", "0x100000000", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sgd5_params.parse_hash(bad)


class ParameterTableAndArchiveTests(unittest.TestCase):
    def test_parse_parameter_table_and_errors(self):
        h_diff_p, h_diff_s = joaat("diffusetexture"), joaat("diffusetex")
        h_bump_p, h_bump_s = joaat("bumptexture"), joaat("bumptexture")
        h_unk = 0x12345678
        body = _make_param_table(
            resources=[
                (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, (1 << 8) | 0, 0, 0, 0, h_diff_p, h_diff_s),
                (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, (0 << 8) | 1, 0, 0, 0, h_bump_p, h_bump_s),
                (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, (0 << 8) | 2, 0, 0, 0, h_unk, h_unk),
                (sgd5_params.SAMPLER_RESOURCE_TYPE, 0, 0, 0, 0, 0, h_diff_p, h_diff_s),
                (sgd5_params.CBUFFER_RESOURCE_TYPE, 0, 3, 0, 0, 0, 0x11111111, 0x11111111),
            ],
            parameters=[
                (5, 0, 0, 64, 0, 0, 0x22222222, 0x22222222),
            ],
            attr_count=2,
        )
        blob = struct.pack("<I", len(body)) + body
        table = sgd5_params.parse_parameter_table(blob, 0)
        self.assertEqual(table["end"], len(blob))
        self.assertEqual(table["attributes"], 2)
        self.assertEqual(len(table["resources"]), 5)
        self.assertEqual(len(table["parameters"]), 1)
        self.assertEqual(table["parameters"][0]["byte_offset"], 64)
        self.assertEqual(table["resources"][0]["binding_slot"], 0)
        self.assertEqual(table["resources"][0]["attribute_count"], 1)
        self.assertEqual(table["resources"][0]["param"].category, "base_color")
        self.assertEqual(table["resources"][1]["param"].category, "normal")
        self.assertNotIn("param", table["resources"][2])
        # Non-texture resource types (sampler/cbuffer) must not attach texture params.
        self.assertNotIn("param", table["resources"][3])
        self.assertNotIn("param", table["resources"][4])

        # Empty table (size == 0) is valid.
        empty = sgd5_params.parse_parameter_table(struct.pack("<I", 0), 0)
        self.assertEqual(empty, {"resources": [], "parameters": [], "attributes": 0, "size": 0, "end": 4})

        # Malformed / truncated tables must raise ValueError.
        for bad in (
            b"\x01\x02",                              # truncated header
            struct.pack("<I", 100) + bytes(4),        # exceeds boundary
            struct.pack("<I", 4) + bytes(4),          # size in 1..8
            struct.pack("<I4H", 16, 1, 0, 0, 99) + bytes(8),  # size != size16
            struct.pack("<I4H", 16, 1, 0, 0, 16) + bytes(8),  # entry count overflows size
        ):
            with self.assertRaises(ValueError):
                sgd5_params.parse_parameter_table(bad, 0)

    def test_scan_sgd5_archive_and_cli(self):
        h_diff_p, h_diff_s = joaat("diffusetexture"), joaat("diffusetex")
        h_spec_p, h_spec_s = joaat("spectexture"), joaat("speculartex")
        h_tint = joaat("tintpalettetex")
        h_unk = 0xCAFEBABE

        t0 = _make_param_table(resources=[
            (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, 0, 0, 0, 0, h_diff_p, h_diff_s),
            (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, 1, 0, 0, 0, h_spec_p, h_spec_s),
            (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, 2, 0, 0, 0, h_tint, h_tint),
            (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, 3, 0, 0, 0, h_unk, h_unk),
            (sgd5_params.SAMPLER_RESOURCE_TYPE, 0, 0, 0, 0, 0, h_diff_p, h_diff_s),
        ])
        t1 = _make_param_table(resources=[
            (sgd5_params.TEXTURE_RESOURCE_TYPE, 0, 0, 0, 0, 0, h_diff_p, h_diff_s),
        ])
        suffix1 = fx_builder.post_pass(words=(1, 2), blocks=(t0, b""))
        suffix2 = fx_builder.post_pass(words=(), blocks=(b"", t1))
        data, layout = fx_builder.database(
            {0: ("MockProg",)},
            [fx_builder.effect("FxOne", {0: (0,)}, suffix=suffix1),
             fx_builder.effect("FxTwo", {0: (0,)}, suffix=suffix2)],
        )

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "mock_shaders.rpf"
            archive.write_bytes(data)

            report = sgd5_params.scan_sgd5_archive(archive, databases=[layout])
            self.assertEqual(report["registry_entries"], 102)
            self.assertEqual(report["effects"], 2)
            self.assertEqual(report["texture_slots"], 5)
            self.assertEqual(report["table_slots"], [4, 1])
            self.assertEqual(report["distinct_texture_pairs"], 4)
            self.assertEqual(report["matched_texture_pairs"], 3)
            self.assertEqual(report["matched_texture_slots"], 4)
            self.assertEqual(report["aliased_texture_pairs"], 2)
            self.assertEqual(report["matched_aliased_pairs"], 2)
            self.assertEqual(report["aliased_texture_slots"], 3)
            self.assertEqual(report["matched_aliased_slots"], 3)
            self.assertEqual(report["category_slots"]["base_color"], 2)
            self.assertEqual(report["category_slots"]["specular"], 1)
            self.assertEqual(report["category_slots"]["tint"], 1)

            # CLI --archive
            with patch.object(sgd5_effects, "DATABASES", [layout]):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = sgd5_params.main(["--archive", str(archive)])
                self.assertEqual(rc, 0)
                self.assertEqual(json.loads(buf.getvalue()), report)

        # CLI --list, --lookup, --pair, --hash
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(sgd5_params.main(["--list"]), 0)
        listed = json.loads(buf.getvalue())
        self.assertEqual(len(listed), 102)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sgd5_params.main(["--lookup", f"{h_diff_s:08x}"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())[0]["match"]["secondary"], "diffusetex")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sgd5_params.main(["--lookup", "00000001"])
        self.assertEqual(rc, 1)
        self.assertIsNone(json.loads(buf.getvalue())[0]["match"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sgd5_params.main(["--pair", f"{h_diff_p:08x}", f"{h_diff_s:08x}"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["match"]["category"], "base_color")

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sgd5_params.main(["--pair", f"{h_diff_s:08x}", f"{h_diff_p:08x}"])
        self.assertEqual(rc, 1)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = sgd5_params.main(["--hash", "diffusetex", "unknown_param_xyz"])
        self.assertEqual(rc, 0)
        hashed = json.loads(buf.getvalue())
        self.assertEqual(hashed[0]["hash"], f"{h_diff_s:08x}")
        self.assertIsNotNone(hashed[0]["match"])
        self.assertIsNone(hashed[1]["match"])

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sgd5_params.main(["--lookup", "not_a_hash"])


if __name__ == "__main__":
    unittest.main()

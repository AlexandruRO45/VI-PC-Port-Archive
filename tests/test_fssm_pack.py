"""runtime/test/lanterns/fssm_pack.py makes FSSM blobs and shader containers that the project's own readers take
apart again: fssm_header.py, fssm_unpack.py and shader_disasm.py's code range see exactly the fields and code that were
packed. Standard library only: no llvm-mc, GPU or game files (tests/test_fssm_pack_chain.py runs the offline chain).

All programs here are a few hand-written instructions; nothing comes from the game.

    python -I tests/test_fssm_pack.py
"""
import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "lanterns"))

import fssm_header  # noqa: E402
import fssm_pack  # noqa: E402
import fssm_unpack  # noqa: E402
import shader_disasm  # noqa: E402

NOP, ENDPGM = 0xBF800000, 0xBF810000
CODE = struct.pack("<3I", NOP, NOP, ENDPGM)


class Pack(unittest.TestCase):
    def test_header_fields_read_back(self):
        blob = fssm_pack.pack(CODE, stage="ps", name="ps_t", wave=32, inputs=[(0x01, False), (0x02, True)])
        self.assertEqual(fssm_header.wave_size(blob), 32)
        self.assertTrue(fssm_header.directory_ok(blob))
        self.assertEqual(fssm_header.read_methods(blob), [0x01, 0x02])
        self.assertEqual(fssm_header.interpolants(blob), (2, 0b10))
        self.assertEqual(fssm_header.binding_table(blob)["name"], "ps_t")
        self.assertEqual(fssm_header.wave_size(fssm_pack.pack(CODE, stage="vs", name="vs_t", wave=64)), 64)
        cs = fssm_pack.pack(CODE, stage="cs", name="cs_t", group=(32, 2, 1), rsrc2=0x998)
        self.assertEqual(struct.unpack_from("<4I", cs, 0x138), (0x998, 32, 2, 1))

    def test_input_slots_and_binding_records_read_back(self):
        # The slot byte need not be position + 1 (the attr number is the table position), and the tail carries the
        # binding records in order before the sentinel and the name.
        bindings = [(1, 2, 3, 4, 5), (0x10, 0, 0x20, 0, 0x30)]
        blob = fssm_pack.pack(CODE, stage="ps", name="ps_t", inputs=[(0x02, False, 7), (0x01, False)],
                              bindings=bindings)
        self.assertEqual([entry["slot"] for entry in fssm_header.pixel_inputs(blob)], [7, 2])
        self.assertEqual(fssm_header.read_methods(blob), [0x02, 0x01])
        table = fssm_header.binding_table(blob)
        self.assertEqual((table["records"], table["name"], table["trailing"]), (bindings, "ps_t", 0))

    def test_header_lines_set_slots_bindings_and_a_first_payload(self):
        text = ("; stage: ps\n; input: interp slot=9\n; input: flat\n; binding: 1 2 3 4 0x5\n"
                "; first-payload: ps_first\ns_endpgm\n")
        fields = fssm_pack.header_fields(text, "ps_t")
        self.assertEqual(fields["inputs"], [(0x02, False, 9), (0x01, False, 2)])
        self.assertEqual(fields["bindings"], [(1, 2, 3, 4, 5)])
        self.assertEqual(fields["first_payload"], "ps_first")
        for line in ("; input: interp slot=256", "; input: flat slot=1 slot=2", "; binding: 1 2 3",
                     "; binding: 0xffffffff 0 0 0 0", "; first-payload: ps_t"):
            with self.subTest(line):
                with self.assertRaises(fssm_pack.PackError):
                    fssm_pack.header_fields("; stage: ps\n" + line + "\ns_endpgm\n", "ps_t")

    def test_second_payload_is_the_one_the_readers_take(self):
        # 195 of the title's containers carry two 0x0D payloads; fssm_unpack.py (and the runtime) take the last.
        first = fssm_pack.pack(CODE, stage="vs", name="vs_first")
        blob = fssm_pack.pack(CODE, stage="vs", name="vs_t")
        data = fssm_pack.container(blob, first=first)
        dxil = fssm_unpack.parts(data)[b"DXIL"]
        self.assertEqual([tag for tag, _ in fssm_unpack.xdat_entries_all(dxil)], [0x0D, 0x0D])
        self.assertEqual(fssm_unpack.unpack(data)[0], blob)
        self.assertEqual(fssm_unpack.unpack(data, payload="first")[0], first)

    def test_code_range_is_the_code_and_data_sits_after_it(self):
        blob = fssm_pack.pack(CODE, stage="ps", name="ps_t", data_at=0x40, data=[0x3F800000, 2])
        start, end, how = shader_disasm.code_range(blob)
        self.assertEqual((blob[start:end], how), (CODE, "header"))
        self.assertEqual(struct.unpack_from("<2I", blob, start + 0x40), (0x3F800000, 2))

    def test_container_unpacks_to_the_blob(self):
        blob = fssm_pack.pack(CODE, stage="vs", name="vs_t")
        unpacked, entries = fssm_unpack.unpack(fssm_pack.container(blob))
        self.assertEqual(unpacked, blob)
        self.assertIn(0x0D, entries)

    def test_refuses_what_would_not_read_back(self):
        bad = {
            "no s_endpgm": dict(code=struct.pack("<I", NOP)),
            "data before the code ends": dict(code=CODE, data_at=4, data=[1]),
            "data that ends the code": dict(code=CODE, data_at=0x20, data=[ENDPGM]),
            "too many inputs": dict(code=CODE, inputs=[(0x02, False)] * (fssm_pack.MAX_INPUTS + 1)),
        }
        for name, kwargs in bad.items():
            with self.subTest(name):
                with self.assertRaises(fssm_pack.PackError):
                    fssm_pack.pack(stage="ps", name="ps_t", **kwargs)


class Names(unittest.TestCase):
    """Every output file is named after its shader, so two names equal but for case would share files on a
    case-insensitive volume. The packer must refuse them before it writes anything."""

    def run_main(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for rel, text in files:
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                paths.append(path)
            out = os.path.join(tmp, "out")
            err = io.StringIO()
            with mock.patch.object(fssm_pack, "assemble", return_value=struct.pack("<I", ENDPGM)), \
                    contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = fssm_pack.main([out] + paths)
            written = sorted(os.listdir(out)) if os.path.isdir(out) else []
            return code, err.getvalue(), written

    def test_names_differing_only_in_case_are_refused_before_any_write(self):
        code, err, written = self.run_main([("a/ps_tile.asm", "s_endpgm\n"), ("b/ps_Tile.asm", "s_endpgm\n")])
        self.assertEqual(code, 1)
        self.assertIn("ps_tile (", err)
        self.assertIn("ps_Tile (", err)
        self.assertEqual(written, [])

    def test_a_native_variant_clashing_in_case_is_refused(self):
        code, err, written = self.run_main([("ps_tile.asm", "; native-variant: PS_TILE\ns_endpgm\n")])
        self.assertEqual(code, 1)
        self.assertIn("PS_TILE (", err)
        self.assertEqual(written, [])

    def test_distinct_names_each_get_their_own_container(self):
        code, err, written = self.run_main([("a/ps_tile.asm", "s_endpgm\n"), ("b/ps_tile2.asm", "s_endpgm\n")])
        self.assertEqual(code, 0, err)
        self.assertEqual(written, ["index.json", "ps_tile.dxbc", "ps_tile2.dxbc"])


if __name__ == "__main__":
    unittest.main()

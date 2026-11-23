"""runtime/test/lanterns/fssm_pack.py makes FSSM blobs and shader containers that the project's own readers take
apart again: fssm_header.py, fssm_unpack.py and shader_disasm.py's code range see exactly the fields and code that were
packed. Standard library only: no llvm-mc, GPU or game files (tests/test_fssm_pack_chain.py runs the offline chain).

All programs here are a few hand-written instructions; nothing comes from the game.

    python -I tests/test_fssm_pack.py
"""
import os
import struct
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()

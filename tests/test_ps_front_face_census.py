"""Synthetic controls for front-face path and lane handling."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from ps_front_face_census import census


class FrontFaceTests(unittest.TestCase):
    def run_cache(self, lines, *, mismatch=False):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            native = base / "fssm"
            native.mkdir()
            (base / "index.json").write_text(json.dumps([{"file": "p.dxbc", "stage": "ps"}]),
                                             encoding="utf-8", newline="\n")
            # ISG1 declares position x/z and front face, assigning the latter to v4.
            sig = bytearray(72)
            struct.pack_into("<II", sig, 0, 2, 8)
            struct.pack_into("<I", sig, 20, 1)
            sig[32:34] = bytes([15, 5])
            struct.pack_into("<I", sig, 52, 9)
            container = bytearray(36)
            container[:4] = b"DXBC"
            struct.pack_into("<II", container, 28, 1, 36)
            (base / "p.dxbc").write_bytes(container + b"ISG1" + struct.pack("<I", len(sig)) + sig)
            start = 0x128
            blob = bytearray(start)
            struct.pack_into("<I", blob, 0xc0, start - 0x28)
            listing = [f"; p code 0x{start:x}..0x{start + 4 * len(lines):x}"]
            for k, line in enumerate(lines):
                # Unique marker words let the census verify listing/native alignment.
                blob.extend(struct.pack("<I", k + 1))
                listing.append(f"\t{line} ; encoding: [0x{k+1:02x},0,0,0]")
            if mismatch:
                blob[start] ^= 0x80
            (native / "p.fssm").write_bytes(blob)
            (base / "p.s").write_text("\n".join(listing), encoding="utf-8", newline="\n")
            return census(base, base)

    def test_branch_and_fallthrough(self):
        result = self.run_cache(["s_cbranch_scc0 2", "v_mov_b32 v4, 1",
                                 "v_cmp_eq_u32_e32 vcc, v4, 0", "v_cmp_ne_u32_e32 vcc, 0, v4"])
        self.assertEqual(result["resolved_reader"], 1)
        self.assertEqual(result["shaders_per_read_kind"], {"v_cmp_ne_u32_e32 immediate zero": 1})

    def test_branch_skips_read(self):
        self.assertEqual(self.run_cache(["s_branch 1", "v_cmp_eq_u32_e32 vcc, v4, 0",
                                         "s_endpgm"])["resolved_without_read"], 1)

    def test_read_and_unknown_paths_in_same_shader(self):
        result = self.run_cache(["s_cbranch_scc0 1", "v_cmp_eq_u32_e32 vcc, 0, v4",
                                 "s_setpc_b64 s[0:1]"])
        self.assertEqual(result["shaders_per_read_kind"], {"v_cmp_eq_u32_e32 immediate zero": 1})
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["resolved_reader"], 0)

    def test_store_address_is_read(self):
        for instruction in ("flat_store_dword v[4:5], v0", "flat_store_dword v[3:4], v0",
                            "scratch_store_dword v4, v0, off"):
            with self.subTest(instruction=instruction):
                result = self.run_cache([instruction, "s_endpgm"])
                self.assertEqual(result["resolved_reader"], 1)
                self.assertEqual(result["shaders_per_read_kind"], {instruction.split()[0]: 1})

    def test_possible_partial_write(self):
        result = self.run_cache(["s_and_saveexec_b64 s[0:1], vcc", "v_mov_b32 v4, 1",
                                 "s_mov_b64 exec, s[0:1]", "v_cmp_ne_u32_e32 vcc, 0, v4"])
        self.assertEqual(result["shaders_per_unresolved_reason"], {"possible lane-partial write": 1})
        self.assertEqual(result["resolved_without_read"], 0)

    def test_unknown_edges_and_sidecar_mismatch(self):
        for lines, mismatch in [(["s_setpc_b64 s[0:1]"], False), (["s_branch -2"], False),
                                (["s_endpgm"], True)]:
            with self.subTest(lines=lines):
                self.assertEqual(self.run_cache(lines, mismatch=mismatch)["unresolved"], 1)


if __name__ == "__main__":
    unittest.main()

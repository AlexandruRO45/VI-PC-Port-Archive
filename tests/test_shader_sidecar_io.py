"""Regression tests for per-shader sidecar I/O deduplication in rdna2spirv / shader_ir / shader_meta."""
import builtins
import hashlib
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402


def _make_dxbc(chunks):
    """Build a minimal valid DXBC container with the given [(fourcc_bytes, payload_bytes), ...]."""
    count = len(chunks)
    header_size = 32 + 4 * count
    offsets = []
    body = bytearray()
    for tag, payload in chunks:
        offsets.append(header_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    total = header_size + len(body)
    hdr = bytearray(b"DXBC")
    hdr += b"\x00" * 16                  # digest
    hdr += struct.pack("<III", 1, total, count)
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr + body)


def _make_isg1(elements):
    """Build an ISG1 payload from [(reg, mask, rw_mask, sysv, ctype), ...]."""
    count = len(elements)
    entry_off = 8
    out = bytearray(struct.pack("<II", count, entry_off))
    for reg, mask, rw_mask, sysv, ctype in elements:
        out += struct.pack("<6IBB2x", 0, 0, 0, sysv, ctype, reg, mask, rw_mask)
    return bytes(out)


def _make_fssm_cs(rsrc2=0x00000084, threads=(64, 1, 1), wave32=True):
    """Minimal FSSM blob with wave-size flag at 0xD4 and CS dispatch state at 0x138..0x144."""
    buf = bytearray(0x160)
    buf[0:4] = b"FSSM"
    struct.pack_into("<I", buf, 0xD4, 0x20 if wave32 else 0x00)
    struct.pack_into("<4I", buf, 0x138, rsrc2, *threads)
    # Put s_endpgm encoding (0xbf810000) at 0x150..0x154 so check_printed_encodings exercises the blob.
    struct.pack_into("<I", buf, 0x150, 0xBF810000)
    return bytes(buf)


def _make_fssm_ps(wave32=True):
    buf = bytearray(0x160)
    buf[0:4] = b"FSSM"
    struct.pack_into("<I", buf, 0xD4, 0x20 if wave32 else 0x00)
    struct.pack_into("<I", buf, 0x150, 0xBF810000)
    return bytes(buf)


class ShaderSidecarIOTest(unittest.TestCase):
    def setUp(self):
        self._old_stages = rdna2spirv._STAGES
        rdna2spirv._STAGES = {}

    def tearDown(self):
        rdna2spirv._STAGES = self._old_stages

    def _count_io(self, fn):
        opens = []
        realpaths = []
        orig_open = builtins.open
        orig_realpath = os.path.realpath

        def wrapped_open(file, *args, **kwargs):
            opens.append(os.path.normcase(os.path.abspath(os.fspath(file))))
            return orig_open(file, *args, **kwargs)

        def wrapped_realpath(path, *args, **kwargs):
            realpaths.append(os.fspath(path))
            return orig_realpath(path, *args, **kwargs)

        builtins.open = wrapped_open
        os.path.realpath = wrapped_realpath
        try:
            result = fn()
        finally:
            builtins.open = orig_open
            os.path.realpath = orig_realpath
        return result, opens, realpaths

    def test_cs_with_fssm_and_dxbc_opens_each_file_once(self):
        with tempfile.TemporaryDirectory() as td:
            disasm_dir = os.path.join(td, "disasm")
            fssm_dir = os.path.join(td, "fssm")
            os.makedirs(disasm_dir)
            os.makedirs(fssm_dir)

            src_path = os.path.join(disasm_dir, "cs_demo.s")
            fssm_path = os.path.join(fssm_dir, "cs_demo.fssm")
            dxbc_path = os.path.join(td, "cs_demo.dxbc")

            src_bytes = b"; cs_demo code 0x150..0x154\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"
            fssm_bytes = _make_fssm_cs()
            dxbc_bytes = _make_dxbc([(b"ISG1", _make_isg1([]))])
            with open(src_path, "wb") as f:
                f.write(src_bytes)
            with open(fssm_path, "wb") as f:
                f.write(fssm_bytes)
            with open(dxbc_path, "wb") as f:
                f.write(dxbc_bytes)

            def run():
                asm = rdna2spirv.translate(src_path)
                return rdna2spirv.shader_meta(
                    rdna2spirv.translate.last_gen, asm, src_path=src_path, fssm_path=fssm_path
                )

            meta, opens, realpaths = self._count_io(run)

            self.assertEqual(meta["stage"], "cs")
            self.assertEqual(meta["local_size"], [64, 1, 1])
            self.assertEqual(meta["src_sha256"], hashlib.sha256(src_bytes).hexdigest())
            self.assertEqual(meta["fssm_sha256"], hashlib.sha256(fssm_bytes).hexdigest())
            self.assertEqual(len(realpaths), 1, f"expected 1 realpath call, got {realpaths}")
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(src_path))), 1)
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(fssm_path))), 1)
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(dxbc_path))), 1)

    def test_ps_with_fssm_and_dxbc_opens_each_file_once(self):
        with tempfile.TemporaryDirectory() as td:
            disasm_dir = os.path.join(td, "disasm")
            fssm_dir = os.path.join(td, "fssm")
            os.makedirs(disasm_dir)
            os.makedirs(fssm_dir)

            src_path = os.path.join(disasm_dir, "ps_demo.s")
            fssm_path = os.path.join(fssm_dir, "ps_demo.fssm")
            dxbc_path = os.path.join(td, "ps_demo.dxbc")

            src_bytes = b"; ps_demo code 0x150..0x154\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n"
            fssm_bytes = _make_fssm_ps()
            dxbc_bytes = _make_dxbc([(b"ISG1", _make_isg1([(0, 0xF, 0xF, 1, 3)]))])
            with open(src_path, "wb") as f:
                f.write(src_bytes)
            with open(fssm_path, "wb") as f:
                f.write(fssm_bytes)
            with open(dxbc_path, "wb") as f:
                f.write(dxbc_bytes)

            def run():
                asm = rdna2spirv.translate(src_path)
                return rdna2spirv.shader_meta(
                    rdna2spirv.translate.last_gen, asm, src_path=src_path, fssm_path=fssm_path
                )

            meta, opens, realpaths = self._count_io(run)

            self.assertEqual(meta["stage"], "ps")
            self.assertEqual(meta["src_sha256"], hashlib.sha256(src_bytes).hexdigest())
            self.assertEqual(meta["fssm_sha256"], hashlib.sha256(fssm_bytes).hexdigest())
            self.assertEqual(len(realpaths), 1, f"expected 1 realpath call, got {realpaths}")
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(src_path))), 1)
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(fssm_path))), 1)
            self.assertEqual(opens.count(os.path.normcase(os.path.abspath(dxbc_path))), 1)

    def test_shader_meta_hashes_explicit_alternate_paths(self):
        with tempfile.TemporaryDirectory() as td:
            disasm_dir = os.path.join(td, "disasm")
            fssm_dir = os.path.join(td, "fssm")
            os.makedirs(disasm_dir)
            os.makedirs(fssm_dir)

            src_path = os.path.join(disasm_dir, "vs_alt.s")
            alt_src = os.path.join(disasm_dir, "other.s")
            alt_fssm = os.path.join(fssm_dir, "other.fssm")

            with open(src_path, "wb") as f:
                f.write(b"; vs_alt synthetic\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(alt_src, "wb") as f:
                f.write(b"; alternate source bytes\n")
            with open(alt_fssm, "wb") as f:
                f.write(b"alternate fssm bytes")

            asm = rdna2spirv.translate(src_path)
            meta = rdna2spirv.shader_meta(
                rdna2spirv.translate.last_gen, asm, src_path=alt_src, fssm_path=alt_fssm
            )
            self.assertEqual(meta["src_sha256"], hashlib.sha256(b"; alternate source bytes\n").hexdigest())
            self.assertEqual(meta["fssm_sha256"], hashlib.sha256(b"alternate fssm bytes").hexdigest())

    def test_direct_helpers_keep_default_signatures(self):
        with tempfile.TemporaryDirectory() as td:
            disasm_dir = os.path.join(td, "disasm")
            fssm_dir = os.path.join(td, "fssm")
            os.makedirs(disasm_dir)
            os.makedirs(fssm_dir)

            src_path = os.path.join(disasm_dir, "cs_direct.s")
            fssm_path = os.path.join(fssm_dir, "cs_direct.fssm")
            with open(src_path, "wb") as f:
                f.write(b"; cs_direct code 0x150..0x154\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(fssm_path, "wb") as f:
                f.write(_make_fssm_cs(threads=(32, 2, 1)))

            info = shader_ir.parse_file(src_path)
            self.assertEqual(info["code_start"], 0x150)
            self.assertEqual(info["code_end"], 0x154)
            rdna2spirv.check_printed_encodings(src_path, info)
            self.assertEqual(rdna2spirv.cs_dispatch_state(src_path, "cs_direct")[1:], (32, 2, 1))
            meta = shader_meta.collect(src_path, "cs_direct", [info["insns"]], [[]])
            self.assertIn("fssm_sha256", meta)

    def test_unreadable_sidecar_propagates_permission_error(self):
        with tempfile.TemporaryDirectory() as td:
            disasm_dir = os.path.join(td, "disasm")
            fssm_dir = os.path.join(td, "fssm")
            fspx_dir = os.path.join(td, "fspx")
            os.makedirs(disasm_dir)
            os.makedirs(fssm_dir)
            os.makedirs(fspx_dir)

            src_path = os.path.join(disasm_dir, "vs_perm.s")
            fssm_path = os.path.join(fssm_dir, "vs_perm.fssm")
            code_path = os.path.join(fspx_dir, "vs_perm.code")
            dxbc_path = os.path.join(td, "vs_perm.dxbc")
            with open(src_path, "wb") as f:
                f.write(b"; vs_perm synthetic\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            with open(fssm_path, "wb") as f:
                f.write(_make_fssm_ps())
            with open(code_path, "wb") as f:
                f.write(b"\x00\x00\x81\xbf")
            with open(dxbc_path, "wb") as f:
                f.write(_make_dxbc([(b"ISG1", _make_isg1([]))]))

            orig_open = builtins.open

            def deny(target):
                target_key = os.path.normcase(os.path.abspath(target))

                def wrapped(file, *args, **kwargs):
                    if os.path.normcase(os.path.abspath(os.fspath(file))) == target_key:
                        raise PermissionError(13, "Permission denied", os.fspath(file))
                    return orig_open(file, *args, **kwargs)

                return wrapped

            # Unreadable FSSM must raise rather than falling back to fspx/.code or omitting fssm_sha256.
            builtins.open = deny(fssm_path)
            try:
                with self.assertRaises(PermissionError):
                    rdna2spirv.translate(src_path)
            finally:
                builtins.open = orig_open

            # Unreadable DXBC must raise in shader_meta.collect as well.
            info = shader_ir.parse_file(src_path)
            builtins.open = deny(dxbc_path)
            try:
                with self.assertRaises(PermissionError):
                    shader_meta.collect(src_path, "vs_perm", [info["insns"]], [[]])
            finally:
                builtins.open = orig_open


if __name__ == "__main__":
    unittest.main()

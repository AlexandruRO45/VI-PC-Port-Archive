"""The translator refuses a shader whose FSSM stage record names another hardware stage (SHADER-33 follow-up).

The VS input layout (`v0` = vertex id, `v3` = instance id) is Mesa's GFX10 legacy-VS layout. It holds only if the
console runs the VS on the legacy hardware VS stage and not as NGG (or as the ES half of a merged ES-GS wave), where
the inputs sit after the GS VGPRs. The FSSM header records the PAL hardware stage each blob was compiled for
(tools/pal_stages.py): over all 26,503 blobs of 1.0.2.1 every VS is on hardware stage 4 (VS), so rdna2spirv.translate()
now checks that record and raises Unsupported instead of reading the wrong registers when it says otherwise.

All blobs, DXBC containers and disassembly listings here are built by hand; nothing needs llvm-mc, spirv-as/val or
a GPU.

    python -I tests/test_pal_stage_guard.py
"""
import gc
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import fssm_header  # noqa: E402
import pal_stages  # noqa: E402
import rdna2spirv  # noqa: E402
from test_fssm_wave_size import GS_DXBC, STRAIGHT_GS, STRAIGHT_PS, make_dxbc  # noqa: E402

TAG = 0x000000C000020000
API = {"vs": 0, "ps": 1, "gs": 2, "cs": 3, "ms": 6}
HW = {"es": 2, "gs": 3, "vs": 4, "ps": 5, "cs": 6, "ngg": 7}

# A VS that reads both legacy-VS input ids: v0 (vertex id) and v3 (instance id).
VS_IDS = [
    "v_cvt_f32_u32_e32 v1, v0 ; encoding: [0x00,0x0d,0x02,0x7e]",
    "v_cvt_f32_u32_e32 v2, v3 ; encoding: [0x03,0x0d,0x04,0x7e]",
    "exp pos0 v1, v2, v1, v2 done ; encoding: [0xcf,0x08,0x00,0xf8,0x01,0x02,0x01,0x02]",
    "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
]


CODE = 0x210   # past the stage table (0x100 + 0x88 * k), where the code of a real blob starts too


def code_bytes(lines):
    """The instruction words of a synthetic listing, from its `; encoding: [...]` comments."""
    out = bytearray()
    for ln in lines:
        out.extend(int(b, 16) for b in re.search(r"encoding: \[([^\]]*)\]", ln).group(1).split(","))
    return bytes(out)


def make_fssm(records, wave_size=64, lines=(), size=0x300):
    """Synthetic FSSM: wave-size flags at 0xD4, one stage record per (api, hardware) name pair at 0x100, and the
    listing's code at CODE (the header's code offset), so the translator's encoding check finds it."""
    data = bytearray(size)
    data[0:4] = b"FSSM"
    struct.pack_into("<H", data, 4, 0x101)
    struct.pack_into("<Q", data, 0xC0, CODE)
    struct.pack_into("<I", data, fssm_header.WAVE_FLAGS_OFFSET, 0x00110020 if wave_size == 32 else 0x00110000)
    for index, (api, hw) in enumerate(records):
        struct.pack_into("<Q", data, 0x100 + 0x88 * index, TAG | (API[api] << 8) | HW[hw])
    code = code_bytes(lines)
    data[CODE:CODE + len(code)] = code
    return bytes(data)


def translate_case(name, lines, records, dxbc=None):
    """Write disasm/<name>.s, <name>.dxbc and fssm/<name>.fssm (with `records`) to a temp folder and translate."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm = os.path.join(tmp, "disasm")
        os.mkdir(disasm)
        s_path = os.path.join(disasm, name + ".s")
        end = CODE + len(code_bytes(lines))
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  code {CODE:#x}..{end:#x} (synthetic)\n")
            for ln in lines:
                f.write("\t" + ln + "\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(make_dxbc() if dxbc is None else dxbc)
        os.mkdir(os.path.join(tmp, "fssm"))
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(records, lines=lines))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gen = rdna2spirv.translate.last_gen
            meta = rdna2spirv.shader_meta(gen, spv)
            gc.collect()
        return spv, gen, meta


class PalStageGuardTest(unittest.TestCase):
    def test_synthetic_blob_reads_back(self):
        blob = make_fssm([("vs", "vs")])
        self.assertEqual(pal_stages.api_stage(blob), "vs")
        self.assertEqual(pal_stages.hardware_stage(blob), "vs")
        self.assertEqual(fssm_header.wave_size(blob), 64)

    def test_legacy_vs_translates_with_legacy_inputs(self):
        spv, gen, meta = translate_case("vs_legacy", VS_IDS, [("vs", "vs")])
        self.assertTrue(gen.wave64)
        self.assertEqual(meta["stage"], "vs")
        # v0 and v3 come from VertexIndex / InstanceIndex minus their draw bases, as the contract row says.
        for builtin in ("VertexIndex", "InstanceIndex", "BaseVertex", "BaseInstance"):
            self.assertIn(f"BuiltIn {builtin}", spv)

    def test_vs_on_another_hardware_stage_is_refused(self):
        # ES: the first payload of a two-payload VS container (`fssm_unpack.py --payload first`); GS: an NGG VS
        # runs on the hardware GS stage; ngg: the id the corpus mesh shaders carry.
        for hw in ("es", "gs", "ngg", "ps"):
            with self.subTest(hardware=hw):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    translate_case(f"vs_on_{hw}", VS_IDS, [("vs", hw)])
                self.assertIn(f"hardware stage {hw}, not vs", str(ctx.exception))

    def test_unknown_hardware_id_is_refused_by_number(self):
        blob = bytearray(make_fssm([("vs", "vs")]))
        struct.pack_into("<Q", blob, 0x100, TAG | (API["vs"] << 8) | 9)
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            rdna2spirv.check_pal_hardware_stage("vs", bytes(blob))
        self.assertIn("hardware stage id 9", str(ctx.exception))

    def test_ps_and_gs_on_their_measured_stages(self):
        _, _, meta = translate_case("ps_hw", STRAIGHT_PS, [("ps", "ps")])
        self.assertEqual(meta["stage"], "ps")
        # A GS blob carries two records, GS then its VS copy; only the first one decides.
        _, _, meta = translate_case("gs_hw", STRAIGHT_GS, [("gs", "gs"), ("gs", "vs")], dxbc=GS_DXBC)
        self.assertEqual(meta["stage"], "gs")
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_case("ps_on_vs", STRAIGHT_PS, [("ps", "vs")])

    def test_cs_and_mesh_expectations(self):
        rdna2spirv.check_pal_hardware_stage("cs", make_fssm([("cs", "cs")]))
        rdna2spirv.check_pal_hardware_stage("ms", make_fssm([("ms", "ngg")]))
        with self.assertRaises(rdna2spirv.Unsupported):
            rdna2spirv.check_pal_hardware_stage("ms", make_fssm([("ms", "gs")]))

    def test_blob_without_records_is_not_checked(self):
        # Synthetic sidecars (and the zero-prefixed headers tests/conftest.py writes) carry no stage table.
        for blob in (make_fssm([]), b"\x00" * 0x300, b""):
            with self.subTest(size=len(blob)):
                rdna2spirv.check_pal_hardware_stage("vs", blob)
        _, gen, _ = translate_case("vs_norecord", VS_IDS, [])
        self.assertTrue(gen.wave64)


if __name__ == "__main__":
    unittest.main()

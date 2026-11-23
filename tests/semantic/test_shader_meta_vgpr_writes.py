"""Descriptor stride metadata follows VGPR swaps and forgets dynamic destination writes."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
from spirv_mini_interp import run_workgroup  # noqa: E402


RELATIVE_WRITES = (
    ("v_movreld_b32", ["s_mov_b32 m0, 4", "v_mov_b32 v1, 48", "v_movreld_b32 v0, v1"]),
    ("v_movrelsd_b32", ["s_mov_b32 m0, 4", "v_mov_b32 v5, 48", "v_movrelsd_b32 v0, v1"]),
    ("v_movrelsd_2_b32", ["s_mov_b32 m0, 0x40000", "v_mov_b32 v1, 48", "v_movrelsd_2_b32 v0, v1"]),
    ("v_swaprel_b32", ["s_mov_b32 m0, 0", "v_mov_b32 v0, 48", "v_swaprel_b32 v0, v4"]),
)


def translate_case(body, source="v4", after_read=()):
    lines = [
        "s_mov_b32 s3, 0",
        "v_mov_b32 v0, 1",
        "v_lshlrev_b32 v4, 5, v0",
        *body,
        f"v_readfirstlane_b32 s4, {source}",
        *after_read,
        "s_load_dwordx4 s[8:11], s[2:3], s4",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v20, 0x1000",
        "v_mov_b32 v21, 0",
        "global_store_dword v[20:21], v0, off",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    asm = support.translate_lines("ps_meta_vgpr_writes", lines)
    meta = support.rdna2spirv.shader_meta(support.rdna2spirv.translate.last_gen, asm)
    return asm, meta


def memory_atomic_body(family, kind, glc):
    body = ["v_lshlrev_b32 v5, 4, v0", "v_mov_b32 v20, 0x1800"]
    pair = kind in ("cmpswap", "swap_x2")
    if family == "global":
        body += ["v_mov_b32 v21, 0", "v_mov_b32 v1, 0", "v_mov_b32 v2, 16"]
        destination = "v[4:5]" if kind.endswith("_x2") else "v4"
        data = "v[1:2]" if pair else "v1"
        operands = (destination + ", " if glc else "") + f"v[20:21], {data}, off"
    else:
        body += ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0",
                 "s_mov_b32 s10, 0x2000", "s_mov_b32 s11, 0x31016fac"]
        data = "v[4:5]" if pair else "v4"
        operands = f"{data}, v20, s[8:11], 0 offen"
    body.append(f"{family}_atomic_{kind} {operands}" + (" glc" if glc else ""))
    return body


class VgprStrideMetadataTests(unittest.TestCase):
    def check_case(self, body, expected_offset, expected_strides, source="v4", after_read=(), memory=None):
        asm, meta = translate_case(body, source, after_read)
        # The first active lane copies the offset to s4; one live lane suffices
        # because every instruction here writes the same value in all lanes.
        memory = run_workgroup(asm, 1, 32, memory=memory)
        self.assertEqual(memory[0x1000], expected_offset)
        self.assertEqual(meta.get("descriptor_strides", []), expected_strides)
        return asm

    def test_swap_clears_overwritten_second_operand(self):
        self.check_case(["v_mov_b32 v0, 48", "v_swap_b32 v0, v4"], 48, [])

    def test_swap_transfers_both_stride_facts(self):
        body = ["v_lshlrev_b32 v0, 4, v0", "v_swap_b32 v0, v4"]
        for source, expected in (("v0", 32), ("v4", 16)):
            with self.subTest(source=source):
                self.check_case(body, expected, [expected], source=source)

    def test_swap_with_itself_retains_stride(self):
        self.check_case(["v_swap_b32 v4, v4"], 32, [32])

    def test_relative_destinations_drop_stale_stride(self):
        for op, body in RELATIVE_WRITES:
            with self.subTest(op=op):
                self.check_case(body, 48, [])

    def test_relative_source_preserves_unmodified_vgpr_stride(self):
        self.check_case(["s_mov_b32 m0, 4", "v_movrels_b32 v8, v0"], 32, [32])

    def test_relative_writes_preserve_copied_sgpr_stride(self):
        for op, body in RELATIVE_WRITES:
            with self.subTest(op=op):
                self.check_case([], 32, [32], after_read=body)

    def test_scaling_after_relative_write_is_tracked(self):
        body = [*RELATIVE_WRITES[0][1], "v_lshlrev_b32 v4, 6, v0"]
        self.check_case(body, 64, [64])

    def test_direct_overwrite_still_drops_stride(self):
        self.check_case(["v_mov_b32 v4, 48"], 48, [])

    def test_memory_atomic_returns_invalidate_only_written_words(self):
        for family in ("global", "buffer"):
            for kind in ("add", "cmpswap", "swap_x2"):
                for glc in (False, True):
                    for source in ("v4", "v5"):
                        with self.subTest(family=family, kind=kind, glc=glc, source=source):
                            written = glc and (source == "v4" or kind.endswith("_x2"))
                            old_stride = 32 if source == "v4" else 16
                            returned = 48 if source == "v4" else 64
                            self.check_case(
                                memory_atomic_body(family, kind, glc),
                                returned if written else old_stride,
                                [] if written else [old_stride], source=source,
                                memory={0x1800: 48, 0x1804: 64},
                            )

    def test_image_atomic_metadata_matches_translated_return_writes(self):
        # The CPU interpreter has no storage-image model. Inspect actual translated
        # destination stores here, including the unchanged comparison word.
        for kind in ("add", "cmpswap", "fcmpswap"):
            pair = kind != "add"
            for glc in (False, True):
                for source in ("v4", "v5"):
                    with self.subTest(kind=kind, glc=glc, source=source):
                        data = "v[4:5]" if pair else "v4"
                        dmask = "0x3" if pair else "0x1"
                        body = [
                            "v_lshlrev_b32 v5, 4, v0",
                            "v_mov_b32 v20, 0", "v_mov_b32 v21, 0",
                            f"image_atomic_{kind} {data}, v[20:21], s[16:23] "
                            f"dmask:{dmask} dim:SQ_RSRC_IMG_2D unorm" + (" glc" if glc else ""),
                        ]
                        asm, meta = translate_case(body, source)
                        after = asm[asm.index(" = OpAtomic"):]
                        for register in (4, 5):
                            # Register pointers may be hoisted before the atomic; only
                            # destination writes need to follow the returned value.
                            pointers = re.findall(
                                rf"(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c{register}\b", asm)
                            stores = any(re.search(rf"OpStore {re.escape(pointer)}\b", after)
                                         for pointer in pointers)
                            self.assertEqual(stores, glc and register == 4)
                        expected = [] if glc and source == "v4" else [32 if source == "v4" else 16]
                        self.assertEqual(meta.get("descriptor_strides", []), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

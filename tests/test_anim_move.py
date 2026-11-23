"""Synthetic tests for tools/anim_move.py (ASSET-86).

    python -I -m unittest tests/test_anim_move.py -v

Every byte here is built synthetically by the test: single-stream and 64 KiB-split multi-stream MoVE v19
containers with request/flag/5-byte descriptor preambles, Type 1 (StateMachine), Type 3 (InlinedStateMachine),
and Type 22 (MotionTree) hierarchies, outgoing and default transitions, and length-prefixed parameter blocks.
No game file or game asset name is used.
"""
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "anim_move.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_move  # noqa: E402
from anim_index import MAX_CHUNK, joaat32  # noqa: E402
from test_anim_clips import write_rpf  # noqa: E402

HASH_SM = joaat32("statemachine")
HASH_MT = joaat32("motiontree")


def build_transition(target_rel_from_48, cond_kind=1, extra_cond_words=(0x11110001,)):
    """Build an outgoing state transition record (`u32 size` at +0x10, `i32 target_rel` at +0x48)."""
    extra = b"".join(struct.pack("<I", w) for w in extra_cond_words)
    size = 0x50 + len(extra)
    buf = bytearray(size)
    struct.pack_into("<I", buf, 0x10, size)
    struct.pack_into("<iI", buf, 0x48, target_rel_from_48, cond_kind)
    buf[0x50:] = extra
    return bytes(buf)


def build_default_transition(target_rel_from_04, cond_count=1, size=20):
    """Build a default transition record (`{u16 size; u16 cond_count; i32 target_rel}` at +0x00)."""
    buf = bytearray(size)
    struct.pack_into("<HHi", buf, 0, size, cond_count, target_rel_from_04)
    return bytes(buf)


def build_param_block(index=0, counts_16b=(1, 0, 1, 0), outputs_8b=1, curve_sizes=(8,)):
    """Build a length-prefixed MotionTree parameter block satisfying `12 + 16*sum(b1..b4) + 8*b5 + sum(curves)`."""
    b1, b2, b3, b4 = counts_16b
    n16 = b1 + b2 + b3 + b4
    body = bytearray()
    for i in range(n16):
        body += struct.pack("<4I", 0x20000001 + i, 1, i, 0)
    for i in range(outputs_8b):
        body += struct.pack("<2I", 0x30000001 + i, i)
    for csz in curve_sizes:
        sub = bytearray(csz)
        struct.pack_into("<H", sub, 4, csz)
        body += sub
    total = 12 + len(body)
    hdr = struct.pack("<I8B", total, index, b1, b2, b3, b4, outputs_8b, len(curve_sizes), 0)
    return hdr + bytes(body)


def build_motion_tree(
    name_hash=HASH_MT,
    declared_children=1,
    transitions=(),
    param_blocks=(),
    leaf_type=11,
    leaf_hash=0x44440001,
    embedded_sm=b"",
    pad_bytes=0,
):
    """Build a Type 22 (`MotionTree`) container with optional transitions, parameter blocks, and embedded SM."""
    tr_blob = b"".join(transitions)
    blk_blob = b"".join(param_blocks)
    first_node = struct.pack("<HHI", 2, 0, 0x44440000) + bytes(4)  # 12-byte Type 2 node at index 0
    leaf_node = struct.pack("<HHI", leaf_type, 0, leaf_hash) + bytes(36)
    if embedded_sm:
        arena = first_node + embedded_sm + bytes(pad_bytes)
        root_off_in_arena = len(first_node)
    else:
        arena = leaf_node + bytes(pad_bytes)
        root_off_in_arena = 0
    tr_start = 0x30
    blk_start = tr_start + len(tr_blob)
    nodes_start = blk_start + len(blk_blob)
    root_target = nodes_start + root_off_in_arena
    hdr = bytearray(0x30)
    struct.pack_into(
        "<HHIiI4B4I",
        hdr,
        0,
        anim_move.TYPE_MOTION_TREE,
        0,
        name_hash,
        root_target - 8,
        0,
        0,
        0,
        declared_children,
        len(transitions),
        0,
        0,
        0,
        0,
    )
    struct.pack_into("<iiI", hdr, 0x24, tr_start - 0x24, blk_start - 0x28, len(param_blocks))
    return bytes(hdr) + tr_blob + blk_blob + arena


def build_state_machine(
    children,
    name_hash=HASH_SM,
    inlined=False,
    input_rel_offsets=(),
    transitions=(),
    default_transitions=(),
    initial_index=0,
):
    """Build a Type 1 (`StateMachine`) or Type 3 (`InlinedStateMachine`) container around `children` `(hash, bytes)`."""
    n_ch = len(children)
    tr_blob = b"".join(transitions)
    def_tr_blob = b"".join(default_transitions)
    base_hdr = 0x5C + 4 * len(input_rel_offsets) if inlined else 0x58
    tr_start = base_hdr + 8 * n_ch
    def_tr_start = tr_start + len(tr_blob)
    first_child_start = def_tr_start + len(def_tr_blob)

    child_offsets = []
    cur = first_child_start
    for _ch_hash, ch_bytes in children:
        child_offsets.append(cur)
        cur += len(ch_bytes)

    init_target = child_offsets[initial_index]
    hdr = bytearray(base_hdr)
    ntype = anim_move.TYPE_INLINED_STATE_MACHINE if inlined else anim_move.TYPE_STATE_MACHINE
    struct.pack_into(
        "<HHIiI4B4I",
        hdr,
        0,
        ntype,
        0,
        name_hash,
        init_target - 8,
        1,
        0,
        0,
        n_ch,
        len(transitions),
        0,
        0,
        0,
        0,
    )
    struct.pack_into("<i", hdr, 0x24, tr_start - 0x24)
    struct.pack_into("<I", hdr, 0x44, 0x3F800000)  # 1.0f
    struct.pack_into("<Ii", hdr, 0x50, len(default_transitions), def_tr_start - 0x54)
    if inlined:
        struct.pack_into("<I", hdr, 0x58, len(input_rel_offsets))
        for i, rel in enumerate(input_rel_offsets):
            struct.pack_into("<i", hdr, 0x5C + 4 * i, rel)

    directory = bytearray()
    for i, (ch_hash, _ch_bytes) in enumerate(children):
        epos = base_hdr + 8 * i
        ch_rel = child_offsets[i] - (epos + 4)
        directory += struct.pack("<Ii", ch_hash, ch_rel)

    return bytes(hdr) + bytes(directory) + tr_blob + def_tr_blob + b"".join(b for _, b in children)


def build_move_resource(
    root_bytes,
    network_hash=0x11110001,
    requests=(0x11110010, 0x11110011),
    flags=(0x11110020,),
    preamble_mode=1,
    descriptors_5b=((0x11110030, 1), (0x11110031, 7)),
    tail=(0, 0, 0),
):
    """Wrap `root_bytes` in a MoVE v19 header + preamble satisfying `total == 46 + 5 * N_desc + body_size`."""
    body_size = 12 + 4 * (len(requests) + len(flags)) + len(root_bytes)
    out = bytearray()
    out += anim_move.MAGIC
    out += struct.pack("<7I", anim_move.VERSION, 0, 0, 0, network_hash, body_size, 0)
    out += struct.pack("<I", len(requests))
    for h in requests:
        out += struct.pack("<I", h)
    out += struct.pack("<I", len(flags))
    for h in flags:
        out += struct.pack("<I", h)
    out += struct.pack("<2I", preamble_mode, len(descriptors_5b))
    for h, kind in descriptors_5b:
        out += struct.pack("<IB", h, kind)
    out += struct.pack("<HII", *tail)
    out += root_bytes
    return bytes(out)


class AnimMoveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _sample_state_machine_resource(self):
        c1_hash, c2_hash = 0x55550001, 0x55550002
        mt1 = build_motion_tree(
            name_hash=c1_hash,
            declared_children=2,
            transitions=(build_transition(0x40, cond_kind=2),),
            param_blocks=(build_param_block(index=0, counts_16b=(1, 1, 0, 0), outputs_8b=1, curve_sizes=(8, 10)),),
        )
        # Second child MotionTree embeds a Type 3 InlinedStateMachine with one backward input pointer.
        inlined_child_hash = 0x66660001
        inlined_leaf = build_motion_tree(name_hash=inlined_child_hash, declared_children=1)
        inlined_sm = build_state_machine(
            [(inlined_child_hash, inlined_leaf)],
            name_hash=0x66660000,
            inlined=True,
            input_rel_offsets=(-0x60,),
            default_transitions=(build_default_transition(0x20, cond_count=2),),
        )
        mt2 = build_motion_tree(
            name_hash=c2_hash,
            declared_children=3,
            param_blocks=(build_param_block(index=0, counts_16b=(0, 0, 1, 0), outputs_8b=0, curve_sizes=()),),
            embedded_sm=inlined_sm,
        )
        root_sm = build_state_machine(
            [(c1_hash, mt1), (c2_hash, mt2)],
            name_hash=HASH_SM,
            transitions=(build_transition(0x80, cond_kind=3),),
            default_transitions=(build_default_transition(0x30, cond_count=1),),
            initial_index=0,
        )
        return build_move_resource(
            root_sm,
            network_hash=0x12345678,
            requests=(0xAAAA0001, 0xAAAA0002),
            flags=(0xBBBB0001,),
            preamble_mode=2,
            descriptors_5b=((0xCCCC0001, 1), (0xCCCC0002, 7)),
            tail=(1, 0xDDDD0001, 0xE469EC59),
        )

    def _sample_large_two_stream_resource(self):
        # Pad the MotionTree node arena so the resource exceeds 64 KiB (MAX_CHUNK = 65,536) and splits across 2 streams.
        mt = build_motion_tree(
            name_hash=HASH_MT,
            declared_children=1,
            param_blocks=(build_param_block(index=0, counts_16b=(1, 0, 0, 0), outputs_8b=0, curve_sizes=()),),
            pad_bytes=66000,
        )
        res = build_move_resource(
            mt,
            network_hash=0x87654321,
            requests=(),
            flags=(),
            preamble_mode=0,
            descriptors_5b=((0xCCCC0003, 2),),
            tail=(0, 0, 0),
        )
        self.assertGreater(len(res), MAX_CHUNK)
        return res

    def test_parse_move_decodes_hierarchy_and_counts(self):
        blob = self._sample_state_machine_resource()
        parsed = anim_move.parse_move(blob)
        self.assertEqual(parsed["version"], 19)
        self.assertEqual(parsed["network_hash"], 0x12345678)
        self.assertEqual(parsed["bytes"], 46 + 5 * 2 + parsed["body_size"])
        self.assertEqual(parsed["preamble"]["mode"], 2)
        self.assertEqual(parsed["preamble"]["tail"]["sub_flag"], 1)
        self.assertEqual(parsed["root"]["kind"], "state_machine")
        c = parsed["counts"]
        self.assertEqual(c["containers"], 5)
        self.assertEqual(c["container:state_machine"], 1)
        self.assertEqual(c["container:inlined_state_machine"], 1)
        self.assertEqual(c["container:motion_tree"], 3)
        self.assertEqual(c["state_machine_children"], 2)
        self.assertEqual(c["inlined_state_machine_children"], 1)
        self.assertEqual(c["inlined_state_machine_inputs"], 1)
        self.assertEqual(c["transitions"], 2)
        self.assertEqual(c["default_transitions"], 2)
        self.assertEqual(c["param_blocks"], 2)
        self.assertEqual(c["param_bindings_16b"], 3)
        self.assertEqual(c["param_outputs_8b"], 1)
        self.assertEqual(c["param_curves"], 2)

    def test_scan_rpf_joins_64k_continuation_stream(self):
        one = self._sample_state_machine_resource()
        two = self._sample_large_two_stream_resource()
        rpf = os.path.join(self.tmp.name, "synthetic.rpf")
        write_rpf(rpf, [one, two[:MAX_CHUNK], two[MAX_CHUNK:]])
        resources, summary = anim_move.scan(rpf)
        self.assertEqual(summary["move_resources"], 2)
        self.assertEqual(summary["move_streams"], 3)
        self.assertEqual(summary["counts"]["move_resources"], 2)
        self.assertEqual(summary["counts"]["multi_stream_resources"], 1)
        self.assertEqual([r["streams"] for r in resources], [1, 2])
        self.assertEqual([r["bytes"] for r in resources], [len(one), len(two)])

    def test_scan_directory_and_cli_json_output(self):
        carved = os.path.join(self.tmp.name, "carved")
        os.makedirs(carved)
        one = self._sample_state_machine_resource()
        two = self._sample_large_two_stream_resource()
        with open(os.path.join(carved, "00010000.bin"), "wb") as f:
            f.write(one)
        with open(os.path.join(carved, "00020000.bin"), "wb") as f:
            f.write(two[:MAX_CHUNK])
        with open(os.path.join(carved, "00030000.bin"), "wb") as f:
            f.write(two[MAX_CHUNK:])

        with open(os.path.join(carved, "00025000.CSV"), "wb") as f:
            f.write(b"offset,bytes")
        out = os.path.join(carved, "move.json")
        for _ in range(2):
            r = subprocess.run([sys.executable, "-I", TOOL, carved, out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["streams"], 3)
        self.assertEqual(doc["summary"]["counts"]["move_resources"], 2)
        self.assertEqual(doc["summary"]["counts"]["multi_stream_resources"], 1)
        self.assertEqual([row["streams"] for row in doc["move_resources"]], [1, 2])
        # Ensure per-asset hash arrays are not dumped in the summary JSON.
        self.assertNotIn("network_hash", json.dumps(doc["move_resources"]))

    def test_corrupt_resource_is_reported_not_raised_in_scan(self):
        bad = bytearray(self._sample_state_machine_resource())
        struct.pack_into("<I", bad, 0x18, 12)  # corrupt body_size
        rpf = os.path.join(self.tmp.name, "bad.rpf")
        write_rpf(rpf, [bytes(bad)])
        resources, summary = anim_move.scan(rpf)
        self.assertEqual(summary["counts"]["move_resources_unparsed"], 1)
        self.assertIn("size mismatch", resources[0]["error"])

    def test_cli_rejects_start_limit_on_directory(self):
        out = os.path.join(self.tmp.name, "out.json")
        r = subprocess.run(
            [sys.executable, "-I", TOOL, self.tmp.name, out, "--start", "0"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 2)

    def test_cli_refuses_to_overwrite_input_and_creates_missing_parent(self):
        one = self._sample_state_machine_resource()
        rpf = os.path.join(self.tmp.name, "in.rpf")
        write_rpf(rpf, [one])
        same = subprocess.run([sys.executable, "-I", TOOL, rpf, rpf], capture_output=True, text=True)
        self.assertNotEqual(same.returncode, 0)
        nested_out = os.path.join(self.tmp.name, "nested", "dir", "move.json")
        ok = subprocess.run([sys.executable, "-I", TOOL, rpf, nested_out], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(os.path.isfile(nested_out))

    def test_cli_refuses_to_overwrite_scanned_directory_stream(self):
        carved = os.path.join(self.tmp.name, "carved")
        os.makedirs(carved)
        stream = os.path.join(carved, "00010000.bin")
        payload = self._sample_state_machine_resource()
        with open(stream, "wb") as f:
            f.write(payload)
        res = subprocess.run([sys.executable, "-I", TOOL, carved, stream], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refuses to overwrite", res.stderr)
        with open(stream, "rb") as f:
            self.assertEqual(f.read(), payload)

    def test_cli_refuses_relative_alias_to_input_archive(self):
        one = self._sample_state_machine_resource()
        rpf = os.path.join(self.tmp.name, "in.rpf")
        write_rpf(rpf, [one])
        with open(rpf, "rb") as f:
            original = f.read()
        alias = os.path.join(self.tmp.name, ".", "in.rpf")
        res = subprocess.run([sys.executable, "-I", TOOL, rpf, alias], capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("refuses to overwrite", res.stderr)
        with open(rpf, "rb") as f:
            self.assertEqual(f.read(), original)

    def test_module_docstring_has_no_double_encoded_characters(self):
        """The preamble layout uses U+00D7; it was once stored as the pair U+00C3 U+2014."""
        with open(TOOL, encoding="utf-8") as f:
            source = f.read()
        for broken in ("\u00c3\u2014", "\u00e2\u20ac", "\u00c2"):
            self.assertNotIn(broken, source)
        self.assertIn("N_req \u00d7 u32 request hashes", source)

    def test_parse_move_rejects_excessive_nesting_without_recursion_error(self):
        cur = build_motion_tree(name_hash=0x55550001)
        prev_hash = 0x55550001
        for i in range(anim_move.MAX_CONTAINER_DEPTH + 5):
            h = 0x55550002 if prev_hash == 0x55550001 else 0x55550001
            cur = build_state_machine([(prev_hash, cur)], name_hash=h)
            prev_hash = h
        blob = build_move_resource(cur, requests=(), flags=(), preamble_mode=0, descriptors_5b=())
        with self.assertRaisesRegex(anim_move.DecodeError, "nesting exceeds"):
            anim_move.parse_move(blob)
        carved = os.path.join(self.tmp.name, "deep")
        os.makedirs(carved)
        with open(os.path.join(carved, "00010000.bin"), "wb") as f:
            f.write(blob)
        resources, summary = anim_move.scan(carved)
        self.assertEqual(summary["counts"]["move_resources_unparsed"], 1)
        self.assertIn("nesting exceeds", resources[0]["error"])

    def test_parse_move_rejects_overlapping_transitions_and_truncated_child_dir(self):
        mt = build_motion_tree(name_hash=0x55550001)

        # n_transitions=0, n_def_tr=1 with def_tr_rel pointing backward into header bytes before child_table_end.
        sm_back = bytearray(build_state_machine([(0x55550001, mt)], name_hash=HASH_SM))
        struct.pack_into("<HHi", sm_back, 0x30, 0x30, 1, 0x20)
        struct.pack_into("<Ii", sm_back, 0x50, 1, 0x30 - 0x54)
        blob_back = build_move_resource(bytes(sm_back), requests=(), flags=(), preamble_mode=0, descriptors_5b=())
        with self.assertRaisesRegex(anim_move.DecodeError, "default-transition offset"):
            anim_move.parse_move(blob_back)

        # n_transitions=1, n_def_tr=0 with def_tr_rel pointing past first_child_off into child bytes.
        tr = build_transition(0x40, cond_kind=1)
        sm_fwd = bytearray(build_state_machine([(0x55550001, mt)], name_hash=HASH_SM, transitions=(tr,)))
        # Extend transition size by 0x10 and move def_tr_rel 0x10 bytes past first_child_off.
        first_child = 0x58 + 8 + len(tr)
        struct.pack_into("<I", sm_fwd, 0x58 + 8 + 0x10, len(tr) + 0x10)
        struct.pack_into("<Ii", sm_fwd, 0x50, 0, (first_child + 0x10) - 0x54)
        blob_fwd = build_move_resource(bytes(sm_fwd), requests=(), flags=(), preamble_mode=0, descriptors_5b=())
        with self.assertRaisesRegex(anim_move.DecodeError, "default-transition offset"):
            anim_move.parse_move(blob_fwd)

        # Truncated child directory: n_children runs past children_end_limit.
        sm_trunc = bytearray(build_state_machine([(0x55550001, mt)], name_hash=HASH_SM))
        sm_trunc[0x12] = 250
        blob_trunc = build_move_resource(bytes(sm_trunc), requests=(), flags=(), preamble_mode=0, descriptors_5b=())
        with self.assertRaisesRegex(anim_move.DecodeError, "child directory"):
            anim_move.parse_move(blob_trunc)

    def test_continuation_joining_stops_at_stop_signatures_oversized_and_bad_headers(self):
        two = self._sample_large_two_stream_resource()

        # expected_total_size rejects wrong version, non-zero reserved words, and invalid preamble_mode.
        bad_ver = bytearray(two[:MAX_CHUNK])
        struct.pack_into("<I", bad_ver, 4, 18)
        self.assertIsNone(anim_move.expected_total_size(bytes(bad_ver)))

        bad_res = bytearray(two[:MAX_CHUNK])
        struct.pack_into("<I", bad_res, 8, 1)
        self.assertIsNone(anim_move.expected_total_size(bytes(bad_res)))

        bad_mode = bytearray(two[:MAX_CHUNK])
        struct.pack_into("<I", bad_mode, 0x28, 3)  # requests=0, flags=0 -> preamble_mode at 0x28
        self.assertIsNone(anim_move.expected_total_size(bytes(bad_mode)))

        # Directory scan must not glue an RBF0 (STOP_SIGNATURES) stream onto a truncated 64 KiB MoVE stream.
        carved_stop = os.path.join(self.tmp.name, "carved_stop")
        os.makedirs(carved_stop)
        with open(os.path.join(carved_stop, "00010000.bin"), "wb") as f:
            f.write(two[:MAX_CHUNK])
        with open(os.path.join(carved_stop, "00020000.bin"), "wb") as f:
            f.write(b"RBF0" + b"\x00" * 128)
        items_stop, _ = anim_move._collect_move_streams_dir(carved_stop)
        self.assertEqual(len(items_stop), 1)
        self.assertEqual(items_stop[0][1], 1)
        self.assertEqual(len(items_stop[0][3]), MAX_CHUNK)

        # Both directory and RPF scans must not join a continuation chunk larger than MAX_CHUNK (64 KiB).
        oversized = b"\x01" * (MAX_CHUNK + 16)
        carved_big = os.path.join(self.tmp.name, "carved_big")
        os.makedirs(carved_big)
        with open(os.path.join(carved_big, "00010000.bin"), "wb") as f:
            f.write(two[:MAX_CHUNK])
        with open(os.path.join(carved_big, "00020000.bin"), "wb") as f:
            f.write(oversized)
        items_big, _ = anim_move._collect_move_streams_dir(carved_big)
        self.assertEqual(items_big[0][1], 1)

        rpf_big = os.path.join(self.tmp.name, "oversized_cont.rpf")
        write_rpf(rpf_big, [two[:MAX_CHUNK], oversized])
        rpf_items, _ = anim_move._collect_move_streams_rpf(rpf_big)
        self.assertEqual(rpf_items[0][1], 1)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_atomic_json_preserves_existing_mode(self):
        out = os.path.join(self.tmp.name, "mode_report.json")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("{}")
        os.chmod(out, 0o644)
        anim_move._write_json_atomic(out, {"ok": True})
        self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o644)



if __name__ == "__main__":
    unittest.main()

"""Parity tests for the vectorised channel decoders of tools/anim_clips.py.

    python -I tests/test_anim_clips_parity.py

`decode_delta2`, the frame columns of `decode_sequence` and `assemble_tracks` read whole columns at a time. The
straight-line frame-by-frame versions they replaced are kept below as references, and every result (floats
compared by their bits, errors by type and message, the stats counters) must equal the reference's. All input is
built by the test with the encoders of test_anim_clips.py; no game file or game asset name is used.
"""
import collections
import math
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_clips  # noqa: E402
from test_anim_clips import encode_delta2, encode_sequence  # noqa: E402

DecodeError = anim_clips.DecodeError


# ---------------------------------------------------------------- references (the frame-by-frame originals)


def ref_decode_delta2(data, at, nf, stats=None):
    words, packed, quantum, offset = struct.unpack_from("<IIff", data, at)
    c1, c2, c3 = packed & 255, (packed >> 8) & 255, (packed >> 16) & 255
    size = 4 * words
    if size < 16 or at + size > len(data):
        raise DecodeError("type-10 channel runs past the sequence")
    blob = bytes(data[at + 16:at + size])
    nbits = 8 * len(blob)
    nc, ng = (nf + 63) // 64, (nf + 6) // 8
    starts = [anim_clips._bits(blob, nc * c1 + i * c2, c2) for i in range(nc)]
    pos = nc * (c1 + c2)
    widths = []
    for _ in range(ng):
        o, k = pos >> 3, 0
        byte = blob[o] >> (pos & 7) if o < len(blob) else 0
        if byte:
            k = (byte & -byte).bit_length() - 1
        else:
            k = 8 - (pos & 7)
            o += 1
            while o < len(blob) and not blob[o]:
                k += 8
                o += 1
            if o >= len(blob):
                raise DecodeError("type-10 width code runs past the channel")
            k += (blob[o] & -blob[o]).bit_length() - 1
        pos += k + 1
        widths.append(k)
    lane_at = nbits - 256 * c3
    if lane_at < pos:
        raise DecodeError("type-10 lanes overlap the width codes")
    if sum(widths) > 32 * c3:
        raise DecodeError("type-10 widths need more lane bits than the channel has")
    lw = struct.unpack_from(f"<{8 * c3}I", blob, lane_at // 8)
    lanes = [b"".join(struct.pack("<I", lw[w * 8 + j]) for w in range(c3)) for j in range(8)]
    raw, lp = [], 0
    for k in widths:
        for j in range(8):
            raw.append(anim_clips._bits(lanes[j], lp, k))
        lp += k
    values = [0.0] * nf
    for ci in range(nc):
        x, v, f0 = starts[ci], 0, ci * 64
        values[f0] = x * quantum + offset
        for j in range(64):
            f = f0 + j + 1
            if f >= nf:
                break
            v += anim_clips.zigzag(raw[f0 + j])
            x += v
            if j == 63:
                if stats is not None:
                    stats["delta2_boundaries"] += 1
                    stats["delta2_boundaries_closed"] += x == starts[ci + 1]
                break
            values[f] = x * quantum + offset
    return values, size


def ref_decode_sequence(s, at, stats=None):
    h = anim_clips.sequence_header(s.d, at)
    data = memoryview(s.d)[at + 0x20:at + 0x20 + h["length"]]
    nf, fo, fl = h["frames"], h["frame_offset"], h["frame_bytes"]
    lists_at = fo + fl * nf
    if lists_at + 22 > len(data):
        raise DecodeError("sequence counts run past its data")
    counts = struct.unpack_from("<11H", data, lists_at)
    lists, p = [], lists_at + 22
    for c in counts:
        lists.append([(x >> 2, x & 3) for x in struct.unpack_from(f"<{c}H", data, p)])
        p += 2 * ((c + 3) // 4 * 4)
    p, channels, frame_fields = 0, [], []
    for t, c in enumerate(counts):
        if c and t in (6, 9):
            raise DecodeError(f"channel type {t} ({anim_clips.CHANNEL_NAMES[t]}) is not decoded")
        for i in range(c):
            track, comp = lists[t][i]
            value = None
            if t == 0 or t == 1:
                value = struct.unpack_from("<3f", data, p)
                p += 12
            elif t == 2:
                value = struct.unpack_from("<f", data, p)[0]
                p += 4
            elif t == 3:
                frame_fields.append((len(channels), 32, "raw", None))
            elif t == 4:
                bits, quantum, offset = struct.unpack_from("<Iff", data, p)
                p += 12
                frame_fields.append((len(channels), bits, quantum, offset))
            elif t == 5:
                fbits, vbits, words, quantum, offset = struct.unpack_from("<IIIff", data, p)
                table_bytes = bytes(data[p + 20:p + 20 + 4 * words])
                p += 20 + 4 * words
                n = (32 * words) // vbits if vbits else 1
                table = [(anim_clips._bits(table_bytes, i * vbits, vbits) if vbits else 0) * quantum + offset
                         for i in range(n)]
                frame_fields.append((len(channels), fbits, "table", table))
            elif t == 10:
                value, size = ref_decode_delta2(data, p, nf, stats)
                p += size
            channels.append([t, track, comp, value])
    if p != fo:
        raise DecodeError(f"channel definitions end at 0x{p:x}, frame data starts at 0x{fo:x}")
    if stats is not None:
        stats["quantize_bits_match"] += sum(b for _i, b, q, _o in frame_fields if q not in ("raw", "table")) \
            == h["quantize_bits"]
    if frame_fields:
        need = sum(b for _i, b, _q, _o in frame_fields)
        if need > 8 * fl:
            raise DecodeError(f"frame fields need {need} bits, frames have {8 * fl}")
        columns = {i: [] for i, *_ in frame_fields}
        for f in range(nf):
            row = int.from_bytes(data[fo + f * fl:fo + (f + 1) * fl], "little")
            for i, bits, quantum, offset in frame_fields:
                x = row & ((1 << bits) - 1)
                row >>= bits
                if quantum == "raw":
                    columns[i].append(struct.unpack("<f", struct.pack("<I", x))[0])
                elif quantum == "table":
                    if x >= len(offset):
                        raise DecodeError("indirect index past its value table")
                    columns[i].append(offset[x])
                else:
                    columns[i].append(x * quantum + offset)
        for i, col in columns.items():
            channels[i][3] = col
    return nf, channels


def ref_assemble_tracks(nf, channels, ntracks, stats=None):
    by = collections.defaultdict(list)
    for t, track, comp, v in channels:
        if track >= ntracks:
            raise DecodeError("channel names a track past the track list")
        by[track].append((t, comp, v))
    out = {}
    for track, chs in by.items():
        whole = [c for c in chs if c[0] in (0, 1)]
        if whole:
            t, _comp, v = whole[0]
            if t == 0:
                x, y, z = v
                r = 1.0 - x * x - y * y - z * z
                if stats is not None:
                    stats["quaternion_static"] += 1
                    stats["quaternion_static_xyz_over_unit"] += r < -1e-3
                v = (x, y, z, math.sqrt(max(0.0, r)))
            out[track] = [tuple(v)] * nf
            continue
        comps, rebuild, normalise = {}, None, False
        for t, comp, v in chs:
            if t == 7:
                rebuild = comp
            elif t == 8:
                normalise = True
            else:
                comps[comp] = v if isinstance(v, list) else [v] * nf
        if rebuild is not None and sorted(comps) != [0, 1, 2]:
            raise DecodeError("type-7 channel on a track without exactly three stored components")
        if normalise and sorted(comps) != [0, 1, 2, 3]:
            raise DecodeError("type-8 channel on a track without four stored components")
        n = 3 if rebuild is not None else (max(comps) + 1 if comps else 0)
        frames, worst = [], 0.0
        for f in range(nf):
            vec = [comps[c][f] if c in comps else 0.0 for c in range(n)]
            if rebuild is not None:
                r = 1.0 - sum(a * a for a in vec)
                worst = max(worst, -r)
                vec.insert(rebuild, math.sqrt(max(0.0, r)))
            elif n == 4:
                worst = max(worst, abs(math.sqrt(sum(a * a for a in vec)) - 1.0))
            if rebuild is not None or normalise:
                norm = math.sqrt(sum(a * a for a in vec)) or 1.0
                vec = [a / norm for a in vec]
            frames.append(tuple(vec))
        if stats is not None and (rebuild is not None or n == 4):
            kind = "rebuilt" if rebuild is not None else ("type8_four_stored" if normalise else "four_stored")
            stats[f"quaternion_{kind}"] += 1
            stats[f"quaternion_{kind}_worst_1e6"] = max(stats[f"quaternion_{kind}_worst_1e6"], round(worst * 1e6))
        out[track] = frames
    return out


# ---------------------------------------------------------------- helpers


def exact(x):
    """`x` with every float replaced by its 8 bytes, so -0.0, inf and nan compare by their bits."""
    if isinstance(x, float):
        return struct.pack("<d", x)
    if isinstance(x, (list, tuple)):
        return [exact(v) for v in x]
    if isinstance(x, dict):
        return {k: exact(v) for k, v in x.items()}
    return x


def outcome(fn, *args):
    """-> ("ok", result, stats) or ("err", exception type, message); the stats are the counters fn filled in."""
    stats = collections.Counter()
    try:
        return "ok", exact(fn(*args, stats)), dict(stats)
    except Exception as e:  # noqa: BLE001 - the type and message are what is compared
        return "err", type(e).__name__, str(e)


def sequence_page(nf, channels):
    seq = encode_sequence(nf, channels)
    return anim_clips.System(seq + bytes(64))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Parity(unittest.TestCase):
    def assert_same_sequence(self, nf, channels, msg=None):
        s = sequence_page(nf, channels)
        want, got = outcome(ref_decode_sequence, s, 0), outcome(anim_clips.decode_sequence, s, 0)
        self.assertEqual(want, got, msg)
        return want

    # ------------------------------------------------------------ type 10

    def test_delta2_matches_the_reference_at_every_chunk_edge(self):
        rnd = random.Random(10)
        for nf in (1, 2, 3, 8, 9, 10, 63, 64, 65, 66, 127, 128, 129, 130, 191, 192, 193, 300):
            for _ in range(3):
                hi = rnd.choice([2, 8, 200, 1 << 14])
                ints = [rnd.randrange(hi) for _ in range(nf)]
                blob = encode_delta2(ints, rnd.choice([0.002, 1.0, 1e-7]), rnd.choice([0.0, -0.1, 7.25]))
                want = outcome(ref_decode_delta2, blob, 0, nf)
                self.assertEqual(want[0], "ok")
                self.assertEqual(want, outcome(anim_clips.decode_delta2, blob, 0, nf), nf)

    def test_delta2_values_are_the_encoded_integers(self):
        ints = [(f * f) // 7 % 211 for f in range(150)]
        values, size = anim_clips.decode_delta2(encode_delta2(ints, 0.5, -3.0), 0, 150)
        self.assertEqual(values, [i * 0.5 - 3.0 for i in ints])
        self.assertEqual(size, len(encode_delta2(ints, 0.5, -3.0)))

    def test_delta2_constant_signal_has_zero_width_codes(self):
        blob = encode_delta2([5] * 100, 0.25, 1.0)
        self.assertEqual(anim_clips.decode_delta2(blob, 0, 100)[0], [5 * 0.25 + 1.0] * 100)
        self.assertEqual(outcome(ref_decode_delta2, blob, 0, 100), outcome(anim_clips.decode_delta2, blob, 0, 100))

    def test_delta2_boundary_counters(self):
        ints = [(3 * f) % 50 for f in range(200)]
        blob = encode_delta2(ints, 1.0, 0.0)
        want, got = outcome(ref_decode_delta2, blob, 0, 200), outcome(anim_clips.decode_delta2, blob, 0, 200)
        self.assertEqual(want, got)
        self.assertEqual(got[2], {"delta2_boundaries": 3, "delta2_boundaries_closed": 3})   # 4 chunks, 3 joins

    def test_delta2_damaged_blobs_fail_the_same_way(self):
        rnd = random.Random(11)
        ints = [rnd.randrange(300) for _ in range(140)]
        blob = encode_delta2(ints, 0.01, 0.5)
        cases = [blob[:n] for n in (0, 4, 15, 16, 17, len(blob) // 2, len(blob) - 4)]
        for _ in range(60):
            damaged = bytearray(blob)
            damaged[rnd.randrange(len(damaged))] ^= 1 << rnd.randrange(8)
            cases.append(bytes(damaged))
        kinds = collections.Counter()
        for case in cases:
            want = outcome(ref_decode_delta2, case, 0, 140)
            self.assertEqual(want, outcome(anim_clips.decode_delta2, case, 0, 140))
            kinds[want[0]] += 1
        self.assertGreater(kinds["ok"], 0)
        self.assertGreater(kinds["err"], 0)

    # ------------------------------------------------------------ frame columns

    def test_frame_columns_match_the_reference(self):
        rnd = random.Random(12)
        for nf in (1, 2, 63, 64, 65, 128, 129, 200):
            chans = [
                (4, 0, 0, (10, 0.002, -1.0, [rnd.randrange(1 << 10) for _ in range(nf)])),
                (4, 0, 1, (7, 0.01, 0.5, [rnd.randrange(1 << 7) for _ in range(nf)])),
                (4, 1, 0, (1, 2.0, 0.0, [rnd.randrange(2) for _ in range(nf)])),
                (4, 1, 1, (31, 1e-6, 0.0, [rnd.randrange(1 << 31) for _ in range(nf)])),
                (5, 2, 0, (3, 5, [0, 3, 9, 17, 25, 31, 2, 1], 0.1, 1.0, [rnd.randrange(8) for _ in range(nf)])),
                (3, 3, 0, [rnd.choice([0.5, -1.0, 0.0, 1e-30, rnd.random()]) for _ in range(nf)]),
                (10, 4, 0, (0.002, -0.1, [rnd.randrange(60) for _ in range(nf)])),
                (2, 5, 0, 0.5), (1, 6, 0, (1.0, 2.0, 3.0)),
            ]
            outcome_ok = self.assert_same_sequence(nf, chans, nf)
            self.assertEqual(outcome_ok[0], "ok")

    def test_raw_floats_keep_their_bits(self):
        nf = 70
        specials = [0.0, -0.0, math.inf, -math.inf, 1e-45, 3.4028234663852886e38, -1.5]
        values = [f32(specials[f % len(specials)]) for f in range(nf)]       # exactly representable
        s = sequence_page(nf, [(3, 0, 0, values)])
        _nf, channels = anim_clips.decode_sequence(s, 0)
        self.assertEqual([struct.pack("<d", v) for v in channels[0][3]], [struct.pack("<d", v) for v in values])

    def test_raw_floats_with_nan_bit_patterns(self):
        """A NaN stays a NaN (with the payload float32 -> float64 gives it), exactly like the one-at-a-time path."""
        nf = 4
        patterns = [0x7FC00000, 0xFFC00001, 0x7F800001, 0x3F800000]
        s = sequence_page(nf, [(3, 0, 0, [0.0] * nf)])
        seq = bytearray(s.d)
        fo = anim_clips.sequence_header(seq, 0)["frame_offset"]
        for f, bits in enumerate(patterns):
            struct.pack_into("<I", seq, 0x20 + fo + 4 * f, bits)
        s = anim_clips.System(bytes(seq))
        want, got = outcome(ref_decode_sequence, s, 0), outcome(anim_clips.decode_sequence, s, 0)
        self.assertEqual(want, got)
        self.assertTrue(math.isnan(anim_clips.decode_sequence(s, 0)[1][0][3][0]))

    def test_zero_bit_field_leaves_empty_frames(self):
        """Quantised with 0 bits: frames are 0 bytes wide and every value is the offset."""
        nf = 5
        want = self.assert_same_sequence(nf, [(4, 0, 0, (0, 0.5, 2.0, [0] * nf))])
        self.assertEqual(want[0], "ok")
        self.assertEqual(anim_clips.decode_sequence(sequence_page(nf, [(4, 0, 0, (0, 0.5, 2.0, [0] * nf))]), 0)[1],
                         [[4, 0, 0, [2.0] * nf]])

    def test_no_frames(self):
        want = self.assert_same_sequence(0, [(4, 0, 0, (5, 1.0, 0.0, [])), (3, 1, 0, []), (2, 2, 0, 1.0)])
        self.assertEqual(want[0], "ok")
        self.assertEqual(want[1][0], 0)

    def test_indirect_index_past_its_table(self):
        # a 2-entry table (2 x 16 bits = 1 word) indexed by a 3-bit field that reaches 5
        chans = [(5, 0, 0, (3, 16, [10, 20], 0.5, 1.0, [0, 1, 1, 0, 5]))]
        want = self.assert_same_sequence(5, chans)
        self.assertEqual(want, ("err", "DecodeError", "indirect index past its value table"))
        chans = [(5, 0, 0, (3, 16, [10, 20], 0.5, 1.0, [0, 1, 1, 0, 1]))]                      # in range: decodes
        want = self.assert_same_sequence(5, chans)
        self.assertEqual(want[0], "ok")
        # past the table only in a later 64-frame block
        idx = [1] * 100 + [4] + [0] * 5
        want = self.assert_same_sequence(106, [(5, 0, 0, (3, 16, [10, 20], 0.5, 1.0, idx))])
        self.assertEqual(want, ("err", "DecodeError", "indirect index past its value table"))

    def test_random_damage_fails_or_decodes_like_the_reference(self):
        rnd = random.Random(13)
        nf = 90
        chans = [(4, 0, 0, (9, 0.01, 0.0, [rnd.randrange(512) for _ in range(nf)])),
                 (5, 1, 0, (2, 8, [1, 2, 3, 4], 1.0, 0.0, [rnd.randrange(4) for _ in range(nf)])),
                 (3, 2, 0, [rnd.random() for _ in range(nf)]),
                 (10, 3, 0, (0.5, 0.0, [rnd.randrange(40) for _ in range(nf)])),
                 (2, 4, 0, 0.25)]
        seq = encode_sequence(nf, chans)
        # Damage the frame data and the lists after it: a damaged header or definition can ask for 2**31-bit masks.
        frame_at = anim_clips.sequence_header(seq, 0)["frame_offset"]
        seen = collections.Counter()
        for _ in range(300):
            damaged = bytearray(seq + bytes(64))
            for _k in range(rnd.choice([1, 1, 2])):
                damaged[rnd.randrange(0x20 + frame_at, len(seq))] = rnd.randrange(256)
            s = anim_clips.System(bytes(damaged))
            want = outcome(ref_decode_sequence, s, 0)
            self.assertEqual(want, outcome(anim_clips.decode_sequence, s, 0))
            seen[want[0]] += 1
        self.assertGreater(seen["ok"], 0)
        self.assertGreater(seen["err"], 0)

    # ------------------------------------------------------------ assemble_tracks

    def test_assemble_matches_the_reference(self):
        rnd = random.Random(14)
        nf = 40

        def column(lo=-0.6, hi=0.6):
            return [rnd.uniform(lo, hi) for _ in range(nf)]

        for rebuild in range(4):                      # the rebuilt component can sit at any of the 4 places
            comps = [c for c in range(4) if c != rebuild]
            chans = [[10, 0, c, column()] for c in comps] + [[7, 0, rebuild, None]]
            self.assert_same_assembly(nf, chans, 1)
        chans = [[4, 0, c, column(-1, 1)] for c in range(4)] + [[8, 0, 0, None]]                    # type 8
        self.assert_same_assembly(nf, chans, 1)
        self.assert_same_assembly(nf, chans[:4], 1)                                                  # four stored
        self.assert_same_assembly(nf, [[4, 0, c, column()] for c in range(3)], 1)                    # three stored
        self.assert_same_assembly(nf, [[4, 0, 0, column()], [4, 0, 2, column()]], 1)                 # a hole: 0.0
        self.assert_same_assembly(nf, [[2, 0, 1, 0.5]], 1)                                           # a static float
        self.assert_same_assembly(nf, [[0, 0, 0, (0.1, 0.2, 0.3)], [1, 1, 0, (1.0, 2.0, 3.0)]], 2)
        self.assert_same_assembly(nf, [[0, 0, 0, (0.9, 0.9, 0.9)]], 1)                               # over unit
        chans = [[4, 0, 1, column()], [10, 0, 2, column()]] + [[3, 2, c, column()] for c in range(4)]
        self.assert_same_assembly(nf, chans, 3)                                                      # two tracks

    def assert_same_assembly(self, nf, channels, ntracks):
        copies = [[list(c) for c in channels] for _ in range(2)]
        want = outcome(ref_assemble_tracks, nf, copies[0], ntracks)
        got = outcome(anim_clips.assemble_tracks, nf, copies[1], ntracks)
        self.assertEqual(want, got, channels[:1])
        self.assertEqual(copies[0], copies[1])                      # the channels are left as they were
        return want

    def test_assemble_edge_values(self):
        nf = 6
        zero = [0.0] * nf
        # all four components 0: the norm is 0 and is replaced by 1.0, not divided by
        want = self.assert_same_assembly(nf, [[4, 0, c, list(zero)] for c in range(4)] + [[8, 0, 0, None]], 1)
        self.assertEqual(want[1][0][0], [struct.pack("<d", 0.0)] * 4)
        nan = [math.nan] * nf
        self.assert_same_assembly(nf, [[4, 0, c, list(nan)] for c in range(4)] + [[8, 0, 0, None]], 1)
        self.assert_same_assembly(nf, [[4, 0, c, list(nan)] for c in range(3)] + [[7, 0, 3, None]], 1)
        big = [3.0] * nf                                                    # x, y, z over unit: sqrt(max(0, r)) is 0
        want = self.assert_same_assembly(nf, [[4, 0, c, list(big)] for c in range(3)] + [[7, 0, 0, None]], 1)
        self.assertEqual(want[2]["quaternion_rebuilt_worst_1e6"], 26_000_000)
        mixed = [-0.0, 0.0, 1e-30, -1e-30, 0.5, -0.5]
        self.assert_same_assembly(nf, [[4, 0, c, list(mixed)] for c in range(4)], 1)
        self.assert_same_assembly(0, [[4, 0, c, []] for c in range(3)] + [[7, 0, 3, None]], 1)       # no frames

    def test_assemble_errors(self):
        nf = 3
        col = [0.1, 0.2, 0.3]
        for chans, ntracks in (([[4, 0, 0, col], [7, 0, 3, None]], 1),                     # type 7 without 3 comps
                               ([[4, 0, 0, col], [8, 0, 0, None]], 1),                     # type 8 without 4 comps
                               ([[4, 5, 0, col]], 2)):                                      # track past the list
            want = self.assert_same_assembly(nf, chans, ntracks)
            self.assertEqual(want[0], "err")

    def test_assemble_output_shares_nothing_it_should_not(self):
        """Static tracks repeat one tuple (as before); computed tracks get a new tuple per frame."""
        out = anim_clips.assemble_tracks(3, [[1, 0, 0, (1.0, 2.0, 3.0)], [4, 1, 0, [1.0, 2.0, 3.0]]], 2)
        self.assertIs(out[0][0], out[0][2])
        self.assertEqual(out[1], [(1.0,), (2.0,), (3.0,)])

    # ------------------------------------------------------------ whole animations

    def test_decode_animation_matches_the_reference(self):
        """The real two-sequence fixture of test_anim_clips, decoded through both implementations."""
        from test_anim_clips import TRACKS, two_sequence_dictionary
        s = anim_clips.System(two_sequence_dictionary())
        anim = anim_clips.parse_dictionary(s)["packs"][0]["animations"][0]
        self.assertEqual(anim["tracks"], TRACKS)
        got_stats = collections.Counter()
        tracks, total = anim_clips.decode_animation(s, anim, got_stats)
        want_stats, want_tracks, want_total = collections.Counter(), {}, 0
        for si, at in enumerate(anim["sequences"]):
            nf, channels = ref_decode_sequence(s, at, want_stats)
            for t, *_ in channels:
                want_stats[f"channel:{t:02d}_{anim_clips.CHANNEL_NAMES[t]}"] += 1
            part = ref_assemble_tracks(nf, channels, len(anim["tracks"]), want_stats)
            for k, frames in part.items():
                have = want_tracks.setdefault(k, [frames[0]] * want_total if frames else [])
                have.extend(frames[1:] if si else frames)
            want_total += nf - (1 if si else 0)
            for have in want_tracks.values():
                if len(have) < want_total:
                    have.extend([have[-1]] * (want_total - len(have)))
        self.assertEqual(total, want_total)
        self.assertEqual(exact(tracks), exact(want_tracks))
        for k in ("delta2_boundaries", "delta2_boundaries_closed", "quantize_bits_match",
                  "quaternion_rebuilt", "quaternion_type8_four_stored"):
            self.assertEqual(got_stats[k], want_stats[k], k)
        self.assertEqual(sorted(tracks), [0, 1, 2, 3, 4])
        self.assertEqual({len(v) for v in tracks.values()}, {total})

    def test_track_missing_from_a_later_sequence_is_padded(self):
        """A track only the first sequence has keeps its last value for the rest (the `have.extend` padding)."""
        a = encode_sequence(3, [(4, 0, 0, (4, 1.0, 0.0, [1, 2, 3])), (4, 1, 0, (4, 1.0, 0.0, [4, 5, 6]))])
        b = encode_sequence(3, [(4, 0, 0, (4, 1.0, 0.0, [3, 7, 8]))])
        s = anim_clips.System(a + bytes((-len(a)) % 16) + b + bytes(64))
        anim = {"sequences": [0, len(a) + (-len(a)) % 16], "tracks": [(0, 0, 4), (1, 0, 4)], "frames": 5}
        tracks, total = anim_clips.decode_animation(s, anim)
        self.assertEqual(total, 5)
        self.assertEqual(tracks[0], [(1.0,), (2.0,), (3.0,), (7.0,), (8.0,)])
        self.assertEqual(tracks[1], [(4.0,), (5.0,), (6.0,), (6.0,), (6.0,)])


if __name__ == "__main__":
    unittest.main()

"""The Rust port of the lane model accepts exactly the streams Python accepts, with the same symbols.

Builds BCPack chunks with the synthetic encoder of `test_bcpack.py` (no game
files), some of them corrupted or cut short, and decodes each one both ways.
Skipped unless `tools/bcpack_calibration` has a release build:

    cargo build --release --locked --offline --manifest-path tools/bcpack_calibration/Cargo.toml
"""
import contextlib
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, HERE)

import bcpack  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402
import test_bcpack as synth  # noqa: E402


def _python(raw, codec, size):
    try:
        lanes, used = bcpack._decode_lanes_raw(memoryview(raw), bcpack.LANE_WIDTHS[codec], size // 8)
    except (bcpack.BcpackError, ZeroDivisionError, IndexError):
        return None
    return None if any(memoryview(raw)[used:]) else (lanes, used)


def _cases(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        codec = rng.choice(sorted(bcpack.LANE_WIDTHS))
        size = rng.choice([1, 2, 7, 64]) * bcpack.BLOCK_BYTES[codec]
        lanes = []
        for w in bcpack.LANE_WIDTHS[codec]:
            rate = rng.random() + 0.05
            lanes.append([min((1 << w) - 1, int(rng.expovariate(rate))) if rng.random() < 0.9
                          else rng.randrange(1 << w) for _ in range(size // 8)])
        q = rng.randrange(16) if rng.random() < 0.7 else 0
        raw = bytearray(synth.encode_lanes(lanes, codec, q=q))
        kind = rng.random()
        if kind < 0.25:
            for _ in range(rng.randint(1, 3)):
                raw[rng.randrange(len(raw))] = rng.randrange(256)
        elif kind < 0.35:
            del raw[rng.randrange(len(raw)):]
        elif kind < 0.4:
            raw += bytes([rng.randrange(1, 256)])
        yield codec, size, bytes(raw), lanes if kind >= 0.4 else None


@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeLaneDecoderTest(unittest.TestCase):
    def test_native_accepts_exactly_what_python_accepts(self):
        accepted = refused = 0
        for codec, size, raw, lanes in _cases(20261123, 160):
            with self.subTest(codec=codec, size=size, raw=raw.hex()[:32]):
                fast = bcpack._decode_lanes_native(raw, bcpack.LANE_WIDTHS[codec], size // 8)
                slow = _python(raw, codec, size)
                self.assertEqual(fast, slow)
                if lanes is not None and slow is not None:
                    self.assertEqual(slow[0], lanes)
                accepted += fast is not None
                refused += fast is None
        self.assertGreater(accepted, 40)
        self.assertGreater(refused, 20)

    def test_uniform_model_and_virtual_zeros(self):
        # q = 0 fragments end on a zero byte the encoder leaves out; the model
        # shifts it back in (docs/textures.md, flush = 1). third_party/bcpack-rs,
        # which follows the hardware RTL, refuses these.
        widths = bcpack.LANE_WIDTHS[4]
        for sym in (0, 1, 3, 63):
            lanes = [[min(sym, (1 << w) - 1)] * 16 for w in widths]
            raw = synth.encode_lanes(lanes, 4, q=0)
            fast = bcpack._decode_lanes_native(raw, widths, 16)
            self.assertIsNotNone(fast)
            self.assertEqual(fast, _python(raw, 4, 128))
            self.assertEqual(fast[0], lanes)

    def test_decode_lanes_falls_back_and_raises_like_python(self):
        for codec, size, raw, _ in _cases(7, 60):
            with self.subTest(codec=codec, size=size):
                slow = _python(raw, codec, size)
                if slow is None:
                    with self.assertRaises(bcpack.BcpackError):
                        bcpack.decode_lanes(raw, codec, size)
                else:
                    self.assertEqual(bcpack.decode_lanes(raw, codec, size), slow)

    def test_switch_off(self):
        saved = list(bcpack._NATIVE), os.environ.get("LEONIDA_BCPACK_NATIVE")
        try:
            bcpack._NATIVE.clear()
            os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
            self.assertIsNone(bcpack._native_unpack())
        finally:
            bcpack._NATIVE[:] = saved[0]
            if saved[1] is None:
                os.environ.pop("LEONIDA_BCPACK_NATIVE", None)
            else:
                os.environ["LEONIDA_BCPACK_NATIVE"] = saved[1]


def _python_chunk(raw, codec, size, provisional, history=None):
    """`decode_with_info` with both native paths switched off, or the exception it raises."""
    saved = bcpack._decode_blocks_native, bcpack._decode_lanes_native
    bcpack._decode_blocks_native = lambda *a, **k: None
    bcpack._decode_lanes_native = lambda *a, **k: None
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional, history=history)
    except Exception as exc:  # noqa: BLE001 - the comparison includes the error
        return type(exc), str(exc)
    finally:
        bcpack._decode_blocks_native, bcpack._decode_lanes_native = saved


def _chunk(raw, codec, size, provisional, history=None):
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional, history=history)
    except Exception as exc:  # noqa: BLE001
        return type(exc), str(exc)


def _split_distance(d):
    """The two 4-bit lane-0 symbols of a BC3/BC5 block that carry history distance `d` (0..255)."""
    code = d << 1 if d <= 127 else ((256 - d) << 1) - 1
    return (sum(((code >> 2 * k) & 1) << k for k in range(4)),
            sum(((code >> 2 * k + 1) & 1) << k for k in range(4)))


def _zig(v):
    return v << 1 if v >= 0 else ((-v) << 1) - 1


def _random_two_element_lanes(rng, codec, n, entries, builtin):
    """Random lanes of `n` BC3 or BC5 blocks (two symbols per block).

    With `builtin`, some distances reach into the built-in history (the entries in `entries` and wild ones).
    """
    widths = bcpack.LANE_WIDTHS[codec]
    lanes = [[rng.randrange(1 << w) if rng.random() < 0.5 else min((1 << w) - 1, int(rng.expovariate(0.3)))
              for _ in range(2 * n)] for w in widths]
    lanes[1] = [0] * (2 * n) if rng.random() < 0.9 else [rng.randrange(4) for _ in range(2 * n)]
    for i in range(n):
        r = rng.random()
        d = 0
        if builtin and r >= 0.9:
            d = rng.randint(1, 255)
        elif builtin and r >= 0.7 and entries:
            d = 256 + i - rng.choice(entries)
            d = d if 1 <= d <= 255 else 0
        elif r >= 0.3 and i:
            d = rng.randint(1, min(i, 20))
        lanes[0][2 * i], lanes[0][2 * i + 1] = _split_distance(d)
    return lanes


def _encode_two_element_blocks(codec, blocks, ref_dists):
    """A codec 3 or 5 chunk for raw 16-byte blocks, each predicted from the block `ref_dists[i]` back (0: none)."""
    inv_pair = {v: i for i, v in enumerate(bcpack._tables.BC3_ALPHA_PAIR if codec == 3 else bcpack._tables.BC5_PAIR)}
    inv_quad = {v: i for i, v in enumerate(bcpack._tables.BC3_COLOUR_QUADRANT)}
    mods = (256, 256, 32, 64, 32, 32, 64, 32) if codec == 3 else (256,) * 4
    none = (128, 128, 0, 0, 0, 0, 0, 0) if codec == 3 else (128,) * 4
    n = len(blocks) // 16
    lanes = [[] for _ in bcpack.LANE_WIDTHS[codec]]

    def pairs(index_bytes):  # six bytes of 3-bit indices -> row symbols of the left and right element
        bits = int.from_bytes(index_bytes, "little")
        out = []
        for r in range(4):
            row = bits >> 12 * r
            out.append((inv_pair[(row & 7, (row >> 3) & 7)], inv_pair[((row >> 6) & 7, (row >> 9) & 7)]))
        return out

    hist = []
    for i in range(n):
        b = blocks[i * 16:i * 16 + 16]
        if codec == 3:
            c0, c1 = int.from_bytes(b[8:10], "little"), int.from_bytes(b[10:12], "little")
            e = (b[0], b[1], c0 >> 11, (c0 >> 5) & 63, c0 & 31, c1 >> 11, (c1 >> 5) & 63, c1 & 31)
        else:
            e = (b[0], b[1], b[8], b[9])
        d = ref_dists[i]
        base = none if d == 0 else hist[i - d]
        hist.append(e)
        a_sym, b_sym = _split_distance(d)
        lanes[0] += [a_sym, b_sym]
        lanes[1] += [0, 0]
        delta = []
        for k in range(len(e)):
            v = (e[k] - base[k]) % mods[k]
            delta.append(_zig(v - mods[k] if v >= mods[k] // 2 else v))
        if codec == 3:
            lanes[2] += [delta[0], delta[1]]
            for lane, (left, right) in zip((7, 8, 9), ((2, 5), (3, 6), (4, 7))):
                lanes[lane] += [delta[left], delta[right]]
        else:
            lanes[2] += [delta[0], delta[1]]
            lanes[3] += [delta[2], delta[3]]
        if codec == 3:
            rows = pairs(b[2:8])
            for r in range(4):
                lanes[3 + r] += list(rows[r])
            q = [b[12] & 15 | (b[13] & 15) << 4, b[12] >> 4 | (b[13] >> 4) << 4,
                 b[14] & 15 | (b[15] & 15) << 4, b[14] >> 4 | (b[15] >> 4) << 4]
            lanes[10] += [inv_quad[q[0]], inv_quad[q[1]]]
            lanes[11] += [inv_quad[q[2]], inv_quad[q[3]]]
        else:
            for ch in range(2):
                rows = pairs(b[8 * ch + 2:8 * ch + 8])
                for r in range(4):
                    lanes[4 + 4 * ch + r] += list(rows[r])
    return synth.encode_lanes(lanes, codec)



@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeBlockRebuildTest(unittest.TestCase):
    """BC1, BC3, BC4 and BC5 chunks give the same bytes, provisional entries and errors with the native rebuild."""

    def test_bc1_bc4_match_python(self):
        rng = random.Random(4242)
        native = 0
        for _ in range(120):
            codec = rng.choice([1, 4])
            n = rng.choice([1, 3, 64, 300])
            widths = bcpack.LANE_WIDTHS[codec]
            lanes = [[rng.randrange(1 << w) if rng.random() < 0.5 else min((1 << w) - 1, int(rng.expovariate(0.3)))
                      for _ in range(n)] for w in widths]
            lanes[1] = [0] * n if rng.random() < 0.9 else [rng.randrange(4) for _ in range(n)]
            for i in range(n):
                r = rng.random()
                d = 0 if r < 0.3 else rng.randint(1, max(1, min(i, 20))) if i and r < 0.9 else rng.randint(1, 255)
                lanes[0][i] = (d << 1) if d <= 127 else ((256 - d) << 1) - 1
            raw = bytearray(synth.encode_lanes(lanes, codec, q=rng.randrange(16)))
            if rng.random() < 0.15:
                raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
            raw = bytes(raw)
            provisional = rng.random() < 0.4
            with self.subTest(codec=codec, n=n, provisional=provisional):
                self.assertEqual(_chunk(raw, codec, n * 8, provisional),
                                 _python_chunk(raw, codec, n * 8, provisional))
            native += bcpack._decode_blocks_native(raw, codec, n * 8, None) is not None
        self.assertGreater(native, 20, "the native rebuild should take many of these chunks")

    def test_bc1_round_trip(self):
        rng = random.Random(5)
        blocks = bytes(rng.randrange(256) for _ in range(8 * 256))
        raw = synth.encode_bc1_blocks(blocks)
        self.assertEqual(bcpack._decode_blocks_native(raw, 1, len(blocks), None), blocks)


    def test_bc3_bc5_match_python(self):
        rng = random.Random(3535)
        native = {3: 0, 5: 0}
        seen = {3: 0, 5: 0}
        for _ in range(240):
            codec = rng.choice([3, 5])
            n = rng.choice([1, 3, 64, 300])
            entries = sorted(set(getattr(bcpack._tables, f"BC{codec}_HISTORY_CONFIRMED"))
                             | set(getattr(bcpack._tables, f"BC{codec}_HISTORY_PROVISIONAL")))
            lanes = _random_two_element_lanes(rng, codec, n, entries, rng.random() < 0.5)
            raw = bytearray(synth.encode_lanes(lanes, codec, q=rng.randrange(16)))
            if rng.random() < 0.15:
                raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
            elif rng.random() < 0.05:
                del raw[rng.randrange(len(raw)):]
            raw = bytes(raw)
            provisional = rng.random() < 0.4
            with self.subTest(codec=codec, n=n, provisional=provisional):
                got = _chunk(raw, codec, n * 16, provisional)
                self.assertEqual(got, _python_chunk(raw, codec, n * 16, provisional))
            fast = bcpack._decode_blocks_native(raw, codec, n * 16, None)
            seen[codec] += 1
            native[codec] += fast is not None
            if fast is not None:  # accepted natively: Python accepts it too, in strict mode, with the same bytes
                self.assertEqual(_python_chunk(raw, codec, n * 16, False), (fast, set()))
        for codec in (3, 5):
            self.assertGreater(native[codec], 15, f"the native rebuild should take many BC{codec} chunks")
            self.assertGreater(seen[codec] - native[codec], 15, f"BC{codec} should have refused chunks too")

    def test_bc3_bc5_round_trip(self):
        rng = random.Random(77)
        for codec in (3, 5):
            for n in (1, 2, 17, 256, 600):
                blocks = bytes(rng.randrange(256) for _ in range(16 * n))
                dists = [0 if i == 0 or rng.random() < 0.3 else rng.randint(1, min(i, 40)) for i in range(n)]
                raw = _encode_two_element_blocks(codec, blocks, dists)
                with self.subTest(codec=codec, n=n):
                    self.assertEqual(bcpack._decode_blocks_native(raw, codec, len(blocks), None), blocks)
                    self.assertEqual(bcpack.decode(raw, codec, len(blocks)), blocks)
                    self.assertEqual(_python_chunk(raw, codec, len(blocks), False), (blocks, set()))

    def test_bc3_bc5_refuse_what_python_decides(self):
        rng = random.Random(8)
        for codec in (3, 5):
            blocks = bytes(rng.randrange(256) for _ in range(16 * 40))
            raw = _encode_two_element_blocks(codec, blocks, [0] * 40)
            # a non-zero lane 1 is Python's error to raise
            lanes = _random_two_element_lanes(rng, codec, 8, [], False)
            lanes[1] = [0] * 15 + [1]
            bad = synth.encode_lanes(lanes, codec)
            self.assertIsNone(bcpack._decode_blocks_native(bad, codec, 8 * 16, None))
            for provisional in (False, True):
                got = _chunk(bad, codec, 8 * 16, provisional)
                self.assertEqual(got, _python_chunk(bad, codec, 8 * 16, provisional))
                self.assertIs(got[0], bcpack.BcpackError)
                self.assertIn("lane 1 is non-zero", got[1])
            # sizes that are not a whole number of blocks
            self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 16 * 40 + 8, None))
            self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 0, None))
            # a corrupted or cut-off chunk
            for broken in (raw[:len(raw) // 2], raw + b"\x01", bytes(len(raw))):
                self.assertIsNone(bcpack._decode_blocks_native(broken, codec, len(blocks), None))
                for provisional in (False, True):
                    self.assertEqual(_chunk(broken, codec, len(blocks), provisional),
                                     _python_chunk(broken, codec, len(blocks), provisional))

    def test_bc3_bc5_history_entries(self):
        def one_block(codec, entry):
            lanes = [[0, 0] for _ in bcpack.LANE_WIDTHS[codec]]
            lanes[0] = list(_split_distance(256 - entry))  # block 0 reaches back to built-in entry `entry`
            return synth.encode_lanes(lanes, codec)

        def table(codec, kind):
            return set(getattr(bcpack._tables, f"BC{codec}_HISTORY_{kind}"))

        confirmed = sorted(table(5, "CONFIRMED"))[0]
        loose = {c: sorted(table(c, "PROVISIONAL") - table(c, "CONFIRMED"))[0] for c in (3, 5)}
        unknown = {c: next(e for e in range(1, 256) if e not in table(c, "CONFIRMED") | table(c, "PROVISIONAL"))
                   for c in (3, 5)}
        # confirmed entry: the native path takes it, nothing is provisional
        raw = one_block(5, confirmed)
        fast = bcpack._decode_blocks_native(raw, 5, 16, None)
        self.assertIsNotNone(fast)
        self.assertEqual(_python_chunk(raw, 5, 16, False), (fast, set()))
        for codec in (3, 5):
            for label, entry in (("unconfirmed", loose[codec]), ("unknown", unknown[codec])):
                raw = one_block(codec, entry)
                with self.subTest(codec=codec, kind=label):
                    self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 16, None))
                    strict = _chunk(raw, codec, 16, False)
                    self.assertEqual(strict, _python_chunk(raw, codec, 16, False))
                    self.assertIs(strict[0], bcpack.BcpackProvisionalError)
                    loosely = _chunk(raw, codec, 16, True)
                    self.assertEqual(loosely, _python_chunk(raw, codec, 16, True))
                    self.assertTrue(loosely[1], "provisional mode records the entry it used")

    def test_bc3_bc5_external_history(self):
        rng = random.Random(1212)
        for codec, mods in ((3, (256, 256, 32, 64, 32, 32, 64, 32)), (5, (256,) * 4)):
            for replace in (False, True):
                history = {f"bc{codec}": {"confirmed": {"9": [rng.randrange(m) for m in mods],
                                                        "200": [7 % m for m in mods]},
                                          "replace": replace}}
                accepted = 0
                for _ in range(20):
                    n = rng.choice([1, 5, 40])
                    lanes = _random_two_element_lanes(rng, codec, n, [9, 200, 105, 2], True)
                    lanes[1] = [0] * (2 * n)
                    raw = synth.encode_lanes(lanes, codec, q=rng.randrange(16))
                    for provisional in (False, True):
                        with self.subTest(codec=codec, replace=replace, n=n, provisional=provisional):
                            self.assertEqual(_chunk(raw, codec, n * 16, provisional, history),
                                             _python_chunk(raw, codec, n * 16, provisional, history))
                            self.assertEqual(_chunk(raw, codec, n * 16, provisional),
                                             _python_chunk(raw, codec, n * 16, provisional))
                    accepted += bcpack._decode_blocks_native(raw, codec, n * 16, history) is not None
                self.assertGreater(accepted, 0)
        # a history the loader rejects is raised after the lanes, as in Python: a broken chunk still gives BcpackError
        broken = {"bc3": {"confirmed": {"9": [999] * 8}}}
        raw = synth.encode_lanes(_random_two_element_lanes(rng, 3, 4, [], False), 3)
        for chunk in (raw, raw[:5]):
            self.assertEqual(_chunk(chunk, 3, 64, False, broken), _python_chunk(chunk, 3, 64, False, broken))
        self.assertIsNone(bcpack._decode_blocks_native(raw, 3, 64, broken))

    def test_bc3_bc5_negative_controls(self):
        """The comparison notices a wrong lookup table, and the switch turns the native rebuild off."""
        rng = random.Random(99)
        swaps = (
            (3, lambda t: t[:128] + t[129:130] + t[128:129] + t[130:]),  # two colour quadrant rows
            (3, lambda t: t[2:4] + t[:2] + t[4:]),  # two alpha pair rows
            (5, lambda t: t[2:4] + t[:2] + t[4:]),  # two pair rows
        )
        for codec, swap in swaps:
            blocks = bytes(rng.randrange(256) for _ in range(16 * 64))
            raw = _encode_two_element_blocks(codec, blocks, [0] * 64)
            spec = bcpack._NATIVE_BLOCKS[codec]
            saved = dict(bcpack._NATIVE_BUILTIN)
            try:
                bcpack._NATIVE_BLOCKS[codec] = (lambda lut=spec[0]: swap(lut()),) + spec[1:]
                bcpack._NATIVE_BUILTIN.pop(codec, None)
                wrong = bcpack._decode_blocks_native(raw, codec, len(blocks), None)
            finally:
                bcpack._NATIVE_BLOCKS[codec] = spec
                bcpack._NATIVE_BUILTIN.clear()
                bcpack._NATIVE_BUILTIN.update(saved)
            self.assertIsNotNone(wrong)
            self.assertNotEqual(wrong, blocks, "a swapped lookup row must show up in the output")
            self.assertEqual(bcpack._decode_blocks_native(raw, codec, len(blocks), None), blocks)
        saved = list(bcpack._NATIVE), os.environ.get("LEONIDA_BCPACK_NATIVE")
        try:
            bcpack._NATIVE.clear()
            os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
            for codec in (3, 5):
                raw = _encode_two_element_blocks(codec, blocks, [0] * 64)
                self.assertIsNone(bcpack._decode_blocks_native(raw, codec, len(blocks), None))
                self.assertEqual(bcpack.decode(raw, codec, len(blocks)), blocks)
        finally:
            bcpack._NATIVE[:] = saved[0]
            if saved[1] is None:
                os.environ.pop("LEONIDA_BCPACK_NATIVE", None)
            else:
                os.environ["LEONIDA_BCPACK_NATIVE"] = saved[1]



# BC6H and BC7 ------------------------------------------------------------------

def _signed(d, bits):
    """`d` modulo 2**bits as a signed value: the delta the encoder zigzags."""
    d %= 1 << bits
    return d - (1 << bits) if d >= 1 << (bits - 1) else d


def _set_distances(lanes, i, de, di):
    """Lane 0 and 1 symbols of BC6H/BC7 block `i` for endpoint distance `de` and index distance `di`."""
    lanes[0][2 * i], lanes[0][2 * i + 1] = synth._deint4(synth._zig(_signed(de - di, 8)))
    lanes[1][2 * i], lanes[1][2 * i + 1] = synth._deint4(di)


def _pick_distance(rng, i, entries, wild=0.003):
    """0, a block earlier in the chunk, a built-in entry from `entries`, or (rarely) any distance."""
    r = rng.random()
    if r < wild:
        return rng.randint(1, 255)
    if entries and r < 0.15:
        d = 256 + i - rng.choice(entries)
        return d if i < d <= 255 else 0
    if i and r < 0.75:
        return rng.randint(1, min(i, 30))
    return 0


def _noise_lanes(rng, codec, n):
    return [[rng.randrange(1 << w) if rng.random() < 0.5 else min((1 << w) - 1, int(rng.expovariate(0.3)))
             for _ in range(2 * n)] for w in bcpack.LANE_WIDTHS[codec]]


def _set_index_rows(lanes, rows, i, deltas):
    for y in range(4):
        d = deltas[4 * y:4 * y + 4]
        lanes[rows[y]][2 * i] = synth._pack_ix_pair(d[0], d[1], y)
        lanes[rows[y]][2 * i + 1] = synth._pack_ix_pair(d[2], d[3], y)


def _bc6h_lanes(rng, n, history=None, entries=(), bad=0.0, *, modes=None, dists=None, wild=0.003):
    """Lanes of `n` BC6H blocks whose anchor indices agree, except with probability `bad` per block.

    Endpoint lanes are noise (every value decodes). Index deltas are chosen against the
    decoder's index state so that the anchors come out right. With `bad`, some blocks get a
    mode symbol outside 0..13 or a wrong anchor. `modes` cycles the mode symbols, `dists`
    fixes `(de, di)` for every block and `wild` is the chance of any distance (a built-in entry).
    """
    ih = bptc.load_bc6h(history)[1]
    lanes = _noise_lanes(rng, 6, n)
    st_i = []
    for i in range(n):
        de, di = dists or (_pick_distance(rng, i, entries, wild), _pick_distance(rng, i, entries, wild))
        _set_distances(lanes, i, de, di)
        if modes:
            m_sym = modes[i % len(modes)]
        else:
            m_sym = rng.randrange(14) if rng.random() >= bad / 2 else rng.randrange(14, 32)
        lanes[15][2 * i], lanes[15][2 * i + 1] = m_sym, rng.randrange(32)
        part = lanes[15][2 * i + 1]
        reg = bcpack._BC6H_INFO[bcpack._BC6H_CODE[m_sym + 1]][3] if m_sym <= 13 else 1
        ibits = 3 if reg == 2 else 4
        ishift = 4 - ibits
        iref = bcpack._BC6H_NONE_I if di == 0 else st_i[i - di] if i >= di else ih[256 + i - di]
        ix = [rng.randrange(1 << ibits) for _ in range(16)]
        if rng.random() >= bad / 2:
            ix[0] &= (1 << (ibits - 1)) - 1
            if reg == 2:
                ix[bptc.BC7_ANCHOR2[part]] &= 3
        _set_index_rows(lanes, (11, 12, 13, 14), i,
                        [synth._zig(_signed(ix[p] - (iref[p] >> ishift), ibits)) for p in range(16)])
        st_i.append([v << ishift for v in ix])
    return lanes


def _bc7_lanes(rng, n, history=None, entries=(), bad=0.0, *, ranks=None, dists=None, wild=0.003):
    """Lanes of `n` BC7 blocks that pass the strict checks, except with probability `bad` per block.

    Endpoints and indices are chosen against the decoder's state, so that shared p-bits and
    anchor indices agree. With `bad`, some blocks get an inferred or unknown signature, a
    p-bit that does not agree or an anchor with its top bit set. `ranks` cycles the
    signatures, `dists` fixes `(de, di)` for every block and `wild` is as for `_bc6h_lanes`.
    """
    t = bptc.load_bc7(history)
    cfg, rank_of = t["config"], t["rank_of"]
    cell_of = {rank: cell for cell, rank in enumerate(rank_of) if rank >= 0}
    good = [r for r in range(len(cfg)) if cfg[r][0] <= 7 and r not in t["inferred"]]
    lanes = _noise_lanes(rng, 7, n)
    eps, ixs = [], []
    for i in range(n):
        a, b = 2 * i, 2 * i + 1
        de, di = dists or (_pick_distance(rng, i, entries, wild), _pick_distance(rng, i, entries, wild))
        _set_distances(lanes, i, de, di)
        r = rng.random()
        if ranks:
            rank = ranks[i % len(ranks)]
        else:
            rank = (rng.choice(good) if r >= bad / 4 else
                    rng.choice(sorted(t["inferred"])) if r < bad / 8 else len(cfg) - 1)
        lanes[15][a], lanes[15][b] = divmod(cell_of[rank], 16)
        if not ranks and rng.random() < bad / 8:
            lanes[15][b] = rng.randrange(16, 32)  # outside the signature table
        mode, part, rot, _ = cfg[rank]
        if mode > 7:
            eps.append([0] * 24)
            ixs.append([0] * 32)
            continue
        rep = [0] * 24 if de == 0 else eps[i - de] if i >= de else t["history_endpoints"][256 + i - de]
        rix = [0] * 32 if di == 0 else ixs[i - di] if i >= di else t["history_indices"][256 + i - di]
        ns, cb, ab, pk = bcpack._B7_NS[mode], bcpack._B7_CB[mode], bcpack._B7_AB[mode], bcpack._B7_PK[mode]
        w = [cb, cb, cb, ab]
        if rot:
            w[rot - 1], w[3] = ab, cb
        sep = [0] * 24
        for sub in range(ns):
            first = 0
            for c in range(4):
                lane = bcpack._B7_LANE[sub][c]
                if not w[c] or lane < 0:
                    continue
                bits = w[c] + (1 if pk and c == 0 else 0)
                shift = 8 - bits
                for e, at in ((0, a), (1, b)):
                    v = rng.randrange(1 << bits)
                    if pk == 2 and c == 0:
                        if e == 0:
                            first = v & 1
                        elif rng.random() >= bad / 4:
                            v = (v & ~1) | first
                    lanes[lane][at] = synth._zig(_signed(v - (rep[sub * 8 + e * 4 + c] >> shift), bits))
                    sep[sub * 8 + e * 4 + c] = v << shift
        for sub in range(ns, 3):
            sep[sub * 8:sub * 8 + 8] = sep[0:8]
        eps.append(sep)
        b1 = bcpack._B7_IB[mode]
        s1 = [rng.randrange(1 << b1) for _ in range(16)]
        s2 = [rng.randrange(4) for _ in range(16)]
        if rng.random() >= bad / 4:
            if mode == 4:
                s2[0] &= 1
                s1[0] &= 3
            elif mode == 5:
                s1[0] &= 1
                s2[0] &= 1
            else:
                sm = t["partition2"][part] if ns == 2 else t["partition3"][part] if ns == 3 else (0,) * 16
                anchors = (0, t["anchor2"][part]) if ns == 2 else (0, t["anchor3a"][part], t["anchor3b"][part])
                for p in range(16):
                    if p == anchors[sm[p]]:
                        s1[p] &= (1 << (b1 - 1)) - 1
        _set_index_rows(lanes, (11, 12, 13, 14), i,
                        [synth._zig(_signed(s1[p] - (rix[p] >> (4 - b1)), b1)) for p in range(16)])
        six = [v << (4 - b1) for v in s1]
        if mode in (4, 5):
            _set_index_rows(lanes, (6, 7, 8, 10), i,
                            [synth._zig(_signed(s2[p] - (rix[16 + p] >> 2), 2)) for p in range(16)])
            six += [v << 2 for v in s2]
        else:
            six += [0] * 16
        ixs.append(six)
    return lanes


def _bc6h_history(rng):
    """A synthetic external BC6H history: wild values, every class, sometimes explicit confirmed bits."""
    keys = rng.sample(range(1, 256), 12)
    doc = {"endpointHistory": {}, "indexHistory": {}, "endpointClass": {}, "indexClass": {}}
    for k in keys:
        doc["endpointHistory"][str(k)] = [
            [rng.randrange(1 << 16) if rng.random() < 0.8 else rng.randrange(-70000, 300000),
             rng.randrange(1 << 10) if rng.random() < 0.8 else rng.randrange(-3000, 9000),
             rng.randrange(64) if rng.random() < 0.8 else rng.randrange(-200, 500), rng.randrange(64)]
            for _ in range(3)]
        doc["indexHistory"][str(k)] = [rng.randrange(16) if rng.random() < 0.9 else rng.randrange(-40, 80)
                                       for _ in range(16)]
        doc["endpointClass"][str(k)] = rng.choice((0, 1, 2, 3, 3, 3, 3))
        doc["indexClass"][str(k)] = rng.choice((0, 1, 2, 3, 3, 3, 3))
    if rng.random() < 0.5:
        full = (16, 10, 6, 6)
        doc["endpointConfirmedBits"] = {
            str(k): [[v if rng.random() < 0.85 else rng.randint(0, v) for v in full] for _ in range(3)]
            for k in keys if rng.random() < 0.9}
        doc["indexConfirmedBits"] = {str(k): [4 if rng.random() < 0.9 else rng.randint(0, 4) for _ in range(16)]
                                     for k in keys if rng.random() < 0.9}
    return {"bc6h": doc}, keys


def _bc7_history(rng):
    """A synthetic external BC7 history: full, partial and wild entries and a few last-resort slots."""
    keys = rng.sample(range(1, 256), 12)
    names = ("historyEndpoints", "historyEndpointBits", "historyEndpointConfirmedBits",
             "historyIndices", "historyIndexBits", "historyIndexConfirmedBits")
    doc = {name: {} for name in names}
    for k in keys:
        full = rng.random() < 0.6
        doc["historyEndpoints"][str(k)] = [rng.randrange(256) if rng.random() < 0.9 else rng.randrange(-300, 1000)
                                           for _ in range(24)]
        doc["historyEndpointBits"][str(k)] = [8 if full or rng.random() < 0.5 else rng.randint(0, 8)
                                              for _ in range(24)]
        doc["historyEndpointConfirmedBits"][str(k)] = [8 if full or rng.random() < 0.5 else rng.randint(0, 9)
                                                       for _ in range(24)]
        doc["historyIndices"][str(k)] = [rng.randrange(16) if rng.random() < 0.9 else rng.randrange(-40, 80)
                                         for _ in range(32)]
        doc["historyIndexBits"][str(k)] = [4 if full or rng.random() < 0.5 else rng.randint(0, 4) for _ in range(32)]
        doc["historyIndexConfirmedBits"][str(k)] = [4 if full or rng.random() < 0.5 else rng.randint(0, 4)
                                                    for _ in range(32)]
    doc["historyEndpointLastResort"] = [[k, rng.randrange(24)] for k in rng.sample(keys, 3)]
    return {"bc7": doc}, keys


def _bptc_lanes(codec):
    return _bc6h_lanes if codec == 6 else _bc7_lanes


def _bptc_case(rng, codec, n=None):
    """`(raw, size, history)`: a random BC6H or BC7 chunk, sometimes with an external history, damaged or cut."""
    n = n or rng.choice([1, 2, 3, 17, 64, 300])
    history, entries = (_bc6h_history if codec == 6 else _bc7_history)(rng) if rng.random() < 0.4 else (None, [])
    lanes = _bptc_lanes(codec)(rng, n, history, entries, bad=rng.choice([0.0, 0.0, 0.01, 0.1]))
    raw = bytearray(synth.encode_lanes(lanes, codec, q=rng.randrange(16)))
    r = rng.random()
    if r < 0.1:
        raw[rng.randrange(len(raw))] ^= 1 << rng.randrange(8)
    elif r < 0.14:
        del raw[rng.randrange(len(raw)):]
    elif r < 0.16:
        raw += bytes([rng.randrange(1, 256)])
    return bytes(raw), n * 16, history


@contextlib.contextmanager
def _switched_off():
    """`LEONIDA_BCPACK_NATIVE=0` with the library cache cleared, restored afterwards."""
    saved = list(bcpack._NATIVE), os.environ.get("LEONIDA_BCPACK_NATIVE")
    try:
        bcpack._NATIVE.clear()
        os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
        yield
    finally:
        bcpack._NATIVE[:] = saved[0]
        if saved[1] is None:
            os.environ.pop("LEONIDA_BCPACK_NATIVE", None)
        else:
            os.environ["LEONIDA_BCPACK_NATIVE"] = saved[1]


@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeBptcRebuildTest(unittest.TestCase):
    """BC6H and BC7 chunks give the same bytes, provisional entries and errors with the native rebuild."""

    def assert_same(self, raw, codec, size, history=None):
        """Native takes the chunk exactly when strict Python accepts it, and every mode decodes alike."""
        fast = bcpack._decode_blocks_native(raw, codec, size, history)
        strict = _python_chunk(raw, codec, size, False, history)
        self.assertEqual(fast is not None, isinstance(strict[0], bytes))
        if fast is not None:
            self.assertEqual(strict, (fast, set()))
        for provisional in (False, True):
            self.assertEqual(_chunk(raw, codec, size, provisional, history),
                             _python_chunk(raw, codec, size, provisional, history))
        return fast, strict

    def test_bc6h_bc7_match_python(self):
        rng = random.Random(6767)
        native = {6: 0, 7: 0}
        seen = {6: 0, 7: 0}
        for _ in range(160):
            codec = rng.choice([6, 7])
            raw, size, history = _bptc_case(rng, codec)
            with self.subTest(codec=codec, size=size, history=history is not None):
                fast, _ = self.assert_same(raw, codec, size, history)
            seen[codec] += 1
            native[codec] += fast is not None
        for codec in (6, 7):
            self.assertGreater(native[codec], 20, f"the native rebuild should take many codec {codec} chunks")
            self.assertGreater(seen[codec] - native[codec], 10, f"codec {codec} should have refused chunks too")

    def test_bc7_mode6_round_trip(self):
        rng = random.Random(66)
        for n in (1, 2, 40, 300):
            blocks = bytearray(rng.randrange(256) for _ in range(16 * n))
            for i in range(n):
                blocks[16 * i] = (blocks[16 * i] & 0x80) | 0x40  # mode 6
            blocks = bytes(blocks)
            raw = synth.encode_bc7_mode6_blocks(blocks)
            with self.subTest(n=n):
                self.assertEqual(bcpack._decode_blocks_native(raw, 7, len(blocks), None), blocks)
                self.assertEqual(bcpack.decode(raw, 7, len(blocks)), blocks)
        example = synth.DocumentedExampleTest
        self.assertEqual(bcpack._decode_blocks_native(example.BC7_PACKED, 7, 16, None), example.BC7_BLOCK)

    def test_every_mode_and_signature(self):
        """Every BC6H mode and every confirmed BC7 signature is taken natively, with Python's bytes."""
        rng = random.Random(12)
        for m_sym in range(14):
            raw = synth.encode_lanes(_bc6h_lanes(rng, 32, modes=[m_sym], wild=0), 6)
            with self.subTest(codec=6, mode=m_sym + 1):
                fast, _ = self.assert_same(raw, 6, 32 * 16)
                self.assertIsNotNone(fast)
        t = bptc.load_bc7()
        good = [r for r in range(len(t["config"])) if t["config"][r][0] <= 7 and r not in t["inferred"]]
        for start in range(0, len(good), 40):
            raw = synth.encode_lanes(_bc7_lanes(rng, 40, ranks=good[start:start + 40], wild=0), 7)
            with self.subTest(codec=7, ranks=start):
                fast, _ = self.assert_same(raw, 7, 40 * 16)
                self.assertIsNotNone(fast)

    def test_refuse_what_python_decides(self):
        rng = random.Random(9)
        t = bptc.load_bc7()

        def refused(lanes, codec, error, history=None):
            raw = synth.encode_lanes(lanes, codec)
            size = len(lanes[0]) * 8
            fast, strict = self.assert_same(raw, codec, size, history)
            self.assertIsNone(fast)
            self.assertIs(strict[0], error)

        # BC6H: a mode symbol outside 0..13, a wrong anchor, a built-in entry with no history
        lanes = _bc6h_lanes(rng, 4, wild=0)
        lanes[15][4] = 14
        refused(lanes, 6, bcpack.BcpackError)
        lanes = [[0, 0] for _ in range(16)]
        lanes[15] = [10, 0]  # mode 11 with no history: index 0 decodes to 8, its anchor bit set
        refused(lanes, 6, bcpack.BcpackError)
        refused(_bc6h_lanes(rng, 1, dists=(0, 256 - 42)), 6, bcpack.BcpackProvisionalError)
        refused(_bc6h_lanes(rng, 1, dists=(256 - 42, 0)), 6, bcpack.BcpackProvisionalError)
        # BC7: unknown and inferred signatures, built-in entries with no history
        lanes = _bc7_lanes(rng, 3, wild=0)
        lanes[15][2:4] = [31, 17]
        refused(lanes, 7, bcpack.BcpackError)
        lanes = _bc7_lanes(rng, 3, wild=0)
        lanes[15][2:4] = divmod(t["rank_of"].index(min(t["inferred"])), 16)
        refused(lanes, 7, bcpack.BcpackProvisionalError)
        mode6 = t["config"].index((6, 0, 0, 0))
        refused(_bc7_lanes(rng, 1, ranks=[mode6], dists=(256 - 9, 0)), 7, bcpack.BcpackProvisionalError)
        refused(_bc7_lanes(rng, 1, ranks=[mode6], dists=(0, 256 - 9)), 7, bcpack.BcpackProvisionalError)
        # BC7: a shared p-bit that does not agree (mode 1), an anchor with its top bit set (mode 6)
        lanes = _bc7_lanes(rng, 1, ranks=[t["config"].index((1, 0, 0, 0))], dists=(0, 0))
        lanes[2][1] ^= 2  # one zigzag step: flips the low bit of subset 0's second red endpoint
        refused(lanes, 7, bcpack.BcpackError)
        lanes = _bc7_lanes(rng, 1, ranks=[mode6], dists=(0, 0))
        lanes[11][0] = synth._pack_ix_pair(synth._zig(-8), 0, 0)  # pixel 0 decodes to 8: top bit set
        refused(lanes, 7, bcpack.BcpackError)
        # both: sizes that are not whole blocks, a damaged or cut chunk
        for codec in (6, 7):
            raw = synth.encode_lanes(_bptc_lanes(codec)(rng, 20, wild=0), codec)
            self.assertIsNotNone(bcpack._decode_blocks_native(raw, codec, 320, None))
            for size in (0, 8, 328, -16):
                self.assertIsNone(bcpack._decode_blocks_native(raw, codec, size, None))
            for broken in (raw[:len(raw) // 2], raw + b"\x01", bytes(len(raw))):
                fast, _ = self.assert_same(broken, codec, 320)
                self.assertIsNone(fast)

    def test_provisional_mode_is_python(self):
        """BC6H and BC7 in provisional mode always run in Python, which adds labels Rust does not track."""
        rng = random.Random(31)
        history, entries = _bc7_history(rng)
        k = entries[0]
        doc = history["bc7"]
        for name, full in (("historyEndpointBits", 8), ("historyEndpointConfirmedBits", 8),
                           ("historyIndexBits", 4), ("historyIndexConfirmedBits", 4)):
            doc[name][str(k)] = [full] * (24 if "Endpoint" in name else 32)
        doc["historyEndpointLastResort"] = [[k, j] for j in range(24)]
        mode6 = bptc.load_bc7()["config"].index((6, 0, 0, 0))
        raw = synth.encode_lanes(_bc7_lanes(rng, 1, history, ranks=[mode6], dists=(256 - k, 256 - k)), 7)
        fast, _ = self.assert_same(raw, 7, 16, history)
        self.assertIsNotNone(fast, "a confirmed entry is taken natively in strict mode")
        self.assertIsNone(bcpack._decode_blocks_native(raw, 7, 16, history, True))
        loosely = _chunk(raw, 7, 16, True, history)
        self.assertEqual(loosely[0], fast)
        self.assertIn(f"BC7 endpoint history {k} (last-resort)", loosely[1])

    def test_external_history(self):
        rng = random.Random(4545)
        accepted = {6: 0, 7: 0}
        for _ in range(60):
            codec = rng.choice([6, 7])
            history, entries = (_bc6h_history if codec == 6 else _bc7_history)(rng)
            n = rng.choice([1, 5, 40])
            raw = synth.encode_lanes(_bptc_lanes(codec)(rng, n, history, entries), codec)
            with self.subTest(codec=codec, n=n):
                fast, _ = self.assert_same(raw, codec, n * 16, history)
            accepted[codec] += fast is not None
        self.assertGreater(accepted[6], 3)
        self.assertGreater(accepted[7], 3)
        # a history the loader rejects is raised after the lanes, as in Python
        for codec, broken in ((6, {"bc6h": {"endpointHistory": {"999": [[0] * 4] * 3}}}),
                              (6, {"bc6h": {"endpointHistory": {}, "indexConfirmedBits": {"3": [9] * 16}}}),
                              (7, {"bc7": {"historyEndpoints": {"3": [1, 2]}}}),
                              (7, {"bc7": ["not", "a", "section"]})):
            raw = synth.encode_lanes(_bptc_lanes(codec)(rng, 4, wild=0), codec)
            self.assertIsNotNone(bcpack._decode_blocks_native(raw, codec, 64, None))
            self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 64, broken))
            for chunk in (raw, raw[:5]):
                self.assertEqual(_chunk(chunk, codec, 64, False, broken), _python_chunk(chunk, codec, 64, False, broken))

    def test_negative_controls(self):
        """The comparison notices a wrong table, and the switch turns the native rebuild off."""
        rng = random.Random(1001)
        swaps = (
            (6, lambda t: t[:32 + 15] + t[32 + 19:32 + 20] + t[32 + 16:32 + 19] + t[32 + 15:32 + 16] + t[32 + 20:]),
            # ^ the stream bits of mode 1's first two plan items swapped
            (6, lambda t: t[:32 + 8] + bytes([t[32 + 8] + 1]) + t[32 + 9:]),  # mode 1 header one bit longer
            (6, lambda t: t[:17] + bytes([t[17] ^ 8]) + t[18:]),  # the anchor of partition 17
            (7, lambda t: t[:6144] + bytes([14]) + t[6145:]),  # partition 0's second anchor: pixel 14, not 15
            (7, lambda t: t[:6385] + t[6386:6387] + t[6385:6386] + t[6387:]),  # subset 0's green and blue lanes
        )
        t = bptc.load_bc7()
        every = [r for r in range(len(t["config"])) if t["config"][r][0] <= 7 and r not in t["inferred"]]
        for codec, swap in swaps:
            n = 300  # BC7: every confirmed signature at least once
            raw = synth.encode_lanes(_bc6h_lanes(rng, n, wild=0) if codec == 6 else
                                     _bc7_lanes(rng, n, ranks=every, wild=0), codec)
            good = bcpack._decode_blocks_native(raw, codec, n * 16, None)
            self.assertIsNotNone(good)
            spec = bcpack._NATIVE_BPTC[codec]
            self.assertEqual(len(swap(spec[0]())), len(spec[0]()))
            self.assertNotEqual(swap(spec[0]()), spec[0]())
            saved = dict(bcpack._NATIVE_BUILTIN), dict(bcpack._NATIVE_LUT)
            try:
                bcpack._NATIVE_BPTC[codec] = (lambda lut=spec[0]: swap(lut()), spec[1])
                bcpack._NATIVE_BUILTIN.pop(codec, None)
                bcpack._NATIVE_LUT.pop(codec, None)
                wrong = bcpack._decode_blocks_native(raw, codec, n * 16, None)
            finally:
                bcpack._NATIVE_BPTC[codec] = spec
                for cache, old in zip((bcpack._NATIVE_BUILTIN, bcpack._NATIVE_LUT), saved):
                    cache.clear()
                    cache.update(old)
            self.assertTrue(wrong != good, "a wrong table must show up as other bytes or a refusal")
            self.assertEqual(bcpack._decode_blocks_native(raw, codec, n * 16, None), good)
            self.assertEqual(_python_chunk(raw, codec, n * 16, False), (good, set()))
        # the history is masked to the bits a block reads: wider values decode the same, other low bits do not
        history, _ = _bc7_history(rng)
        k = 200
        doc = history["bc7"]
        for name, v in (("historyEndpoints", 0x5A), ("historyEndpointBits", 8), ("historyEndpointConfirmedBits", 8),
                        ("historyIndices", 5), ("historyIndexBits", 4), ("historyIndexConfirmedBits", 4)):
            doc[name][str(k)] = [v] * (24 if "Endpoint" in name else 32)
        mode6 = bptc.load_bc7()["config"].index((6, 0, 0, 0))
        raw = synth.encode_lanes(_bc7_lanes(rng, 1, history, ranks=[mode6], dists=(256 - k, 256 - k)), 7)
        base = bcpack._decode_blocks_native(raw, 7, 16, history)
        self.assertIsNotNone(base)
        doc["historyEndpoints"][str(k)] = [0x5A + 0x300] * 24
        doc["historyIndices"][str(k)] = [5 - 32] * 32
        self.assertEqual(bcpack._decode_blocks_native(raw, 7, 16, history), base)
        self.assertEqual(_python_chunk(raw, 7, 16, False, history), (base, set()))
        doc["historyEndpoints"][str(k)] = [0x5B] * 24
        self.assertNotEqual(bcpack._decode_blocks_native(raw, 7, 16, history), base)
        with _switched_off():
            for codec in (6, 7):
                raw = synth.encode_lanes(_bptc_lanes(codec)(rng, 8, wild=0), codec)
                self.assertIsNone(bcpack._decode_blocks_native(raw, codec, 128, None))
                self.assertEqual(_chunk(raw, codec, 128, False), _python_chunk(raw, codec, 128, False))

    def test_ffi_guards(self):
        """`decode_blocks` refuses lane counts, sizes and table lengths that do not fit, and null pointers."""
        import ctypes
        _, tables, blocks = bcpack._native_unpack()
        raw = synth.encode_lanes(_bc7_lanes(random.Random(2), 4, dists=(0, 0)), 7)
        lut, builtin = bcpack._native_tables(7, None)
        out = ctypes.create_string_buffer(64)
        good = dict(data=raw, size=len(raw), codec=7, widths=bytes(bcpack.LANE_WIDTHS[7]), lanes=16, target=8,
                    tables=tables, tables_len=len(tables), lut=lut, lut_len=len(lut), builtin=builtin,
                    builtin_len=len(builtin), output=out, output_len=64)
        self.assertGreater(blocks(*good.values()), 0)
        lut6, builtin6 = bcpack._native_tables(6, None)
        for change in (dict(lanes=12), dict(lanes=17), dict(codec=2), dict(codec=8), dict(target=7),
                       dict(target=0, output_len=0), dict(output_len=63), dict(lut_len=len(lut) - 1),
                       dict(builtin_len=len(builtin) - 1), dict(tables_len=len(tables) - 1), dict(data=None),
                       dict(widths=None), dict(output=None), dict(builtin=None), dict(lut=None), dict(tables=None),
                       dict(size=len(raw) - 1), dict(target=1 << 61, output_len=(1 << 64) - 8),
                       dict(codec=6, lut=lut6, lut_len=len(lut6), builtin=builtin6, builtin_len=len(builtin6))):
            with self.subTest(change=sorted(change)):
                self.assertEqual(blocks(*{**good, **change}.values()), -1)


if __name__ == "__main__":
    unittest.main()

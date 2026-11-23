"""The Rust port of the lane model accepts exactly the streams Python accepts, with the same symbols.

Builds BCPack chunks with the synthetic encoder of `test_bcpack.py` (no game
files), some of them corrupted or cut short, and decodes each one both ways.
Skipped unless `tools/bcpack_calibration` has a release build:

    cargo build --release --locked --offline --manifest-path tools/bcpack_calibration/Cargo.toml
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, HERE)

import bcpack  # noqa: E402
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


def _python_chunk(raw, codec, size, provisional):
    """`decode_with_info` with both native paths switched off, or the exception it raises."""
    saved = bcpack._decode_blocks_native, bcpack._decode_lanes_native
    bcpack._decode_blocks_native = lambda *a: None
    bcpack._decode_lanes_native = lambda *a: None
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional)
    except Exception as exc:  # noqa: BLE001 - the comparison includes the error
        return type(exc), str(exc)
    finally:
        bcpack._decode_blocks_native, bcpack._decode_lanes_native = saved


def _chunk(raw, codec, size, provisional):
    try:
        return bcpack.decode_with_info(raw, codec, size, allow_provisional=provisional)
    except Exception as exc:  # noqa: BLE001
        return type(exc), str(exc)


@unittest.skipIf(bcpack._native_unpack() is None, "tools/bcpack_calibration is not built")
class NativeBlockRebuildTest(unittest.TestCase):
    """BC1 and BC4 chunks give the same bytes, provisional entries and errors with the native rebuild."""

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


if __name__ == "__main__":
    unittest.main()

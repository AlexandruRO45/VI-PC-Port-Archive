"""The Rust lane decoder agrees with the Python one wherever it accepts a stream.

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
    except bcpack.BcpackError:
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
    def test_native_never_disagrees_with_python(self):
        accepted = 0
        for codec, size, raw, lanes in _cases(20261123, 160):
            with self.subTest(codec=codec, size=size, raw=raw.hex()[:32]):
                fast = bcpack._decode_lanes_native(raw, bcpack.LANE_WIDTHS[codec], size // 8)
                slow = _python(raw, codec, size)
                if fast is not None:
                    accepted += 1
                    self.assertIsNotNone(slow, "native accepted a stream Python refuses")
                    self.assertEqual(fast, slow)
                if lanes is not None and slow is not None:
                    self.assertEqual(slow[0], lanes)
        self.assertGreater(accepted, 20, "the native decoder should accept many of these streams")

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


if __name__ == "__main__":
    unittest.main()

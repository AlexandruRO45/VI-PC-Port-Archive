"""Parity tests for the optimised hot paths of `bcpack_history_calib`.

Each test compares the current function with a reference copied here from the
straightforward version it replaced, on hand-built, seeded random inputs, so nothing
depends on game data, on a platform, or on pinned bytes:

* `decode_lanes_prefix`: the cycle-by-cycle prefix scheduler (a copy of
  `bcpack._decode_lanes_raw` that stops after `max_sym` emitted elements);
* `Bc6hCalibrator._fit_signed_delta` and `Bc7Calibrator._fit_ep_pair`: the per-candidate
  error searches of the table fit;
* `_pretty_json`: `json.dumps(obj, sort_keys=True, indent=2)`.

    python -I tests/test_bcpack_history_calib_parity.py
"""
import json
import os
import random
import sys
import tempfile
import unittest
from bisect import bisect_right

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bcpack  # noqa: E402
import bcpack_history_calib as calib  # noqa: E402
import bcpack_tables as tables  # noqa: E402
import test_bcpack_history_calib as stream_builder  # noqa: E402  (synthetic rANS encoder helpers)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def reference_decode_lanes_prefix(raw, codec, out_size, max_sym):
    """Decode up to `max_sym` symbols per lane from a BCPack chunk (reference scheduler)."""
    nsym = out_size // 8
    if max_sym <= 0 or max_sym >= nsym:
        return bcpack.decode_lanes(raw, codec, out_size)[0]
    widths = bcpack.LANE_WIDTHS[codec]
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = bcpack.RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [False] * nl
    full = [False] * nl
    held = [False] * nl
    vz = [0] * nl
    inbuf = [0] * nl
    sym = [0] * nl
    in_rem = [0] * nl
    p0 = [0] * nl
    p1 = [0] * nl
    model = [None] * nl
    slotfull = [False] * nl
    slotval = [0] * nl
    stall = [False] * nl
    out = [[] for _ in range(nl)]
    pos = 0
    nraw = len(raw)
    data = memoryview(raw)
    emitted = 0
    while emitted < max_sym:
        left = 8
        for i in range(nl):
            if done[i] or (held[i] and slotfull[i]):
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                if pos >= nraw:
                    raise bcpack.BcpackError("BCPack lane stream ended before prefix symbols finished")
                inbuf[i] = data[pos]
                pos += 1
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False
        for i in range(nl):
            if done[i] or stall[i]:
                continue
            c = ctrl[i]
            if c == HDR0:
                if not full[i]:
                    continue
                h = inbuf[i]
                q[i] = h & 15
                flush[i] = (h >> 4) & 1
                h3[i] = (h >> 5) & 1
                w[i] = widths[i] - (h >> 6)
                full[i] = False
                ctrl[i] = HDR1
                vz[i] = 0
            elif c == HDR1:
                if not full[i]:
                    continue
                full[i] = False
                model[i] = bcpack._get_model(w[i], q[i])
                if h3[i]:
                    in_rem[i] = inbuf[i]
                    ctrl[i] = HDR2
                else:
                    in_rem[i] = inbuf[i] + 1
                    ctrl[i] = PROC
            elif c == HDR2:
                if not full[i]:
                    continue
                in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257
                ctrl[i] = PROC
                full[i] = False
            else:
                st = p0[i]
                if st < L:
                    if in_rem[i] > 0:
                        if full[i]:
                            st = (st << 8) | inbuf[i]
                            in_rem[i] -= 1
                            full[i] = False
                            if in_rem[i] == 0 and flush[i]:
                                while st < L and st != 0 and vz[i] < 8:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        while st < L and st != 0 and vz[i] < 8:
                            st <<= 8
                            vz[i] += 1
                    else:
                        p1[i] = st
                        if out_rem[i] > 0:
                            ctrl[i] = HDR0
                        else:
                            done[i] = True
                        continue
                nxt = st > L and out_rem[i] > 0
                if nxt:
                    sym[i] = model[i].lookup(st & 0xFFFF)
                    out_rem[i] -= 1
                held[i] = nxt
                p1[i] = st
                if in_rem[i] == 0:
                    if flush[i]:
                        if st == L:
                            if out_rem[i] > 0:
                                ctrl[i] = HDR0
                                p0[i] = 0
                                p1[i] = 0
                            else:
                                done[i] = True
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
        ready = True
        anyfull = False
        for i in range(nl):
            if slotfull[i]:
                anyfull = True
            elif not (done[i] and not held[i]):
                ready = False
                break
        if ready and anyfull:
            for i in range(nl):
                if slotfull[i]:
                    out[i].append(slotval[i])
                    slotfull[i] = False
            emitted += 1
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False
    return out


# ---------------------------------------------------------------------------
# Stream builders
# ---------------------------------------------------------------------------

def _lane_symbols(rng, width, nsym, mode):
    top = (1 << width) - 1
    if mode == "uniform":
        return [rng.randrange(top + 1) for _ in range(nsym)]
    if mode == "sparse":
        return [0 if rng.random() < 0.85 else rng.randrange(top + 1) for _ in range(nsym)]
    return [min(top, int(rng.expovariate(0.4))) for _ in range(nsym)]


def _lane_fragments(rng, symbols, default_width):
    """Encode `symbols` as one to four fragments, each with its own width offset and q."""
    cuts = sorted(rng.sample(range(1, len(symbols)), rng.choice([0, 0, 1, 2, 3])))
    blob = b""
    prev = 0
    for cut in cuts + [len(symbols)]:
        part = symbols[prev:cut]
        prev = cut
        need = max(max(part).bit_length(), 1)
        options = [
            (default_width - z, q)
            for z in range(4)
            for q in range(16)
            if default_width - z >= need and (default_width - z, q) in tables.PROB_TABLES
        ]
        width, q = rng.choice(options)
        blob += stream_builder._encode_fragment(part, default_width, width, q)
    return blob


def build_stream(seed, codec, nsym):
    rng = random.Random(seed)
    mode = rng.choice(("uniform", "sparse", "geometric"))
    widths = bcpack.LANE_WIDTHS[codec]
    lanes = [_lane_fragments(rng, _lane_symbols(rng, w, nsym, mode), w) for w in widths]
    return stream_builder._interleave_lanes(lanes, widths, nsym)


class DecodeLanesPrefixParityTest(unittest.TestCase):
    def assert_same(self, raw, codec, nsym, max_sym):
        self.assertEqual(
            calib.decode_lanes_prefix(raw, codec, nsym * 8, max_sym),
            reference_decode_lanes_prefix(raw, codec, nsym * 8, max_sym),
            f"codec {codec} nsym {nsym} max_sym {max_sym}",
        )

    def test_every_codec_matches_the_reference_for_many_prefix_lengths(self):
        for codec in sorted(bcpack.LANE_WIDTHS):
            for seed, nsym in ((codec, 60), (100 + codec, 300)):
                raw = build_stream(seed, codec, nsym)
                for max_sym in sorted({1, 2, 3, 7, 64, nsym // 2, nsym - 2, nsym - 1}):
                    with self.subTest(codec=codec, seed=seed, max_sym=max_sym):
                        self.assert_same(raw, codec, nsym, max_sym)

    def test_long_fragments_with_the_three_byte_header_match_the_reference(self):
        # More than 256 payload bytes in a fragment selects the 3-byte header (`h3`).
        for codec in (6, 7):
            raw = build_stream(7000 + codec, codec, 1100)
            for max_sym in (5, 512, 1099):
                with self.subTest(codec=codec, max_sym=max_sym):
                    self.assert_same(raw, codec, 1100, max_sym)

    def test_full_length_requests_fall_back_to_the_complete_decoder(self):
        raw = build_stream(3, 7, 40)
        for max_sym in (0, -1, 40, 41):
            with self.subTest(max_sym=max_sym):
                self.assertEqual(
                    calib.decode_lanes_prefix(raw, 7, 40 * 8, max_sym),
                    bcpack.decode_lanes(raw, 7, 40 * 8)[0],
                )

    def test_prefix_is_a_prefix_of_the_full_decode(self):
        raw = build_stream(11, 6, 80)
        full = bcpack.decode_lanes(raw, 6, 80 * 8)[0]
        for max_sym in (1, 9, 79):
            prefix = calib.decode_lanes_prefix(raw, 6, 80 * 8, max_sym)
            self.assertEqual(prefix, [lane[:max_sym] for lane in full])

    def test_truncated_streams_fail_like_the_reference(self):
        rng = random.Random(5)
        raised = 0
        for codec in (1, 4, 6, 7):
            nsym = 40
            raw = build_stream(500 + codec, codec, nsym)
            for _ in range(8):
                cut = raw[:rng.randrange(1, len(raw))]
                max_sym = rng.randrange(1, nsym)
                outcomes = []
                for fn in (calib.decode_lanes_prefix, reference_decode_lanes_prefix):
                    try:
                        outcomes.append(("ok", fn(cut, codec, nsym * 8, max_sym)))
                    except bcpack.BcpackError as exc:
                        outcomes.append(("error", str(exc)))
                self.assertEqual(outcomes[0], outcomes[1])
                raised += outcomes[0][0] == "error"
        self.assertGreater(raised, 0)


class ModelIdentitiesTest(unittest.TestCase):
    """The two model identities the inlined scheduler relies on."""

    def test_bisect_finds_the_band_of_model_lookup(self):
        for (w, q) in sorted(tables.PROB_TABLES):
            model = bcpack._get_model(w, q)
            self.assertEqual(len(model.base), w + 1)
            probes = set(range(0, 65536, 251)) | {65535}
            for b in model.base:
                probes.update(v for v in (b - 1, b, b + 1) if 0 <= v < 65536)
            for cf in sorted(probes):
                p = bisect_right(model.base, cf) - 1
                sym = ((1 << p) >> 1) + (cf - model.base[p]) // model.freq[p]
                self.assertEqual(sym, model.lookup(cf), (w, q, cf))

    def test_advance_equals_the_state_folded_at_lookup_time(self):
        rng = random.Random(9)
        for (w, q) in sorted(tables.PROB_TABLES):
            model = bcpack._get_model(w, q)
            for _ in range(24):
                cf = rng.randrange(65536)
                hi = rng.randrange(1 << 8, 1 << 16)
                x = (hi << 16) | cf
                p = bisect_right(model.base, cf) - 1
                bp = model.base[p]
                fp = model.freq[p]
                qq = (cf - bp) // fp
                sym = ((1 << p) >> 1) + qq
                c = bp + fp * qq
                if c == 0:
                    fp -= 1
                    c = 1
                self.assertEqual(fp * (x >> 16) + (cf - c), model.advance(sym, x), (w, q, cf))


def reference_fit_signed_delta(obs_list, total_bits):
    if not obs_list:
        return 1 << (total_bits - 1)
    limit = 1 << total_bits
    best_err = 1e30
    best_r = 0
    for r in range(limit):
        err = 0.0
        for dc, dz in obs_list:
            sh = total_bits - dc if dc < total_bits else 0
            mD = (1 << dc) - 1
            v = ((r >> sh) + dz) & mD
            err += abs(calib._sign_extend(v, dc)) / float(1 << (dc - 1))
        if err < best_err:
            best_err = err
            best_r = r
    return best_r


def reference_fit_ep_pair(obs_list):
    ev = calib._eval_bc7_ops
    cands = []
    for h0 in range(0, 256, 16):
        for h1 in range(0, 256, 16):
            err = 0.0
            for ops0, ops1, tgt, wt in obs_list:
                e0 = ev(ops0, h0 | 8)
                e1 = ev(ops1, h1 | 8)
                d = abs(0.5 * (e0 + e1) - tgt)
                err += wt * ((d if d < 40.0 else 40.0) + 0.01 * abs(e0 - e1))
            cands.append((err, h0, h1))
    cands.sort()
    top = cands[:4]
    for step in (8, 4, 2, 1):
        nxt = {}
        mid = step // 2
        for _, b0, b1 in top:
            for d0 in (0, step):
                for d1 in (0, step):
                    h0, h1 = (b0 + d0) & 255, (b1 + d1) & 255
                    key = (h0, h1)
                    if key in nxt:
                        continue
                    err = 0.0
                    for ops0, ops1, tgt, wt in obs_list:
                        e0 = ev(ops0, h0 | mid)
                        e1 = ev(ops1, h1 | mid)
                        d = abs(0.5 * (e0 + e1) - tgt)
                        err += wt * ((d if d < 40.0 else 40.0) + 0.01 * abs(e0 - e1))
                    nxt[key] = err
        top = sorted((e, a, b) for (a, b), e in nxt.items())[:4]
    return top[0][1], top[0][2]


class TableFitParityTest(unittest.TestCase):
    def test_fit_signed_delta_matches_the_reference(self):
        rng = random.Random(21)
        for _ in range(120):
            total_bits = rng.choice((10, 6, 5, 12))
            obs = [
                (
                    rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, total_bits, total_bits + 1, total_bits + 2)),
                    rng.choice((0, rng.randrange(-70, 70), rng.randrange(-5000, 5000))),
                )
                for _ in range(rng.choice((0, 1, 2, 5, 30, 96)))
            ]
            with self.subTest(total_bits=total_bits, observations=len(obs)):
                self.assertEqual(
                    calib.Bc6hCalibrator._fit_signed_delta(obs, total_bits),
                    reference_fit_signed_delta(obs, total_bits),
                )

    def test_fit_signed_delta_rejects_a_zero_width_delta_like_the_reference(self):
        for fn in (calib.Bc6hCalibrator._fit_signed_delta, reference_fit_signed_delta):
            with self.assertRaises(ValueError):
                fn([(0, 1)], 6)

    def test_fit_ep_pair_matches_the_reference(self):
        rng = random.Random(22)

        def ops():
            return tuple((rng.choice((4, 5, 6, 7, 8)), rng.randrange(-300, 300)) for _ in range(rng.choice((1, 1, 2, 3))))

        for _ in range(40):
            obs = [
                (ops(), ops(), rng.uniform(0, 255), rng.choice((4.0, 2.0, 0.75)))
                for _ in range(rng.choice((0, 1, 2, 5, 20, 60)))
            ]
            with self.subTest(observations=len(obs)):
                self.assertEqual(calib.Bc7Calibrator._fit_ep_pair(obs), reference_fit_ep_pair(obs))


class PrettyJsonTest(unittest.TestCase):
    @staticmethod
    def random_value(rng, depth=0):
        r = rng.random()
        if depth > 4 or r < 0.35:
            return rng.choice((
                rng.randrange(-10**6, 10**6), rng.randrange(-5, 5), 2**70, True, False, None,
                rng.uniform(-1e5, 1e5), 1e300, float("inf"), "x", "é☃\"\\\n", "",
            ))
        if r < 0.65:
            return [PrettyJsonTest.random_value(rng, depth + 1) for _ in range(rng.choice((0, 1, 2, 5, 20)))]
        if r < 0.7:
            return tuple(PrettyJsonTest.random_value(rng, depth + 1) for _ in range(rng.choice((0, 1, 3))))
        if r < 0.8:
            return [rng.randrange(-300, 300) for _ in range(rng.choice((1, 3, 16, 32)))]
        keys = ("b", "a", "é", "Z", "", "☃", "k1", "k10", "k2")
        return {
            rng.choice(keys) + str(rng.randrange(3)): PrettyJsonTest.random_value(rng, depth + 1)
            for _ in range(rng.choice((0, 1, 3, 8)))
        }

    def test_matches_json_dumps(self):
        rng = random.Random(23)
        for _ in range(400):
            value = self.random_value(rng)
            self.assertEqual(
                calib._pretty_json(value), json.dumps(value, sort_keys=True, indent=2), repr(value)[:80]
            )

    def test_calibrator_documents_match_json_dumps(self):
        for calibrator in (calib.Bc6hCalibrator(), calib.Bc7Calibrator()):
            doc = calibrator.build_document()[0]
            self.assertEqual(calib._pretty_json(doc), json.dumps(doc, sort_keys=True, indent=2))

    def test_written_file_matches_json_dumps_including_the_fallback(self):
        docs = (
            {"endpointHistory": [[1, 2], [3, 4]], "name": "x", "flag": True},
            {1: [2, 3], 3: {"a": None}},  # int keys: the writer falls back to `json.dumps`
            [float("nan"), (1, 2), {}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            for i, doc in enumerate(docs):
                path = os.path.join(tmp, f"doc{i}.json")
                calib._write_json_deterministic(path, doc)
                with open(path, "r", encoding="utf-8", newline="") as fh:
                    self.assertEqual(fh.read(), json.dumps(doc, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()

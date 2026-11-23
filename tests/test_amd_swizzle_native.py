"""tools/amd_swizzle_native gives the same arrays as amd_swizzle's numpy code, and every fallback keeps numpy.

    python -I tests/test_amd_swizzle_native.py
    python -m pytest tests/test_amd_swizzle_native.py

Build the library first (`cargo build --release` in tools/amd_swizzle_native); without it the comparisons skip.
Synthetic, seeded random equations and data only: no addrlib header, no game data.
"""
import contextlib
import os
import random
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT, "tools")
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for d in (TOOLS_DIR, TESTS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    import numpy as np
    import amd_swizzle as A
    from test_amd_swizzle_detile import make_equation, reference_detile, reference_offsets
except ModuleNotFoundError as e:
    if e.name != "numpy":
        raise
    np = A = None

LIB = A._native() if A is not None else None


@contextlib.contextmanager
def numpy_only():
    """Run amd_swizzle with the native library switched off."""
    saved = list(A._NATIVE)
    A._NATIVE[:] = [None]
    try:
        yield
    finally:
        A._NATIVE[:] = saved


@contextlib.contextmanager
def native_lib(lib):
    saved = list(A._NATIVE)
    A._NATIVE[:] = [lib]
    try:
        yield
    finally:
        A._NATIVE[:] = saved


def outcome(fn, *args):
    """('ok', value) or ('raise', exception type) of one call."""
    try:
        return "ok", fn(*args)
    except Exception as e:  # noqa: BLE001 - the type is what is compared
        return "raise", type(e)


def random_eq(rng, bits):
    return [(rng.getrandbits(16) if rng.random() < 0.8 else 0, rng.getrandbits(16) if rng.random() < 0.8 else 0)
            for _ in range(bits)]


@unittest.skipIf(np is None, "numpy is not installed")
@unittest.skipIf(LIB is None, "tools/amd_swizzle_native is not built (or LEONIDA_AMD_SWIZZLE_NATIVE=0)")
class NativeMatchesNumpy(unittest.TestCase):
    def compare(self, data, eq, bpe, wb, hb, block_bytes, label):
        with numpy_only():
            want_off = outcome(A.offsets, eq, bpe, wb, hb, block_bytes)
            want = outcome(A.detile, data, eq, bpe, wb, hb, block_bytes)
        got_off = outcome(A.offsets, eq, bpe, wb, hb, block_bytes)
        got = outcome(A.detile, data, eq, bpe, wb, hb, block_bytes)
        for w, g, what in ((want_off, got_off, "offsets"), (want, got, "detile")):
            self.assertEqual(w[0], g[0], (what, label))
            if w[0] == "raise":
                self.assertIs(w[1], g[1], (what, label))
        if want_off[0] == "ok":
            (wo, wm), (go, gm) = want_off[1], got_off[1]
            self.assertEqual(gm, wm, label)
            self.assertEqual((go.shape, go.dtype), (wo.shape, wo.dtype), label)
            self.assertTrue(np.array_equal(go, wo), label)
        if want[0] == "ok":
            if want[1] is None:
                self.assertIsNone(got[1], label)
            else:
                self.assertIsNotNone(got[1], label)
                self.assertEqual((got[1].shape, got[1].dtype), (want[1].shape, want[1].dtype), label)
                self.assertEqual(got[1].tobytes(), want[1].tobytes(), label)
        return want

    def test_native_path_is_taken(self):
        eq = make_equation("R_X", 8, 16)
        args = A._native_args(eq, 8, 64, 64, 65536)
        self.assertIsNotNone(args)
        self.assertIsNot(A._native_offsets(LIB, eq, 8, 64, 64, 65536), A._REFUSED)
        data = bytes(65536 * 2)
        self.assertIsNot(A._native_detile(LIB, data, eq, 8, 64, 64, 65536), A._REFUSED)

    def test_1500_random_layouts(self):
        rng = random.Random(20261123)
        sizes = (0, 1, 2, 3, 5, 7, 16, 31, 64, 100, 129, 257)
        for case in range(1500):
            bpe = rng.choice((1, 2, 3, 4, 5, 6, 8, 12, 16))
            bits = rng.choice((8, 12, 16))
            block_bytes = 1 << bits
            if rng.random() < 0.5:
                eq = make_equation(rng.choice(("S", "D_X", "R_X")), bpe if bpe & (bpe - 1) == 0 else 4, bits)
            else:
                eq = random_eq(rng, bits)
            if rng.random() < 0.7:
                wb, hb = rng.choice(sizes), rng.choice(sizes)
            else:
                wb, hb = rng.randint(0, 300), rng.randint(0, 300)
            with numpy_only():
                off = A.offsets(eq, bpe, wb, hb, block_bytes)[0]
            need = (int(off.max()) // bpe + 1) * bpe if off.size else 0
            size = max(0, need + rng.choice((-bpe - 1, -1, 0, 0, 0, 1, bpe - 1, 77)))
            data = np.random.default_rng(case).integers(0, 256, size, dtype=np.uint8).tobytes()
            data = rng.choice((bytes, bytearray, memoryview))(data)
            self.compare(data, eq, bpe, wb, hb, block_bytes, (case, bpe, bits, wb, hb, size))

    def test_against_the_per_bit_reference(self):
        rng = random.Random(7)
        for case in range(120):
            bpe = rng.choice((1, 2, 4, 8, 16, 3, 12))
            bits = rng.choice((8, 12, 16))
            eq = random_eq(rng, bits)
            wb, hb = rng.randint(1, 150), rng.randint(1, 150)
            want_off, want_meta = reference_offsets(eq, bpe, wb, hb, 1 << bits)
            got_off, got_meta = A.offsets(eq, bpe, wb, hb, 1 << bits)
            self.assertEqual(got_meta, want_meta)
            self.assertTrue(np.array_equal(got_off, want_off), case)
            data = np.random.default_rng(case).integers(0, 256, int(want_off.max()) + bpe, dtype=np.uint8).tobytes()
            self.assertEqual(A.detile(data, eq, bpe, wb, hb, 1 << bits).tobytes(),
                             reference_detile(data, eq, bpe, wb, hb, 1 << bits).tobytes(), case)

    def test_large_grids_use_threads(self):
        for bpe, n, mode in ((4, 1024, "R_X"), (8, 700, "D_X"), (16, 513, "S"), (1, 1500, "R_X")):
            eq = make_equation(mode, bpe, 16)
            with numpy_only():
                off = A.offsets(eq, bpe, n, n - 3)[0]
            data = np.random.default_rng(bpe).integers(0, 256, int(off.max()) + bpe, dtype=np.uint8).tobytes()
            self.compare(data, eq, bpe, n, n - 3, 65536, (bpe, n))
            self.compare(data[: int(off.max()) // bpe * bpe + bpe - 1], eq, bpe, n, n - 3, 65536, ("short", bpe))

    def test_edge_cases(self):
        eq = make_equation("R_X", 4, 16)
        data = bytes(range(256)) * 1024
        cases = [
            (data, eq, 4, 0, 0, 65536), (data, eq, 4, 0, 9, 65536), (data, eq, 4, 9, 0, 65536),
            (b"", eq, 4, 1, 1, 65536), (b"", eq, 4, 0, 5, 65536), (data[:3], eq, 4, 1, 1, 65536),
            (data, [], 4, 33, 17, 65536), (data, tuple(eq), 4, 20, 20, 65536),
            (data, [list(m) for m in eq], 4, 20, 20, 65536),
            (data, [(mx, my, 99) for mx, my in eq], 4, 20, 20, 65536),
            (data, eq, 4, 20, 20, 4096), (data, eq, 16, 3, 3, 256), (data, eq, 7, 13, 11, 65536),
            # Unusual arguments the native path refuses: numpy keeps its own result or error.
            (data, eq, 4, -1, 5, 65536), (data, eq, 4, 5, -1, 65536), (data, eq, 0, 4, 4, 65536),
            (data, eq, 32, 4, 4, 16), (data, eq, 4, 4, 4, 0), (data, eq, 4, 4, 4, -256),
            (data, [(True, False)] * 8, 4, 9, 9, 256), (data, [(np.int64(1), 2)] * 8, 4, 9, 9, 256),
            (data, eq, np.int64(4), 9, 9, 65536), (data, eq, 4, np.int64(9), 9, 65536), (data, eq, 4.0, 9, 9, 65536),
            (data, [(1 << 40, 1 << 41)] * 16, 4, 9, 9, 65536), (data, [(1, 1)] * 70, 4, 9, 9, 65536),
            (data, [(-1, 0)] * 8, 4, 9, 9, 256), (data, [(1,)] * 8, 4, 9, 9, 256),
            (np.frombuffer(data, np.uint8), eq, 4, 30, 30, 65536),
            (memoryview(data).cast("I"), eq, 8, 30, 30, 65536), (memoryview(data)[::2], eq, 4, 8, 8, 65536),
            (data, [(1 << 62, 0)] * 63, 4, 129, 1, 65536),  # an offset past int64: native refuses
        ]
        for i, args in enumerate(cases):
            with self.subTest(case=i):
                self.compare(*args, label=i)
        # Offsets past int64 (numpy would wrap): refused before anything is allocated.
        self.assertIs(A._native_offsets(LIB, [(1 << 62, 0)] * 63, 4, 129, 1, 65536), A._REFUSED)
        self.assertIs(A._native_detile(LIB, data, [(1 << 62, 0)] * 63, 4, 129, 1, 65536), A._REFUSED)
        self.assertIsNone(A._native_args(iter(eq), 4, 9, 9, 65536))
        self.assertIsNone(A._native_args(eq, 32, 4, 4, 16))  # an element larger than the block

    def test_generator_equation_keeps_numpy_behaviour(self):
        eq = make_equation("D_X", 8, 16)
        data = bytes(range(256)) * 2048
        with numpy_only():
            want = A.detile(data, iter(eq), 8, 40, 40)
        got = A.detile(data, iter(eq), 8, 40, 40)
        self.assertEqual(got.tobytes(), want.tobytes())

    def test_benchmark(self):
        rows = []
        for bpe, n, mode in ((8, 512, "R_X"), (16, 512, "R_X"), (4, 2048, "R_X"), (1, 1024, "D_X"), (8, 64, "S")):
            eq = make_equation(mode, bpe, 16)
            with numpy_only():
                off = A.offsets(eq, bpe, n, n)[0]
            data = np.random.default_rng(3).integers(0, 256, int(off.max()) + bpe, dtype=np.uint8).tobytes()

            def best(fn):
                t_best = float("inf")
                for _ in range(5):
                    t = time.perf_counter()
                    fn()
                    t_best = min(t_best, time.perf_counter() - t)
                return t_best
            with numpy_only():
                t_np = best(lambda: A.detile(data, eq, bpe, n, n))
                o_np = best(lambda: A.offsets(eq, bpe, n, n))
            t_nat = best(lambda: A.detile(data, eq, bpe, n, n))
            o_nat = best(lambda: A.offsets(eq, bpe, n, n))
            rows.append(f"bpe {bpe:2} {n}x{n} {mode:3}: detile {t_np * 1e3:7.3f} -> {t_nat * 1e3:7.3f} ms "
                        f"({t_np / t_nat:5.1f}x), offsets {o_np * 1e3:7.3f} -> {o_nat * 1e3:7.3f} ms "
                        f"({o_np / o_nat:5.1f}x)")
        print("\n" + "\n".join(rows), file=sys.stderr)


class _FakeLib:
    """Stands in for the ctypes library: a wrong ABI, or kernels that give a different answer."""

    def __init__(self, abi=1, status=0, scribble=False):
        self.abi, self.status, self.scribble = abi, status, scribble

    def amd_swizzle_native_abi(self):
        return self.abi

    def amd_swizzle_offsets(self, *args):
        if self.scribble:
            import ctypes
            n = args[3] * args[4]
            if n:
                ctypes.memset(args[-1], 0x5A, n * 8)
        return self.status

    def amd_swizzle_detile(self, *args):
        if self.scribble:
            import ctypes
            n = args[6] * args[7] * args[2]
            if n:
                ctypes.memset(args[-1], 0x5A, n)
        return self.status


@unittest.skipIf(np is None, "numpy is not installed")
class Fallbacks(unittest.TestCase):
    def run_hex(self, env_value):
        code = ("import sys, hashlib; sys.path[:0] = [%r, %r]; import amd_swizzle as A; "
                "from test_amd_swizzle_detile import make_equation; "
                "eq = make_equation('R_X', 8, 16); d = bytes(range(256)) * 4096; "
                "print(A._native() is not None, hashlib.sha256(A.detile(d, eq, 8, 100, 77).tobytes() "
                "+ A.offsets(eq, 8, 100, 77)[0].tobytes()).hexdigest())") % (TOOLS_DIR, TESTS_DIR)
        env = dict(os.environ, LEONIDA_AMD_SWIZZLE_NATIVE=env_value)
        out = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, check=True)
        loaded, digest = out.stdout.split()
        return loaded == "True", digest

    def test_opt_out_env_var_gives_the_same_bytes(self):
        off_loaded, off_hex = self.run_hex("0")
        self.assertFalse(off_loaded)
        on_loaded, on_hex = self.run_hex("1")
        self.assertEqual(on_loaded, A._native_load() is not None)  # built or not, regardless of this process's env
        self.assertEqual(on_hex, off_hex)

    def test_missing_library_is_none(self):
        saved = list(A._NATIVE)
        real_isfile = os.path.isfile
        try:
            A._NATIVE[:] = []
            os.path.isfile = lambda p: False if "amd_swizzle_native" in str(p) else real_isfile(p)
            self.assertIsNone(A._native())
        finally:
            os.path.isfile = real_isfile
            A._NATIVE[:] = saved

    def test_wrong_abi_and_failed_self_check_turn_native_off(self):
        import ctypes
        saved_cdll, saved_isfile, saved = ctypes.CDLL, os.path.isfile, list(A._NATIVE)
        for fake in (_FakeLib(abi=2), _FakeLib(scribble=True), _FakeLib(status=-1), _FakeLib(status=1)):
            try:
                A._NATIVE[:] = []
                ctypes.CDLL = lambda path, fake=fake: fake
                os.path.isfile = lambda p: True if "amd_swizzle_native" in str(p) else saved_isfile(p)
                with self.subTest(abi=fake.abi, status=fake.status, scribble=fake.scribble):
                    self.assertIsNone(A._native())
            finally:
                ctypes.CDLL, os.path.isfile = saved_cdll, saved_isfile
                A._NATIVE[:] = saved

    def test_refusing_library_keeps_numpy_results(self):
        eq = make_equation("R_X", 4, 16)
        data = bytes(range(256)) * 1024
        with numpy_only():
            want = A.detile(data, eq, 4, 50, 50).tobytes(), A.offsets(eq, 4, 50, 50)[0].tobytes()
        with native_lib(_FakeLib(status=-2, scribble=True)):
            got = A.detile(data, eq, 4, 50, 50).tobytes(), A.offsets(eq, 4, 50, 50)[0].tobytes()
        self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()

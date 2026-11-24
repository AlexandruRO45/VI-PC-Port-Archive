"""Compare complete cache lines, adjacent bytes, RAX and the condition flags."""
import ctypes
from pathlib import Path
import subprocess
import sys

native_path, lifted_path, translator, out = map(Path, sys.argv[1:])
native = ctypes.CDLL(str(native_path))
lifted = ctypes.CDLL(str(lifted_path))

# An ordinary Windows reference remains executable on Intel hosts.
def run(fn, start, end):
    buf = ctypes.create_string_buffer(512)
    ctypes.memset(buf, 0xa5, len(buf))
    origin = (ctypes.addressof(buf) + 63) & ~63
    state = (ctypes.c_ubyte * 3)()
    fn.argtypes = (ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p)
    fn.restype = ctypes.c_size_t
    result = fn(origin + start, origin + end, state)
    return bytes(buf)[origin - ctypes.addressof(buf):][:384], result - origin, tuple(state)


for start, end in [(64, 64), (92, 91), (64, 65), (127, 128), (65, 194), (128, 320)]:
    observed = run(lifted.lifted_counted, start, end)
    assert observed == run(native.reference_counted, start, end)
    for symbol in ('loop_dx', 'loop_r8', 'loop_r9', 'loop_r10'):
        plain = run(getattr(lifted, 'lifted_' + symbol), start, end)
        assert plain[:2] == observed[:2] and plain[2] == (0, 0, 0), (symbol, start, end)
    lines = max(1, (end - start + 63) // 64)
    first = start & ~63
    final = start + 64 * lines
    expected = bytearray([0xa5] * 384)
    expected[first:first + lines * 64] = bytes(lines * 64)
    assert observed == (expected, final, (int(start < end), int(final == end), int(final > end))), (start, end)
for name, diagnostic in [('prefix_addr32', 'unsupported CLZERO'), ('unknown_rax', 'rax')]:
    res = subprocess.run([sys.executable, '-B', str(translator), str(native_path), name, str(out / (name + '.cpp'))],
                         capture_output=True, text=True)
    assert res.returncode and diagnostic.lower() in (res.stdout + res.stderr).lower(), (name, res.stdout, res.stderr)
print('six instrumented and 24 four-instruction loop comparisons; two input rejections passed')

"""Shared helpers for the translator tests (standard library only)."""
import atexit
import gc
import os
import shutil
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# The mutation tests point this at a sabotaged copy of tools/; normally it is the repository's own.
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "disasm")

sys.dont_write_bytecode = True      # never drop __pycache__ into the tree under test
sys.path.insert(0, TOOLS_DIR)
import rdna2spirv  # noqa: E402  (path set above)
from spirv_mini_interp import run_fragment  # noqa: E402

# translate() reads <dir>/<name>.s and, for pixel shaders, an input signature from <dir>/../<name>.dxbc.
# A DXBC container with zero parts (magic, 24 zero bytes, part count 0, 4 more bytes) declares no inputs.
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def translate_fixture(name, dxbc=STUB_DXBC, fssm=None):
    """SPIR-V assembly text for tests/semantic/fixtures/disasm/<name>.s, translated by the translator under test.

    `dxbc` overrides the empty input signature; pixel shaders that read attributes need a real ISG1 part.
    `fssm` supplies the compute-shader header blob (RSRC2 and group size) for cs_* fixtures.
    """
    src = os.path.join(FIXTURE_DIR, name + ".s")
    if not os.path.isfile(src):
        raise AssertionError(f"missing fixture {src}")        # a missing fixture is a failure, never a skip
    with open(src, "rb") as f:
        return translate_listing(name, f.read(), dxbc, fssm)


def translate_lines(name, lines, dxbc=STUB_DXBC, fssm=None):
    """Like translate_fixture for a hand-written program given as instruction lines (no fixture file)."""
    text = f"; {name}  hand-written test program\n" + "".join("\t" + ln.strip() + "\n" for ln in lines)
    return translate_listing(name, text.encode("utf-8"), dxbc, fssm)


_scratch = {"dir": None, "files": {}}


def _scratch_dir():
    """One temporary directory for the whole test process, removed at exit (a fresh one per translation cost a
    mkdir, two file writes and an rmtree each time: ~4 ms on Windows with a virus scanner, 3 000 times a file)."""
    if _scratch["dir"] is None:
        _scratch["dir"] = tempfile.mkdtemp()
        atexit.register(_remove_scratch)
    return _scratch["dir"]


def _remove_scratch():
    path, _scratch["dir"] = _scratch["dir"], None
    try:
        shutil.rmtree(path)
    except OSError:
        gc.collect()        # a leaked file object still holds the directory (Windows); close it and retry
        shutil.rmtree(path, ignore_errors=True)


def _put(path, data):
    """Make `path` hold `data` (None: absent), writing only when it differs from what this process left there."""
    if _scratch["files"].get(path) == data:
        return
    if data is None:
        os.remove(path)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    _scratch["files"][path] = data


def translate_listing(name, data, dxbc=STUB_DXBC, fssm=None):
    """SPIR-V assembly text for a listing in the translator's input format (bytes of a .s file)."""
    tmp = _scratch_dir()
    listing = os.path.join(tmp, "disasm", name + ".s")
    _put(listing, data)
    _put(os.path.join(tmp, name + ".dxbc"), dxbc)
    if fssm is not None or os.path.join(tmp, "fssm", name + ".fssm") in _scratch["files"]:
        _put(os.path.join(tmp, "fssm", name + ".fssm"), fssm)
    with warnings.catch_warnings():
        # translate() leaves its .dxbc handle to the garbage collector; harmless, and not ours to change here.
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(listing)


def mrt0_words(spvasm, **run_options):
    """Run the translated pixel shader once; return the four 32-bit words it writes to mrt0."""
    out = run_fragment(spvasm, **run_options)
    if list(out) != ["%out_mrt0"]:
        raise AssertionError(f"expected exactly one output, %out_mrt0; got {sorted(out)}")
    return list(out["%out_mrt0"])


def varyings(spvasm, attrs, bary):
    """Input values for a translated PS, given {attribute n: three per-vertex vec4s of the producer's raw 32-bit
    words} and the barycentric weights `bary` of vertices 0..2, as Vulkan hands them over: a PerVertexKHR input gets
    the three vertices, a Flat one vertex 0, any other the words read as f32 and interpolated; %bary gets `bary`."""
    f = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]    # noqa: E731
    out = {"%bary": tuple(bary)}
    for n, verts in attrs.items():
        name, fv = f"%in_attr{n}", [tuple(f(w) for w in v) for v in verts]
        if f"OpDecorate {name} PerVertexKHR" in spvasm:
            out[name] = tuple(fv)
        elif f"OpDecorate {name} Flat" in spvasm:
            out[name] = fv[0]
        else:
            out[name] = tuple(sum(b * v[c] for b, v in zip(bary, fv)) for c in range(4))
    return out

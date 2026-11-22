"""Shared helpers for the translator tests (standard library only)."""
import gc
import os
import re
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

# The VOP3 clamp bit on an f32 result (Gen.clamp01, SHADER-186): v > 0 selects v or +0, then lo < 1 selects lo or 1.0.
# Group 1 is the indent, 3 the value being clamped, 6 the clamped result.
CLAMP01_RE = re.compile(r"(?m)^( *)(%\w+) = OpFOrdGreaterThan %bool (%\w+) %f0\n *(%\w+) = OpSelect %float \2 \3 %f0\n"
                        r" *(%\w+) = OpFOrdLessThan %bool \4 %f1\n *(%\w+) = OpSelect %float \5 \4 %f1$")


def clamp01_count(spvasm):
    """How many f32 clamp-bit sequences the module has."""
    return len(CLAMP01_RE.findall(spvasm))


def drop_clamp01(spvasm):
    """A negative control: each f32 clamp-bit sequence replaced by its unclamped input. Returns (text, count)."""
    return CLAMP01_RE.subn(lambda m: f"{m.group(1)}{m.group(6)} = OpCopyObject %float {m.group(3)}", spvasm)


# The same sequence on a double (clamp_mods_f64): the 0.0 and 1.0 constants are bitcast ids, not %f0 / %f1.
CLAMP01_F64_RE = re.compile(r"(?m)^ *(%\w+) = OpFOrdGreaterThan %bool (%\w+) (%\w+)\n *(%\w+) = OpSelect %double \1 \2 \3\n"
                            r" *(%\w+) = OpFOrdLessThan %bool \4 (%\w+)\n *(%\w+) = OpSelect %double \5 \4 \6$")


def clamp01_f64_count(spvasm):
    """How many f64 clamp-bit sequences the module has."""
    return len(CLAMP01_F64_RE.findall(spvasm))


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


def translate_listing(name, data, dxbc=STUB_DXBC, fssm=None):
    """SPIR-V assembly text for a listing in the translator's input format (bytes of a .s file)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", name + ".s"), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(dxbc)
        if fssm is not None:
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(fssm)
        with warnings.catch_warnings():
            # translate() leaves its .dxbc handle to the garbage collector; harmless, and not ours to change here.
            warnings.simplefilter("ignore", ResourceWarning)
            result = rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
            gc.collect()        # release the handle before the temporary directory is removed (Windows)
        return result


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

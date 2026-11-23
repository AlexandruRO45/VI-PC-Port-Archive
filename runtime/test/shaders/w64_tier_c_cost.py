"""SHADER-836, synthetic: how much of a tier C pixel fixture's output depends on how the wave is partitioned.

    python3 -I runtime/test/shaders/w64_tier_c_cost.py <build dir with spirv/> [tools dir] [--ref DIR] [--save DIR]

Tier C means the shader's output changes when a wave64 is cut into two 32-lane subgroups (docs/design.md, "Wave64
graphics: partition invariance"). The ps_w64_c_* fixtures are small hand-written examples of each tier C reason; this
tool counts, per fixture, how many of the 64 x 64 pixels get a different value in a word the fixture's header names as
partition dependent (the None entries of FIXTURES in w64_graphics.py), against a reference run:

  default      the reference is this device's own run with the fragment subgroup size required to 64 (a device that can
               require it: RADV, AMD on Windows), and the run is the one required to 32
  --ref DIR    the reference is read from DIR: `<name>.ref64.raw` as W64G_SAVE_REF=DIR writes it on a 64-lane device, or
               `<name>.raw` as --save writes it on any device; the run is this device's default-size draw

--save DIR writes each default-size draw of this device to DIR/<name>.raw, so a second device can be compared with it.
Two 32-wide devices give no 64-lane reference. Comparing them with each other, or an Intel iGPU that runs 16 live lanes
with an NVIDIA GPU that runs 32, measures how much the words move with the partition width, not the cost against a real
wave64: the header says which reference a row used. Only counts are printed (no shader code, no game data).
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w64_graphics as g  # noqa: E402


def tier_c_pixel_fixtures():
    return sorted(n for n, (tier, _, _) in g.FIXTURES.items() if tier == "C" and n.startswith("ps_"))


def dependent_words(name):
    """The word positions (0..3) the fixture's header names as partition dependent, in any pixel."""
    return sorted({k for i in range(g.W * g.H) for k, v in enumerate(g.FIXTURES[name][1](i)) if v is None})


def moved(name, ref, run):
    """(pixels whose partition-dependent words differ, pixels whose other words differ) between two draws."""
    dep = dependent_words(name)
    changed = other = 0
    for i, (a, b) in enumerate(zip(ref, run)):
        want = g.FIXTURES[name][1](i)
        changed += any(a[k] != b[k] for k in dep if want[k] is None)
        other += any(a[k] != b[k] for k in range(4) if want[k] is not None)
    return changed, other


def read_ref(directory, name):
    for suffix in (".ref64.raw", ".raw"):
        path = Path(directory) / (name + suffix)
        if path.is_file() and path.stat().st_size == g.W * g.H * 16:
            return list(struct.iter_unpack("<4I", path.read_bytes())), suffix == ".ref64.raw"
    return None, False


def measure(tools, spirv, name, ref_dir=None, save_dir=None):
    """One result row: (reference, pixels, moved pixels, pixels with another word moved), or a message string."""
    with tempfile.TemporaryDirectory(prefix="leonida-w64cost-") as tmp:
        if save_dir:
            default, _, _ = g.draw(tools, spirv, name, None, tmp)
            if default is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                (Path(save_dir) / (name + ".raw")).write_bytes(b"".join(struct.pack("<4I", *w) for w in default))
        if ref_dir:
            ref, is64 = read_ref(ref_dir, name)
            if ref is None:
                return f"no {name}.ref64.raw or {name}.raw in {ref_dir}"
            words, heap, _ = g.draw(tools, spirv, name, None, tmp)
            label = "64-lane run from --ref" if is64 else "run from --ref (partition unknown)"
        else:
            ref, heap, applied64 = g.draw(tools, spirv, name, 64, tmp)
            if ref is None:
                return heap
            if not applied64:
                return "skipped: this device cannot require 64 lanes; use --ref with a W64G_SAVE_REF run"
            words, heap, applied32 = g.draw(tools, spirv, name, 32, tmp)
            if words is not None and not applied32:
                return "skipped: this device cannot require 32 lanes"
            label = "required 64 vs required 32"
        if words is None:
            return heap
        changed, other = moved(name, ref, words)
        return label, len(words), changed, other


def main(argv):
    args, ref_dir, save_dir = [], None, None
    it = iter(argv)
    for a in it:
        if a == "--ref":
            ref_dir = next(it, None)
        elif a == "--save":
            save_dir = next(it, None)
        else:
            args.append(a)
    if not args or (("--ref" in argv) and not ref_dir) or (("--save" in argv) and not save_dir):
        sys.exit(__doc__)
    spirv = os.path.join(args[0], "spirv")
    tools = os.path.abspath(args[1]) if len(args) > 1 else os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build"))
    with tempfile.TemporaryDirectory(prefix="leonida-w64cost-") as tmp:
        live = g.live_width(tools, spirv, tmp)
    print(f"live lanes per fragment subgroup: {live if live is not None else 'not measured'}")
    print(f"{'fixture':28} {'dependent words':16} {'pixels moved':>14}  reference")
    failed = 0
    for name in tier_c_pixel_fixtures():
        if not os.path.isfile(os.path.join(spirv, name + ".spv")):
            print(f"{name:28} missing {name}.spv")
            failed += 1
            continue
        row = measure(tools, spirv, name, ref_dir, save_dir)
        if isinstance(row, str):
            print(f"{name:28} {row}")
            failed += not row.startswith("skipped")
            continue
        label, pixels, changed, other = row
        extra = f"; {other} pixels moved in a word that must not" if other else ""
        print(f"{name:28} {str(dependent_words(name)):16} {changed:5} of {pixels} ({100 * changed / pixels:5.1f}%)  "
              f"{label}{extra}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env python3
"""One real drawable through its own translated game mesh shader, checked against the CPU (TEST-362, #1889).

A static drawable from the user's own package (exported by `rpf9dump models`) is read raw: its `42 40` vertex
block and its meshlet group go into the vk_harness heap unchanged, next to a hand-built descriptor table, an object
matrix and a view-projection. The game's mesh shader then draws it on the GPU (VKH_MESH) and the covered pixels are
compared with the same triangles projected and rasterised on the CPU (intersection over union, plus every
mismatching pixel that is not on a triangle edge). See docs/ms_pass.md.

    python -I tools/ms_pass.py --list --game <package> --models <export>/data/models/manifest.json
    python -I tools/ms_pass.py --game <package> --work <work> --models <manifest> --drawable <file> --run
    python -I tools/ms_pass.py --check --out <work>/ms_pass/out          # compare an earlier draw again

Only counts, the IoU and hashes are printed; the drawable's bytes and the shader stay in the work folder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import zlib

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import first_light as fl  # noqa: E402  (run(), find_harness())
import meshlet_page as mp  # noqa: E402
import mesh_pass_coverage as mpc  # noqa: E402  (#4809: snapped CPU coverage)

CLEAR = -1.0                       # vk_harness clears the target to -1.0 in every channel
COLOR = (1.0, 0.5, 0.25, 1.0)      # what the fixture fragment shader writes on every covered pixel
MIN_IOU = 0.99
SUBPIXEL_BITS = 8                  # VkPhysicalDeviceLimits::subPixelPrecisionBits on the GPUs measured so far
HEAP_BYTES = 0x10000               # vk_harness's heap (VKH_HEAP), runtime/test/vk_harness.cpp
HEAP_ENV_LIMIT = 32000             # stay under the 32,767-character environment limit of Windows
INFLATE_CAP = 64 << 20

# Mesh shaders whose descriptor tables this tool can build, keyed by the SHA-256 of their FSSM program. The table
# offsets, the draw-word bit and the view-projection offset were read from each shader's own listing (the scalar
# loads off the table pointer and the s_buffer_load offsets); the matrix layout was measured (docs/ms_pass.md).
SHADERS = {
    "17464335e29c1f4f8ea80c0d65eb3a455cf71803fd88deb751fb3ccabb9bf959": {
        "what": "position-only meshlet mesh shader of the 'standard' effect",
        "table_sgpr": 17,              # 32-bit pointer to a table of 16-byte V#s
        "position_sgprs": (19, 20),    # low/high VA of the 42 40 vertex block; the shader builds its V# itself
        "draw_word_sgpr": 21,          # packed draw word
        "draw_word": 0,                # bit 28 clear: meshlet = workgroup index while below the group's meshlet
                                       # count (else 0); bit 28 set reads the meshlet from an indirection list
        "heap_sgpr": 23,               # resource heap base, unused on that path but read
        "slots": {"meshlet_group": 0x40, "object": 0x80, "view": 0xA0},
        "object_bytes": 0x40,          # 4 x float4 object (world) matrix at +0
        "view_bytes": 0x1A0,           # view constants; the view-projection is the 4 x float4 at +0x160
        "view_proj": 0x160,
        "subgroup_size": 64,
    },
}


class CameraError(ValueError):
    """The camera would clip a vertex: the CPU side has no clipper, so such a pose is refused."""


# ------------------------------------------------------------------------------------------------ the package side

def inflate_chain(fh, start: int, max_blocks: int = 64, cap: int = INFLATE_CAP) -> list[bytes]:
    """The zlib blocks of one resource from `start`, each starting on a 16-byte boundary after zero padding, as
    rpf9dump's model export reads them. Output is capped at `cap` bytes in all."""
    blocks, pos, total = [], start, 0
    while len(blocks) < max_blocks:
        fh.seek(pos)
        head = fh.read(2)
        if len(head) < 2 or head[0] & 0x0F != 8 or ((head[0] << 8) | head[1]) % 31:
            break
        fh.seek(pos)
        d, out, consumed = zlib.decompressobj(), bytearray(), 0
        try:
            while not d.eof:
                chunk = d.unconsumed_tail
                if not chunk:
                    chunk = fh.read(1 << 16)
                    consumed += len(chunk)
                    if not chunk:
                        break
                room = cap - total - len(out)
                if room <= 0:
                    raise ValueError(f"resource at {start:#x} inflates past {cap} bytes")
                out += d.decompress(chunk, room)
        except zlib.error:
            break
        if not d.eof:
            break
        end = pos + consumed - len(d.unused_data)
        blocks.append(bytes(out))
        total += len(out)
        nxt = (end + 15) & ~15
        fh.seek(end)
        if any(fh.read(nxt - end)):
            break
        pos = nxt
    return blocks


def load_drawable(rpf: str, item: dict) -> dict:
    """The geometry of a single-geometry static drawable: its raw vertex block and meshlet group, parsed by
    meshlet_page. A drawable with more than one geometry is refused: this tool draws one geometry, and a pass on the
    first alone would cover part of the drawable without saying so."""
    with open(rpf, "rb") as fh:
        blocks = inflate_chain(fh, int(item["Offset"]))
    for split in range(1, len(blocks) + 1):
        system, graphics = b"".join(blocks[:split]), b"".join(blocks[split:])
        try:
            geometries = list(mp.drawable_geometries(system, int(item.get("Drawable") or 0)))
            if len(geometries) > 1:
                raise SystemExit(f"{item.get('File')}: {len(geometries)} geometries; this tool draws "
                                 "single-geometry drawables only")
            bind = mp.draw_binding(system, graphics, geometries[0])
            block, vend = mp.parse_other_block(graphics, bind["vertex_block"])
            group, gend = mp.parse_group(graphics, bind["meshlet_group"])
        except (mp.PageError, struct.error, IndexError, ValueError):
            continue
        vblob = graphics[bind["vertex_block"]:vend]
        gblob = graphics[bind["meshlet_group"]:gend]
        positions = mp.positions_4240(graphics, block)
        triangles = list(mp.group_triangles(graphics, group))
        return {"geometries": len(geometries), "vertex_block": vblob, "meshlet_group": gblob,
                "positions": positions, "triangles": triangles, "meshlets": group["meshlets"],
                "axis": block["axis"], "vertices": block["vertices"]}
    raise SystemExit(f"{item.get('File')}: no static meshlet geometry this tool can read")


def candidates(manifest: dict) -> list[dict]:
    """Static single-geometry drawables (no bones, not a fragment, one entry in MeshGeoms), smallest first."""
    out = [i for i in manifest.get("Items", []) if not i.get("HasBoneBindings") and not i.get("IsFragment")
           and i.get("Verts") and i.get("Tris") and len(i.get("MeshGeoms") or [None]) == 1]
    return sorted(out, key=lambda i: (i["Tris"], i["Verts"]))


# ------------------------------------------------------------------------------------------------ the GPU side

def vsharp(va: int, size: int) -> list[int]:
    """A raw buffer V#: base VA, no stride, NUM_RECORDS = size in bytes, OOB_SELECT raw (as vk_sandbox.cpp)."""
    if not 0 <= va < 1 << 32 or not 0 < size < 1 << 32:
        raise ValueError("V# base and size must fit in 32 bits")
    return [va, 0, size, 3 << 28]


def build_heap(spec: dict, vblob: bytes, gblob: bytes, view_proj: list[float], size: int = HEAP_BYTES):
    """The heap image (Xbox VA 0 = offset 0) and where each region went. Every region is 256-byte aligned and
    stays below 4 GiB, which the 32-bit table pointers need."""
    table, layout, cursor = 0x100, {}, 0x200

    def place(name, nbytes):
        nonlocal cursor
        layout[name] = cursor
        cursor = (cursor + nbytes + 0xFF) & ~0xFF
        if cursor > size:
            raise ValueError(f"heap regions need {cursor:#x} bytes, more than the {size:#x}-byte heap")

    place("meshlet_group", len(gblob))
    place("vertex_block", len(vblob))
    place("object", spec["object_bytes"])
    place("view", spec["view_bytes"])
    place("heap_base", 0x20)
    heap = bytearray(size)
    heap[layout["meshlet_group"]:layout["meshlet_group"] + len(gblob)] = gblob
    heap[layout["vertex_block"]:layout["vertex_block"] + len(vblob)] = vblob
    struct.pack_into("<16f", heap, layout["object"], *[1.0 if r == c else 0.0 for r in range(4) for c in range(4)])
    struct.pack_into("<16f", heap, layout["view"] + spec["view_proj"], *view_proj)
    sizes = {"meshlet_group": len(gblob), "object": spec["object_bytes"], "view": spec["view_bytes"]}
    for name, slot in spec["slots"].items():
        struct.pack_into("<4I", heap, table + slot, *vsharp(layout[name], sizes[name]))
    layout["table"] = table
    return heap, layout


def heap_env(heap: bytes) -> str:
    """VKH_HEAP for vk_harness: the non-zero words as addr=word pairs."""
    words = struct.unpack(f"<{len(heap) // 4}I", heap)
    text = ",".join(f"{4 * i:#x}={w:#x}" for i, w in enumerate(words) if w)
    if len(text) > HEAP_ENV_LIMIT:
        raise ValueError(f"VKH_HEAP would be {len(text)} characters, over {HEAP_ENV_LIMIT}")
    return text


def parse_heap_env(text: str, size: int = HEAP_BYTES) -> bytes:
    heap = bytearray(size)
    for pair in filter(None, text.split(",")):
        addr, word = pair.split("=")
        struct.pack_into("<I", heap, int(addr, 0), int(word, 0))
    return bytes(heap)


def sgpr_words(spec: dict, layout: dict) -> list[int]:
    """s0..s31 for VKH_SGPRS (vk_harness's push constants)."""
    s = [0] * 32
    s[spec["table_sgpr"]] = layout["table"]
    lo, hi = spec["position_sgprs"]
    s[lo], s[hi] = layout["vertex_block"], 0
    s[spec["draw_word_sgpr"]] = spec["draw_word"]
    s[spec["heap_sgpr"]] = layout["heap_base"]
    return s


FS_ASM = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %out_v
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %out_v Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
  %ptr_out_v = OpTypePointer Output %v4float
      %out_v = OpVariable %ptr_out_v Output
         %c0 = OpConstant %float {0!r}
         %c1 = OpConstant %float {1!r}
         %c2 = OpConstant %float {2!r}
         %c3 = OpConstant %float {3!r}
        %col = OpConstantComposite %v4float %c0 %c1 %c2 %c3
       %main = OpFunction %void None %fn
      %entry = OpLabel
               OpStore %out_v %col
               OpReturn
               OpFunctionEnd
""".format(*COLOR)


# ------------------------------------------------------------------------------------------------ camera and CPU oracle

def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def camera(lo, hi, yaw_deg: float, pitch_deg: float, fov_deg: float = 50.0, aspect: float = 1.0):
    """A perspective camera looking at the bounds' centre from (yaw, pitch), framing the bounding sphere. Column-
    vector convention, clip = M * (x, y, z, 1), Vulkan clip space (NDC y = -1 is the top row, depth 0..1). `aspect`
    is the target's width / height: the x scale is divided by it, so a square stays square on a rectangular target
    (fov_deg is the vertical field of view)."""
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = max(math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3))) / 2, 1e-6)
    yaw, pitch, half = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(fov_deg) / 2
    dist = radius / math.sin(half) * 1.6
    eye = (centre[0] + dist * math.cos(pitch) * math.sin(yaw), centre[1] + dist * math.sin(pitch),
           centre[2] + dist * math.cos(pitch) * math.cos(yaw))
    f = [centre[i] - eye[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in f))
    f = [x / n for x in f]
    s = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]      # f x up, up = +y
    n = math.sqrt(sum(x * x for x in s))
    if n < 1e-9:
        raise CameraError("camera looks straight up or down")
    s = [x / n for x in s]
    u = [s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0]]
    view = [[s[0], s[1], s[2], -sum(s[i] * eye[i] for i in range(3))],
            [-u[0], -u[1], -u[2], sum(u[i] * eye[i] for i in range(3))],
            [-f[0], -f[1], -f[2], sum(f[i] * eye[i] for i in range(3))],
            [0.0, 0.0, 0.0, 1.0]]
    t, near, far = 1 / math.tan(half), dist * 0.1, dist * 4
    proj = [[t / aspect, 0, 0, 0], [0, t, 0, 0], [0, 0, far / (near - far), near * far / (near - far)],
            [0, 0, -1, 0]]
    return mat_mul(proj, view)


def view_proj_words(m) -> list[float]:
    """The 16 floats at view + view_proj: the shader multiplies a row vector by the stored matrix, so it holds the
    transpose of the column-vector matrix, row by row (measured: the other order draws nothing)."""
    return [float(m[c][r]) for r in range(4) for c in range(4)]


def project(m, p, width: int, height: int):
    c = [sum(m[r][k] * v for k, v in enumerate((p[0], p[1], p[2], 1.0))) for r in range(4)]
    if not (c[3] > 0 and 0 <= c[2] <= c[3] and abs(c[0]) <= c[3] and abs(c[1]) <= c[3]):
        raise CameraError("a vertex is outside the view volume")
    return ((c[0] / c[3] * 0.5 + 0.5) * width, (c[1] / c[3] * 0.5 + 0.5) * height)


def _edge_key(p, q):
    return tuple(sorted((tuple(round(v, 6) for v in p), tuple(round(v, 6) for v in q))))


def segment_distance(p, a, b) -> float:
    """Distance from point p to the segment a-b (not to the infinite line through it)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if not length2 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def silhouette_band(tris2d, width: int, height: int) -> set:
    """The pixels whose centre lies within one pixel of a silhouette edge segment (an edge only one triangle has),
    where the GPU and the CPU projection may disagree by a pixel. The distance is to the segment, so a pixel near the
    line's extension past a vertex is not in the band. An edge two triangles share is not in it either: one of them
    covers the pixel either way, so a mismatch there is a crack or an overlap. Returns a set of y * width + x."""
    tris2d = [(a, b, c) for a, b, c in tris2d                       # a triangle with no area covers nothing
              if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) >= 1e-12]
    uses = {}
    for a, b, c in tris2d:
        for p, q in ((a, b), (b, c), (c, a)):
            uses[_edge_key(p, q)] = uses.get(_edge_key(p, q), 0) + 1
    band = set()
    for p, q in ((p, q) for a, b, c in tris2d for p, q in ((a, b), (b, c), (c, a)) if uses[_edge_key(p, q)] == 1):
        x0 = max(0, int(math.floor(min(p[0], q[0]) - 1.5)))
        x1 = min(width - 1, int(math.ceil(max(p[0], q[0]) + 1.5)))
        y0 = max(0, int(math.floor(min(p[1], q[1]) - 1.5)))
        y1 = min(height - 1, int(math.ceil(max(p[1], q[1]) + 1.5)))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if segment_distance((x + 0.5, y + 0.5), p, q) < 1.0:
                    band.add(y * width + x)
    return band


def rasterize(tris2d, width: int, height: int, subpixel_bits: int = SUBPIXEL_BITS):
    """CPU coverage, from #4809's mesh_pass_coverage.rasterize (vertices snapped to the device's subpixel grid, top-left
    rule at pixel centres), and the silhouette band. Returns (covered, band) as sets of y * width + x."""
    mask = mpc.rasterize(tris2d, width, height, subpixel_bits)
    return {i for i, v in enumerate(mask) if v}, silhouette_band(tris2d, width, height)


def read_raw(path: str, width: int, height: int):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != width * height * 16:
        raise ValueError(f"{path}: {len(data)} bytes, expected {width * height * 16} (RGBA32F)")
    return struct.unpack(f"<{width * height * 4}f", data)


def read_mask(path: str, width: int, height: int) -> set:
    """GPU coverage from vk_harness's VKH_COVERAGE_MASK: one stencil byte per pixel, non-zero where a fragment ran."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != width * height:
        raise ValueError(f"{path}: {len(data)} bytes, expected {width * height} (one per pixel)")
    return {i for i, v in enumerate(data) if v}


def compare(gpu: set, pixels, cpu: set, edge: set) -> dict:
    """`gpu` is the stencil coverage, so a covered pixel counts even if its colour equals the clear value. Every
    covered pixel must hold COLOR in all four channels (a NaN is a wrong colour)."""
    if len(pixels) // 4 <= max(gpu | cpu, default=-1):
        raise ValueError("readback smaller than the coverage")
    wrong_colour = sum(1 for i in gpu if not all(math.isclose(pixels[4 * i + k], COLOR[k], rel_tol=0.0, abs_tol=1e-6)
                                                 for k in range(4)))
    union = gpu | cpu
    return {"gpu": len(gpu), "cpu": len(cpu), "iou": len(gpu & cpu) / len(union) if union else 0.0,
            "missing": len(cpu - gpu), "extra": len(gpu - cpu), "off_edge": len((gpu ^ cpu) - edge),
            "wrong_colour": wrong_colour}


def passed(result: dict) -> bool:
    return (result["gpu"] > 0 and result["cpu"] > 0 and result["off_edge"] == 0 and result["wrong_colour"] == 0
            and result["iou"] >= MIN_IOU)


# ------------------------------------------------------------------------------------------------ the run

def wave64_mode(choice: str) -> bool:
    """Whether to translate with --wave64-emulate. "auto" asks tools/vk_subgroup_info.py, as make_game.py does: a
    wave64 mesh shader runs natively only where the GPU can require 64-wide mesh subgroups (`native64`). Where the
    probe gives no answer this emulates, while make_game.py then builds native: emulation runs at any size, so the
    check still means something on a device nobody could ask."""
    if choice in ("on", "off"):
        return choice == "on"
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "vk_subgroup_info.py")], capture_output=True,
                           text=True, timeout=60)
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return True
    return not info.get("native64")


def translate_ms(work: str, sha: str, emulate: bool = False) -> str:
    """Translate the one container whose FSSM is `sha` with this checkout's translator (with --wave64-emulate when
    `emulate`); return the .spv path."""
    with open(os.path.join(work, "shaders", "spirv", "index.json"), encoding="utf-8") as f:
        stems = [k for k, v in json.load(f).items() if v.get("fssm_sha256") == sha]
    if len(stems) != 1:
        raise SystemExit(f"FSSM {sha[:16]}: {len(stems)} containers in the work folder's index, expected 1")
    base = os.path.join(work, "ms_pass")
    for sub in ("dxbc", "spirv"):
        path = os.path.join(base, sub)
        if os.path.islink(path):
            raise SystemExit(f"{path} is a link; refusing to clear it")
        shutil.rmtree(path, ignore_errors=True)
    dxbc = os.path.join(base, "dxbc")
    os.makedirs(dxbc)
    name = stems[0] + ".dxbc"
    shutil.copy(os.path.join(work, "shaders", "dxbc", name), os.path.join(dxbc, name))
    with open(os.path.join(dxbc, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": stems[0], "file": name, "stage": "ms"}], f)
    fl.run([os.path.join(TOOLS, "fssm_unpack.py"), dxbc])
    fl.run([os.path.join(TOOLS, "shader_disasm.py"), os.path.join(dxbc, "fssm"), "--out",
            os.path.join(dxbc, "disasm"), "--mcpu", "gfx1013"])
    spirv = os.path.join(base, "spirv")
    fl.run([os.path.join(TOOLS, "rdna2spirv.py"), os.path.join(dxbc, "disasm"), "--out", spirv]
           + (["--wave64-emulate"] if emulate else []))
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        meta = json.load(f).get(stems[0])
    if not meta or meta.get("fssm_sha256") != sha or meta.get("stage") != "ms":
        raise SystemExit("the translator did not produce the expected mesh shader")
    if bool(meta.get("wave64_emulated")) != emulate:
        raise SystemExit(f"the translation is {'' if meta.get('wave64_emulated') else 'not '}wave64-emulated, "
                         f"expected {'emulated' if emulate else 'native'}")
    return os.path.join(spirv, stems[0] + ".spv")


def draw(harness: str, ms: str, out: str, spec: dict, heap: bytes, sgprs: list[int], width: int, height: int,
         emulate: bool = False, groups: int = 1):
    """Draw once, `groups` mesh workgroups (one per meshlet: with s21 bit 28 clear the shader takes its meshlet from
    the workgroup index). A native wave64 module needs the GPU to require 64-wide mesh subgroups; an emulated one runs at
    whatever size the GPU picks, so no size is requested."""
    fs_asm, fs_spv = os.path.join(out, "fs.spvasm"), os.path.join(out, "fs.spv")
    with open(fs_asm, "w", encoding="utf-8", newline="\n") as f:
        f.write(FS_ASM)
    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", fs_asm, "-o", fs_spv], check=True)
    env = dict(os.environ, VKH_MESH="1", VKH_SGPRS=",".join(hex(x) for x in sgprs),
               VKH_COVERAGE_MASK=os.path.join(out, "coverage.bin"))
    env.pop("VKH_HEAP", None)
    env.pop("VKH_HEAP_FILE", None)
    try:
        env["VKH_HEAP"] = heap_env(heap)
    except ValueError:
        # Too many non-zero words for the environment: vk_harness reads the same image from a file instead.
        heap_file = os.path.join(out, "heap.bin")
        with open(heap_file, "wb") as f:
            f.write(heap)
        env["VKH_HEAP_FILE"] = heap_file
    env.pop("VKH_SUBGROUP_SIZE", None)
    env.pop("VKH_MESH_GROUPS", None)
    if groups > 1:
        env["VKH_MESH_GROUPS"] = str(groups)
    if not emulate:
        env["VKH_SUBGROUP_SIZE"] = str(spec["subgroup_size"])
    raw = os.path.join(out, "image.raw")
    r = subprocess.run([harness, ms, fs_spv, raw, str(width), str(height)], env=env, timeout=120,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"vk_harness exited {r.returncode}: {r.stderr.strip()[-500:]}")
    if groups > 1 and f"mesh workgroups: {groups} x 1 x 1\n" not in r.stderr + "\n":
        # The harness prints the grid it drew; without that line it drew one workgroup (an older vk_harness without
        # VKH_MESH_GROUPS), so the meshlets past the first are missing from the frame.
        raise SystemExit(f"vk_harness did not confirm {groups} mesh workgroups (no 'mesh workgroups' line); it may "
                         "predate VKH_MESH_GROUPS")
    if not emulate and f"mesh subgroup size: {spec['subgroup_size']}\n" not in r.stderr + "\n":
        # vk_harness prints "mesh subgroup size: N" only when the mesh stage got the required size, and
        # "... not available" otherwise; anything else is no confirmation, so the run is not recorded as native.
        raise SystemExit(f"this GPU did not require {spec['subgroup_size']}-wide subgroups for the native wave64 "
                         "mesh shader (no 'mesh subgroup size' line from vk_harness); run with --wave64-emulate on")
    return raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One real drawable through its own translated game mesh shader, "
                                             "coverage against the CPU (TEST-362).")
    ap.add_argument("--game", help="the package folder (reads data.rpf)")
    ap.add_argument("--work", help="the make_game.py work folder (shaders/dxbc and shaders/spirv)")
    ap.add_argument("--models", help="manifest.json written by `rpf9dump models <game>/data.rpf <dir>`")
    ap.add_argument("--drawable", help="the manifest File of the drawable to draw (see --list)")
    ap.add_argument("--out", help="output folder (default <work>/ms_pass/out)")
    ap.add_argument("--yaw", type=float, default=35.0)
    ap.add_argument("--pitch", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--harness", help="vk_harness binary (default LEONIDA_VK_HARNESS or runtime/build/...)")
    ap.add_argument("--list", action="store_true", help="list static drawables, smallest first, and exit")
    ap.add_argument("--run", action="store_true", help="draw and compare")
    ap.add_argument("--check", action="store_true", help="compare the draw in --out again and fail if it differs")
    ap.add_argument("--subpixel-bits", type=int, default=SUBPIXEL_BITS,
                    help="the GPU's subPixelPrecisionBits, for the CPU's vertex snap (default 8)")
    ap.add_argument("--wave64-emulate", choices=("auto", "on", "off"), default="auto",
                    help="translate the mesh shader with --wave64-emulate: auto (default) does when the GPU can't "
                         "require 64-wide mesh subgroups (tools/vk_subgroup_info.py), as make_game.py does")
    ap.add_argument("--controls", action="store_true",
                    help="with --run, also run three negative controls that must fail (wrong camera, swapped axes, "
                         "a V# in the wrong table slot)")
    a = ap.parse_args(argv)

    if a.check and not a.run:
        out = a.out or (a.work and os.path.join(a.work, "ms_pass", "out"))
        if not out:
            ap.error("--check needs --out or --work")
        with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
            man = json.load(f)
        result = check_folder(out, man)
        print(summary(result))
        return 0 if passed(result) else 1
    if not (a.game and a.models):
        ap.error("--game and --models are required")
    with open(a.models, encoding="utf-8") as f:
        manifest = json.load(f)
    if a.list:
        for item in candidates(manifest)[:40]:
            print(f"{item['File']}\tverts {item['Verts']}\ttris {item['Tris']}")
        return 0
    if not (a.work and a.drawable and a.run):
        ap.error("--work, --drawable and --run are required to draw")
    item = next((i for i in manifest.get("Items", []) if i.get("File") == a.drawable), None)
    if item is None:
        ap.error(f"{a.drawable} is not in {a.models}")
    sha, spec = next(iter(SHADERS.items()))
    d = load_drawable(os.path.join(a.game, "data.rpf"), item)
    lo = [min(p[i] for p in d["positions"]) for i in range(3)]
    hi = [max(p[i] for p in d["positions"]) for i in range(3)]
    m = camera(lo, hi, a.yaw, a.pitch, aspect=a.width / a.height)
    tris2d = [tuple(project(m, d["positions"][i], a.width, a.height) for i in t) for t in d["triangles"]]
    heap, layout = build_heap(spec, d["vertex_block"], d["meshlet_group"], view_proj_words(m))
    out = a.out or os.path.join(a.work, "ms_pass", "out")
    os.makedirs(out, exist_ok=True)
    emulate = wave64_mode(a.wave64_emulate)
    a.emulate = emulate
    ms = translate_ms(a.work, sha, emulate)
    draw(fl.find_harness(a.harness), ms, out, spec, heap, sgpr_words(spec, layout), a.width, a.height, emulate,
         groups=d["meshlets"])
    mode = "wave64 emulated" if emulate else f"native wave64 (required subgroup {spec['subgroup_size']})"
    man = {"fssm_sha256": sha, "subgroup_mode": mode, "subpixel_bits": a.subpixel_bits, "width": a.width, "height": a.height, "yaw": a.yaw,
           "pitch": a.pitch,
           "vertices": d["vertices"], "triangles": len(d["triangles"]), "meshlets": d["meshlets"],
           "axis_word": d["axis"], "triangles_2d": tris2d}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f)
    result = check_folder(out, man)
    print(f"{d['vertices']} vertices, {len(d['triangles'])} triangles, {d['meshlets']} "
          f"meshlet{'s' if d['meshlets'] != 1 else ''}, axis word "
          f"{d['axis']}, {mode}: {summary(result)}")
    ok = passed(result)
    if a.controls:
        ok = run_controls(a, d, m, lo, hi, spec, ms, out) and ok
    return 0 if ok else 1


def run_controls(a, d, m, lo, hi, spec, ms, out) -> bool:
    """Negative controls: each must fail, or the comparison could not tell a wrong draw from a right one."""
    drawn = (read_mask(os.path.join(out, "coverage.bin"), a.width, a.height),
             read_raw(os.path.join(out, "image.raw"), a.width, a.height))
    swapped = [(p[1], p[0], p[2]) for p in d["positions"]]
    controls = [("CPU camera 3 degrees off", camera(lo, hi, a.yaw + 3.0, a.pitch, aspect=a.width / a.height),
                 d["positions"], drawn),
                ("CPU x and y axes swapped", m, swapped, drawn)]
    heap, layout = build_heap(spec, d["vertex_block"], d["meshlet_group"], view_proj_words(m))
    slot = layout["table"] + spec["slots"]["meshlet_group"]
    heap[slot + 16:slot + 32], heap[slot:slot + 16] = heap[slot:slot + 16], bytes(16)   # V# one slot too far
    wrong = os.path.join(out, "control_wrong_slot")
    os.makedirs(wrong, exist_ok=True)
    draw(fl.find_harness(a.harness), ms, wrong, spec, heap, sgpr_words(spec, layout), a.width, a.height, a.emulate,
         groups=d["meshlets"])
    controls.append(("meshlet V# in the next table slot", m, d["positions"],
                     (read_mask(os.path.join(wrong, "coverage.bin"), a.width, a.height),
                      read_raw(os.path.join(wrong, "image.raw"), a.width, a.height))))
    all_failed = True
    for name, mat, pos, (gpu, pixels) in controls:
        try:
            tris2d = [tuple(project(mat, pos[i], a.width, a.height) for i in t) for t in d["triangles"]]
        except CameraError:
            print(f"control '{name}': camera clips, refused (counts as failed)")
            continue
        cpu, edge = rasterize(tris2d, a.width, a.height, a.subpixel_bits)
        r = compare(gpu, pixels, cpu, edge)
        print(f"control '{name}': {summary(r)}")
        all_failed = all_failed and not passed(r)
    print("controls: " + ("all fail, as they must" if all_failed else "a control PASSED: the comparison is too weak"))
    return all_failed


def check_folder(out: str, man: dict) -> dict:
    w, h = man["width"], man["height"]
    cpu, edge = rasterize([tuple(map(tuple, t)) for t in man["triangles_2d"]], w, h,
                          man.get("subpixel_bits", SUBPIXEL_BITS))
    return compare(read_mask(os.path.join(out, "coverage.bin"), w, h), read_raw(os.path.join(out, "image.raw"), w, h),
                   cpu, edge)


def summary(r: dict) -> str:
    return (f"GPU {r['gpu']} px, CPU {r['cpu']} px, IoU {r['iou']:.4f}, missing {r['missing']}, extra {r['extra']}, "
            f"off the edge band {r['off_edge']}, wrong colour {r['wrong_colour']}: "
            f"{'pass' if passed(r) else 'FAIL'}")


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

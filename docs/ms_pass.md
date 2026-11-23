# One game mesh shader pass against the CPU (TEST-362)

**Status: measured** on the 1.0.2.1 package and an AMD Radeon 760M (RDNA3 iGPU, gfx1103, subgroup 32..64), RADV
Mesa 26.3.1, CachyOS Linux. A real drawable from the package is drawn by its own translated game **mesh shader**
on `vk_harness`, with the descriptor table, the object matrix and the view-projection written by hand into the
heap, and the covered pixels are compared with the same triangles projected and rasterised on the CPU. This is the
method #1889 describes. The note records counts, hashes and the recipe; the drawable, the shader and the image stay
in the user's work folder.

## 1. What is drawn

- **The drawable** is one static drawable exported from `data.rpf` by `rpf9dump models`, using the `standard`
  effect: 24 vertices, 12 triangles, one meshlet, axis word 2 (`42 40` vertex block of 608 bytes, meshlet group of
  124 bytes). Its OBJ export has SHA-256
  `10066f48f1d7aeac50f59dd7ed2ba40ecf2691a1d094a78b2b67b39743806aa8`. `tools/ms_pass.py --list` ranks the
  static candidates of an export by size.
- **The mesh shader** is the position-only meshlet mesh shader of the `standard` effect, FSSM SHA-256
  `17464335e29c1f4f8ea80c0d65eb3a455cf71803fd88deb751fb3ccabb9bf959` (the only container with that program).
  It is wave64, 128 invocations, structured control flow, no descriptor bindings, and reads user SGPRs s17, s19,
  s20, s21 and s23 (`index.json` `user_sgpr_reads`). The tool translates it with the checkout's own translator.
- **The pixel shader** is a fixture with no inputs that writes one constant colour. The position-only technique
  has no pixel shader of its own; pairing a game pixel shader is the next step (section 5).

## 2. The tables (read from the shader's listing)

Everything the shader reads sits in the one heap at heap base + Xbox VA (`docs/design.md`, "Xbox GPU addresses"):

| What | Where | Content |
|---|---|---|
| s17 | table VA | a table of 16-byte raw V#s (base, 0, size in bytes, `OOB_SELECT` raw) |
| table +0x40 | V# | the drawable's meshlet group, unchanged: count word, 16-byte meshlet records, packed triangles, u16 vertex references |
| table +0x80 | V# | object constants: the object (world) matrix as the first four float4 rows; identity here |
| table +0xA0 | V# | view constants: the view-projection is the 4 x float4 at +0x160 |
| s19 / s20 | low / high VA | the drawable's `42 40` vertex block, unchanged; the shader builds the position V# itself |
| s21 | draw word | 0. With bit 28 clear the meshlet is the workgroup index while it is below the group's meshlet count (the low 17 bits of the group's first word), else 0; with bit 28 set it is read from an indirection list instead (section 4) |
| s23 | VA | resource heap base; read but unused on that path (a zeroed region) |

The view-projection is stored so that the shader's row-vector multiply gives clip = M · p for the column-vector
matrix M: the transpose of M, row by row. Measured: the other order draws nothing.

## 3. The check

- The GPU side is the stencil coverage `vk_harness` writes with `VKH_COVERAGE_MASK` (#4809), so a covered pixel
  counts whatever colour it holds. Every covered pixel must hold the fixture colour in all four channels
  (`math.isclose`, absolute 1e-6; a NaN is a wrong colour).
- The CPU side projects the same triangles (`meshlet_page.group_triangles`, positions from
  `meshlet_page.positions_4240`) with the same matrix in double precision and rasterises them with #4809's
  `mesh_pass_coverage.rasterize`: vertices snapped to the device's subpixel grid (`--subpixel-bits`, default 8),
  top-left rule at pixel centres.
- **Pass:** IoU of at least 0.99, no wrong colour, and **no mismatching pixel whose centre is more than one pixel
  from a silhouette edge segment** (an edge only one triangle has; the distance is to the segment, not to its line). IoU alone is not enough: a camera 3 degrees off still gives
  0.98 on this drawable. An edge two triangles share is not in the edge band, because one of them covers the pixel
  either way; a mismatch there is a crack or an overlap.
- **Subgroup mode.** The mesh shader is wave64. `--wave64-emulate auto` (the default) translates it natively only
  where the GPU can require 64-wide mesh subgroups (`tools/vk_subgroup_info.py` `native64`, as `make_game.py`
  decides) and otherwise with `--wave64-emulate`; the mode is printed with the result and kept in the output's
  `manifest.json`. Where the probe gives no answer, `auto` emulates, while `make_game.py` then builds native:
  emulation runs at any subgroup size, so the check still means something on a device nobody could ask. A native
  draw counts only when `vk_harness` confirms the mesh stage got the required size (`mesh subgroup size: 64`); its
  "not available", or no line at all (a device whose `requiredSubgroupSizeStages` leaves out the mesh stage), fails
  instead of running at another size. On the 760M both modes give byte-identical frames in both poses.
- The camera is a perspective view of the bounding sphere from a chosen yaw and pitch. A pose that clips a vertex is
  refused, so the CPU side needs no clipper.

## 4. Results

| Pose (yaw, pitch) | GPU | CPU | IoU | missing / extra | off the edge band |
|---|---:|---:|---:|---:|---:|
| 35°, 25° | 1,457 px | 1,457 px | **1.0000** | 0 / 0 | 0 |
| -60°, 40° | 1,131 px | 1,131 px | **1.0000** | 0 / 0 | 0 |

128 x 128 target. Every covered pixel holds the fixture colour exactly. The negative controls (`--controls`) all
fail, as they must:

| Control | 35°, 25° | -60°, 40° |
|---|---|---|
| CPU camera 3 degrees off | IoU 0.9836, 24 off the edge band | IoU 0.9566, 51 |
| CPU x and y axes swapped | IoU 0.2665 | IoU 0.3574 |
| meshlet V# one table slot too far | nothing drawn | nothing drawn |

### Every candidate (census)

Every static single-geometry candidate in `data.rpf` (274; 5 have no static meshlet geometry the tool reads, so
269 are drawn), one draw each at 35°, 25° on a 128 x 128 target, native wave64 on the 760M. The translator is
`main` at `5d74c2723`, alone and with #3910 merged; the TEST-362 tools are this branch's.

| | `main` | `main` + #3910 |
|---|---:|---:|
| all | 85 / 269 | **254 / 269** |
| one meshlet | 80 / 141 | 135 / 141 |
| several meshlets (2 to 18) | 5 / 128 | 119 / 128 |
| axis word 0 / 1 / 2 | 17 / 43, 0 / 46, 68 / 180 | 34 / 43, 46 / 46, 174 / 180 |

**What #3910 fixes here.** The shader marks a primitive culled when one of its vertex indices is out of range,
with a compare whose wave64 lane mask lands in an SGPR pair. `shader_disasm.py` prints that pair as one SGPR
(wave32 syntax), so without #3910 lanes 32-63 read the other SGPR's stale value, and the primitive of every lane
32-63 in a wave is culled. On `main`, 55 of the 61 failing one-meshlet draws drew exactly the primitives of lanes
0-31 of each wave, nothing more or less; the GPU drew more pixels than the CPU in only 5 of the 184 failing draws.
The largest drawable that now passes has 1,836 triangles in 15 meshlets.

**Still open (15 with #3910).** 10 draws have only extra pixels (2 to 37), 3 only missing ones (1, 2 and 163) and
2 both; 9 of the 15 use axis word 0, and most are small on screen (136 to 1,586 pixels). The one with 163
missing pixels (8 triangles, one meshlet) is not dropping primitives: no subset of its triangles reproduces the GPU
coverage (the closest is all eight, 163 pixels off). It covers less than the CPU at three poses (163, 121 and 140
pixels at 35°/25°, -60°/40° and 120°/-20°) and never more. Why, and on which side, is not measured yet: the CPU's
`42 40` position decode and the translation are both candidates.

**Emulated.** With `--wave64-emulate` the census gives the same 254, and all 269 coverage masks are byte-identical
to the native ones.

**The draw word.** With s21 bit 28 set, the shader reads each workgroup's meshlet from an indirection list. In this
heap that list is zeroed memory, so every workgroup drew meshlet 0. A one-meshlet drawable hid this. The census
draws with bit 28 clear, which section 2 now records; one 2-meshlet drawable gives IoU 0.9746 with the bit set
and 1.0000 with it clear. With `--controls` (main + #3910) that drawable (175 vertices, 62 triangles, axis word 2)
gives IoU 1.0000 at both poses, and its three controls fail (IoU 0.9717 / 0.9534 with the camera 3 degrees off,
0.9906 / 0.9911 with the axes swapped but mismatching off the edge band, nothing drawn with the V# a slot off).
The section 4 tables were re-measured with bit 28 clear and are unchanged.

## 5. Reproduce

```sh
export DOTNET_ROOT=~/.dotnet DOTNET_ROLL_FORWARD=Major           # where the .NET SDK is installed
tools/rpf9dump/bin/Release/net8.0/rpf9dump models <package>/data.rpf ~/leonida-work/assets
python -I tools/ms_pass.py --list --game <package> --models ~/leonida-work/assets/data/models/manifest.json
python -I tools/ms_pass.py --game <package> --work ~/leonida-work \
    --models ~/leonida-work/assets/data/models/manifest.json --drawable '<the 24-vertex standard prop>' \
    --run --controls
LEONIDA_MS_PASS_DIR=~/leonida-work/ms_pass/out python -I tests/test_ms_pass.py
```

`tests/test_ms_pass.py` checks the V# and heap layout, the SGPRs, the camera, the rasteriser, the comparison and the
zlib reader on synthetic inputs without the package or a GPU; with `LEONIDA_MS_PASS_DIR` it also re-checks a draw.

## 6. Limits and next steps

- **One geometry.** A drawable with more than one geometry is refused, and `--list` leaves those out (`MeshGeoms`
  in the export's manifest): drawing only the first would pass on part of the drawable without saying so. 274 of
  the 282 static candidates in `data.rpf` have one geometry.
- **Rectangular targets.** `--width` and `--height` may differ; the camera divides its x scale by the aspect, so a
  square stays square (the GPU and the CPU share the projection, so a stretch would not show in the comparison).
- **Several meshlets.** The shader takes its meshlet from the workgroup index (s21 bit 28 clear), so a drawable with
  k meshlets is drawn as k mesh workgroups: `vk_harness`'s `VKH_MESH_GROUPS=k`, checked against the device's
  `maxMeshWorkGroupCount` / `maxMeshWorkGroupTotalCount` (exit 77 past them). The tool counts a draw only when
  `vk_harness` prints `mesh workgroups: k x 1 x 1`; that line confirms the dispatch, not that every workgroup drew.
- **One mesh shader.** The table offsets are per shader (`SHADERS` in `tools/ms_pass.py`, keyed by FSSM SHA-256);
  other variants read other offsets.
- **A fixture pixel shader.** The deferred pass pairs a mesh shader that reads s16 and s17 as pointers with a pixel
  shader that reads s16 as a pointer and s17 as a constant, and `vk_harness` pushes one SGPR block to both stages.
- **64 KiB heap.** Everything the shader reads must fit the 64 KiB heap. The heap goes to `vk_harness` as
  `VKH_HEAP` words while the list fits the environment (32,000 characters); a larger image goes through
  `VKH_HEAP_FILE` (`<out>/heap.bin`, the same image, non-zero words written), which most multi-meshlet drawables
  need.

# G-buffer sandbox: placed props through the deferred material shaders

Follows [sandbox.md](sandbox.md): the same placed props, drawn by a translated material VS/PS pair that pulls
its own vertices and writes the five G-buffer targets of [gbuffer.md](gbuffer.md) (PROC-146). Every target is
read back and checked. There is no lighting or shadow pass. Measured on 2026-11-22 from one retail Xbox Series install (1.0.2.1), on an
NVIDIA GeForce RTX 4070 Ti SUPER (driver 621.24). Nothing below reproduces game data; every number is a count.

## 1. What runs

`tools/gbuffer.py` packs the props with `tools/sandbox_scene.py`, picks a pair from the work folder and runs
`runtime/test/vk_gbuffer`. That tool repacks each mesh into the game's compressed vertex stream, fills the tables
the pair reads, draws one non-indexed `vkCmdDraw` per prop mesh (the VS loads its index itself) into five colour
targets plus D32, and reads every target back. It prints counts only:

| Count | Meaning | PASS when |
|---|---|---|
| covered | pixels with depth < 1 | ≥ 1 % of the frame |
| shaded | covered pixels where the PS wrote mrt2 (alpha 0.5) | ≥ 99 % of covered |
| albedo_colours | distinct mrt0 colours | ≥ 256 when the scene has textures |
| normals_facing | decoded mrt1 normals pointing against the camera's forward axis | ≥ 60 % of covered |
| motion_still | mrt4 motion under 0.1 px (the camera does not move) | ≥ 99 % of covered |

## 2. How the pair is chosen

By what the shaders read, never by name:

| | Vertex shader | Pixel shader |
|---|---|---|
| inputs | `SV_VertexID` only | the VS's outputs, linked by semantic (below) |
| user data | ptr32 s8, s9, s18; const32 s11–s16 | ptr32 s8, s9, s15, s16; const32 s14, s17 |
| bindings | none, no image reads | 0, 1, 2; exactly one `image_sample` |
| listing | stream V# built from s11/s12 (`0x11014fac`), index through the V# table at s18 + s14·32; s9 + 0x60/0x80/0xa0; s8 + 0x6e0; exports param0–5 | s8 + 0x6e0/0xac0; s9 + 0x20/0x60; T# at s15 + index·32; S# at s16 + 0x520; exports mrt0–4 |

On the 1.0.2.1 corpus 360 VS and 112 PS have the user-data layout. 14 VS and 8 PS also match the listing, and 24
of those pairs link.

**Linking.** The translator puts a VS output at location = param number and a PS input at location = attribute
number. The game links them by semantic instead. So `gbuffer.py` reads the DXBC signatures: for each PS attribute
(the non-system ISG1 elements it reads, in register order) it finds the VS param with the same semantic, index and
minimum precision (non-system OSG1 elements, in register order). It passes the result as `--link`, and
`vk_gbuffer` rewrites the PS input locations to match. A VS that exports a different number of params than it
declares outputs is skipped, because its order is unknown.

## 3. The tables

| Where | Contents |
|---|---|
| s8 + 0x6e0 | V# of the frame constants; 0x52c = 1.0 turns off a +1e-5 depth bias; 0x5f0 = frame counter (dither) |
| s8 + 0x1780 | V# the VS stores three debug dwords to (lanes with vertex 0) |
| s8 + 0xac0 | V#; the dword at s14 is the PS's T# number |
| s9 + 0x20 | material bytes read into mrt3 |
| s9 + 0x60 | viewport: width 0x44, height 0x48 |
| s9 + 0x80 | object: world rows at 0x00 (world = x·r0 + y·r1 + z·r2 + r3), previous world at 0x40 |
| s9 + 0xa0 | view: view-projection at 0x160, previous at 0x1a0 |
| s11/s12 | 48-bit address of the vertex stream (below) |
| s18 + s14·32 | index V# (R32_UINT); vertex = index[VertexIndex] |

**Vertex stream.** Attribute words sit at 0x0 (position), 0x8 (normal) and 0xc (UV). Each is
`offset << 7 | stride << 3 | format`, with UV format 2 = float2 (otherwise 16_16 SNORM). At 0x60 come the bias
xyz, the axis mode (0 = xyz), the scale xyz and the UV scale. A position is 8 bytes: a 24-bit field in dword 0
bits 8–31, a 24-bit field split across dword 0 bits 0–7 and dword 1 bits 0–15, and a 16-bit field in dword 1 bits
16–31. Each field is q·scale + bias. The VS weights world row 2 by the first field, row 1 by the second and row 0
by the third, so the first field holds model z and the third model x. A normal is one dword. With bit 31 set it
is 10:10 octahedral (q·2/1023 − 1); otherwise it is 8:8. A length² under 0.1 falls back to (0, 0, 1).

**Targets.** mrt0–3 are R8G8B8A8_UNORM and mrt4 is R16G16_SFLOAT. mrt0 holds the albedo. mrt1 holds a 24-bit
octahedral normal across g/b/a (12 bits per axis). mrt2 alpha is 0.5. mrt3 holds the dither and material bytes.
mrt4 holds the motion in pixels.

## 4. Reproduce it

Prepare the work folder as in [sandbox.md](sandbox.md) §4, then:

```sh
cmake --build runtime/build --target vk_gbuffer
python -I tools/gbuffer.py --work <work> --entities <work>/entities.jsonl --models <work>/assets \
    --textures <work>/ttpf_maps --textures <work>/ttpf_terrain [--center X Y] [--dump <work>/gb] [--shadow]
```

`--shadow` adds a sun shadow pass (§5). With `--dump`, `<prefix>_lit.ppm` (first sun) is a preview: albedo × (0.3 + 0.7 ·
N·L · visibility).

It needs `VK_KHR_fragment_shader_barycentric`, `shaderDrawParameters`, `bufferDeviceAddress`, descriptor
indexing and `shaderInt64` (exit 77 without them). It also needs
shaders translated after the 16-bit interpolant fix of 2026-11-22. An older translation reads the packed f16
normal as one smooth float, and `normals_facing` falls to a few percent of `covered`.

## 5. Sun shadow pass

`--shadow` draws every prop into a 2048² D32 map from an orthographic sun with the game's depth-only VS, then
tests each covered G-buffer pixel against it.

**Which VS.** `pick_shadow_vs` takes the first VS in name order with: SV_VertexID as the only input, no bindings,
user SGPRs `[ptr32_lo 9, const32 11, const32 12]`, no image instructions, exports pos0 and param0 only, and every
pattern in `SHADOW_LAYOUT`. In the current corpus two VS match. Both do the same transform.

**Tables.** s9 points to a table with a V# at +0x40 for the object rows (r0, r1, r2 at 0x00–0x20, translation at
0x30) and a V# at +0x60 for the view, with the view-projection at 0x160. These are the same object and view
layouts as the G-buffer VS, so the object CB is shared. s11/s12 hold the stream base. The header dword 0 gives the
offset (bits 7+) and the stride (bits 3–6). The VS reads float3 positions at offset + stride · SV_VertexID, so the
tool draws indexed with the mesh's index buffer. There is no fragment shader.

**Test.** Each covered pixel goes back to world space through the inverse view-projection. It is offset 1.5 map
texels along its normal, against self-shadowing, and compared with the map at a depth bias of 2 texels in world
units. A non-finite position counts as `outside`. A pixel is
`away` when N·L ≤ 0, `shadowed` when the map is nearer, and `lit` otherwise.

**PASS** (with `--shadow`): for both suns (azimuth/elevation 135°/50° and 315°/50°) the map is non-empty, `lit`
> 0 and `shadowed` ≥ 1 % of the tested pixels. The two `lit` counts differ by at least 10 % of the tested pixels,
because opposite suns light opposite faces.

## 6. Measured

| | |
|---|---|
| test scene (as in sandbox.md) | 150 props, 446 draws, 442 textured, 107 textures |
| 640×360 | 114,503 pixels covered (the same as `vk_sandbox --check` at that camera), 114,503 shaded, 7,325 albedo colours, 98,778 normals facing the camera, 114,503 still motion vectors (largest 0.003 px) |
| `--shadow`, sun 135°/50° | 82,784 lit, 10,529 shadowed, 21,190 facing away |
| `--shadow`, sun 315°/50° | 13,420 lit, 17,164 shadowed, 83,919 facing away |
| synthetic cube | each face decodes to its own rotated normal, and the near faces win the depth test |

## 7. Limits and next

* The shadow test and the N·L preview run on the CPU. The game's own shadow-mask and deferred lighting shaders are
  the next step.
* One pair draws every prop. A prop's own material program comes from the executable's table (#2519).
* Positions are repacked from the exported model, not read from the game's own vertex buffers.

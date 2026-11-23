# Deferred lighting pass: the game's tiled-lighting pixel shaders over a synthetic G-buffer

Follows [gbuffer.md](gbuffer.md) and [sandbox.md](sandbox.md): after the material shaders write the G-buffer, the
game lights it in a full-screen (tiled) pass. `tools/lighting_pass.py` runs that pass's own translated pixel shader on
synthetic inputs through `runtime/test/vk_lighting.cpp` and checks how it responds to the normal, the sun, the shadow mask
and the albedo. Measured on 2026-11-23 from one retail Xbox Series install (1.0.2.1), shaders translated with
`--wave64-emulate --f16-interp-fallback`, on an NVIDIA Tesla V100-SXM2-32GB (driver 580.95.05, Linux). Every number
below is a count or a mean of synthetic outputs; nothing reproduces game data.

**This is a response test, not a reproduction of the game's lighting.** The camera and view constants of the pass are
not decoded: every constant the shader reads is set to 0.5 except the few named in §3, so the view direction is
arbitrary and a specular term is mixed into every output. The G-buffer is synthetic (one uniform normal and albedo
per run), not the targets of the G-buffer sandbox ([#4119](https://github.com/ItsNotRexed/Leonida/pull/4119)), because that sandbox
needs `VK_KHR_fragment_shader_barycentric`, which the V100 does not have.

## 1. What runs

`vk_lighting` is a generic full-screen pass runner. It reads a plan written by the Python tool: the VS/PS pair, the raw
bytes of the heap the shaders' table pointers address, the user SGPRs, the images (their T# is encoded with
`gfx10::encode` and written into the heap at each slot the plan names, then mapped in the descriptor hash table of a
`leonida::BindlessSet`), the S#s and the colour targets. It draws one triangle with `SV_VertexID` 0..2 and writes each
target's raw readback next to the plan. The tool needs `shaderDrawParameters`, `bufferDeviceAddress`, descriptor
indexing and `shaderInt64` (exit 77 without them); unlike `vk_gbuffer` it does not need fragment barycentrics.

## 2. How the shaders are chosen

By what they read, never by name (`tools/lighting_pass.py`, `ps_matches`, `vs_matches`, `is_directional`):

| | Vertex shader | Pixel shader |
|---|---|---|
| inputs | `SV_VertexID` only, no bindings, no scalar loads | `SV_Position` only; outputs 0 and 1 (`exp mrt0`, `exp mrt1`, both `compr`) |
| user data | `const32` s10 only (the triangle's depth, s10 / 65535) | `ptr32_lo` s8 (table A), s9 (table B; the listing copies it into another pair, `s[10:11]` in 84 of the 120 directional variants, `s[12:13]` in 34, `s[92:93]` in 2; the tool follows that copy), s14 (the heap of the dynamic T# lookups) |
| listing | exports `pos0` and `param0` | T#s at A + 0x1240, 0xbc0 and 0x1400; a V# at A + 0x2a0; the 12-bit split of the normal (`v_and_b32 0xfff`, `v_bfe_u32 12, 12`) |

A PS that also loads the sun colour (`s_buffer_load_dwordx4 … 0x148`) and the sun intensity (`s_buffer_load_dword …
0x8`) is *directional*; the shortest directional listing and the shortest other listing are used. On the 1.0.2.1
corpus 3 VS, 120 directional PS and 120 other PS match.

## 3. The table the PS reads

Found by perturbing one input at a time on the GPU (a T# slot set to black, white or red; a constant dword set to
0, 1, −1 or 4) and by comparing the directional and non-directional variants' loads. Offsets are in table A (s8) or
table B (s9).

| Where | What it is | How it was found |
|---|---|---|
| A + 0x1240 (T#) | G-buffer mrt1: the 24-bit octahedral normal in g, b, a (12 bits per axis, as in [gbuffer.md](gbuffer.md)) | the listing decodes it with the 12-bit split; six axis normals give six different sun terms |
| A + 0xbc0 (T#, rgb) | G-buffer mrt0: albedo | a one-hot albedo puts its own output channel highest and leaves the other two equal |
| A + 0x1400 (T#, one channel) | sun shadow mask, per pixel | 0 removes the sun term, 255 doubles it against mid-grey |
| A + 0x900 (T#, 2D array) | likely an ambient probe (spherical-harmonic coefficients); a hypothesis | read only by the directional variant; the factors next to its samples are the SH band constants √π/2 and √(π/3) |
| A + 0x2a0 (V#) | pass scales: 0x0 a pre-scale, 0x4 the output scale (multiplies both exports), 0x8 the sun intensity | 0x4 is the last multiply before the export; 0x8 scales the sun term linearly (with the sun colour 10: 0, 0.5, 1 give 0.0026, 8.65, 17.30) |
| A + 0x2c0 (V#), dword 0x244 | attenuates the sun (0 = none) | 0 raises the sun term about 50×, 0.5 almost removes it |
| B + 0x0 (V#), 0x110..0x118 | sun direction (the shader lights where −dot(n, d) > 0, so the tool stores −L) | flipping it swaps the lit and the unlit normal |
| B + 0x0 (V#), 0x148..0x150 | sun colour, per channel | read only by the directional variant; 0 removes the sun term |

Many other slots (screen-space T#s, the light-loop heap at s14, the camera constants) are filled with mid-grey or 0.5
and not decoded.

## 4. The check

The sun term is the mean of output 0's red channel with the sun colour 1 minus the same run with the sun colour 0, for
each of six uniform normals (±x, ±y, ±z) at 32×18. With the sun along +x, **PASS** needs:

* the normal facing the sun gets at least 3× the sun term of the opposite normal;
* a flipped sun swaps them (the same 3× the other way);
* a shadow mask of 0 leaves at most 1 % of the sun term;
* the non-directional variant has no sun term (at most 0.1 % of it);
* a red, green and blue albedo each put their own output channel at least 5 % above the other two, which stay equal;
* every output pixel is finite.

## 5. Measured

| | |
|---|---|
| matched | 3 VS, 120 directional PS, 120 other PS |
| sun term (sun +x) | facing 3.059, away 0.2955 |
| sun −x | facing 0.5276, away 6.973 |
| shadow mask 0 | 0 |
| non-directional variant | 0 |
| one-hot albedo r/g/b | red 3.925/3.137/3.137, green 3.087/3.875/3.087, blue 3.151/3.151/3.939 |
| result | **PASS**, 51 runs |

Negative controls (each one input moved to a wrong offset; all FAIL):

| change | effect |
|---|---|
| the normal written to A + 0x920 | facing = away (5.144) |
| the shadow mask written to A + 0x920 | a zero mask keeps a sun term of 1.336 |
| the sun colour written at 0x12c (the other light colour, read by both variants) | sun term ≈ 0.0008 |
| the albedo written to A + 0x1480 | all three one-hot albedos give 3.21/3.21/3.21 |

The output is not monotonic in a grey albedo (black 3.558, 128 grey 3.210, white 3.848 with the sun on): with the
material slots at mid-grey part of the specular follows the albedo, which is why the albedo check looks at channel
selectivity instead.


## 6. Other material variants

The sun checks of §4 (facing ≥ 3× away, a flipped sun swaps them, a zero shadow mask removes the sun term; not the
albedo and non-directional checks, which use the picked pair) were run on every one of the 120 directional variants,
each with the same table filling. They are 30 material models, each with four variants (plain, `GLOBAL_GLINTS`,
`UNDERWATER`, `WETNESS`), and every model's four variants land in the same row:

| Result | Variants | Material models | Why |
|---|---:|---|---|
| PASS | 68 | STANDARD, CLEARCOAT, CLEARCOAT_ANISOTROPY, CLEARCOAT_DIFFRACTION, CLEARCOAT_GLINTS, CLEARCOAT_IRIDESCENCE, CLEARCOAT_RADIAL_DIFFRACTION, EMISSIVE, EYE, GLINTS, HAZE, RETROREFLECTION, RETROREFLECTION_DIFFRACTION, SHEEN_ANISOTROPY_SSS, SKIN, TERRAIN_GLINTS, TERRAIN_GLINTS_TRANSLUCENCY | — |
| the sun's sign is right, the ratio is below 3 | 36 | ANISOTROPY, BLURTRANSMISSION, DIFFRACTION, IRIDESCENCE, RADIAL_DIFFRACTION, SHEEN, SHEEN_ANISOTROPY, SHEEN_IRIDESCENCE, TRANSLUCENCY | facing/away is 2.19–2.44 and a flipped sun swaps it, with no sun term under a zero mask: these models light the side away from the sun as well (sheen, transmission) or add a strong view-dependent lobe. TRANSLUCENCY reads the same table as STANDARD but for one dynamic offset. The 3× threshold does not fit them |
| no sun term | 12 | EMISSIVE_DIFFRACTION, EMISSIVE_RETROREFLECTION, HAIR | a model-specific constant gates the sun. For HAIR, B + 0x160 dword 0x158 or B + 0x80 dword 0x4 at 0 (instead of the 0.5 fill) gives a sun term of 3.79; these constants are not decoded |
| the device was lost | 4 | EYE_RETROREFLECTION | `vkQueueWaitIdle` returned `VK_ERROR_DEVICE_LOST` on the V100 with these inputs (likely a loop that does not end with the 0.5 constants); not rerun |

The first full run had 45 variants without a sun term. 35 of them copy table B into `s[12:13]` (34) or `s[92:93]`
(1) instead of `s[10:11]`, so the tool did not fill table B. It now follows the copy (§2). Rerun, 25 of the 45 pass,
8 move to the second row and the 12 above remain.

The PASS claim of this note is for the 68 variants in the first row only.

## 7. Reproduce it

```sh
python -I make_game.py --stages shaders --wave64-emulate on --f16-interp-fallback ...   # or any translated work folder
cmake --build runtime/build --target vk_lighting
python -I tools/lighting_pass.py --work <work> [--build runtime/build] [--keep <folder>]
```

`--keep` leaves every plan, heap and readback in the folder (they are built from the user's own shaders; keep them
there). `python -I tests/test_lighting_pass.py` checks the matching, the table layout, the plan and the checks on
hand-written listings without a GPU.

## 8. Limits and next

* The view and camera constants: decoding them (the position reconstruction from depth) would turn the specular into
  a known function of the view and let the test compare against a CPU BRDF instead of ratios.
* The real G-buffer: running this pass on `vk_gbuffer`'s targets needs a GPU with fragment barycentrics.
* The shadow mask here is a constant image; the game builds it with the `shadowmask` compute shaders.
* The albedo and non-directional checks run on the picked pair only; 52 of the 120 directional variants do not pass
  the sun checks (§6); local lights (the dynamic T# lookups at s14) are
  not exercised.

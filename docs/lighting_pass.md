# Deferred lighting pass: the game's tiled-lighting pixel shaders over a synthetic G-buffer

Follows [gbuffer.md](gbuffer.md) and [sandbox.md](sandbox.md): after the material shaders write the G-buffer, the
game lights it in a full-screen (tiled) pass. `tools/lighting_pass.py` runs that pass's own translated pixel shader on
synthetic inputs through `runtime/test/vk_lighting.cpp` and checks how it responds to the normal, the light direction,
the mask that scales the whole lighting result and the diffuse albedo. Measured on 2026-11-23 from one retail Xbox Series install (1.0.2.1), shaders translated with
`--wave64-emulate --f16-interp-fallback`, on an NVIDIA Tesla V100-SXM2-32GB (driver 580.95.05, Linux). Every number
below is a count or a mean of synthetic outputs; nothing reproduces game data.

**This is a response test, not a reproduction of the game's lighting.** The check fills every constant the shader reads
with 0.5 except the few named in §3, including the camera constants that §3.1 decodes, so the view direction is
arbitrary and a specular term is mixed into every output. The G-buffer is synthetic (one uniform normal, F0 and albedo
per run), not the targets of the G-buffer sandbox ([#4119](https://github.com/ItsNotRexed/Leonida/pull/4119)), because that sandbox
needs `VK_KHR_fragment_shader_barycentric`, which the V100 does not have.

## 1. What runs

`vk_lighting` is a generic full-screen pass runner. It reads a plan written by the Python tool: the VS/PS pair, the raw
bytes of the heap the shaders' table pointers address, the user SGPRs, the images (their T# is encoded with
`gfx10::encode` and written into the heap at each slot the plan names, then mapped in the descriptor hash table of a
`leonida::BindlessSet`), the S#s and the colour targets. It draws one triangle with `SV_VertexID` 0..2 and writes each
target's raw readback next to the plan. The tool needs `shaderDrawParameters`, `bufferDeviceAddress`, descriptor
indexing and `shaderInt64` (exit 77 without them; `tools/lighting_pass.py` passes that on as its own exit 77, a skip);
unlike `vk_gbuffer` it does not need fragment barycentrics.

## 2. How the shaders are chosen

By what they read, never by name (`tools/lighting_pass.py`, `ps_matches`, `vs_matches`, `is_directional`):

| | Vertex shader | Pixel shader |
|---|---|---|
| inputs | `SV_VertexID` only, no bindings, no scalar loads | `SV_Position` only; outputs 0 and 1 (`exp mrt0`, `exp mrt1`, both `compr`) |
| user data | `const32` s10 only (the triangle's depth, s10 / 65535) | `ptr32_lo` s8 (table A), s9 (table B; the listing copies it into another pair, `s[10:11]` in 84 of the 120 directional variants, `s[12:13]` in 34, `s[92:93]` in 2; the tool follows that copy), s14 (the heap of the dynamic T# lookups) |
| listing | exports `pos0` and `param0` | T#s at A + 0x1240, 0xbc0 and 0x1400; a V# at A + 0x2a0; the 12-bit split of the normal (`v_and_b32 0xfff`, `v_bfe_u32 12, 12`) |

A PS that also loads the tint (`s_buffer_load_dwordx4 … 0x148`, §3) and the lighting-term scale (`s_buffer_load_dword …
0x8`) is *directional*; the shortest directional listing and the shortest other listing are used. On the 1.0.2.1
corpus 3 VS, 120 directional PS and 120 other PS match.

## 3. The table the PS reads

Found by perturbing one input at a time on the GPU (a T# slot set to black, white or red; a constant dword set to
0, 1, −1 or 4) and by comparing the directional and non-directional variants' loads. Offsets are in table A (s8) or
table B (s9).

| Where | What it is | How it was found |
|---|---|---|
| A + 0x1240 (T#) | G-buffer mrt1: the 24-bit octahedral normal in g, b, a (12 bits per axis, as in [gbuffer.md](gbuffer.md)) | the listing decodes it with the 12-bit split; six axis normals give six different lighting terms |
| A + 0x920 (T#, rgb) | the diffuse albedo | linear and channel-selective: a channel at 255 instead of 128 raises only its own output channel (+0.429, the other two +0.000); a one-hot albedo gives 4.073 in its own channel and 3.14 in the other two |
| A + 0xbc0 (T#, rgb) | behaves as the specular F0 (named albedo in the first version of this note) | with the diffuse albedo at 0, F0 1 minus 0 correlates 0.77 with N·H and saturates at N·H ≥ 0.8; raising one channel lowers that channel's diffuse share |
| A + 0x1400 (T#, one channel) and B + 0x0 (V#), 0x148..0x150 | a mask m and a tint c per channel that multiply the whole lighting result by lerp(1, c, m) (named the sun shadow mask and the sun colour in the first version) | with c = 1 the mask has no effect (difference exactly 0 over 2,304 pixels); with c = 0.5, mask 255 minus mask 0 is −0.5× the lit output. The tint is read only by the directional variant |
| A + 0x900 (T#, 2D array) | a light probe (spherical-harmonic coefficients) that scales all the directional variant's lighting | linear: 0, 64, 128, 255 give 0, 0.660, 1.321, 2.633; evaluated along A + 0x2c0 dwords 0xec..0xf4; the factors next to its samples are the SH band constants √π/2 and √(π/3) |
| A + 0xc80 (T#, 2D) | a diffuse lookup table addressed by the light angle | sampled at u = 0.5 + 0.5·sign(N·L)·√\|N·L\|: a horizontal ramp in it makes the direct diffuse follow k·u (correlation 0.9973, 6.0 % rms residual) |
| A + 0x2a0 (V#) | pass scales: 0x0 a pre-scale, 0x4 the output scale (multiplies both exports), 0x8 scales the lighting term | 0x4 is the last multiply before the export; 0x8 scales the lighting term linearly (with the tint 10: 0, 0.5, 1 give 0.0026, 8.65, 17.30) |
| A + 0x2c0 (V#), dword 0x244 | attenuates the lighting term (0 = none) | 0 raises the term about 50×, 0.5 almost removes it |
| B + 0x0 (V#), 0x110..0x118 | light direction (the shader lights where −dot(n, d) > 0, so the tool stores −L) | flipping it swaps the lit and the unlit normal |
| B + 0x0 (V#), 0x12c..0x134, with A + 0x3a0 dword 0x54 and the A + 0x1320 1D table | another light colour, on a separate branch | ≈0 with the inputs of §4 (why the tint control of §5 moved to 0x12c gives 0.0008) |

Many other slots (screen-space T#s, the light-loop heap at s14) are filled with mid-grey or 0.5 and not decoded.

### 3.1 The camera: B + 0x80

The pixel's camera-relative position is rebuilt from `SV_Position` and depth with the V# at B + 0x80 (offsets are
dwords of that buffer):

| Dwords | Use |
|---|---|
| 0x4c, 0x50 | pixel → NDC: x = 2·(px + 0.5)·[0x4c] − 1, y = 1 − 2·(py + 0.5)·[0x50] |
| 0x1c, 0x20, 0x24 | ray x = [0x1c]·x + [0x20]·y + [0x24] |
| 0x10, 0x14, 0x18 | ray y = [0x10]·x + [0x14]·y + [0x18] |
| 0x28, 0x2c, 0x30 | ray z = [0x28]·x + [0x2c]·y + [0x30] |
| 0x34, 0x38 | near n and far f: the depth d in A + 0xf80 (r) becomes n·f / (n + (f − n)·\|d\|), which multiplies the ray |

With every dword at 0.5 (the check of §4) the ray is the same for all pixels and the view direction is not meaningful.
With these set (for example a constant ray (0, 0, 1) and n = f = 1) the response follows the half vector of the light
and the eye, as a specular lobe should. A value-for-value CPU reference is not ready: with the ramp of A + 0xc80 above
the direct diffuse still has a 6 % rms residual, and no single extra factor removes it ((1 − N·L)⁵ is the best, 4.6 %).

## 4. The check

The total lighting term is the mean of output 0's red channel with the tint 1 minus the same run with the tint 0, with
the mask at 255: by lerp(1, c, m) (§3) that is the whole lit result X, diffuse, specular and ambient together. It is
taken for each of six uniform normals (±x, ±y, ±z) at 32×18. With the light along +x, **PASS** needs:

* the normal facing the light gets at least 3× the lighting term of the opposite normal;
* a flipped light swaps them (the same 3× the other way);
* a mask of 0 leaves at most 1 % of the lighting term (the tint is then no longer applied, so the two runs are equal;
  this checks the mask and tint slots, not a shadow);
* the non-directional variant has no lighting term (at most 0.1 % of it);
* a red, green and blue diffuse albedo (A + 0x920) each put their own output channel at least 5 % above the other two,
  which stay equal;
* every pixel of both outputs is finite.

## 5. Measured

| | |
|---|---|
| matched | 3 VS, 120 directional PS, 120 other PS |
| total lighting term (light +x) | facing 3.059, away 0.2955 |
| light −x | facing 0.5276, away 6.973 |
| mask 0 | 0 |
| non-directional variant | 0 |
| one-hot diffuse albedo r/g/b | red 4.073/3.14/3.14, green 3.14/4.073/3.14, blue 3.14/3.14/4.073 |
| result | **PASS**, 51 runs |

Negative controls (each one input moved to a wrong offset; all FAIL):

| change | effect |
|---|---|
| the normal written to A + 0x920 | facing = away (5.144) |
| the mask written to A + 0x920 | a zero mask keeps a lighting term of 1.336 |
| the tint written at 0x12c (the other light colour, §3) | lighting term ≈ 0.0008 |
| the diffuse albedo written to A + 0x1480 | all three one-hot albedos give 3.608/3.608/3.608 |

The first version of this note put the one-hot albedo into A + 0xbc0 (then read as the albedo; red 3.925/3.137/3.137).
That slot behaves as the specular F0 (§3), and its output is not monotonic in a grey value (black 3.558, 128 grey
3.210, white 3.848 with the light on); the check now uses the diffuse albedo, and the other numbers above are
unchanged.


## 6. Other material variants

The lighting checks of §4 (facing ≥ 3× away, a flipped light swaps them, a zero mask removes the lighting term; not the
albedo and non-directional checks, which use the picked pair) were run on every one of the 120 directional variants,
each with the same table filling. They are 30 material models, each with four variants (plain, `GLOBAL_GLINTS`,
`UNDERWATER`, `WETNESS`), and every model's four variants land in the same row:

| Result | Variants | Material models | Why |
|---|---:|---|---|
| PASS | 68 | STANDARD, CLEARCOAT, CLEARCOAT_ANISOTROPY, CLEARCOAT_DIFFRACTION, CLEARCOAT_GLINTS, CLEARCOAT_IRIDESCENCE, CLEARCOAT_RADIAL_DIFFRACTION, EMISSIVE, EYE, GLINTS, HAZE, RETROREFLECTION, RETROREFLECTION_DIFFRACTION, SHEEN_ANISOTROPY_SSS, SKIN, TERRAIN_GLINTS, TERRAIN_GLINTS_TRANSLUCENCY | — |
| the light's sign is right, the ratio is below 3 | 36 | ANISOTROPY, BLURTRANSMISSION, DIFFRACTION, IRIDESCENCE, RADIAL_DIFFRACTION, SHEEN, SHEEN_ANISOTROPY, SHEEN_IRIDESCENCE, TRANSLUCENCY | facing/away is 2.19–2.44 and a flipped light swaps it, with no lighting term under a zero mask: these models light the side away from the light as well (sheen, transmission) or add a strong view-dependent lobe. TRANSLUCENCY reads the same table as STANDARD but for one dynamic offset. The 3× threshold does not fit them |
| no lighting term | 12 | EMISSIVE_DIFFRACTION, EMISSIVE_RETROREFLECTION, HAIR | all 12 read the T# at A + 0x13c0 (91 of the 120 directional variants do). The tool fills unnamed T#s with 128/128/128/255, so its alpha is 1, which removes the term for these models. With only that alpha at 128 the 8 EMISSIVE variants give 2.31–3.48 (the 79 other readers are not affected by it at 255); the 4 HAIR variants give only 0.004–0.006, and B + 0x160 dword 0x158 or B + 0x80 dword 0x4 at 0 (instead of 0.5) gives HAIR a term of 3.79; those constants are not decoded |
| the device was lost | 4 | EYE_RETROREFLECTION | `vkQueueWaitIdle` returned `VK_ERROR_DEVICE_LOST` on the V100 with these inputs (likely a loop that does not end with the 0.5 constants); not rerun |

The first full run had 45 variants without a lighting term. 35 of them copy table B into `s[12:13]` (34) or `s[92:93]`
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

* A CPU BRDF reference: the camera is decoded (§3.1), but the remaining factors (the second coordinate of the
  A + 0xc80 table, the energy and sun-disc terms) leave a 6 % residual, so the test still compares ratios.
* The real G-buffer: running this pass on `vk_gbuffer`'s targets needs a GPU with fragment barycentrics.
* The mask here is a constant image. Whether the game fills it from the `shadowmask` compute shaders is not checked.
* The albedo and non-directional checks run on the picked pair only; 52 of the 120 directional variants do not pass
  the lighting checks (§6); local lights (the dynamic T# lookups at s14) are
  not exercised.

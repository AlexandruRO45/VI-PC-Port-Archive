# First light: a real game model with its own translated shaders

**Status: measured** on the 1.0.2.1 package and an NVIDIA RTX 2060 SUPER (driver 620.73.04, Vulkan 1.4.357,
CachyOS). This is the first draw of a real game model with its own translated VS and PS on a real GPU. The image
stays in the user's work folder; this note records the counts and the recipe, not the model, the shaders or a
screenshot.

## 1. What was drawn

The model is one static drawable exported from `data.rpf` by `rpf9dump models`: one mesh, 4,213 vertices and
7,968 triangles, with positions and UVs and no bones or fragments. Its OBJ is the only `data.rpf` export with
those counts and has SHA-256
`a7178cf4b7a00c9df838fb072c59e2c048ceed52306620d859faae4d3991d6b9`; `manifest.json` records it. The tool that
assembles it, `tools/first_light.py`, centres and fits X/Y to NDC and flattens Z (the chosen VS applies no
matrix), so the draw is an orthographic projection of the model's own geometry.

The shaders are two ordinary containers from `shaders.rpf`, carved and translated by the repository's own tools:

| stage | FSSM SHA-256 | what it does |
|---|---|---|
| VS | `128867d3233a97ba5b203c313a24ab8a77df20c2a308091a5595bb9f669677f4` | forwards POSITION and TEXCOORD from the vertex fetch to clip space and a varying |
| PS | `ed4147a5f7514947c2eb8d79255bd80f72761ac1ba2205dd811d3e050cf1b495` | writes the interpolated TEXCOORD as the colour; no bindings, no texture |

Both disassemble with no invalid word, translate, and pass `spirv-val`; `tools/shader_link_check.py check` reports
the pair compatible. At 128×128 the draw covers **12,804 of 16,384 pixels (0.781)** and the centre pixel is
`(0.498, 0.496, 0.0, 1.0)`: the model is the disc its geometry projects to, and the colour is its own UV map. The
test derives the expected centre UV from the mesh on the CPU (`0.4978, 0.4965`), so the GPU pixel has to match
the geometry, not just be non-clear; it also requires the `[0, 0]` corner to keep the clear value and bounds the
coverage at 0.5–0.99, which an all-opaque image cannot pass. Four extra probes where U and V differ by more than
0.15 catch a swapped-U/V image even when the centre is symmetric (this model's centre has U ≈ V). The
`OracleTests` run those checks on synthetic images without a GPU or a package: the accepted fixture, an
all-opaque image, a swapped UV and a short buffer. The same `check_image` function backs the tool's `--check`, so
the command line and the test agree. The PPM the harness writes is a real image of the model; it stays in the
user's folder.

## 2. Why it matters

Until now the D3D12 draw tests (M1, M1.5, M2, M3) were met with hand-written RDNA2 or GLSL shaders and synthetic
geometry, and the pipeline check compiles the game's 26,503 shaders without ever drawing with one
([pipeline_check.md](pipeline_check.md)). This closes the "shaders of the game itself" gap for the native Vulkan
path: a real model, a real translated VS/PS pair, a vertex buffer, a pipeline and a readback, end to end, on a
consumer GPU. It does not need the executable and it does not use a synthetic stand-in for either side.

The pieces it proves are the ones the D3D12 layer also relies on: ISG register → Vulkan vertex input location,
the 144-byte push-constant block, the translator's SPIR-V for a graphics pair, and `vk_harness`'s device,
pipeline and readback code. The same pair also draws through the port's D3D12 layer (section 5).

## 3. Reproduce it

```sh
git submodule update --init
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
DOTNET_ROLL_FORWARD=Major tools/rpf9dump/bin/Release/net8.0/rpf9dump models \
    "<package>/data.rpf" ~/leonida-work/assets

python -I tools/first_light.py --game "<package>" --work ~/leonida-work --list
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj ~/leonida-work/assets/data/models/<model>.obj --no-fit --run
# <model>.obj: 4,213 vertices, 7,968 triangles; the manifest records its SHA-256
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py -v
python -I tools/first_light.py --check --out ~/leonida-work/first_light/out
```

`--check` on an existing folder re-runs the harness through the same fresh path and `check_image` as the test; a
harness that exits 0 without writing, an all-opaque image and a swapped-U/V image all exit 1. The test has
36 cases. Three are package-gated and skip without `LEONIDA_FIRST_LIGHT_DIR`. The other 33 need no package:
the oracle fixtures, the CPU checks of `--camera` (section 6) and, from section 7, the layout derivation, the
constant-colour oracle and the prop rule. The prop rule is checked end to end on a box built in the test (six 7 x 7
grids with UVs, 588 triangles): manifest, `pick()`, the CLI line, `read_obj` and the coverage masks.

`first_light.py` carves `shaders.rpf` if the work folder does not have it, translates only the chosen pair, converts
the OBJ to the harness's mesh file (`u32 vertex_count, u32 index_count`, `(x,y,z,u,v) f32` vertices, `u32`
indices), writes `manifest.json` with the counts and FSSM hashes, and runs `vk_harness`. `--vs` and `--ps` take
other names from the carve index; `--no-fit` keeps a model that is already in NDC (the sky dome is). The test is
package-gated: without `LEONIDA_FIRST_LIGHT_DIR` it skips, and it writes its image to that folder.

## 4. The harness mode

`runtime/test/vk_harness.cpp` gained two environment options, inert when unset (the existing three-vertex and
mesh-shader paths are unchanged): `VKH_DRAW_MESH=<file>` with `VKH_VERTEX_LAYOUT=0:3f,1:2f` reads the binary mesh,
creates a vertex and an index buffer, and draws it with `vkCmdDrawIndexed`. The layout lists `location:components`
and only float attributes. The first-light pair above uses no other harness option: the VS reads no user SGPRs and
the PS has no bindings.

## 5. Through the D3D12 layer

`runtime/test/d3d12_first_light_test.cpp` (ctest `d3d12_first_light`) draws the same model and pair through the
Xbox-ABI device proxy over the Vulkan device, natively on Windows. `CreateGraphicsPipelineState` gets the game's
own containers and finds both stages by the FSSM hash inside them, in the index `first_light.py` writes next to
the SPIR-V (`vs.dxbc`, `ps.dxbc`, `index.json`); the input layout (POSITION float3, TEXCOORD float2) is matched
to the VS signature, the mesh goes through DEFAULT vertex and index buffers, and one `DrawIndexedInstanced` draws
it into a cleared R32G32B32A32 target read back with `CopyTextureRegion`.

D3D12 clip space is y-up and the layer flips the viewport, so `vk_harness` row r is row 127 − r here. On an RTX
3060 (Windows 11, driver 616.64, Vulkan SDK 1.4.363.1) the draw covers the same **12,804 of 16,384 pixels**, the
centre pixel is `(0.49756, 0.49634, 0.0, 1.0)` and the corner keeps the clear (−1). Against `vk_harness`'s
`image.raw`, 16,383 pixels are bit-exact and one is one f16 step apart (0.74219 here, 0.74170 there): the PS
exports with `v_cvt_pkrtz_f16_f32`, which rounds toward zero, so an f32 interpolant a few ulps lower under the
opposite viewport direction drops one step. `vk_harness` drawing the mesh mirrored in y reproduces the D3D12
image bit for bit. The test allows that one step and nothing else; it is clean under `LEONIDA_VK_VALIDATION=1`.

```sh
python -I tools/first_light.py ... --no-fit --run          # as in section 3; writes vs.dxbc, ps.dxbc, index.json
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out ctest --test-dir <build> -R d3d12_first_light -V
```

Without `LEONIDA_FIRST_LIGHT_DIR` the test prints SKIP and exits 77.

## 6. A camera (TEST-523)

`first_light.py --camera` draws the same model in perspective with a game VS that applies a world matrix and a
view-projection matrix of its own, read from two constant buffers. The PS is the one from section 1:

| stage | FSSM SHA-256 | what it does |
|---|---|---|
| VS | `e909a9e5a0096e00bc2b621ae3bdc96ac904f3047aed3917835774cbccb09716` | POSITION × world × view-projection to clip space (computed w); TEXCOORD to a varying |
| PS | `ed4147a5f7514947c2eb8d79255bd80f72761ac1ba2205dd811d3e050cf1b495` | as in section 1 |

The VS's scalar loads give its constant layout. The 32-bit address of a descriptor table is in s9 (the high half is
0). The V# at table + 0x20 points at the world rows: X, Y, Z and translation, one float4 each at 0x00–0x30, of
which the VS reads xyz. The V# at table + 0x60 points at a constant buffer whose view-projection rows sit at
0x160–0x19f. Both matrices multiply a row vector: `world = [x y z 1] W`, `clip = [world 1] VP`. The view-projection
block is common in the game: on 1.0.2.1, all 9 translated VS modules with exactly POSITION and TEXCOORD inputs and
a clip w that is not the constant 1.0 read 16 dwords at 0x160 of a constant buffer, as do 123 of the 254 VS
modules with a POSITION input and a computed w, and 1,266 of all 2,236 VS modules. That it is the view-projection
everywhere is a hypothesis; for this VS the arithmetic shows it.

The mesh keeps the model's own coordinates. The tool builds a world matrix that centres the model and scales its
bounding sphere to radius 1. The camera sits 3 radii away, `--yaw` degrees around Z (the game's up axis) from −Y
and `--pitch` degrees above the XY plane, with a vertical `--fov` (defaults 30, 25 and 50). The projection is
Vulkan's, with the near and far planes just outside the sphere, so depth clips no vertex. The tool writes the two
matrices, the two V#s (raw buffers, stride 0, sized to what the VS reads) and the table into `vk_harness`'s heap
(`VKH_HEAP`) and the table address into s9 (`VKH_SGPRS`). The manifest's `camera` block keeps both, so `--check`
and the test draw with the same constants. The prediction projects the same vertices with the same float32
matrices on the CPU and interpolates UV perspective-correctly, dividing each screen-space weight by its vertex's
clip w. The output goes to `<work>/first_light/camera`, leaving the orthographic folder for section 5.

On an Apple M4 (macOS, MoltenVK 1.4.2), with the section 1 model and the default camera at 128×128:

* The draw covers **6,851 of 16,384 pixels (0.418)**.
* The centre pixel matches the predicted UV `(0.8100, 0.3096)`, and so do the four U ≠ V probes, within the
  oracle's 2e-3. The corner keeps the clear value.
* On a full 32×32 grid of pixel centres, the predicted coverage agrees on all 1,024. The UV agrees within 2e-3 on
  all 428 covered ones, with the worst at 5.05e-4.
* The same draw without the constants covers nothing and is rejected. With the camera turned 2° in yaw, the
  centre pixel is off and the check is rejected too.
* The pair passes `spirv-val`, and `shader_link_check.py check` reports it compatible.

The model's triangles are small, so perspective-correct and affine UV differ by at most 1e-4 at the probes. The
image therefore does not tell the two apart; `CameraTests` in `tests/test_first_light.py` checks the
perspective-correct weights on one large triangle instead. Those CPU tests also read the constants back the way
the VS's loads do (table, V#, num_records, rows) and compare the result with the matrices the prediction uses.

```sh
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj ~/leonida-work/assets/data/models/<model>.obj --camera --run   # coverage window 0.2-0.99
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/camera python -I tests/test_first_light.py -v
```

## 7. A world prop from levels.rpf (TEST-525)

The same camera path draws a world prop from `levels.rpf` with a second game VS, `sga_dr_vs_pos_tex`, which applies
its own world and view-projection matrices too. **Measured** on an RTX 4090 (Windows 11, driver 617.42,
`vk_harness` built RelWithDebInfo with MSVC).

**The prop**, chosen by rule by `tools/first_light_prop.py` from the `rpf9dump models` export of `levels.rpf`
(`manifest.json` is several GB; the tool streams it): a static drawable (`IsFragment` and `HasBoneBindings` false),
`Meshes` = 1, 300 to 3,000 triangles, every bounding-box extent at least 5% of the longest (so no flat card or thin
rod), and an OBJ with UVs; of those, the triangle count closest to 1,000, ties by file name. Of the 149,811 items in
the manifest, 4,174 pass; four are at exactly 1,000. The pick is one mesh of **543 vertices and 1,000 triangles**,
(`first_light_prop.py` prints the pick and its SHA-256 on the user's machine; they are not recorded here). The census of `levels.rpf` layouts
(`tools/fvf_layouts.py`) shows its vertex streams hold quantized positions (16-bit formats) and packed normals and
UVs; the OBJ already holds the dequantized floats. They are fed as float3 and float2 (`VKH_VERTEX_LAYOUT=0:3f,1:2f`),
which is what the VS reads: in the game the input layout's format does that conversion, and `vk_harness` has no
packed-format path to exercise it.

**The pair** is `sga_dr_vs_pos_tex` with section 1's UV-writing PS: both pass `shader_link_check.py check`
("1 compatible, 0 incompatible"). The VS reads POSITION and TEXCOORD, exports a clip position and two parameters, and param0 is not the UV: it is four bytes
unpacked from a constant (`v_cvt_f32_ubyte0..3` of one SGPR, times 1/255), UV goes to param1. The PS writes
param0 as the colour, so the image is a constant colour, not a UV map. The task's alternative, `sga_dr_vs_pos_feff6b5f`
with `sga_dr_ps_feff6b5f`, has the same constant-colour shape (also param0 from `v_cvt_f32_ubyte` of an SGPR); it was
run as a cross-check. What the pair gives up: the UV is read from the vertex buffer but never reaches a pixel, so
section 6's UV oracle does not apply; the texture is TEST-524's.

**The constant layout is derived, not hard-coded.** `first_light.constant_layout` reads the VS's disassembly: the
descriptor table is the SGPR pair `s_load_dwordx4` loads from (the low half followed through `s_mov_b32` copies,
the high half required to be set to 0); the world V# is the one read as four float4 at 0x00, 0x10, 0x20 and 0x30;
the view-projection V# is the one with a 16-dword read; `v_cvt_f32_ubyte*` of an SGPR loaded from the world buffer
names the colour dword (it must be the w of a row, which the VS never reads as a matrix element). A VS that does not
fit is rejected with an error naming it (`VS_main_170e32e6`, which loads no V#, is). The three VSs checked:

| VS | table SGPR | world V# | view V# | view-projection | colour dword |
|---|---|---|---|---|---|
| `VS_TransformVT_e6ebb353` (section 6) | s9, copied to s2 | table + 0x20 | table + 0x60 | 0x160 | none |
| `sga_dr_vs_pos_tex_90997d1b` | s8 | table + 0x00 | table + 0x40 | 0x160 | world + 0x0C |
| `sga_dr_vs_pos_feff6b5f` | s8 | table + 0x00 | table + 0x20 | 0x160 | world + 0x0C |

`tests/test_first_light.py`'s `LayoutTests` derive the same table from hand-written listings, no package needed,
and its error cases (no table, high half not 0, missing world rows, missing view-projection, a V# not loaded from a
table, a colour outside a row's w).

**The oracle** is CPU-derived, like section 6: the prop is projected with the same float32 matrices, and a
rasterization of the projected triangles gives two masks over the 128×128 pixel centres. A pixel is *certain* when
its centre is more than 0.02 px inside some triangle and *possible* when it is not more than 0.02 px outside every
one (the band stands for the GPU's sub-pixel snap and top-left rule; 0.02 px assumes 8 `subPixelPrecisionBits`, as on the
RTX 4090, and a 4-bit device would need about 1/32 px). A pixel in the band must be exactly the clear value or exactly
the constant. The image must cover every certain pixel,
leave every non-possible pixel at the clear value, and carry the set constant **exactly** (all four channels,
bit-equal) on every covered pixel. The constant is packed `0xC0C08040`, so the expected pixel is the CPU's
`ubyte / 255` in float32 rounded toward zero to f16 (the PS's `v_cvt_pkrtz_f16_f32`): `(0.2509766, 0.5019531,
0.7529297, 0.7529297)`. The existing coverage window, clear corner and `check_image` NaN handling are unchanged.

| quantity | predicted | GPU |
|---|---|---|
| covered pixels (of 16,384) | 1,794 certain, 1,953 possible | **1,946 (0.119)** |
| pixels the oracle did not judge (edge band) | 159 | - |
| centre (64, 64) | the constant | the constant, exact |
| clear corner (0, 0) | clear (-1) | clear |
| covered pixels that differ from the constant | 0 | 0 |

The same prop with `VS_TransformVT_e6ebb353` and the UV oracle of section 6 also passes (1,946 covered, the centre UV
and the probes within 2e-3), and the `feff6b5f` pair covers the same 1,946 pixels with the same constant. Negative
controls: a draw without the constants makes the harness fail, a table SGPR of 0 covers nothing, and a wrong expected
channel is rejected. The pair also passes `spirv-val`.

**A constant that is not exact.** With alpha `0xFF`, the first run failed on 905 of the 1,946 covered pixels: alpha
was 0.99951 where the constant is 1.0. The translated PS interpolates the varying with barycentric weights that sum
to 1 within an ulp, so a constant 1.0 arrives as 1.0 - 1 ulp, and the round-toward-zero f16 pack turns that into the
next value down. Only a value on a half-float boundary is affected: the other three channels were exact on every
pixel. The default constant therefore avoids one; a PS that needs 1.0 exactly would need the interpolation of a
constant to be exact, which is an open question about the translator's lowering, not about this tool.

```sh
python -I tools/first_light_prop.py ~/leonida-work/assets/levels/models      # the pick and its SHA-256, printed locally
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj <the pick> --vs "#sga_dr_vs_pos_tex_90997d1b_Wrapped" --camera --min-coverage 0.05 --run
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/camera python -I tests/test_first_light.py -v
```

(The carve index names this VS with the leading `#`; `--list` shows the exact spelling. `--min-coverage` is lowered
because a prop is a small part of the frame; the mask check above is the oracle.)

## 8. Limits and next

* The camera draws (sections 6 and 7) run in `vk_harness` only. Through the D3D12 layer, the table SGPR and the table
  it points at have to come from root arguments, which `d3d12_first_light_test` does not set up yet.
* Some candidate VSs (one of the `Rage_TransformPosTex` pair among them) are carved under a renamed `~1` file,
  which `first_light.py` does not follow when copying the SPIR-V (#2419).
* No texture: the PS writes the UV map. A texture path needs a TTPF albedo, BCPack decode, a T# in the bindless
  set and the game's sampler.
* One model and one draw per run; the prop's quantized vertex formats and packed normals are not exercised (the OBJ
  already holds the floats), and the `levels.rpf` props with COLOUR or NORMAL inputs need the other `sga_dr_*`
  families and a vertex-input layout with more attributes.
* The game still cannot boot: the executable is not in the package (RT-5), so no call site selects these programs.

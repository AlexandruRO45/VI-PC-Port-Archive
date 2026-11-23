# InnerCoverage: implemented contracts and remaining work

This work has two independently reviewable parts: a runtime path for caller-owned SPIR-V libraries using
`FullyCoveredEXT`, and a synthetic underestimation-mask implementation for investigating devices without that
input. The latter is a test executable, **not an enabled runtime fallback**. Neither part establishes the Xbox
compiler's physical input ABI. Standalone devices keep `inner_coverage_abi=false`.

## Runtime contract

The explicit-library path requires the conservative-raster extension, the Tier 1 and Tier 2 geometric properties,
`fullyCoveredFragmentShaderInputVariable`, and the caller's `inner_coverage_abi` opt-in. It inspects actual SPIR-V
capability declarations before graphics module creation, even if optional index metadata is missing. Unsupported
FullyCovered modules are refused; PostDepthCoverage additionally requires its physical-device property.

The synthetic input converts a boolean to uint 0/1. This is deliberately an explicit library contract, not a
guessed Xbox VGPR initialization rule. See [the layer documentation](d3d12_vulkan.md).

The existing [PR #4290 RTX 3090 report](https://github.com/ItsNotRexed/Leonida/pull/4290#issuecomment-7088610406)
tested head `a55ba1fe` with driver 621.24.0.0, Vulkan 1.4.357: 48 positive Vulkan/native pairs, four focused tests,
and zero validation errors. That report applies to its stated head, not automatically to subsequent commits.

## Underestimation-mask implementation

`runtime/test/inner_coverage/` contains a standalone Vulkan implementation and synthetic shader generators.
It needs primitive underestimation, fragment stores/atomics and buffer device addresses. Geometry capture also
requires geometry shaders and vertex-pipeline stores/atomics. It does not need FullyCoveredEXT.

The algorithm is:

1. Allocate private storage indexed by pixel, emitted primitive and instance. The batched representation uses
   one bit per entry, packed into uint32 words; the mask shader uses atomic OR to avoid lost writes.
2. Clear scratch on the GPU outside the render pass, preserving unrelated counter/sentinel storage. A transfer
   barrier makes the clear visible to fragment reads/writes. Replayed batches preserve color/depth/stencil.
3. Rasterize the same geometry with conservative **underestimation**, a dedicated mask fragment shader, no color
   writes, and depth/stencil tests and writes disabled. The original fragment shader is never used for this pass.
4. Synchronize mask writes, then perform one conservative **overestimation** draw using a fragment shader that
   reads its private bit as the InnerCoverage value. Fragment effects, discard and depth/stencil occur here.

`runtime/src/inner_coverage_mask.h` defines checked bitset layout/address arithmetic, rejects zero dimensions,
32-bit shader-index overflow and insufficient budgets. The harness uses that layout for scratch capacity checks.
This component alone does not alter device capability reporting or turn on an emulation path.

### Generated geometry and side effects

Guest PrimitiveId cannot be the unique private mask key: a GS can emit multiple primitives with the same ID.
The fixture carries a separate flat emitted-primitive ID and preserves the guest-visible ID.

Executing a side-effecting VS/GS once per mask and final draw is incorrect. The capture fixture runs the original
VS/GS once with rasterization disabled, stores its output, then replays the stored vertices for both passes.
It checks a GS whose atomic return value changes geometry. The correct capture has VS=3, GS=1, PS=69; the
deliberately unsafe replay executes VS=6, GS=2 and changes the output.

This capture is **bounded to one synthetic input triangle and two emitted triangles**, with a hand-authored
record and replay shader. It is not a general SPIR-V stage instrumenter. Larger capture counts are rejected.
Mask storage begins at byte 4096 of the fixture's 64 KiB heap; captured vertices begin at byte 8192. Capacity
checks keep the mask below the capture region when capture is active. These offsets are a synthetic test
contract, not addresses reserved in a guest application.

### Regression matrix

The portable runner exercises 73 comparisons plus negative controls:

| Suite | Comparisons | Checks |
| --- | ---: | --- |
| Basic batched path | 8 | Early/late tests, discard, 1x/4x MSAA, geometric interior |
| Serial versus batched | 22 | Overlap order, independent primitive bits, instances and nonzero base instance |
| Semantic regressions | 27 | Sample-rate shading, per-sample counters, stencil, derivatives/helpers, shader depth |
| Scratch reuse | 6 | 2/3/5 batches, dirty allocation, preserved counter and sentinel |
| Generated geometry | 4 | Private emitted IDs, preserved guest PrimitiveId, both primitive orders |
| Capture once | 2 | Captured VS/GS effects and geometry at 1x/4x MSAA |
| Explicit helper guard | 4 | Non-helper atomic effects and derivative output |

Controls deliberately collapse primitive IDs, skip clearing, reuse stale masks, and execute side-effecting
geometry twice. A control succeeds only when the test observes the expected semantic difference. Validation
alone cannot detect those algorithmic errors.

The unguarded derivative/discard case counts 52 physical atomic increments on the local driver in both the
reference and mask paths, rather than 26. This is a baseline anomaly, **not proof of correct unguarded helper
side effects**. Explicit helper guards produce 26 on both paths; the anomaly is retained visibly in the suite.

Local environment: Windows build 26200, RX 5700 XT, driver 32.0.21045.5002, MSVC 19.44, validation layer 1.4.363.
This GPU reports primitiveUnderestimation=1 and fullyCoveredFragmentShaderInputVariable=0. The mask fixture
can run positively here; the runtime FullyCoveredEXT positive test must skip. Every fixture subprocess requires
both core and synchronization validation, and fails on validation/performance warnings or errors.

### Reproduce

Use Vulkan headers that declare `VK_KHR_pipeline_binary` (the dedicated harness retains the base harness's
pipeline-binary support), glslang, SPIRV-Tools, and an installed Khronos validation layer. No game files are used.

```powershell
cmake -S runtime/test/inner_coverage -B runtime/build-inner -DINNER_VULKAN_INCLUDE=<headers>/include
cmake --build runtime/build-inner --config Release
ctest --test-dir runtime/build-inner -C Release --output-on-failure
python -I runtime/test/inner_coverage/run.py --harness runtime/build-inner/Release/vk_inner_coverage.exe --glslang <path>/glslangValidator.exe --spirv-as <path>/spirv-as.exe
```

Set `VK_LAYER_PATH` if the layer is not installed globally; select an adapter with `LEONIDA_VK_DEVICE`.
Missing GPU properties or validation are reported as an unavailable prerequisite, not a successful comparison.
The default run deletes temporary synthetic artifacts; `--work <directory>` retains them. Do not run with
Python optimization, since assertions are the readback oracle. The CPU layout test is also registered in the
normal runtime CTest build on all platforms. The Vulkan fixture executable is also built by the normal runtime
build when Vulkan headers are available.

`--benchmark` adds reference, serial-mask and batched-mask GPU timestamps at 8/32 pixels with 1/16/64 triangles,
one warmup and five measurements per mode. Pipeline creation/readback are excluded; batch scratch clearing is
included. The consolidated fixture's local 32x32/64-triangle medians were 37.36 us reference, 742.36 us serial and 60.24 us batched
(8192 mask bytes). This measures synthetic work, not application performance; geometry capture is not timed.

## Public ABI evidence and its limits

The following identifiers are different namespaces and must not be conflated:

| Representation | InnerCoverage value |
| --- | --- |
| Shader feature information bit | 0x400 |
| Program-signature system value | 70 |
| DXIL semantic kind | 15 |
| DXIL opcode | 92 |

DXIL classifies this semantic as NotInSig. Missing ISG1 value 70 therefore does not establish absence of use.
The public [DXIL specification](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst)
and [DXC definitions](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DXIL/DxilConstants.h)
describe the logical input; they do not identify an Xbox VGPR. Direct3D false is zero; true has bit 0 set,
with other bits potentially undefined. Coverage and InnerCoverage are mutually exclusive.

[AMD LLPC](https://github.com/GPUOpen-Drivers/llpc/blob/40cb8d95ad8d6f7f1652e3fd47d39667594cce08/lgc/state/PalMetadata.cpp#L424)
selects INPUT_INNER_COVERAGE when its rasterizer state requests it. Public Mesa/PAL implementations support
the architecture of a coverage argument controlled by pipeline state. They do not demonstrate Xbox encoding.
ENA and ADDR must both be accounted for: enabled fields and reserved fields are not interchangeable, and
preceding input fields have different VGPR widths. A fixed coverage VGPR is unjustified.

Earlier local synthetic PC pipeline-binary experiments varied input masks and selectors. Coverage appeared
at v2 with a barycentric pair, v4 with position XY added, and v5 when Z was reserved in ADDR without enabling
it in ENA. Across 27 geometry/layout controls, the PC behavior was consistent with reserved positions.
Selecting inner coverage while using overestimation still produced false geometric positives; underestimation
was needed for the measured mask. These are PC compiler/driver observations, not a recovered Xbox ABI.

Public GDK samples also show alternate compiler and precompiled-pipeline paths, including separately loaded
metadata/instructions. Their existence does not establish an FSSM encoding. Header census failures cannot rule
out encoded fields, trailing metadata or pipeline-state information. The metadata census is tracked separately
in [PR #4498](https://github.com/ItsNotRexed/Leonida/pull/4498).

## Handoff: work required before automatic Xbox or general AMD Tier 3

| Remaining work | Required evidence / acceptance condition |
| --- | --- |
| Identify Xbox input-state metadata | A legally shareable synthetic program and independently known ENA, ADDR and coverage selector, varying preceding inputs to expose reservations |
| Confirm VGPR and value encoding | A register-use observation distinguishing false/true, input placement and helper/MSAA behavior; PC-only DXIL/GLSL tests are insufficient |
| General stage instrumentation | Capture/replay all live outputs, topology, instances, clipping/culling and guest IDs; execute effects exactly once; bound output growth and refuse unsupported stages |
| Fragment lowering | Rewrite only the proven logical InnerCoverage input, preserving early/late tests, derivatives, discard, sample masks and physical side effects |
| Runtime resource integration | Private allocation/lifetime, budget/chunking, predication, render-pass transitions and command-list replay without aliasing guest storage |
| Broader raster contract | Validate viewports/scissors, layers, clipping, tessellation/mesh, transform feedback, queries and 8x or other supported MSAA configurations |
| Independent hardware validation | Repeat exact-head comparisons on additional AMD/NVIDIA/Intel devices, comparing geometric guarantees and native D3D12 where available |

Until these are demonstrated, do not report Tier 3 merely because primitiveUnderestimation is supported, do not
run arbitrary original geometry stages twice, and do not enable a guessed Xbox input mapping. The existing
explicit-library native-property route can be reviewed and integrated independently of this remaining work.

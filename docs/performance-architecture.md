# Performance Architecture: Shader Translation, Pipeline Management, and CPU–GPU Synchronization

This review covers ZTRX main at [`90b529f878a470fec9bf07b7be316bc769a12232`][baseline] and contribution status as of November 25, 2026. The most useful first comparison is a warm-cache trace of the default submission path and the existing asynchronous experiments. That comparison can show where CPU preparation, queue execution, synchronization, and memory transfers constrain sustained performance. Separately, cold-cache and first-use traces can show where shader translation and host pipeline construction interrupt otherwise steady rendering.

**Evidence.** Source-confirmed describes the pinned implementation. Repository-reported measurements come from project instrumentation or contributor reports. Literature-supported describes a public API or hardware contract. Pending contributions are open at the review date. Architectural hypotheses are measurable proposals. Research directions require additional design or correctness work. This review adds no gameplay measurements.

## Execution model

```text
Xbox offline      HLSL → compiler IR → RDNA 2 code generation → console GPU
Windows D3D12     HLSL → DXIL → driver pipeline compilation → host GPU code
ZTRX GPU path     console code → decode → SSA → recover control flow and resources
                  → emit DXIL → host D3D12 PSO → NVIDIA GPU code
ZTRX CPU path     mapped x86-64 title code → native CPU execution
                  → console API bindings and host Windows services
```

**Literature-supported and source-confirmed.** Microsoft's public [Xbox HLSL compilation sample][gdk] demonstrates offline console shader compilation and optional graphics PSO serialization. It establishes a supported workflow. The title's exact toolchain and the full behavior of the console runtime remain undocumented here. Windows DXIL is an LLVM-derived intermediate language consumed by graphics drivers. ZTRX receives containers at title pipeline creation, including containers supplied from memory, and works backward from native GPU instructions. Reconstructing lane participation, resource bindings and shader interfaces is additional work beyond the host driver's ordinary DXIL consumption. The loader maps the title's native x86-64 code and binds imports, while the graphics layer translates the GPU ISA and adapts the console API. [DXIL specification][dxil] · [ZTRX architecture][arch] · [Runtime shader emitter][emitter]

## Submission and memory visibility

**Source-confirmed.** Logical title queues feed a shared host D3D12 direct queue. The queue layer already holds and splits work at guest fence or memory waits. In the default non-timing path, submission holds a shared lock across preparation, batched command-list submission, host fence wait and copy-back. Completion writes are issued after required landing work. Present can force an unresolved guest queue wait, so preserving ordering here involves both explicit queue dependencies and CPU-visible memory. Runtime batching already exists. Its interaction with lock scope, logical queue contention and guest-visible completion is the question to measure. [Host queue][queue-host] · [Guest dependency handling][queue-waits] · [Submit and completion][submit] · [Present waits][present-waits] · [Guest fence behavior][fence]

**Repository-reported measurement.** The comment for run 86 records 446 ms per frame, including 134 ms CPU preparation, 194 ms waiting and 78 ms copy-back. Those listed components account for 406 ms. The timed configuration used synchronous submission and per-list waits. The 194 ms is elapsed CPU-side waiting in that instrumented path. Hardware execution time and queue idle time require separate observations, and run 86 gives no ordinary-path speedup estimate. [Run 86 instrumentation][run86]

| Existing experiment | Mechanism and constraints |
| --- | --- |
| `VICTORY_LATE_WAIT` | Stages 1–3 progressively defer the host wait around submission and present. Outstanding work still settles at relevant dependency and visibility points. Stage 3 can queue the present copy separately and retain its resources. Instrumented GPU timing disables this path, and captures, probes, and resize can require synchronization. [Stages and settle logic][late] · [Present copy and retirement][present] |
| `VICTORY_RELAY` | Stage 1 moves ordered CPU landing work to a separate thread. Stage 2 narrows waiting around overlapping declared memory and bounds outstanding work. Relay selects late-wait stage 3 unless explicitly overridden, and has timing and snapshot restrictions. Declared memory coverage remains central to correctness. [Relay][relay] · [Shadow options][shadow-options] |
| `VICTORY_WIDE_BACK` | Opt-in CPU parallelism for snapshot copies before submission and declared-range comparisons after completion. These tasks compete for CPU and memory bandwidth. Relay separately controls asynchronous landing into title memory. [Copy-back workers][wide] · [Copy-back dispatch][snapshots] |

**Architectural hypothesis.** In a warm library run, measure whether narrower visibility waits and less time under the submission lock recover more critical-path time than additional cache coverage. Trace dependency edges by logical queue, overlapping resource range, alias and CPU consumer. First test bounded overlap on the existing direct queue. Increase batch size or flight depth only where traces show useful overlap and where command allocators, descriptor slices, snapshots, aliased allocations and back buffers survive their last GPU use. Confirm signal/wait ordering, present pacing and resize behavior at each step. D3D12 multi-engine support is relevant if independent copy or compute work can later be demonstrated. [Multi-engine synchronization][multi-engine] · [Fence-based lifetimes][fence-lifetime]

**Literature-supported and source-confirmed.** Xbox Series X and S share physical GDDR6 between CPU and GPU within their respective SoCs, with published bandwidth regions. A conventional discrete NVIDIA host separates system RAM and VRAM. Software cannot reproduce the console's physical unified memory architecture. The runtime mirrors selected guest ranges, uses CPU write watch or page guards where available, uploads stale ranges, and returns GPU writes when required. Large eligible shadows can already reside in GPU-local memory. GPU-side comparisons produce 64 KiB changed-piece information for declared ranges, with a local-memory budget check. Unknown or untracked accesses limit how aggressively this can be used. Some texture aliases have a validated buffer-mediated path for matching layouts, while changed tiled textures can still incur full detiling. [Xbox Series X][series-x] · [Xbox Series S][series-s] · [Shadow tracking and admission][shadow] · [Texture refresh][texture] · [Alias path][alias]

**Repository-reported measurement.** A former snapshot strategy repeatedly allocated and populated snapshots of overlapping views of a 174 MB heap, producing about 2.3 million page faults per world frame according to the source comment. These were attributed to touching fresh CPU snapshot memory. The current code pools snapshot buffers and captures bytes shared by overlapping ranges once for a snapshot. Measure residual faults and overlap on the current path before prioritizing more snapshot work. [Snapshot implementation and historical observation][snapshots]

**Research directions.** Refine the existing GPU dirty-piece mechanism by measuring declared versus changed and transferred bytes, then test range coalescing or finer tracking where it reduces work. Extend GPU-local residency where producers and consumers stay on the GPU. Defer CPU readback only to a proven visibility point, with conservative handling for unknown descriptors, aliasing and CPU polling. Track both local and nonlocal DXGI budgets alongside private commit, working set and page faults. These changes may reduce transfers or CPU landing time while increasing metadata, tail latency or residency pressure. [Readback visibility][readback] · [DXGI video memory budgets][residency]

The existing `VICTORY_INPLACE` experiment uses eligible aligned, private committed address ranges at level 1 and includes smaller title buffers at level 2. Address-backed heaps pin pages, which rules out decommit, free or protection changes while the heap lives. File-mapping probes have restrictions, and the existing reserved pool views are excluded. The source instructs against combining level 2 with late wait because the in-place writes bypass shadow dependency tracking. Separately, compare ordinary staging against D3D12 GPU Upload Heaps for selected CPU-write/GPU-read buffers. Feature support and local budget are prerequisites. ReBAR and mapped GPU Upload Heaps still carry CPU writes across PCIe and CPU reads from VRAM can be expensive. [In-place implementation][inplace] · [GPU Upload Heaps specification][gpu-upload] · [NVIDIA upload heap analysis][nvidia-upload]

## Translation quality and Wave64 compatibility

**Source-confirmed.** The translator builds register-dword SSA with explicit EXEC/SCC and preserved masked values, reconstructs reducible control flow and per-lane predicates, traces descriptors and addresses to root parameters or heaps, and raises shader interfaces for host emission. Symbolic wave analysis recognizes reductions and prefixes. Specific lane-local bit algorithms and uniform branches can already be simplified. DXIL cleanup includes dead-code elimination, integer folding, dominance-based expression reuse, branch facts and same-block redundant load removal with effect boundaries. These are comparable to standard compiler passes applied after binary semantic reconstruction. [SSA][ir] · [Control flow][structure] · [Resource provenance][resource] · [Symbolic wave recovery][wave] · [Uniform threading][threading] · [Lane-local bit analysis][own-bits] · [Cleanup passes][opt] · [LLVM passes][llvm]

**Literature-supported and source-confirmed.** A guest Wave64 may span two NVIDIA 32-lane warps. Wave-local collectives alone cannot reproduce cross-half lane exchange. The emitter attempts exact forms, including independent halves where it can prove independence and shared-memory exchange at suitable group points. Fallback paths can use approximate half-wave behavior. The opt-in wide Wave64 path on 32-lane hosts has a recorded device-loss history involving group barriers in branches or loops. At the pinned revision group-sync mode 2 is the default. [AMD RDNA 2 ISA][rdna] · [NVIDIA warp model][warp] · [Wave selection and fallback][wave-select] · [Lowering][lower] · [Group-sync modes][group-sync] · [Device-loss fix][device-fix]

**Pending contribution.** The [ZTRX #20][pr20] contribution from rivaldodev adapts [VIctory #1133][victory1133] and [victory-fork #608][fork608]. It reserves a consistent final null descriptor slot on capped heaps and proposes automatic mode 3. For classified divergent title barriers on a known narrower host wave it drops the group rendezvous while retaining memory fences. Uniform barriers, translation-added synchronization and explicit overrides keep their behavior. The PR also keys translations on host wave width and descriptor heap cap and reports synthetic NVIDIA coverage. D3D12 group synchronization requires whole-group convergence, and fence-only lowering loses rendezvous semantics. Cross-wave producer–consumer behavior and full title validation remain open. God489's [#10][pr10] expands VALU coverage with synthetic hardware/WARP comparisons, and [#11][pr11] changes queue reuse in the GPU test harness. Both address test or translation coverage. [Group barrier contract][group-barrier]

**Research direction.** Prefer recoveries with a demonstrated equivalence, starting with uniform-value propagation, redundant EXEC-mask simplification and recognized collectives that can reduce shared-memory traffic. Profile register usage, spills, occupancy, shared memory and barrier cost on representative translated kernels after validation. Use synthetic cross-half exchanges, sparse EXEC masks, nonconverging control flow, helper lanes, and exact integer and memory checks. Specify floating-point rounding, NaN/denormal behavior and error bounds before tolerating differences. Lowering loop limits can truncate useful work. Changing the guest wave size or removing a barrier requires proof of preserved lane selection and memory visibility. [Loop diagnostics and barrier modes][group-sync] · [AMD occupancy guidance][occupancy]

## Pipeline preparation, scheduling and identity

**Source-confirmed.** Runtime shader containers are deduplicated by full content hash and each shared shader memoizes translation. The persistent translation cache uses a content-addressed converter. The separate D3D12 library keys host PSOs by pipeline kind, root signature, translated bytecode and state, with driver validation and fallback creation. Its mapped blob is retained for the library's lifetime. The early list stores bounded keys for requested title pipelines, including first uses that ran out of wait budget. A subsequent launch starts matching jobs when the title recreates the objects. Independently replaying a PSO would require recording its description in addition to these keys. [Runtime shader objects][shader-objects] · [Converter cache][cache] · [Pipeline library design][library] · [Early requests][early] · [Prewarm commit][prewarm-commit] · [D3D12 pipeline library contract][d3d-library]

**Source-confirmed.** `victory translate` walks accessible inputs, deduplicates the native-code-bearing container part, emits and reports translated shaders, and can validate DXIL or attempt diagnostic host pipelines. The graphics checks can use stand-ins or explicitly selected stage pairs. At this revision it does not write the runtime translation cache or reconstruct complete title PSOs into the runtime library. Its census dedup key should not substitute for a complete runtime translation identity. [Translate file walk][translate-walk] · [Container deduplication][translate-dedup] · [Validation and diagnostic PSOs][translate-checks]

**Research direction.** Explore a local preparation workflow in two steps. First, discover accessible containers and populate the existing translation cache through a shared compile, validation and serialization path using the same effective target settings as runtime. Second, record complete normalized pipeline descriptions from representative local playthroughs and replay those descriptions with the same host pipeline constructor. Serialize compatible libraries, and retain a runtime fallback for misses. A graphics description needs stage associations, the root signature and effective fixed state, including render-target/depth formats, samples and input or stream state where relevant. Static discovery cannot cover containers or permutations only encountered at runtime, and known shaders alone cannot establish every pipeline combination. All derived artifacts belong outside the repository. [Asset policy](asset-policy.md) · [Host PSO construction][pso] · [Epic PSO precaching][epic-precache] · [Epic bundled PSO caches][epic-bundle] · [Valve Fossilize][fossilize]

Epic's automatic precaching and bundled caches illustrate two ways to obtain candidate PSOs, from loaded engine state and captured usage. Fossilize illustrates capture and replay of Vulkan creation descriptions to warm a target-device cache. D3D12 libraries serialize compatible driver artifacts after host creation. The closed, unmerged [victory-fork #282][fork282] by t4l03 already prototyped local runtime-cache population and pipeline preparation, reporting warm-up gains alongside persistent slow heavy scenes. Repository [issue #16][issue16] specifically points to it. Treat that work as precedent and verify current behavior independently before adapting it. [D3D12 library compatibility][d3d-create-library]

**Source-confirmed.** The runtime pool starts half the logical processors, clamped to 2–8 workers, below normal thread priority. First-use requests can promote prior-run early jobs. The default shared wait budget is **500 ms per title frame** across concurrent callers. It is a maximum allowance for waiting and roughly 30 budgets of a 60 Hz frame. On expiry, a pending draw or dispatch can be skipped. A one-time compute pass may initialize or update persistent state that later work consumes, so compiling it in time for a later frame cannot restore an initialization that never ran. [Budget and accounting][budget] · [Queue promotion and workers][workers] · [Dropped work semantics][make]

**Architectural hypothesis.** Measure scheduling policies by phase. Initial preparation can use bounded parallel batches with a commit and residency cap. Loading screens can prioritize observed upcoming work, and menus can consume spare capacity while protecting UI and audio. During gameplay, track critical compute and its consumers before considering any deferral. Classify a draw as optional only with evidence that its omission cannot change later rendering or state. Admission should respond to queue depth, available commit, compile latency, contention in the driver or pipeline library, and title render and audio scheduling. Below-normal priority alone cannot guarantee audio deadlines when the process or driver is under CPU or memory pressure. The closed, unmerged [#6][pr6] already proposed a universal 24 ms budget alongside untested async defaults and a parallel dirty-piece comparator.

**Source-confirmed and architectural hypothesis.** Runtime shader identity includes an executable-wide fingerprint and selected translation settings. An executable relink can invalidate translations after an unrelated edit. At this baseline, host wave width, probed descriptor cap and the effective Wave64 local fallback are absent from the key, despite their relevance to emitted code or fallback selection. #20 covers the first two. Related victory-fork proposals cover [translator identity and early scheduling][fork147], [host wave width][fork35] and [local wave mode][fork199]. If finer versioning is pursued, first audit and normalize every output-affecting input, including decoder, IR recovery, translation ABI, serialized schema, optimization settings and host shader capabilities. Native PSO identity separately requires normalized pipeline descriptions and adapter/driver compatibility. Keep rejection and rebuild paths and test cache reuse across app-only, translator, device and driver changes. [Runtime translation key][shader-key] · [Wave fallback][wave-select] · [Host name and load][library-key]

Each title pipeline has its own `Pending` and `OnceLock`. The library serializes loads by name, but a miss compiles outside that lock, so separately created equivalent requests can miss concurrently and compile twice. Count concurrent misses for identical full host descriptions before testing shared in-flight work scoped to the host device. Keep unrelated compilations concurrent, share completion only for equivalent requests, and make failure and retry behavior explicit. The source records a historical convoy under a global library lock. [Pending identity][pending] · [Library lock history][library] · [Miss and store][library-key]

## API options and ray-tracing correctness

| Technology | Applicability at this baseline |
| --- | --- |
| D3D12 Pipeline Libraries | Shipping and already used here. Highest immediate value is measuring real load latency, misses, saves and mapped-blob memory pressure before changing the cache. [Microsoft API contract][d3d-library] |
| Partial Graphics Programs | Retail in Agility SDK 1.620 since October 28, 2026. It separates reusable pre-raster and pixel shader portions. Microsoft's retail announcement lists Radeon RX 7000/9000 support and other vendors as planned. Treat NVIDIA adoption as conditional on a feature probe and driver support. [Retail announcement][partial-retail] · [Specification][partial] |
| DirectX Advanced Shader Delivery | Available on supported Windows 11 hardware per Microsoft's October 21 announcement. A title/store workflow captures state objects and delivers precompiled data selected for adapter family and compiler ABI. The announcement lists the Xbox PC app/Microsoft Store and describes Steam and Epic rollout as forthcoming. Compatibility-layer eligibility, installation and pipeline coverage require verification. Misses retain local compiler fallback. [Availability][asd] · [Registration API][asd-api] · [Explicit PSDBs][psdb] |
| Vulkan pipeline extensions | `VK_EXT_graphics_pipeline_library` supports separate parts and linking. `VK_EXT_shader_object` exposes a shader-object and dynamic-state model. `VK_KHR_pipeline_binary` supports application-managed per-pipeline binaries with compatibility keys. These are published extensions with device-dependent support and useful design precedents. [Graphics pipeline library][vk-gpl] · [Shader object][vk-so] · [Pipeline binary][vk-bin] |

Keeping D3D12 primary preserves a close fit for DXIL, root signatures, descriptor heaps and the title's pipeline and synchronization model. ZTRX [#2][pr2] by LazyHatGuy proposes running the Windows build under Proton, alongside its [#1][pr1] Linux tooling work. That work exercises a host compatibility path and introduces no separate ZTRX Vulkan backend. The Wine report in [#23][issue23] records a host failure that belongs in a separate test population. Wilu222's [#21][pr21] covers app-local Agility SDK loading on Windows 10 and relates to [VIctory #463][victory463]. It addresses runtime availability, with no evidence of a pipeline construction speedup.

**Source-confirmed and repository-reported.** The title's ray traversal is implemented in compute over guest BVH nodes. AMD BVH instructions accelerate individual intersection tests. This fork lowers those to software DXIL node tests and gates the experimental builder on complete availability of the pipelines it has observed. The ray-tracing note reports synthetic GPU comparisons with the project's CPU reference and incomplete translation or host acceptance. The path is off by default and end-to-end lighting remains incomplete. No measured cost of that disabled path explains the reported frame times. [Ray tracing status](fork-ray-tracing.md) · [Software intersection emitter][intersect] · [Builder gate][bvh-gate]

**Research direction.** DXR consumes driver-managed acceleration structures and exposes traversal through TraceRay or RayQuery. It does not expose a native AMD node layout or per-node intersection interface for direct substitution with NVIDIA RT Cores. A viable investigation needs recovered geometry and instances, update/refit rules, and differential checks for candidate ordering, intersection precision, flags and dynamic content. Establish correct lighting and resource state on the software path first. [DXR specification][dxr]

## Measurement and correctness gates

Use the same revision, Windows build, adapter, driver, title settings, save or scripted input, and scene interval. Compare first-use and steady-state frames separately. Preserve a known warm cache, use independent working copies, and record driver-cache state and run order. Repeat identical runs and report sample counts. Capture separately when instrumentation changes submission semantics. Yoshi-233's [#22][pr22] documents direct diagnostics launch because the normal launcher clears capture and scripted-input variables ([#19][issue19]). Issue [#8][issue8] and [#16][issue16] provide useful scene and stability context with different configurations.

| Controlled comparison | What it can establish |
| --- | --- |
| Cold runtime shader cache and empty PSO library | Translation misses and first-use costs, with driver cache state recorded separately. |
| Warm DXIL cache, empty PSO library | Residual host construction and driver compilation costs. Confirm translation hits and distinguish create from library load. |
| Warm DXIL cache and compatible pipeline library | Sustained execution and memory costs, with remaining PSO misses counted. |
| Same warm state, default versus one async change | Overlap, fence wait, present pacing and any sign or timing of incorrect visibility. |
| Same warm state, one memory strategy at a time | Snapshot, alias, upload/readback and residency effects independently of pipeline warm-up. |

Record CPU frame and displayed-frame interval median/P95/P99, CPU preparation and fence blocking, CPU/GPU utilization, hardware GPU execution and idle intervals, queue depth, title render/audio thread scheduling, shader and PSO hit rates, library load versus creation latency, upload/readback and snapshot byte counts, RAM/private commit and VRAM budgets, descriptor overhead and page faults. Record skipped draws and dispatches, pipeline refusals, device removals, graphical regressions and scene progression. The existing host frame line provides CPU phase fields and transfer counters. Its fence-based activity estimate has a separate meaning from hardware utilization. Use PIX timing captures, GPUView/ETW and Nsight Graphics GPU Trace to cross-check queue overlap and GPU counters. Use PIX GPU captures to inspect resource state and work ordering. Add missing counters only in a future instrumentation change. [Existing host instrumentation][host-metrics] · [Host activity estimate][host-activity] · [PIX timing][pix] · [PIX GPU capture][pix-capture] · [GPUView][gpuview] · [Nsight GPU Trace][nsight]

Require stable synchronization and device health, correct presentation, preserved compute side effects and CPU-visible readbacks, and matching images or buffer results within a predeclared tolerance. Exercise resize, scene transitions, streaming and repeated launches, and validate on a narrow-wave NVIDIA adapter. Compare against synthetic references for features whose baseline output is already incomplete. Keep capture data and generated cache artifacts local.

## Proposed investigation order

| Priority | Expected type of benefit | Prerequisite, risk and validation |
| --- | --- | --- |
| 0 · Measurement and correctness | Identify compilation stalls and sustained critical-path costs independently. | Reproducible scene and cache fixture, correlated timestamps and correctness checks. Quantify instrumentation overhead and repeat cold and warm runs. |
| 1 · CPU–GPU execution | Reduce serialized CPU waits and improve overlap or pacing. | Trace dependency and lifetime edges first. Test default, late wait and relay in warm runs, with bounded flight depth and completion, resize and presentation checks. |
| 2 · Memory | Reduce snapshot, CPU landing and PCIe traffic where data stays resident or untouched. | Quantify dirty ranges and residency. Check alias and CPU-consumer behavior and stress commit and local budgets during scene changes. |
| 3 · Translation | Improve fidelity and reduce cost of selected wave and mask patterns. | Require equivalence tests before hardware timing. Track precision, cross-half behavior, spills, occupancy and barriers. |
| 4 · Pipeline preparation | Reduce cold startup or first-use stalls and preserve compatible cache entries. | Reuse runtime identities and captured descriptions. Validate misses, one-time compute, memory limits, worker interference and driver changes. |
| 5 · Longer-term research | Evaluate semantics-preserving DXR mapping and reusable pipeline parts. | Require geometry and traversal evidence or feature/driver support. Compare correctness and costs per workload before expanding scope. |

This order is an investigation proposal. Reorder it when representative measurements identify a different bottleneck.

### Primary reference index

- Shader execution and compiler contracts. [Xbox offline HLSL sample][gdk] · [DXIL][dxil] · [RDNA 2 ISA][rdna] · [NVIDIA warps][warp] · [LLVM passes][llvm].
- Memory and execution. [D3D12 queue synchronization][multi-engine] · [Fence-based management][fence-lifetime] · [Readback][readback] · [GPU Upload Heaps][gpu-upload].
- Pipeline preparation. [D3D12 Pipeline Libraries][d3d-library] · [Unreal PSO precaching][epic-precache] · [Unreal bundled caches][epic-bundle] · [Fossilize][fossilize].
- API extensions. [Partial Graphics Programs][partial] · [Shader cache registration][asd-api] · [DXR][dxr] · [Vulkan graphics pipeline library][vk-gpl] · [Vulkan shader object][vk-so] · [Vulkan pipeline binary][vk-bin].

[alias]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/resource.rs#L1179-L1224
[arch]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/docs/architecture.md#L7-L25
[asd]: https://devblogs.microsoft.com/directx/advanced-shader-delivery-is-available-now-across-hardware/
[asd-api]: https://microsoft.github.io/DirectX-Specs/d3d/ShaderCacheRegistrationAPI.html
[baseline]: https://github.com/Zettrox/ztrx/commit/90b529f878a470fec9bf07b7be316bc769a12232
[budget]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/make.rs#L94-L206
[bvh-gate]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline.rs#L261-L299
[cache]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-pipeline/src/cache.rs#L9-L74
[d3d-create-library]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device1-createpipelinelibrary
[d3d-library]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12pipelinelibrary
[device-fix]: https://github.com/Zettrox/ztrx/commit/e9765a73a68a75536b1c10ad5ef746d7f6ec0bcc
[dxil]: https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst#introduction
[dxr]: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html
[early]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/make.rs#L118-L147
[emitter]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/mod.rs#L270-L297
[epic-bundle]: https://dev.epicgames.com/documentation/unreal-engine/manually-creating-bundled-pso-caches-in-unreal-engine
[epic-precache]: https://dev.epicgames.com/documentation/unreal-engine/pso-precaching-for-unreal-engine
[fence]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/fence.rs#L78-L110
[fence-lifetime]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/fence-based-resource-management
[fork147]: https://github.com/fxceless/victory-fork/pull/147
[fork199]: https://github.com/fxceless/victory-fork/pull/199
[fork282]: https://github.com/fxceless/victory-fork/pull/282
[fork35]: https://github.com/fxceless/victory-fork/pull/35
[fork608]: https://github.com/fxceless/victory-fork/pull/608
[fossilize]: https://github.com/ValveSoftware/Fossilize
[gdk]: https://learn.microsoft.com/en-us/samples/microsoft/xbox-gdk-samples/hlslcompile/
[gpu-upload]: https://microsoft.github.io/DirectX-Specs/d3d/D3D12GPUUploadHeaps.html
[gpuview]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/using-gpuview
[group-sync]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/mod.rs#L124-L169
[group-barrier]: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/groupmemorybarrierwithgroupsync
[host-activity]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L297-L308
[host-metrics]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L5658-L5688
[inplace]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/inplace.rs#L9-L156
[intersect]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/intersect.rs#L1-L92
[ir]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/ir.rs#L1-L16
[issue16]: https://github.com/Zettrox/ztrx/issues/16
[issue19]: https://github.com/Zettrox/ztrx/issues/19
[issue23]: https://github.com/Zettrox/ztrx/issues/23
[issue8]: https://github.com/Zettrox/ztrx/issues/8
[late]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L1623-L1728
[library]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/library.rs#L1-L46
[library-key]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/library.rs#L498-L563
[llvm]: https://llvm.org/docs/Passes.html
[lower]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/lower.rs#L1272-L1334
[make]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/make.rs#L1-L21
[multi-engine]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization
[nsight]: https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-activity.html
[nvidia-upload]: https://developer.nvidia.com/blog/optimizing-dx12-resource-uploads-to-the-gpu-using-gpu-upload-heaps/
[occupancy]: https://gpuopen.com/learn/occupancy-explained/
[opt]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/bitcode/opt.rs#L1-L35
[own-bits]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/own_bits.rs#L16-L39
[partial]: https://microsoft.github.io/DirectX-Specs/d3d/PartialGraphicsPrograms.html
[partial-retail]: https://devblogs.microsoft.com/directx/agilitysdk-1-620-retail-and-1-722-preview/
[pending]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/make.rs#L27-L70
[pix]: https://learn.microsoft.com/en-us/windows/pix/timing-captures/timing-captures-overview
[pix-capture]: https://devblogs.microsoft.com/pix/gpu-captures/
[pr1]: https://github.com/Zettrox/ztrx/pull/1
[pr10]: https://github.com/Zettrox/ztrx/pull/10
[pr11]: https://github.com/Zettrox/ztrx/pull/11
[pr2]: https://github.com/Zettrox/ztrx/pull/2
[pr20]: https://github.com/Zettrox/ztrx/pull/20
[pr21]: https://github.com/Zettrox/ztrx/pull/21
[pr22]: https://github.com/Zettrox/ztrx/pull/22
[pr6]: https://github.com/Zettrox/ztrx/pull/6
[present]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L5890-L5975
[present-waits]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/queue.rs#L757-L776
[prewarm-commit]: https://github.com/Zettrox/ztrx/commit/d279e69ae999aedfd3d463b0c45c8fd125c1ad7b
[psdb]: https://microsoft.github.io/DirectX-Specs/d3d/ExplicitPSDBs.html
[pso]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/host.rs#L498-L665
[queue-host]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L565-L595
[queue-waits]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/queue.rs#L67-L125
[rdna]: https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture
[readback]: https://learn.microsoft.com/en-us/windows/win32/direct3d12/readback-data-using-heaps
[relay]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L1733-L1778
[residency]: https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info
[resource]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/resource.rs#L1-L26
[run86]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L1623-L1639
[series-s]: https://www.xbox.com/en-US/consoles/xbox-series-s
[series-x]: https://www.xbox.com/en-US/consoles/xbox-series-x
[shader-key]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/shader_cache.rs#L34-L117
[shader-objects]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline.rs#L59-L173
[shadow]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/shadow.rs#L1-L223
[shadow-options]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/shadow.rs#L457-L474
[snapshots]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/shadow.rs#L1749-L1849
[structure]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/structure.rs#L1-L24
[submit]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/host.rs#L3851-L4137
[texture]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/texture.rs#L1-L13
[threading]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/threading.rs#L1-L72
[translate-checks]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/apps/victory/src/translate.rs#L510-L568
[translate-dedup]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/apps/victory/src/translate.rs#L309-L395
[translate-walk]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/apps/victory/src/translate.rs#L155-L230
[victory1133]: https://github.com/Marvisak/VIctory/pull/1133
[victory463]: https://github.com/Marvisak/VIctory/pull/463
[vk-bin]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_pipeline_binary.html
[vk-gpl]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_graphics_pipeline_library.html
[vk-so]: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_object.html
[warp]: https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
[wave]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/wave.rs#L1-L15
[wave-select]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-shader/src/emit/mod.rs#L312-L463
[wide]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/shadow.rs#L938-L962
[workers]: https://github.com/Zettrox/ztrx/blob/90b529f878a470fec9bf07b7be316bc769a12232/crates/victory-gpu/src/pipeline/make.rs#L220-L327

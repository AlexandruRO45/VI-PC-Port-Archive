# Experiment notes and reproduction commands

Run commands from the repository root. Start with the [project overview](../README.md) and [contribution guide](../CONTRIBUTING.md). Package-specific observations below used local inputs. Community regression tests use fixtures built from this repository.

## Local tools

For optional static dependency inspection, build Microsoft's pinned version of xbdepends from PowerShell with x64 Visual Studio Build Tools, CMake and the Windows SDK.

```powershell
.\tools\setup-xbdepends.ps1
```

Load the toolchain in each new development shell, then configure, build and test.

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

The tests exercise desktop D3D12 root signature serialization and deserialization. Use the resulting test binary to try the static inspection tools.

```powershell
xbdepends -v .\out\build\dev\d3d12_sdk_check.exe
dumpbin /imports .\out\build\dev\d3d12_sdk_check.exe
```

Build outputs and downloaded tool sources live under `out/`. [Microsoft xbdepends documentation](https://learn.microsoft.com/en-us/samples/microsoft/xbox-gdk-samples/xbdepends/)

## Offline CPU leaf recompilation

With the Python-enabled Windows dev preset configured, run:

```powershell
cmake --build --preset dev --target cpu_leaf_original_lifted cpu_leaf_inverted_lifted
ctest --preset dev --no-tests=error -R '^cpu_leaf_aot_branches$' -V
```

MSVC's x64 assembler and linker build [the original function](../tests/cpu/branch.asm) into a DLL. The [translator](../tools/cpu_leaf_aot.py) reads its exported entry RVA and the matching runtime-function interval from the PE. It translates those bytes into C++, which MSVC compiles to a separate x64 object and links into a second DLL. The intermediate source is in `out/build/dev/cpu_leaf_original_lifted.cpp`.

The supported signature takes an unsigned 32-bit integer in ECX and a writable word pointer in RDX, and returns a 32-bit integer in EAX. The decoder covers the fixture's register move, immediate compare paired with unsigned short JB/JAE, short forward jump, immediate add/subtract, output store and return. It rejects unknown encodings and branches outside the decoded instruction boundaries. Leaf functions can omit unwind metadata. This experiment uses an explicit interval provided by the assembler.

The Windows test calls both DLLs with the same 263 inputs. Five take the below-five branch and 258 take the other branch, including equality, the unsigned sign boundary and wraparound. It compares the returned word, the stored word and guards on either side, and checks the original against independent unsigned arithmetic. A control assembles the same source with the conditional jump inverted and runs that PE through the same translator. Its results must differ from the original on every input. [Direct-call follow-up](https://github.com/R0Y00/Xbox2Windows/issues/22)

## Chained function ranges and early returns

The range reader groups runtime-function entries by their complete unwind-chain parent records. The early-return translator uses iced-x86 to decode each owned interval separately. It follows instruction-aligned branches and fallthrough, models stack slots and preserved registers, and checks their restoration at each return. A neighboring function can occupy the gap. The single-interval translators report split functions explicitly. [Microsoft chained unwind information](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#chained-unwind-info-structures)

```powershell
cmake --build --preset dev --target cpu_chained_lifted cpu_register_probe
ctest --preset dev --no-tests=error -R '^cpu_aot_chained_ranges$' -V
```

The source-owned fixture has a primary interval, two levels of chaining and a neighboring export. Both null inputs return 7, and two non-null inputs return 21. A Windows probe compares native returns with the generated result, physical RBX/RSI, and guards. Negative controls cover broken and cyclic parent links, jumps into a neighbor or instruction interior, and an indirect call.

For the local FidelityFX loader, run `cpu_chained_aot.py <loader.dll> ffxDestroyContext out/local_context.cpp --partial` using the Python dev environment. Compile the generated C++ as a Windows x64 DLL. Its `try_lifted(first, second, output)` returns 1 and writes the guest result for a completed path. Status 0 leaves the output untouched at an unsupported indirect call. Pointer arguments must reference readable fixture storage on paths that dereference them. The local inspection recovered four owned intervals. Running the generated code produced 6 for a null first argument and for a pointer to a null context. A valid non-null control stopped at the first indirect call with its output sentinel intact. The original package DLL was inspected as a file. Native differential results above are from the repository fixture. [Callback follow-up](https://github.com/R0Y00/Xbox2Windows/issues/29)

### Object-table callbacks on Windows

The source-built caller follows a handle to its context and loads a table pointer from that context. Slot 1 takes the context address and a 64-bit input and returns a 64-bit value. The test selects two tables with the providers in opposite orders. It compares the observed provider, both arguments, return and output guards with native Windows execution, including inputs with bit 63 set. Both null paths return 6 without calling either provider.

```powershell
cmake --build --preset dev --target cpu_callback_lifted
ctest --preset dev --no-tests=error -R '^cpu_aot_object_callback$' -V
```

For this fixture, `cpu_chained_aot.py` takes `--callback-offset 8` and emits `try_lifted_callback(first, second, output, binding)`. The host supplies readable fixture storage and a binding containing the expected table address and typed function pointer. Dispatch requires both the table address and the selected slot's target to match. A missing or mismatched binding returns status 0 and leaves the output untouched. The existing three-argument partial profile still stops at an unbound indirect call.

### FidelityFX destroy provider and cleanup

In the inspected Xbox loader, `ffxCreateContext` masks the effect ID and calls the internal resolver at RVA `0x1000`. For effect `0x10000` the resolver gets or loads `amd_fidelityfx_upscaler_xbox.dll`. It resolves the named export `ffxDestroyContext` and stores the address at loader RVA `0x4098`, slot 1 of the table at `0x4090`. After a successful provider create, the loader puts that table pointer and the provider context in the two words of a wrapper. The packaged upscaler export directory independently identifies the destroy entry at provider RVA `0x13a0`. The frame-generation route uses another table and its module is absent from this package.

The loader passes the **address of the wrapper's provider-context field** in RCX and the optional allocation-callbacks pointer in RDX. The upscaler dereferences the first argument and saves the second in a local slot. Its internal destroy method receives the effect object, the context-field address and the address of that saved allocator pointer. On the provider-object release path the upscaler also uses the allocator for deallocation or calls CRT free. These register uses agree with AMD's two-argument destroy declaration and unsigned 32-bit status. [AMD FidelityFX API header](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/api/include/ffx_api.h)

After provider destroy, the **loader** frees its own wrapper, including when the provider returned an error. Custom cleanup receives the allocator's user data and the wrapper address. The default branch calls the imported CRT free. Both branches return the saved provider status. The explicit `--ffx-destroy` profile binds a stable provider table, allocator and cleanup routines for the duration of the call. The host must supply live storage compatible with the chosen deallocator. An unresolved binding stops before invoking the provider.

```powershell
cmake --build --preset dev --target cpu_destroy_lifted cpu_destroy_provider
ctest --preset dev --no-tests=error -R '^cpu_aot_destroy_contract$' -V
```

CI compares a source-built native caller with its static translation. On Windows the same harness also compared that native reference with the translation of all four owned ranges of the local loader export. Both null exits returned 6. With a source-built provider resolved by export name, both cleanup routes preserved 0, 6, 0x80000001 and 0xffffffff. The test checks the addresses passed to destroy and custom cleanup, and that the provider clears the inner field before the wrapper is released. The packaged upscaler's own effect teardown was inspected statically.

To repeat the local comparison in a VS x64 developer shell, translate the loader with `tools/cpu_chained_aot.py <loader.dll> ffxDestroyContext out/local_destroy.cpp --ffx-destroy`, compile it with `cl /nologo /LD /O2 /Fo:out/local_destroy.obj out/local_destroy.cpp /link /OUT:out/local_destroy.dll /IMPLIB:out/local_destroy.lib`, and run `tests/cpu/destroy_check.py` with absolute paths to `cpu_destroy.dll`, `local_destroy.dll` and `cpu_destroy_provider.dll`, in that order.

### First device calls in the packaged upscaler

For the upscaling effect (`0x10000`), the local upscaler's `ffxCreateContext` export at RVA `0x11d0` takes an output handle pointer in RCX, a descriptor chain in RDX, and an optional allocator in R8. On the inspected path, the backend extension has type `0x01000002` at offset 0, a next pointer at +8 and the device pointer at +16. The provider selection call at `0x129c` passes that pointer to a predicate that returns true without reading it. At `0x1349` the export calls the selected provider's create method at `0xc9b0`.

The provider reads the device from the extension at `0xcdca`, sets up a fresh 0x18f160-byte scratch buffer for one context and invokes its backend callback at `0x2720`. At the call site RCX points to the backend interface, EDX is `0x14`, R8 points to four 32-bit words `{0, 0, 0, 1}`, and R9 points to storage for the output context index. The callback reads the device at interface +0xd0 and scratch at +0xc0. On its first use, the device receives `IUnknown::AddRef(this)` at `0x2803`. It takes only the device in RCX. The callback stores the device at scratch +8 and continues without inspecting the returned reference count.

The first resource creation at `0x2893` uses device vtable slot 14 (+0x70). RCX holds the device, RDX points to a 16-byte `D3D12_DESCRIPTOR_HEAP_DESC`, R8 points to the IID of `ID3D12DescriptorHeap` (`8efb471d-616c-4f49-90f7-127bb763fa51`), and R9 points to the output slot. All five calls request node mask 0. [D3D12 CreateDescriptorHeap](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createdescriptorheap) · [IUnknown AddRef](https://learn.microsoft.com/en-us/windows/win32/api/unknwn/nf-unknwn-iunknown-addref)

| Call RVA | Heap type | Descriptors | Shader visible | Output in scratch |
|---|---|---:|---|---|
| `0x2893` | CBV/SRV/UAV | 512 | No | +0x28 |
| `0x28ac` | CBV/SRV/UAV | 512 | No | +0x30 |
| `0x28cd` | CBV/SRV/UAV | 512 | Yes | +0x38 |
| `0x28fd` | CBV/SRV/UAV | 167,936 | Yes | +0x48 |
| `0x2935` | RTV | 8 | No | +0x20 |

On Windows the probe starts at RVA `0x27f8` with a live device, zero-initialized scratch, a context capacity of one and R15 zero. It executes the offline translation of the package's argument construction and binds the two COM methods to desktop D3D12. The RTX 4060 Laptop (driver 32.0.16.1656) and Microsoft WARP (10.0.26100.9818) each returned `S_OK` five times. The probe verified five distinct live heap handles, their output slots and their descriptions. Both debug-layer runs reported zero errors and zero warnings. The host releases the five heaps and its device reference on exit.

After the fifth heap the test continues into the backend slot scan at RVA `0x2938`. The host stops on a failed HRESULT so that a desktop failure identifies the exact heap. The package slice itself has no intervening HRESULT branches.

To run the local translation in a VS x64 developer shell, set `$upscaler` to the full path of the package's upscaler DLL and use a separate build directory. The translator checks the image hash before generating C++.

```powershell
cmake --preset dev -B out/build/ffx-device "-DFFX_UPSCALER_IMAGE:FILEPATH=$upscaler"
cmake --build out/build/ffx-device --target offscreen_check
ctest --test-dir out/build/ffx-device -R '^ffx_device_init_' --no-tests=error -V
```

The log identifies this run with `source=translated-package`. The default build uses a source-owned desktop control for the five D3D12 calls (`source=desktop-control`). Public CI runs that control on WARP and compares native and translated x64 heap-size arithmetic for six inputs, including 32-bit wraparound.

### Backend context slot selection

The provider takes the address of its 32-bit index field at +0xec (`0xcf94`) and passes it in R9. The backend callback saves that pointer in R14. On first use it finishes setting up the shared descriptor range at `0x2938`. Subsequent calls skip heap creation and rejoin at `0x2942`. The callback increments the 32-bit scratch counter before checking capacity, then scans from index zero. The table pointer is at scratch +0x1078, records are 96 bytes each, and bit 0 of the byte at record +0x3c means occupied.

At the first free slot, the store at `0x2980` writes the index directly to the provider. The callback writes effect `0x14`, sets flag bit 0, clears bit 1 and preserves the other six bits. It initializes two pairs of descriptor cursors to `512*i+1` and `512*i+511`. On the package's one-slot input, the four sizes `{0,0,0,1}` reserve the half-open range `[0x19000,0x19001)`. When all slots are occupied, the scan branches to the same zero-return exit at `0x2a9f` without writing a record or the output index.

| Windows input | Backend return | Provider index after call | Scratch counter | Slot change |
|---|---:|---|---|---|
| Fresh package table after five heaps | 0 | 0 | 0 → 1 | Slot 0 becomes occupied |
| Same table, second request | 0 | `0xfeedface` preserved | 1 → 2 | None |
| Controlled two-slot table, slot 0 occupied | 0 | 1 | 1 → 2 | Slot 1 becomes occupied. Its range starts after slot 0 at `0x32008` |
| Both controlled slots occupied | 0 | `0xfeedface` preserved | 2 → 3 | None |

The package translation and an independently compiled C++ control ran on the RTX 4060 Laptop and WARP using the commands above. The test compares the entire scratch buffer and provider output region, including bytes around the index. The exhaustion cases supply a new output field initialized to the sentinel. The dual-slot setup exercises a nonzero index with a controlled capacity and four nonzero sizes. Public CI runs the desktop control plus a source-built x64 scan and four-lane sum in `cpu_aot_heap_layout`.

The provider tests EAX at `0xcfc2`. A zero backend return continues initialization. It later loads the same index field into R8D at `0xd1f4` for the resource callback at `0x2e30`. On the default path the first descriptor identifies `FSR4UPSCALER_Recurrent` (ID 13), with its allocation mode and placed-resource flag both zero and the optional allocation callback unset. The intervening descriptor-builder call at `0xc4ff` resolves to a CPU-only helper at `0x7900`, and the import at `0x2ea1` is an XMem working-set query. The next call **on the device** is at `0x33cc`, Xbox vtable slot 76 (+0x260). Its arguments include heap properties, zero heap flags, a resource description, initial state, a null clear value, the `ID3D12Resource` IID and an output pointer. Those arguments match the shape of desktop committed-resource creation. The Windows replay below binds that call to the desktop device. [Desktop CreateCommittedResource signature](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createcommittedresource)

### Creating the first recurrent texture

For the default provider, the builder at `0xbf60` receives the output-size pair in R8. The create descriptor supplies render width/height at +0x14/+0x18 and output width/height at +0x1c/+0x20. The first resource uses the latter pair and has ID 13 and name `FSR4UPSCALER_Recurrent`. The local replay translates the first descriptor stores, the provider's index-output address calculation and the default committed-resource path. It supplies zero for the optional Xbox working-set statistics and uses the default allocator route with the optional resource allocation callback unset.

| Field | Package path | Desktop D3D12 argument |
|---|---|---|
| Dimensions | Kind 2, output width and height, one layer, one mip | Texture2D, one sample, quality 0 |
| Format | Format code 10 selects value 28 in the format switch at `0x2160` | `DXGI_FORMAT_R8G8B8A8_UNORM` |
| Flags | Usage bits 0 and 1 produce flags 0x5 | `ALLOW_RENDER_TARGET` and `ALLOW_UNORDERED_ACCESS` |
| Initial state | Provider state 2 selects value 8 at `0x325a` | `D3D12_RESOURCE_STATE_UNORDERED_ACCESS` |
| Allocation | Default heap, zero heap flags, null optimized clear value | Committed resource, default alignment, single-node masks 1/1 |
| Layout | Xbox texture layout 0x101, extension word 3 and trailing zero | Driver-chosen `D3D12_TEXTURE_LAYOUT_UNKNOWN`. The adapter checks the Xbox extension fields before omitting them from the desktop descriptor |

With a render size of 1280×720, the RTX 4060 Laptop (driver 32.0.16.1656) and WARP (10.0.26100.9818) each ran two fresh contexts. For outputs of 1920×1080 and 1600×900, all four calls returned `S_OK`. Each resource's `GetDesc` reported the requested output dimensions, format 28, flags 0x5, one layer/mip/sample, quality zero, desktop layout 0 and 65,536-byte alignment. Initial state is a creation argument. The probe transitions the texture from UAV to shader read, then copy source, then back to UAV and waits for completion. Both debug layers reported zero errors and zero warnings.

On context slot 0 the provider passes +0x17c as the output for ID 13. The resource callback takes cursor 1, advances it to 2 and writes 1 to the output **before** the device call. On success it records provider state 2 in both state fields and stores the returned COM pointer in backend record 1 at `0x3408`. The test checks that pointer against the live texture and verifies all adjacent resource indices retain their sentinels.

Use the package build directory configured above in a VS x64 developer shell. Public CI runs the source-owned WARP fixture with the same desktop mapping and readback checks. It also compares native and translated descriptor-copy instructions using the MASM fixture.

```powershell
cmake --build out/build/ffx-device --target offscreen_check
ctest --test-dir out/build/ffx-device -R '^ffx_recurrent_' --no-tests=error -V
```

### Recurrent texture views and pixel readback

After the resource pointer is stored, the callback reads its description and builds views for the single-layer, single-mip Texture2D path. The SRV and UAV format switches both preserve `DXGI_FORMAT_R8G8B8A8_UNORM` (28). The SRV exposes mip 0 only, plane 0, minimum LOD 0 and the default RGBA component mapping (`0x1688`). Both UAV descriptors select mip 0 and plane 0 and use a null counter.

| Device call RVA | Descriptor | Heap pointer in scratch | Slot in fresh context 0 |
|---|---|---|---:|
| `0x37aa` | SRV | +0x28, CPU-only | 1 (resource index) |
| `0x3844` | UAV | +0x38, shader-visible | 1 (UAV base + mip) |
| `0x388a` | UAV mirror | +0x30, CPU-only | 1 (UAV base + mip) |

The resource record stores the UAV base at +0x34 and the count at +0x38, both 1. After creating both UAV descriptors, the callback advances the context's UAV cursor at +0x0c from 1 to 2. Each destination is the heap's CPU start plus the slot times the device-reported descriptor increment. That increment was 32 bytes on the RTX 4060 Laptop and 64 bytes on WARP.

The same `ffx_recurrent_hardware` and `ffx_recurrent_warp` commands above run the pixel check. A source-built compute shader writes a coordinate-dependent RGBA pattern through the original shader-visible UAV. A second pass writes a different pattern through a copy of the CPU UAV mirror. For shader access the probe copies the CPU-only SRV and UAV mirror into a separate, shader-visible test heap. A read shader samples the SRV at texel centers and writes packed RGBA bytes to a buffer. A separate `CopyTextureRegion` reads the original resource pointer directly. After a fence, both paths must match the CPU's expected RGBA value for every pixel. [D3D12 descriptor copying](https://learn.microsoft.com/en-us/windows/win32/direct3d12/copying-descriptors)

Both adapters passed both patterns at 1920×1080 and 1600×900. Each adapter checked 7,027,200 pixels through the SRV and the same number through direct copies, with zero mismatches, debug errors or warnings. The local build executes the package-derived view creation through the cursor update at RVA `0x389b`. Public Windows CI runs the source-owned WARP control. The test shaders exercise descriptor access. The provider loop is covered below.

### Provider resource loop and returned handles

At 1280×720 render and 1920×1080 output, the default provider visits twelve candidate descriptors in order. The builder at RVA `0xbf60` emits 104 bytes per candidate. The loop at `0xd1d0` skips ID 23 when bit 9 of the create flags is clear. The index destination is `provider + 0x148 + 4 * ID`. Three callbacks also allocate an upload resource from the same cursor.

| ID | Resource (FSR4UPSCALER_ prefix) | Shape | Default index | Index with flag 0x200 |
| ---: | --- | --- | ---: | ---: |
| 13 | Recurrent | 1920×1080 RGBA8 | 1 | 1 |
| 14 | History | 1920×1080 RGBA16F | 2 | 2 |
| 15 | HistoryReprojected | 1920×1080 RGBA16F | 3 | 3 |
| 16 | RcasIntermediary | 1920×1080 RGB9E5 | 4 | 4 |
| 5 | AutoExposure | 8-byte raw buffer | 5 | 5 |
| 17 | DefaultExposure | 1×1 RG32F | 7 | 7 |
| 6 | SpdAtomicCounter | 1×1 R32_UINT | 9 | 9 |
| 7 | Luma_Mip_5 | 80×45 R32F | 11 | 11 |
| 22 | ScratchBuffer | 21,045,760-byte raw buffer | 12 | 12 |
| 20 | InitializerBuffer | 131,072-byte raw buffer | 13 | 13 |
| 21 | InitializerBuffer_Upload | 131,072-byte upload buffer | 14 | 14 |
| 23 | DebugInformation | 1920×1080 R16_UNORM | 0 (skipped) | 15 |

The uploads for IDs 5, 17 and 6 occupy slots 6, 8 and 10 respectively. ID 21 is a named provider resource with its own index. The package row loop treats the zero height of ID 5 as zero rows to initialize, despite its eight-byte input payload. ID 17 supplies two float ones and ID 6 supplies a zero fill. The desktop binding uses committed resources, creates SRVs and paired UAVs for eligible resources, and submits the initialization copies with a fence. Default-heap buffers use desktop COMMON state and implicit promotion.

On success, the translated loop copies all 25 indices to the mirror at +0x1ac and writes the provider pointer through the output argument at RVA `0xd263`. The loader returns a separate 16-byte wrapper containing the callback table and that same provider pointer. With flags 0 and 0x200 both adapters returned 0, with next resource indices 15 and 16 respectively. Injecting a callback return of 7 at ID 20 returned 3 through the provider and loader, left both output handles null, preserved the previous resource indices, and left the mirror untouched. The test checks every live backend record against its D3D12 object, both index tables, and the loader wrapper fields. All three cases completed with zero debug errors or warnings on the RTX 4060 Laptop and WARP.

The first desktop format difference occurs at ID 16. WARP exposed typed RGB9E5 UAV stores. The RTX driver (32.0.16.1656) lacked that capability, so the binding created an R32_UINT texture and views for the packed bits. Shader packing, filtering and complete upscaling output remain unverified.

The local build translates the descriptor builder, debug flag, loop and writeback, resource-slot reservation, upload-descriptor construction, and selected create-export branches. Registry selection and the backend device callbacks are bound on the host. When the pinned loader DLL is beside the upscaler DLL, the probe also executes the recompiled loader create path against that provider. Public CI runs the source-owned provider control on WARP. Configure the local package build as in the recurrent texture test above and run:

```powershell
cmake --build out/build/ffx-device --target offscreen_check
ctest --test-dir out/build/ffx-device -R '^ffx_provider_loop_' --no-tests=error -V
```

### FidelityFX context creation and ownership

The Windows test calls the translated loader create function, then passes its returned handle directly to the existing destroy translation. Create binds a source-owned upscaler provider to the two table slots. On success, the loader allocates a 16-byte wrapper and stores the table pointer and provider context in it. A provider create error propagates its 32-bit return code, clears the caller's handle and skips wrapper allocation. Destroy calls the provider with the address of the inner field before releasing the wrapper, including on a provider destroy error.

Fourteen cases cover default CRT allocation and custom allocation with user data. For each route the test checks success, two destroy errors, three create errors, and failure to allocate the wrapper after the provider succeeds. It observes allocation and release addresses, order, sizes and the live-allocation count. In the allocation-failure case the host adapter calls provider destroy once and returns memory error 5. This rollback is a Windows adapter behavior. The inspected loader proceeds to a wrapper store after its allocator returns. The provider used in this comparison owns real CRT memory and has controlled return codes; creation of a live upscaling effect on a graphics device remains open.

```powershell
cmake --build --preset dev --target cpu_create_lifted cpu_destroy_lifted
ctest --preset dev --no-tests=error -R '^cpu_aot_context_lifecycle$' -V
```

CI compares an independently assembled native create caller and its translation, each paired with the fixture's lifted destroy. To check the package translation from a VS x64 developer shell, set `$loader` to the path of the local FidelityFX loader and run:

```powershell
New-Item -ItemType Directory -Force out/ffx-lifecycle | Out-Null
./out/pe-env/Scripts/python.exe -B tools/ffx_create_context.py $loader out/ffx-lifecycle/create.cpp
./out/pe-env/Scripts/python.exe -B tools/cpu_chained_aot.py $loader ffxDestroyContext out/ffx-lifecycle/destroy.cpp --ffx-destroy
foreach ($unit in 'create','destroy') {
    $prefix = "out/ffx-lifecycle/$unit"
    cl /nologo /O2 /EHsc /LD "/Fo:$prefix.obj" "$prefix.cpp" /link "/OUT:$prefix.dll" "/IMPLIB:$prefix.lib"
    if ($LASTEXITCODE) { throw "Compilation failed for $unit" }
}
./out/pe-env/Scripts/python.exe -B tests/cpu/context_lifecycle.py out/build/dev/cpu_create.dll out/ffx-lifecycle/create.dll out/ffx-lifecycle/destroy.dll
```

### FidelityFX version enumeration

The Windows query returned one upscaler, version ID `0xf5a5ca1e01001001` and name `4.1.1`. The first call requested the count. The second supplied one ID slot and one name slot. Both returned status 0. The descriptor uses AMD's null-context, type-4 version query. [AMD FidelityFX query descriptor](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/api/include/ffx_api.h)

`tools/ffx_version_query.py` checks both input DLL hashes, decodes the selected x64 query intervals and the provider's two filter functions, and emits their integer operations and branches as C++. The generated Windows DLL combines both modules' selected paths. Its host binding specializes module resolution to one available upscaler and no frame-generation module. It reconstructs the single provider entry from the initializer's stores, including the full ID and the referenced name. Static storage keeps the returned name valid for the lifetime of the DLL.

On this build, capacity zero reports the available count without writing array elements. ID-only and name-only calls both succeed. Capacity two writes one result and leaves the second slot untouched. A missing frame-generation module returns 4 and an unknown effect returns 2. In both cases the loader clears the count **pointer in the descriptor** and leaves the caller's count value untouched. This entry point covers version enumeration only.

Run these commands in an x64 VS developer shell with the Python development environment available. Set `$loader` and `$upscaler` to the two local input paths.

```powershell
New-Item -ItemType Directory -Force out/ffx-query | Out-Null
./out/pe-env/Scripts/python.exe -B tools/ffx_version_query.py $loader $upscaler out/ffx-query/query.cpp
cl /nologo /O2 /EHsc /LD /Fo:out/ffx-query/query.obj out/ffx-query/query.cpp /link /OUT:out/ffx-query/query.dll /IMPLIB:out/ffx-query/query.lib
./out/pe-env/Scripts/python.exe -B tests/cpu/version_query_check.py out/ffx-query/query.dll query_versions 0xf5a5ca1e01001001 4.1.1
```

The public CTest `cpu_aot_version_query` runs the same output checks against an independent MASM fixture and its translation. The fixture uses a different name and 64-bit ID. The two proprietary inputs and generated code stay local.

## Recompiling a direct x64 call

The [caller and callee](../tests/cpu/direct.asm) each accept two unsigned 64-bit integers and an output pointer. The [translator](../tools/cpu_call_aot.py) starts at the caller export, follows the signed displacement in its E8 call and reads the target's runtime-function interval. It generates two C++ functions from the linked machine code. MSVC emits a separate x64 DLL with a non-inlined call. The Windows test checks the relative call against the two export addresses in both the input and output DLLs.

```powershell
cmake --build --preset dev --target cpu_call_original_lifted cpu_call_altered_lifted cpu_register_probe
ctest --preset dev --no-tests=error -R '^cpu_aot_direct_call$' -V
```

The caller changes both arguments and keeps its original inputs live across the call. The callee records its arguments and arithmetic result through R8. The caller uses the returned RAX and both live inputs to compute its result. A separate assembly probe seeds physical RBX and RSI and reads them immediately after each invocation. Tests exercise the whole call and the callee in isolation using the same 249 input pairs, including wraparound and high-word differences. They compare the return, three output words, adjacent guards and both register sentinels with the original and independent arithmetic.

A rebuilt callee that increments only its return after storing the result exposes a mismatch for every pair. A register-clobber control checks the probe itself. Targets redirected into the middle of the callee or back to the caller are rejected during translation. Coverage is limited to this integer ABI, one direct call, its two fixed frame layouts and the decoded register operations. [Windows x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention) · [Import-call follow-up](https://github.com/R0Y00/Xbox2Windows/issues/24)

## Calling another DLL through a named PE import

The source-built caller uses an x64 RIP-relative indirect call (`FF 15 disp32`). The translator computes the import address table (IAT) slot RVA from the end of the instruction and looks up the exact slot in the ordinary import descriptors. The build supplies an explicit binding for the fixture's integer ABI. The DLL name and case-sensitive symbol must match the slot, and the provider must have a local executable export for that name. MSVC compiles the generated caller and links the provider's import library. The test independently checks that the call in each linked caller addresses its own slot for `cpu_import_provider.dll!mix_args`. [PE import address table](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#import-address-table)

```powershell
cmake --build --preset dev --target cpu_import_original_lifted cpu_import_altered cpu_register_probe
ctest --preset dev --no-tests=error -R '^cpu_aot_import_call$' -V
```

Each provider configuration runs in a fresh Windows process. The provider records the two arguments it received and its arithmetic result in three output words. For 249 pairs the test compares both callers with independent arithmetic, checks the adjacent guard words and samples RBX and RSI. Replacing the provider with one that increments its return changes the caller's result on every pair and preserves the output writes.

With the provider absent, both callers fail to load with error 126. With a DLL of the expected name that lacks the requested export, both fail with error 127. The translator also diagnoses the missing provider or export before compilation and rejects a call displaced one byte into the IAT slot. The verified profile covers one ordinary named import with two 64-bit integers and an output pointer. [Ordinal-only import follow-up](https://github.com/R0Y00/Xbox2Windows/issues/26)

## Calling an ordinal-only export

The source-built provider exposes only ordinal 77 through a NONAME export. The build binds `#77` with the local import-library symbol `mix_args`. The translator checks the DLL and ordinal from the call's IAT slot against the actual export address table. The test reads both linked callers' import lookup words, checks the ordinal flag and value, and verifies the provider has no exported name for the entry.

```powershell
cmake --build --preset dev --target cpu_ordinal_original_lifted cpu_ordinal_altered cpu_register_probe cpu_import_provider
ctest --preset dev --no-tests=error -R '^cpu_aot_ordinal_import$' -V
```

For 249 input pairs both callers match the provider's observed arguments, guarded writes, register sentinels and expected return. Replacing the provider changes the return on all inputs. Both callers fail to load with error 126 when the DLL is absent and error 182 when the DLL exports the same link name but lacks ordinal 77. The static translator diagnoses a mismatched binding, a missing link name, a missing ordinal and a displacement into the middle of a slot.

The local Bink DLL imports ordinal 1 from XAudio2_9.dll. A read-only check confirmed an executable export at that ordinal in this workstation's System32 provider. The Bink call's argument and result contract remains unverified.

## Offline pixel shader recompilation

`tools/pixel_aot.py` extracts one named pixel program from a local archive, decodes gfx1013 instructions with LLVM and emits HLSL. DXC compiles the result to desktop DXIL before the draw. The lowerer tracks register writes across whole-quad interpolation of attribute 0, a scalar texture-descriptor load, a 2D RGBA sample, packed round-toward-zero binary16 conversions and one compressed color export. Unknown instructions, mismatched bindings and reads before the required wait cause translation to fail. [LLVM disassembler API](https://llvm.org/docs/doxygen/group__LLVMCDisassembler.html) · [Leonida public format reference](https://github.com/ItsNotRexed/Leonida/blob/56f023a10a726da20702d4e121656ceda2391d93/tools/fssm_unpack.py)

The caller identifies the entry registers for the texture address and interpolation base, and supplies the four sampler descriptor words to match. The generated pixel shader uses `TEXCOORD0`, texture `t0` and sampler `s0`. The test binds a point-clamp sampler, one mip and a full-screen triangle. These bindings are an explicit specialization. The test measures finite UNORM inputs under full pixel coverage. Execution with a title's original pipeline state, helper-lane behavior at triangle edges and other texture formats require separate validation.

For a shader that reads constants, pass `--constant-descriptor OFFSET` to identify its four-word descriptor in the same table as the texture descriptor. The desktop test binds 256 bytes at `b0`. A four-word read may span two HLSL constant vectors. Arithmetic support includes log2, exp2, reciprocal, min/max, subtraction, legacy zero-preserving multiplication, multiply-add, destination accumulation and saturation. Compile arithmetic shaders with DXC `-Gis`. Transcendental precision and multiply-add rounding may vary by device. [AMD RDNA instruction reference](https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Shader_ISA_5August2019.pdf) · [HLSL mad precision](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/mad)

From the repository root on Linux with LLVM 15 installed, assemble the three original test shaders, wrap them in synthetic records, then extract, decode and translate them. The test compares the result with lowering the source operands directly.

```sh
python3 -B tests/pixel_aot_check.py binary llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
```

On Windows, use the Python-enabled build from the project README. Each backend draws a copy, a red/blue swap and a nonlinear curve with two sets of constants. It compares all four channels of 4,096 pixels bitwise with an independent CPU reference. Negative controls change the expected channel order or the bound gain and must report a mismatch.

To supply a reference for another arithmetic program, run `offscreen_check.exe hardware --check-pixel-case shader.cso reference.case`. The case contains 256 bytes of constants followed by 64 expected RGBA colors in row-major little-endian binary32, one for each texel of the test texture. Generate the expected colors after round-toward-zero half export and widening back to float.

```powershell
ctest --preset dev -R '^pixel_aot_' -V
```

For a compatible local record, substitute the archive, exact record name, two register numbers and four descriptor words below. Run the translator from the repository root in WSL, then compile and draw from a Windows development shell. The JSON printed by the translator identifies each input and output by SHA-256 for local reproduction. The renderer's `rgba` expectation checks a copy and `bgra` checks a red/blue swap.

```sh
python3 -B tools/pixel_aot.py "$archive" "$record" --llvm libLLVM-15.so.1 \
    --texture-sgpr "$texture_reg" --interp-sgpr "$interp_reg" \
    --sampler-words "$word0" "$word1" "$word2" "$word3" --output out/pixel-aot/translated.hlsl
```

```powershell
.\tools\dev.ps1
dxc -T ps_6_0 -E main -Fo out/pixel-aot/translated.cso out/pixel-aot/translated.hlsl
.\out\build\dev\offscreen_check.exe hardware --check-pixel out/pixel-aot/translated.cso rgba
.\out\build\dev\offscreen_check.exe warp --check-pixel out/pixel-aot/translated.cso rgba
```

## Package integrity

For a local build with a multi-file v1 torrent and `layout*.xml`, run the offline verification with Python 3.9 or newer. Substitute your local directory name.

```powershell
python tools/verify_build.py "game-files/<build-directory>"
```

The script verifies SHA-1 pieces across file boundaries in torrent order and compares layout destinations separately with disk and torrent entries. The report is `out/integrity/<build-directory>.json`. Exit code 0 means the payload passed piece verification. `layout.absent_from_disk` lists missing layout entries separately.

## PE dependency baseline

The scanner uses x64 Windows, Python 3.11 or newer and pefile. Set up an isolated environment once.

```powershell
python -m venv out/pe-env
.\out\pe-env\Scripts\python.exe -m pip install pefile==2024.8.26
```

Capture a DLL-only baseline. When an EXE becomes available, write a separate report and compare it with the saved baseline.

```powershell
$pePython = '.\out\pe-env\Scripts\python.exe'
$buildDir = 'game-files/<build-directory>'
& $pePython tools/pe_inventory.py $buildDir --output out/dependencies/pre-exe.json
# Run the next command after adding the main executable.
& $pePython tools/pe_inventory.py $buildDir --baseline out/dependencies/pre-exe.json --output out/dependencies/with-exe.json
```

`summary` gives counts. `unresolved_edges` lists requested exports missing from both the package and the host. Each `imports` entry identifies the caller, import mechanism, provider and symbol. `#N` denotes an ordinal. `images` includes architecture, SHA-256, export names, ordinals and forwarders. `diff` distinguishes new caller relationships from dependencies seen for the first time in the build.

The scanner reads System32 export tables and applies the host API-set v6 map at each hop, using the forwarding DLL as the importer at the next hop. Each import's `desktop_resolution.chain` records the requested module, selected provider, symbol and any advertised forwarder. Its final status identifies a terminal export, missing provider or export, malformed forwarder, or cycle. `desktop_export` retains the first-hop result for older report consumers. The comparison covers System32 DLLs on the measured machine. Package selection at runtime, parameter layouts and behavior require separate evidence. [PE export address table](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format#export-address-table) · [Wine API-set lookup](https://github.com/wine-mirror/wine/blob/master/dlls/ntdll/loader.c)

On Windows build 26200, the pre-executable inventory covered 16 package DLLs and 1,337 import edges. Of these, 171 edges (78 distinct requested module/symbol pairs) went through a single export forwarder. All 171 reached a terminal export. In total 1,329 edges resolved on the desktop. The eight remaining edges requested six distinct interfaces across `xmem.dll` and `d3d12_xs.dll`. The 16 package DLLs had zero forwarded exports of their own. The historical baseline was preserved for comparison.

## Heap and process memory

`src/era_heap.cpp` implements ordinary desktop heap requests for source-built callers. The alignment exponent uses attribute bits 24–28 following SlimEra's Xbox One ERA definition. The fixture accepts ordinary heap requests with all other attribute bits clear. Scarlett attribute encodings, error conventions and graphics memory semantics remain open questions. [SlimEra attributes](https://github.com/XWine1/SlimEra/blob/main/era.h#L105-L137) · [SlimEra default heap](https://github.com/XWine1/SlimEra/blob/main/kernelx.cpp#L600-L657)

| CTest case | Inputs and checks |
|---|---|
| `era_heap_allocation` | Allocate 257 bytes using six alignment exponents. Check alignment of at least 16 bytes and the requested alignment, initial zeroing, writes and release. Debug builds also compare CRT heap snapshots. |
| `era_heap_failure` | Zero size, excessive size, graphics memory, nondefault pages and reserved bits return a null pointer under the fixture's error policy. |
| `era_heap_concurrency` | Four threads run 100 rounds each. They hold simultaneous allocations, write distinct values, compare every byte and release the memory. |
| `era_heap_caller_layout` | Owned desktop storage covers all 32 base residues at eight payload sizes. Each Bink-style wrapper recovers its original base after a payload write and preserves the surrounding guard bytes. Ten AMD-style requests check 256-byte size rounding and the unused tail. These cases establish caller-side arithmetic. The special allocation attributes still have unknown semantics. |

`desktop_memory_query` supplies Windows observations for `XMemVirtualQuery` and `XMemGetWorkingSetStatistics`. The test reserves memory, commits and writes 1,024 pages, checks the read-only/writable boundary, then decommits. It verifies residency of a locked page and records working set, peak working set and private commit at three points. Values vary between runs. Scarlett structure layout, units and accounting scope remain unconfirmed. [Windows working sets](https://learn.microsoft.com/en-us/windows/win32/memory/working-set)

## Offscreen rendering and readback

`offscreen_hardware` selects a high-performance adapter. `offscreen_warp` selects Microsoft's software rasterizer. Both draw the same DXC-compiled triangle to a 64×64 RGBA8 texture. After GPU completion the test reads the texture using its row pitch and compares colors and coverage pixel by pixel. Outputs are `triangle_hardware.bmp` and `triangle_warp.bmp` in the build directory. `ctest --preset dev -V` reports the adapter, driver, debug layer and pixel differences. Error or corruption messages from an available debug layer fail the test.

The original test machine had Windows Graphics Tools installed. A second build in `out/build/agility` used Microsoft Agility SDK 1.620.2. Set `-DOFFSCREEN_AGILITY_620_DIR=<directory containing the x64 SDK DLLs>` to reproduce it. This configuration verifies debug interface activation. [Agility SDK package](https://www.nuget.org/packages/Microsoft.Direct3D.D3D12/1.620.2)

## Static executable intake

By default, the intake tool locates the Scarlett entry in the sole build's `MicrosoftGame.Config`. It accepts the layout path or a separate candidate file and opens the game files for reading only.

```powershell
.\out\pe-env\Scripts\python.exe -B tools/intake_exe.py
# Inspect a separately supplied candidate.
.\out\pe-env\Scripts\python.exe -B tools/intake_exe.py "E:\incoming\game_scarlett_master.exe"
```

Each run writes a new JSON report in `out/intake/`. Exit codes are 3 for a missing layout executable, 2 for an input or inspection failure and 0 for completed inspection. The report links the previous verification time, torrent SHA-256, install chunks and baseline summary. The inspected torrent omitted the EXE. The 13,345 matching pieces and 223,880,932,870 bytes verified in the historical report cover that original payload. A later EXE needs separate provenance. Intake rescans the package PEs and lists changed DLLs.

For each candidate the tool records SHA-256, architecture, entry RVA and section, TLS directory and callback order, then compares imports, modules and unresolved items with the baseline. Source-built PE fixtures and a miniature torrent cover named, ordinal and delayed imports, missing providers, TLS callbacks, invalid callback addresses, truncation, an absent entry and path conflicts. These checks establish static properties. Authenticity and startup remain unverified.

`pe.startup.locations` lists TLS callbacks in table order followed by the executable entry. Each location includes its RVA, section and file offset. When the x64 exception directory supplies a covering function interval, the report includes its inclusive start, exclusive end and unwind data RVA. A location can legitimately lack a covering record. An absent or unusable table leaves the independently observed startup locations available. In the local debug fixture the table had 110 records, one covering its TLS callback. The entry had no covering record. CTest cross-checks against pefile and exercises both absent and incomplete directories. [Microsoft x64 unwind data](https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64)

## Six API call sites

After loading the toolchain, inspect three local DLLs. `out/calls/six-apis.json` records image SHA-256, import slots, callers of thunks, embedded container sections and code references. Addresses below are RVAs in the relevant DLL. Indirect calls need execution evidence to identify the selected path.

```powershell
.\tools\dev.ps1
.\out\pe-env\Scripts\python.exe -B tools/call_evidence.py "game-files/<build-directory>"
```

| Import | Observed inputs, return handling and desktop check |
|---|---|
| `XMemAlloc` | At `0x25c35`, Bink's fallback requests the length plus 64 with attributes `0x05e50000` and checks for null. It advances the base by `64 - (base & 31)` bytes (33–64), giving a 32-byte-aligned user pointer. The two preceding bytes hold the offset at `[-1]` and a tag at `[-2]`. At `0x565d`, AMD rounds the nonzero request up to a multiple of 256, reads attributes at `0x6eee0` (on-disk initial value `0xa8004000`) and checks for null. The ordinary-heap fixture rejects both observed attribute words. |
| `XMemFree` | At `0x25be4`, Bink uses the saved offset to recover the allocation base and passes the same attributes. Custom allocations use a matching callback. AMD cleanup at `0x5712` and release at `0x5921` both use the allocation's attribute slot. The latter also clears the stored pointer. No observed caller consumes the return value. |
| `XMemVirtualQuery` | Four BVH callers pass the current process, target address and a zeroed 64-byte buffer. Each checks a nonzero return, bit `0x1000` of the word at `+0x20` and either low bit of the byte at `+0x30`. Source acceleration structure validation at `0x19c69` also requires the byte at `+0x24` to intersect mask `6`. Destination acceleration structure (`0x19ccb`), index buffer (`0x19d9d`) and vertex buffer (`0x19df9`) use the common checks. Diagnostics describe the whole check. The individual tail bits have no confirmed meaning. |
| `XMemGetWorkingSetStatistics` | Seven AMD calls pass selector `0x20000001` and a local output address. At `0x26b3` and elsewhere a zero return contributes zero to a calculation. Success uses the 64-bit value at `+8`. Code near `0x74d2/75b1` uses a before/after difference to update a counter. The units, scope and selector meaning remain open. |
| `D3D12SerializeVersionedRootSignature` | At `0xc2d0` the BVH code copies a caller's 48-byte descriptor and adds flags `0x03000000` at `+0x28`. At `0xc32f` it passes the descriptor, a blob output pointer and a null error output. Negative HRESULT enters diagnostics. Success uses the blob length and data pointer. A synthetic v1.1 descriptor returned `S_OK` with baseline flags and `E_INVALIDARG` with the two added bits. |
| `D3D12CreateRootSignatureDeserializer` | At `0x4f88`, AMD reads a byte pointer and a 32-bit length from record offsets `+0/+8`, then supplies an interface IID and object output address. A negative HRESULT or null object triggers partial cleanup. Success retrieves the descriptor via the object. Desktop code read the root signature from a complete Microsoft sample shader and returned `E_INVALIDARG` for a synthetic shader without one. |

### Graphics input paths

- The BVH serializer helper has 16 direct calls. At `0xd967–0xd998` the caller builds a version, six parameters and the parameter-array address on the stack. At `0x2151d` another path references container `0x25700` and passes it downstream with length `0x1134`. The DLL contains 52 code-referenced DXBC containers with structurally valid DXIL programs and `RTS0` sections. All 52 root signature headers carry `0x03000000`. The desktop failure case reproduces these flag bits. Their meaning remains unconfirmed.
- AMD registers the deserialization function in a callback table at `0xcea3–0xceaa`. A producer at `0xe9b8/0xe9c3` selects container `0x714f0` of length `0x1910`, copies its descriptor record and calls through the table at `0xebad`. The DLL has 29 structurally complete containers, each with a matching pointer-table entry and `RTS0` and `ACCC` sections. None contains DXIL. These observations support an embedded source. Runtime initialization determines the indirect path.
- Counts cover the inspected DLLs. Scene shaders, archive read order and the graphics path used at startup have no execution record here. An entry executable would allow its imports and initialization calls to be examined.

### Microsoft sample on desktop

The test uses Microsoft's SimpleTriangle at `Xbox-GDK-Samples` commit `b1e77b6ac10f95b07296025fda171ca2ddc42dbe`. The project includes a `Gaming.Xbox.Scarlett.x64` configuration and a published reference image. Desktop builds compile the two HLSL files using the sample's three vertices, colors, sRGB render target and clear color. The test checks SHA-256 for both shaders and the image. Set `SIMPLE_TRIANGLE_DIR` to that revision's sample directory and run `ctest --preset dev -R reference_triangle -V`. [Microsoft sample overview](https://learn.microsoft.com/en-us/samples/microsoft/xbox-gdk-samples/simpletriangle/) · [Pinned source and reference image](https://github.com/microsoft/Xbox-GDK-Samples/tree/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle)

RTX 2060 and WARP rendered at 64×64. Excluding triangle edges by normalized coordinates left 10,812 color channels per adapter. Differences from the source-derived colors were at most 2 on an 8-bit scale. Against the published image the mean absolute difference was 6.88 and the maximum was 17. The two desktop images differed at 94 pixels, by at most 1 per channel. The published image came without a device model, build hash or lossless GPU readback. Source-derived colors supply the acceptance values. Microsoft distributes console GDKX separately. [Microsoft developer resources](https://developer.microsoft.com/en-us/games/resources/)

## Compute and ray tracing before executable intake

The inspection reads bytes from two local DLLs. Desktop parsers inspect serialized descriptors and DXIL. GPU tests execute HLSL built from this repository. After building, run:

```powershell
$pkg = (Get-ChildItem game-files -Directory)[0].FullName
.\out\pe-env\Scripts\python.exe -B tools/probe_roots.py $pkg --controls out/build/dev
ctest --preset dev -R 'compute_|dxr_query|shader_metadata|bvh_buffer_ops' -V
```

`out/calls/desktop-roots.json` records per-container return codes, root signature summaries, matching desktop fixtures, static call counts and validator diagnostics. It also hashes the input inventory and probe executable. Each container was checked independently.

| Windows check | XboxBvhBuilder | AMD upscaler |
|---|---|---|
| Both D3D12 deserializers read the embedded signature from the full container | 52/52 accepted, 16 distinct signatures | 29/29 accepted, 3 distinct signatures |
| Reserialize unmodified v1.1 descriptor | 52/52 returned `E_INVALIDARG` | 29/29 accepted |
| Clear only `0x03000000` in a memory copy and reserialize | 52/52 accepted. Console flag semantics are unknown. | Original flags were zero. |
| Validate the entire container and the internal bitcode separately using the Windows SDK | Both checks rejected all 52 Shader Model 6.5 compute programs. | `ACCC` present, no DXIL input. |

Both source-built control shaders passed container and module validation. For the BVH modules the validator reported an unrecognized `dxil-ms-xb2` target and `xbox.compiler.options` metadata. DXC disassembly showed 43 modules directly calling 33 distinct `xbox.` functions, including cache-controlled reads, cross-lane operations and floating-point atomics. The other nine had no such direct calls. Diagnostics retain up to 2 KiB per message. Counts use the complete disassembly. [DXC validator interface](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/dxcapi.h)

The first synthetic compute layout uses an SRV, a UAV and 16 root constants. Its `RTS0` payload matches 11 BVH containers byte for byte after clearing the two flags. The second uses 13 UAVs, 21 SRVs, four root CBVs and a static sampler. It matches 27 unmodified AMD containers. Both tests access the last registers in the descriptor tables. RTX 2060 and WARP wrote 257 correct integer words per layout and preserved 63 sentinel words.

The native DXR fixture builds a bottom-level structure (BLAS) from one triangle and a top-level structure (TLAS) from two instances. Five inline Shader Model 6.5 queries cover hits, a geometric miss, distance clipping, translation and instance masks. Both backends hit instances 7 and 11 at distance 1 and missed the other three rays. Integration of the game's acceleration structure layout and consumers with desktop DXR remains unverified. [Microsoft DXR specification](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)

Four priority BVH containers use the first layout, are referenced by the function starting at `0x20270` and use 32×1×1 thread groups. Source-built HLSL reproduces the data operations inferred from static instructions. Independent CPU results supply the expected values.

| Container RVA | Observed input and write behavior |
|---|---|
| `0xb3ae0` | Copy the specified count of 16-byte input blocks to matching output offsets. |
| `0xb82e0` | Write the same constant to all four 32-bit words of each valid block. |
| `0xb4bd0` | Write consecutive unsigned 32-bit integers starting at a seed. |
| `0xb5be0` | Read the length parameter from the second input word and copy `64 × (parameter + 1)` bytes in steps. When reset is enabled, write the positive and negative maximum finite float bit patterns to six words at output offsets 16–39. |

Tests use disjoint buffers and cover zero length, thread-group boundaries, integer wraparound and untouched memory. Each desktop backend runs 26 cases and checks 384 words per case. The fourth operation schedules its optional reset in a second dispatch with a UAV barrier. Source cross-group synchronization, aliasing and actual game call parameters remain open.

### BVH reads and cross-lane operations

In `out/calls/desktop-roots.json`, `static_operands` from `tools/probe_roots.py` groups functions by module count, call site and operand form. The counts below describe static instructions across the 52 BVH programs. Runtime frequency is unknown.

| Read form | Modules | Static calls |
|---|---:|---:|
| Global, one word | 17 | 227 |
| Global, two words | 3 | 129 |
| Global, three words | 12 | 184 |
| Global, four words | 10 | 44 |
| Scalar, one word | 16 | 32 |

Across 27 distinct modules, all 616 reads use 64-bit addresses and end with two LLVM `undef` arguments. They provide no definite cache policy. At RVA `0xa4780` a count read from base plus 12 limits active threads. A loop in `0xc6260` reads a word at base plus four times the index and writes it to the destination. The fixture exercises both data behaviors and the scalar offset `(index << 8) | 44`.

Two successive writes to the same GPU buffer are each ordered by a UAV-to-SRV transition before reads of all four widths. Each round submits 128 synthetic requests covering low-word carry, shared and scattered addresses, a final valid vector, overrun, misalignment and different high words. The CPU uses independent 64-bit range arithmetic and compares each returned word. Out-of-range sentinels are a fixture rule. Console cache hit rate, address ownership, invalid access behavior and visibility without explicit ordering remain unknown.

Eight modules contain 75 calls to `LaneSwizzle.i32`, with immediates 30, 60, 120, 240 and 480 occurring 15 times each. The `0x95e20` sequence conditionally accumulates exchanged values and reads lane 31. This motivated a candidate five-stage prefix sum within each 32-lane subgroup. Tests exercise native waves and group-shared memory separately. The logical 64-lane path also checks both halves of the ballot mask, prefix counts, dynamic indices and lane 31. The native path checks the active minimum, broadcast, divergent votes and dynamic reads. The CPU reconstructs expectations from GPU-reported native lane IDs.

Each backend runs 38 groups with empty, full and sparse ballots and 32 single-point inputs covering every position in each half. RTX 2060 reported native width 32–32 and WARP 4–4. Both passed the group-shared 64-logical-lane test and skipped forced native wave64 on capability grounds. A capable adapter will run that case. Single-point inputs, CPU prefix sums and final ray hits provide the PC acceptance checks. [WaveReadLaneAt semantics](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/wavereadlaneat)

```powershell
ctest --preset dev -R 'bvh_reads|bvh_waves|bvh_wave64|bvh_hits' -V
```

### From buffer reads to ray hits

`bvh_hits_hardware` and `bvh_hits_warp` connect bounded address reads and half-group prefix sums to ray intersections. Box and triangle tests follow AMD GPURT's software approach. A synthetic root holds 64 child bounds with logical 64-bit triangle addresses. The 64 logical lanes test bounds and compact candidates using both half-group sums. Receiving lanes fetch vertices and reduce nearest hits within and across native waves. The fixture breaks distance ties by the smallest primitive ID. [AMD GPURT intersections](https://github.com/GPUOpen-Drivers/gpurt/blob/dev/src/shaders/IntersectCommon.hlsl)

Three successive scenes occupy the same GPU buffer. Each is written as a UAV and then read as an SRV. A CPU brute-force, double-precision plane intersection independently checks each ray's box count, triangle hit count, closest distance and primitive ID. Each round uses the same 130 rays, covering each leaf in isolation, forward, reverse and oblique rays, edges, vertices, zero distance and box-only intersections.

| Scene | Hits | Misses |
|---|---:|---:|
| Sparse layout, isolate all 64 leaves | 112 | 18 |
| Permuted addresses, overlapping geometry and equal-distance hits | 14 | 116 |
| Permuted again, with invalid address high bits, an out-of-range tail and a degenerate triangle | 108 | 22 |

RTX 2060 and WARP agreed with the CPU on every field. Closed intervals, tie breaking and invalid-leaf handling are fixture rules. The tests establish synchronized desktop reads and hit results. Scarlett node layouts and cache hints remain hypotheses to check against producer writes and consumer reads.

### Independent BVH checks on PC

Each scene sends the same 136 rays through a double-precision brute-force CPU test, a custom shader reading packed nodes, and standard DXR. DXR builds its own BLAS and TLAS from the original accessible vertices and runs inline queries. All paths check hit status and closest distance. The CPU also checks that the specific DXR-selected primitive intersects at that distance, allowing valid ties. Comparisons use triangle interiors and distance margins. Edge and endpoint cases use the preceding fixture's fixed conventions. [DXR ray extents](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#ray-extents)

RTX 2060 and WARP each completed 408 comparisons. In fault controls only the custom shader's input changes. Swapping child 0 and 32 addresses changed four results, replaying the previous scene changed 102 and aliasing invalid address high bits changed one. Each fault must alter its designated witness ray. Run `ctest --preset dev -R bvh_differential -V` for hit counts, changed results and witness indices.

Project acceptance uses reproducible PC behavior. The next game-specific check is to identify the builder selected by the main executable, its node-buffer inputs and outputs, and the downstream queries, then add the observed constraints to PC tests.

### Building bounds under contention

Static disassembly identified 51 direct calls each to `xbox.global_atomic_fmin` and `xbox.global_atomic_fmax` across seven modules each. Each call takes a 64-bit address and a float after the opcode. Integer atomic add and subtract each occurred at four sites with a 64-bit address and a 32-bit operand. Regenerate `out/calls/desktop-roots.json` with `probe_roots.py` to inspect per-module counts and operand forms.

`bvh_atomics` builds seven sets of axis-aligned bounds in GPU memory. Contending threads use 32-bit compare-and-exchange to extend six raw float fields. Two counters cross the unsigned wrap boundary. The CPU checks every field and verifies that each accepted point received a unique prior value from each counter. Inputs cover positive-only and negative-only bounds, both signed zeros, subnormals, finite extremes, invalid points and an empty bucket. The fixture skips a point with any non-finite coordinate and orders negative zero before positive zero. These are explicit desktop conventions.

The query dispatch reads the accumulated bounds after a UAV barrier and tests 54 axis-aligned rays. The CPU computes each expected bucket mask by projecting the original points onto the three axes. RTX 2060 and WARP each passed 270 ray masks over five input batches, including the empty input and a reversed 1,025-point submission. Each full submission accepted 897 points. This exercises a source-built producer and consumer with shared bounds. The observed atomic calls alone do not identify the game's node layout or special-value rules.

### Packed leaves and initialization stores

The per-module `validation.ir.packing` report follows the immediate definition of the byte selector at all 106 `V_ALIGNBYTE_B32` sites. Each selector is masked with 3. The three modules contribute 44, 44 and 18 sites. Across seven other modules, 47 `BFI` sites include variable masks, zero and contiguous low-bit masks through 1023. The desktop test uses these constraints and AMD's bit selection and byte alignment operations to build a source-owned leaf format. Its field offsets and flag assignments are fixture conventions. [AMD RDNA 2 ISA reference](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

`bvh_packed` uploads 64 packed leaf records per scene. A compute shader reconstructs their fields at four byte offsets, selects the ID and flags through a mask, then writes bounds, triangle vertices and logical addresses. The next dispatch traces the existing 130 rays directly against that buffer. A separate readback compares all 1,280 scene words plus four guard words per scene against the CPU layout. Three scenes cover address carry, an invalid high word, a truncated vertex span, degeneracy and ties. Flipping one selected ID bit in the first scene changed exactly four returned IDs. Distances and counts matched the independent geometry reference in both runs on RTX 2060 and WARP.

The static inventory records one scalar, six three-word and nine four-word cached stores. All have two trailing undefined operands. At module RVA `0xb92d0`, two pairs write the largest finite float three times, then its negative three times at the same base plus 12 bytes. `bvh_atomics` now initializes the six bounds on the GPU from dirty storage with this pair of stores. A UAV barrier orders initialization before accumulation. Empty buckets expose incomplete resets, and independent checks cover the two adjacent counters. Cache policy and the game's allocation layout remain unknown.

### Numeric operations on bounds

All 180 observed `Med3_F32` calls had two constant operands. In 156 calls they were the negative and positive largest finite floats. Six used the positive lower bound with float bits `0x0f000000` and the same upper bound. The remaining 18 used zero and `0x3f7ff972`. For ordered values these medians clamp the variable input to the two constants. The operand inventory now includes all 33 distinct direct Xbox intrinsic names.

`bvh_numeric` checks those three clamps at their boundaries, including neighboring float encodings and infinite inputs. It also tests three-input float extrema, three-input unsigned min/median/max and two independent packed binary16 lanes. For half precision the CPU decodes the exponent and fraction in double precision. Inputs cover every finite nonzero binary16 encoding and positive zero. The fixture maps signed zeros to positive zero and excludes non-finite half inputs. Each desktop backend agreed with the CPU on all 655,360 words from 65,536 input rows. NaN propagation and native Scarlett behavior remain open.

### Updating a live DXR scene

`dxr_update` moves a triangle from depth 1 to 3, changes two instance transforms and changes one visibility mask. It updates BLAS and TLAS in place using separate build and update scratch-size requirements. Seven rays cover old and new positions, visibility masks and distance clipping. Four change hit status. Both desktop backends matched the expected hit, instance, primitive and distance fields before and after the change. Each in-place result matched a fresh full build of the changed scene. [DirectX acceleration structure updates](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#acceleration-structure-update-constraints)

```powershell
ctest --preset dev -R 'bvh_packed|bvh_atomics|bvh_numeric|dxr_update|era_heap_caller_layout|static_exe_intake' -V
```

### Desktop checks for the query tail and root flags

On x64 Windows, `desktop_memory_query` observed a 48-byte write from `VirtualQueryEx`. State and protection occupy offsets `0x20` and `0x24`. All 16 trailing sentinel bytes survived. With the caller's zero initialization, even a committed read/write page fails the tail check. The test supplies seven tail bytes after the query and compares the four callers' two predicate shapes on reserved, committed read/write, read-only and no-access pages. It also checks the zero-return path.

`bvh_root_roundtrip` uses a source-built compute shader with a v1.1 root signature. The desktop serializer rejected the added `0x03000000` bits with `E_INVALIDARG`. Clearing just those bits and serializing, decoding and serializing again produced the same bytes as the original control. RTX 2060 and WARP both created the root signature and pipeline from the control shader and recovered signature. Each dispatch matched 257 CPU reference words and preserved 63 guard words. This establishes the desktop behavior for that shader. The two flags' Scarlett semantics remain open.

### Creating root signatures on a second workstation

The second maintainer workstation ran Windows 26200.9974 with SDK 10.0.26100.0, an RTX 4060 Laptop GPU (driver 32.0.16.1656) and WARP (10.0.26100.9818). At commit `127fa86`, both backends passed 408 differential ray queries with the debug layer enabled. Forced native wave64 skipped on both. The hardware reported 32 native lanes and WARP reported 4. [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7058907095)

The device probe groups containers in each DLL by the SHA-256 of their serialized root signature (`RTS0`). For one representative per group it deserializes the v1.1 descriptor, changes the flags in a memory copy, and records the serializer HRESULT. For each successful serialization it calls `CreateRootSignature` on each selected device and records that HRESULT separately.

| Local input | Containers / distinct layouts | Serializer and device results |
|---|---:|---|
| BVH builder | 52 / 16 | All 16 returned `E_INVALIDARG` in each of three cases: original flags, clear only `0x01000000`, or clear only `0x02000000`. Clearing both returned `S_OK` for serialization and device creation on both backends for every layout. |
| AMD upscaler | 29 / 3 | All three original layouts returned `S_OK` for serialization and device creation on both backends. |

The observed boundary is root signature creation. The desktop DXIL validator still rejected all 52 BVH programs, and the 29 upscaler containers contained no DXIL. An open binding test is the BVH layout shared by seven containers, including the one at RVA `0x99d60`. A source-built dispatch with CPU-checked readback would extend coverage beyond device creation for this layout.

After the first build with the Python dependencies, run the two source-built controls on WARP:

```powershell
ctest --preset dev -R '^root_device_controls_warp$' -V
```

With local inputs, generate the call inventory first. The runner verifies the DLL hashes and writes per-layout HRESULTs to `out/calls/root-devices.json`. Use `--warp-only` for a software-only report in `out/calls/root-devices-warp.json`.

```powershell
.\tools\dev.ps1
$pkg = 'game-files/<build-directory>'
$python = '.\out\pe-env\Scripts\python.exe'
& $python -B tools/call_evidence.py $pkg
& $python -B tools/probe_root_devices.py $pkg
```

### Dynamic dependencies at startup

Five `GetProcAddress` calls were inspected in the local AMD loader. The `0x10000` descriptor selects the packaged upscaler, whose five exports match. `0x20000` and `0x30000` select `amd_fidelityfx_framegeneration_xbox.dll`, absent from the inspected directory and torrent manifest. The loader checks for an already-loaded module before calling `LoadLibraryA`. Whether the main executable requests frame generation remains unknown. [AMD FidelityFX API](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/main/Kits/FidelityFX/api/include/ffx_api.h)

The executable intake report's `dependency_paths` gives an explanatory path from the entry to each reachable dependency, preferring ordinary imports and marking delay imports and confirmed AMD branches. The AMD branch data applies only to a SHA-256-matched loader. New inventories include System32 forwarder chains. Older inventories provide first-hop evidence only. Dynamic lookups and execution order remain open. With DLLs alone, trace from each one:

```powershell
.\out\pe-env\Scripts\python.exe -B tools/dependency_paths.py out/dependencies/pre-exe.json |
    Set-Content out/dependencies/pre-exe-paths.json -Encoding utf8
```

To rehearse intake with the locally built candidate, keep its report in a separate directory:

```powershell
.\out\pe-env\Scripts\python.exe -B tools/intake_exe.py out/build/dev/intake_candidate.exe --reports out/intake-rehearsal
```

The tool reads the candidate as a file and uses the manifest slot as its graph name. The local rehearsal matched the candidate's SHA-256 to the fixture and preserved all 16 baseline DLL hashes. The fixture supplied ordinary imports by name and ordinal from the missing `intake_provider.dll`, a delay import from the missing `intake_delay.dll`, and an ordinary import of `ffxQuery` from the packaged AMD loader. From that loader, descriptor `0x10000` reached the packaged upscaler. Descriptors `0x20000` and `0x30000` reached the absent frame-generation module conditionally. The independent CTest uses a temporary package without that loader and checks that the report omits the frame-generation branch. The game's startup dependencies still require its own entry image.

## Resource chunks and shader formats

### State-linked settings in a script archive

`python -B tools/script_state_probe.py path/to/script.rpf --state <label>` checks aligned zlib pages, section lengths through CHKS, schema and array extents, and parent-to-child attribute ranges. It emits aggregate counts and at most one matching example. The supported profile has five root arrays. Other layouts and entries outside the recognized zlib framing require separate inspection.

In the local base script archive, 532 children of state nodes carried orbit distance, heading and pitch bounds, blend duration and priority together. Reformatting the stored integers, floats and vectors as decimal strings reproduced all five saved value hashes for each child. This establishes authored camera settings associated with states. Activation conditions, time units, interpolation and native calls still need execution evidence. The source-built regression changes a scalar while preserving its saved hash and separately introduces a parent cycle. [Public RPF9 observations](https://github.com/walkthestars/GTA6-Asset-Info/blob/main/formats/01-rpf9.md) · [CodeWalker field-name reference](https://github.com/dexyfex/CodeWalker/blob/master/CodeWalker.Core/GameFiles/MetaTypes/MetaNames.cs)

The Windows path replay locates a record by an anchor state ID. It consumes that record's node and attribute rows in memory. After building with the dev preset, run this command with the local base archive:

```powershell
python -B tools/script_path_replay.py path/to/script.rpf --host out/build/dev/script_path_host.dll --record-with bumper_f motorbike ./bumper_f ../bumper_r ../bumper_f car/bumper_f
```

Bare paths begin at the record's `states` container. A leading `./` starts at the active node and `../` moves to its parent. Each input is supplied by the experiment. The native host walks the stored first-child and sibling links, checks reverse links and selects a unique matching child ID. A second lookup uses parent indices to verify the resulting node. A missing or ambiguous input leaves the test host at its last selection. The local motorbike record yields distinct front and rear camera bounds and restores the original bounds on returning to the front. The runtime event that supplies the state ID remains unresolved.

### Drawing BC7 textures with a recompiled pixel shader

`linearize_4k_s_tail(tile, width, height, tail_level)` reads a single BC7 level from a shared GFX10 4 KiB mip-tail tile. Supply the decoded tile, the requested pixel dimensions and its zero-based position within the tail. Whole-tile storage and per-slot bounds are checked before reading. [AMD mip-tail address calculation](https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/imported/addrlib/src/gfx10/gfx10addrlib.cpp)

To test the published fixed-symbol rANS hypothesis against an isolated BCPack prefix, run `python -B tools/bcpack_probe.py packet.bin --blocks N` with an independently established block count for that stream. A chunk can contain several streams. A header match checks the predicted byte budget and initial state. The probe reports a verified constant lane only when it can consume all symbols and reach the expected terminal state without refill bytes. An empty result leaves the lane models unresolved. This probe produces diagnostics only. [SixNG BCPack research notes](https://github.com/walkthestars/GTA6-Asset-Info/blob/HEAD/formats/06-bcpack.md) · [Microsoft BCPack modes](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_bcpack_mode)

`tools/bc7_tiles.py` converts one padded GFX10 `SW_4K_S` surface of 16-byte BC7 blocks to desktop block rows. The caller supplies the pixel dimensions and the bytes of one mip level. Each 4 KiB tile contains 16 × 16 BC7 blocks. The address equation applies to the standard non-XOR mode. [AMD GFX10 swizzle tables](https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/imported/addrlib/src/gfx10/gfx10SwizzlePattern.h)

The desktop check uploads those rows as `DXGI_FORMAT_BC7_UNORM`, runs the translated copy shader, and checks its floating-point readback against CPU-decoded RGBA values after half-precision export. Its case file starts with `BC7C` and two little-endian 32-bit pixel dimensions, followed by tightly packed BC7 block rows and one reference RGBA8 value per texel. The draw check accepts dimensions divisible by four, up to 4096 on either axis and 4 Mi pixels total.

```powershell
ctest --preset dev -R '^bc7_draw_' -V
.\out\build\dev\offscreen_check.exe hardware --check-bc7 copy.cso base.case capture.rgba
.\out\build\dev\offscreen_check.exe hardware --check-bc7-mip1 copy.cso base.case mip1.case
.\out\build\dev\offscreen_check.exe hardware --check-bc7-tail copy.cso 3 mip3.case mip4.case mip5.case mip6.case mip7.case
```

The tail check uses one logical texture per draw. It uploads all supplied consecutive levels at their original subresource indices and sets the SRV's `MostDetailedMip` to the first supplied level. The command above takes a 64×64 first resident level and tests viewports from 512×512 down to 4×4. Point sampling at the three largest sizes must reproduce magnified pixels from the 64×64 reference. Each smaller viewport must match its own reference. The test uses committed resources and measures sampling correctness.

To check a proposed first read round in the BCPack probe, append `--prefetch-mask MASK`. Set bit *i* for each of the first eight lanes expected to read a byte in that round. The probe uses the proposed lane order to locate the byte and compares it with the last byte emitted by the reverse encoder. A matching first byte tests a single refill prediction. Later refill positions and the assignment of BC7 block fields require separate evidence.

The optional capture contains verified, top-down RGBA8 pixels converted from the GPU readback. The mip test binds both levels in one texture and renders to a half-size target so implicit derivatives select the lower mip. An original mode-6 fixture varies endpoints by block and indices by texel. The negative control shifts only the lower level's compressed blocks and requires a pixel mismatch. [Microsoft BC7 mode reference](https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference)

### Reading logical ranges and probing structured resources

`tools/paged_resource.py` reads a logical byte range from an open archive and a supplied segment map. Each segment gives a physical offset, stored byte count excluding padding, decoded byte count, and codec. Raw and zlib segments may be mixed. The reader verifies the complete compressed stream and decoded size before returning bytes from it. Reads that touch an unknown codec fail. Segments after it remain independently addressable.

`tools/section_probe.py` samples up to 8 MiB at the start, middle and end of each input. It looks for 16-byte-aligned zlib streams, with stored and decoded sizes capped at 256 KiB. A section record must start and end with the requested tags, and its big-endian section sizes must exactly cover the decoded buffer. Streams cut off by a window or size cap are omitted. Counts establish record boundaries within sampled regions. Interpreting payload fields or validating a checksum inside a section requires its format.

```powershell
python -B tools/section_probe.py --first PSIN --last CHKS "game-files/<build-directory>/levels.rpf"
ctest --preset dev -R resource_range_controls -V
```

### Source-built binding, wave and texture checks

`dual_uav` dispatches through two root UAVs with a count supplied as a root constant. The CPU checks the input, active output and unwritten tail over seven counts. Decreasing the count from 129 to 128 leaves word 128 untouched. The wave tests compare a full-group mirror with a half-local mirror at lanes 31 and 32 and check an inclusive sum across both halves. With native wave64 support they also check a native mirror using the lane assignment returned by the GPU.

`texture_mips` writes an original RGBA8 checker pattern in padded 4×4 Morton tiles, decodes it into the device's copy footprints and point-samples every texel through a texture SRV. It checks all 148 texels of a 13×9 texture and its three smaller mips, plus a guard word after each level. Morton order is the synthetic input convention. Establishing a particular title's texture format and addressing mode requires separate evidence.

```powershell
ctest --preset dev -R 'dual_uav|texture_mips|bvh_waves|bvh_wave64' -V
```

### Optional compressed-record inventory

Run the following after building the native validator. The optional pass scans named records in the shader archive, decodes one raw-DEFLATE stream at a time and writes only aggregate metadata to the resource report. A candidate must fit the file, have matching stored sizes, reach the stream end, match the declared output length and contain a bounded DXBC section table with a DXIL program header. The pass caps stored and decoded spans at 8 MiB and 16 MiB. It records shader stages, section counts, LLVM bitcode framing, unused bytes at stream end and rejection reasons. The native validator samples the first accepted container per stage. These are content-based counts and depend on the recognized record format. [External record-format reference](https://github.com/ItsNotRexed/Leonida/blob/56f023a10a726da20702d4e121656ceda2391d93/tools/shader_carve.py)

```powershell
python -B tools/resource_inventory.py "game-files/<build-directory>" --scene-records --scene-validator out/build/dev/root_probe.exe
ctest --preset dev -R 'compressed_scene_metadata|static_exe_intake|desktop_memory_query|era_heap_failure' -V
```

The synthetic intake test compares the candidate's direct unresolved imports against a DLL baseline and against a second baseline with a peer caller. Each `entry_import_gaps` item records ordinary or delayed binding and whether the same module/symbol pair was already unresolved. Transitive and conditional routes remain in `dependency_paths`. The memory tests check a real one-byte Windows query failure and unsupported heap attribute bits in combination with a supported alignment.

### Raw archive scan

After building, use the source-built pixel shader as a known DXIL control.

```powershell
python -B tools/resource_inventory.py "game-files/<build-directory>" --shader-control out/build/dev/triangle_ps.cso
```

`out/resources/<build-directory>.json` records install chunks, flags, byte counts and missing layout destinations. It records file-header signatures for the build and scans all raw bytes in `appdata0.rpf`, `common.rpf` and `shaders.rpf` for format markers. Candidate DXBC containers are checked for length, section boundaries and DXIL program headers. `sample_streams` covers at most eight qualifying zlib streams per file within the first 4 MiB. Plaintext stays in memory. The sample cannot characterize the rest of the archive. Installation order and runtime I/O need independent checks. [DXIL container format](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst) · [RPF9 observation source](https://gtamodx.com/columns/eXk_fiBUaYta)

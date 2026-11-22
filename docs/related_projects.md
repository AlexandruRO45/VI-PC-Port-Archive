# Related projects and prior art

Leonida is **GPL-3.0-or-later**. Some useful reverse-engineering work lives under other licences, and some of it
cannot be mixed with this repository. This page records what we may learn from each project and where the line is.

The rule is the same for every entry: **facts are fine; code is not.** A format layout, an ABI observation or a
measured behaviour can be reimplemented independently. The source that expressed it cannot be copied.

## rpatty-dev/PercyStar

<https://github.com/rpatty-dev/PercyStar>

A source-available Rust explorer for RPF9/TTPF, with an RDNA2 disassembler, texture decoders and an `ADAT` Opus audio
decoder. Its format observations are relevant to the archive, texture and audio work here.

**Licence:** the "PercyStar Non-Commercial Attribution & Copyleft License (v1.2)". It is source-available, says
"All rights reserved", and adds non-commercial, attribution, naming and share-alike conditions. It is not
GPL-compatible, so **no PercyStar code, comments or tables may be copied into Leonida**.

**What we use it for:** reading about format facts, then implementing them independently. If a fact is useful,
record the observation in our own words, cite the project and verify it against the package or a public
specification. Do not paste even a small function "to save time".

## realdody/americas-recomp

<https://github.com/realdody/americas-recomp>

A **GPL-2.0-only** fork of [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), on its `xbox-series` branch. It
recompiles the same Xbox Series shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI
proxy.

**Licence:** GPL-2.0-only is **incompatible with GPL-3.0-or-later**. GPLv2-only code cannot be combined into a
GPLv3 work, so **no code may be copied**, and there is no "just this one file" exception.

**What we use it for:** the measured ABI facts only. The useful findings are summarised in
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi): the Xbox device entry point, the slot
deviations, the descriptor-heap start-handle return convention, the root-signature flag word and the BVH builder's
compute-pipeline path. Those are observations to verify, not code to port. The document paraphrases them; no code or
long passages are copied.

## breadbbbb/rpf9-explorer

<https://github.com/breadbbbb/rpf9-explorer>

A Windows viewer for RPF archives, and the source of the Core library behind `tools/rpf9dump` (models, textures,
TTPF pools and AWC audio). It is a `third_party` **submodule**, not code copied into this repository.

**Licence:** the repository has **no licence file** (GitHub reports no licence), so all rights are reserved by
default. A submodule is an external dependency; it is not a grant to copy its code. Do not paste its functions,
tables or comments into Leonida. If a format fact learned from it is useful, verify it against the package and
write our own implementation. The model-recovery gap measured in [models.md](models.md) is in that submodule; the
fix should go upstream, not into a copy.

## p-portela/SCARAB

<https://github.com/p-portela/SCARAB>

A GPL-3.0 "Scarlett Compatibility And Runtime API Bridge" that forwards the console D3D12 runtime to PC D3D12 (and
to vkd3d-proton on Linux). It carries a measured catalogue of the console `d3d12_xs.dll` export surface, standard
and console-only IIDs, extension interfaces, command-list interception and presentation entry points.

**Licence:** GPL-3.0. That is compatible with this project's GPL-3.0-or-later for study, and adapted code can be
used with attribution and the original licence notice, unlike the two projects above. Even so, **facts first**:
verify a claim against SCARAB's code and, where possible, this package; write the implementation in our style; and
say in the pull request which file a borrowed fact came from. Do not copy a whole component just because the
licence allows it.

**What we use it for:** the measured console runtime surface (see `docs/d3d12_xs_surface.md` when that study
lands), the IID/extension tables, and the call-sequence observations that Leonida's own
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi) still lists as unknown. SCARAB forwards to
D3D12; Leonida translates to Vulkan. The ABI facts carry over, the forwarding architecture does not.

## microsoft/libHttpClient

<https://github.com/microsoft/libHttpClient>

Microsoft's HTTP/WebSocket library for GDK titles. Its `Source/Task` folder is the implementation of the GDK's
`XTaskQueue` and `XAsync`.

**Licence:** MIT, which is compatible with GPL-3.0-or-later.

**What we use:** the task queue's headers and Win32 sources, copied unmodified into `third_party/xtaskqueue` (commit
and file list in its README) with the licence file. `runtime/shims/xgameruntime.cpp` registers those functions as
`xgameruntime.dll` imports. The HTTP and WebSocket stack is not included. Unlike the projects above, this is code
use under a compatible licence, so the MIT notice must stay with the files.

## AMD machine-readable GPU ISA specification

<https://gpuopen.com/machine-readable-isa> (reader: <https://github.com/GPUOpen-Tools/isa_spec_manager>)

AMD's XML description of the RDNA and CDNA instruction sets: instructions, encodings, operands, data formats.

**Licence:** MIT (`<License>MIT</License>` in each XML file; `isa_spec_manager` is MIT too), compatible with
GPL-3.0-or-later.

**What we use it for:** `tools/isa_coverage.py` reads `amdgpu_isa_rdna2.xml` from a path the user supplies and counts
which instructions the translator has a rule for ([isa_coverage.md](isa_coverage.md)). The XML is not vendored (11 MB);
tests use a small fixture of our own in the same schema.

## Mesa (AMD register database and format tables) and LLVM's AMDGPU backend

<https://gitlab.freedesktop.org/mesa/mesa> (`src/amd/registers/*.json`, `src/amd/common/ac_descriptors.c`),
<https://github.com/llvm/llvm-project> (`llvm/lib/Target/AMDGPU/SIDefines.h`)

**Licence:** Mesa's AMD code is MIT, LLVM is Apache-2.0 with LLVM exceptions. Both are GPL-3.0-or-later compatible,
but nothing is copied.

**What we use:** field positions and enum values, written into this repository's own code. `tools/hw_oracle/tsharp.py`
builds the T# for the hardware oracle of `tools/image_fuzz.py` from Mesa's SQ_IMG_RSRC_WORD1..4 bit ranges
(`gfx10-rsrc.json`, `gfx11-rsrc.json`), the way `ac_build_gfx10_texture_descriptor` and `ac_set_mutable_tex_desc_fields`
fill them (linear images carry their pitch in WORD4), and the GFX10 and GFX11 format numbers (Mesa's `GFX10_FORMAT`,
and for GFX11 the order of LLVM's `UfmtGFX11::UnifiedFormat`). The GFX11 numbers were confirmed on a gfx1150 by the
hardware oracle itself (a wrong FORMAT changes the fetched words); the GFX10 numbers and layout are Mesa's and not yet
run on RDNA2 hardware.

## Other Xbox Series compatibility projects (licences checked 2026-11-22)

Several other efforts target the same console surface, and some have run retail Series titles, which Leonida cannot
do until it has a game executable. The licence column is what each repository's own `LICENSE` file says (read from a
shallow clone, not from GitHub's label, which was wrong for one of them). The rule is the one above: **facts first,
cite the project and file, verify, implement independently.** What "code may be adapted" means for the compatible ones
is only that the licence allows it with attribution; it is still not the default (see the entries above for SCARAB).

| Project | Licence | Against Leonida's GPL-3.0-or-later | What it is, as far as it matters here |
|---|---|---|---|
| [sammwyy/xvalkyrie](https://github.com/sammwyy/xvalkyrie) | GPL-3.0 | compatible | C++20 layer for Xbox One and Series on D3D11/D3D12; reports that retail Series titles create their device and present frames, with shaders translated to HLSL. Its `docs/research/` records measured behaviour (the device's slot order, descriptor handles, rectangle lists, frame pacing); the D3D12.X note covers the Xbox One device, not `d3d12_xs`. |
| [omonkeyz/xbox-emulator](https://github.com/omonkeyz/xbox-emulator) | GPL-3.0 | compatible | Rust runtime continuing TomBox; reports a retail title reaching its lobby. `docs/BLOCKERS.md` holds per-slot findings for the console device and command list with the console DLL offsets they came from. Those DLLs are console system files that are not in our package, so treat each finding as a hypothesis until a Leonida measurement agrees. |
| [p-portela/SCARAB](https://github.com/p-portela/SCARAB) | GPL-3.0 | compatible | Listed above. |
| [LooperSalty/AnyXboxX](https://github.com/LooperSalty/AnyXboxX) | GPL-3.0 | compatible | Python/C++ prototype with a table of the 396 GDK runtime functions. |
| [holdmysocks/SixNG](https://github.com/holdmysocks/SixNG), [nullbrik/RageForge](https://github.com/nullbrik/RageForge) | GPL-3.0 | compatible | RPF9 asset extractors for the same package. Format facts only come from the archives; check any claim against `docs/rpf.md` and the package. |
| [californiaAlmonds/OpenScarlett](https://github.com/californiaAlmonds/OpenScarlett), [R0Y00/Xbox2Windows](https://github.com/R0Y00/Xbox2Windows) | MIT | compatible | Translation layer on SDK samples, and static-recompilation research with source-built fixtures. |
| [jadeowaowa/scarlett-layer](https://github.com/jadeowaowa/scarlett-layer) | Apache-2.0 | compatible | Loader plus an injected hook DLL, tested on self-written programs only. |
| [ivall/xcl](https://github.com/ivall/xcl) | **GPL-2.0-only** | **incompatible** | Reports retail titles running their menus. Same rule as `americas-recomp`: no code, comments, tables or prose, and no "just one file". |
| [Freminet/ScarlettX](https://github.com/Freminet/ScarlettX), [brcly/TuxBox](https://github.com/brcly/TuxBox) | none | all rights reserved | No licence file, so nothing may be copied. |
| [edupode/anyxboxseries](https://github.com/edupode/anyxboxseries) | CPAL-1.0 | not compatible with the GPL | Attribution and network-use terms that the GPL does not allow us to add. Facts only. |

How to use this page when you read one of them:

- Quote a fact in your own words with the project and file, as the entries above do, and say what you checked it
  against. A console-DLL offset in someone else's notes is a lead, not a measurement of ours.
- Their notes may describe a different console generation. The `d3d12_x.dll` material is Xbox One; Leonida's package
  has `d3d12_xs.dll`. State which one a fact was measured on.
- If you read an incompatible project (the `GPL-2.0-only`, no-licence and CPAL rows) while writing a change, say so
  in the pull request so the provenance can be checked.

## The line

| Fine | Not fine |
|---|---|
| A format layout, field offset or enum value | Copying the function that parses it |
| An ABI measurement (slot number, return register, flag word) | Copying the proxy, thunk or struct definitions |
| A behaviour observed by running a tool | Copying comments, identifiers or test code |
| A public specification or vendor documentation | A line-by-line port, a translated port, or a "cleaned up" copy |
| A table of facts you rebuilt from your own observations | A table copied from the source, even with renamed columns |

A table of constants is still expression. Rebuild it from your own measurements or a public specification. A
struct definition is still expression even if the field names are changed. If the only way to use a fact is to
copy the surrounding code, stop and find the fact another way.

When in doubt:

1. Write down the fact, not the code.
2. Cite the project and the section or file where you read it.
3. Verify the fact against the game package, a public specification or a test of your own.
4. Implement it independently, in this repository's style.

Never paste code from either project into a pull request, an issue, a commit message or a chat log that becomes
part of the repository history. If a contribution was written while looking at incompatible source, say so in the
pull request so the provenance can be checked.

Other dependencies keep their own licences; see the Credits and License sections of [README.md](../README.md).

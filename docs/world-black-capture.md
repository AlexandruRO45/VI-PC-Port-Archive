# One world-frame capture: black after fade

Do not start a title run without separate approval. Use the approved diagnostic binary containing this branch, Series X, the same pipeline cache and settings as the comparison run, no `VICTORY_WAIT_VALUE=1`, and no `VICTORY_SHOW_TARGET` so the title's actual source goes to present. Do not copy game data or captures into Git.

## What the historical run establishes

`20261125-084014-upwv/stderr.log:3868` records the `255 -> 0` fade at present 4154. At line 5352 (12:43:35.107610Z), in the recording interval labelled 4227 by nearby first binds, the old `copy_texture_region` warning was emitted after `host_resource(dst)` failed. It was a once-only generic warning; it contains no source format, dimensions, subresource, box or destination. The nearby computes include `e74c876bd581859b` immediately before and `1c02f8131c508c75` after; the luminance binds `23d507cbb6a22e86` / `b0095eb50222e808` are ~96 ms earlier. Logs across recording threads cannot attribute the copy to a producer or consumer. In particular the warning is after the fade, not evidence of the cause of the first black present.

Upstream PR [#67](https://github.com/Marvisak/VIctory/pull/67) was open and main still had the stub when checked. This port uses `GetCopyableFootprints` and a compact aligned READBACK copy of the source box, retaining source/destination and publishing only the copied rows after the host fence, before bottom-of-pipe completion writes. It handles supported single-sample color formats (including mip/array/volume subresources), not depth/stencil, BC, packed video or incompatible texel sizes. It does not make a subsequent GPU operation in the **same** host submission see a CPU-published title buffer; if the title does that, it needs a separate GPU-side path. The first 16 eligible calls log `texture-to-buffer copy requested` (full descriptors and destination VA), then `texture-to-buffer fence completed` for completed copies (including raw first bytes, before publication). `n` joins the two events; `recording_after_present` is a recording-time counter, not a submission ID.

## Capture settings

After approval, build/preflight under `with-build-lock.ps1` before starting the game. In a fresh PowerShell session, apply the following environment to the existing approved Series X launcher/preflight for the exact diagnostic commit (do not use `-Build up` with an older binary by accident):

```powershell
$dir = 'D:\GTA 6\wt\victory-data\captures\world-black-' + (Get-Date -Format yyyyMMdd-HHmmss)
New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
@('VICTORY_CAPTURE_FRAMES','VICTORY_DUMP_WRITERS','VICTORY_SHOW_TARGET',
  'VICTORY_GPU_TIMING','VICTORY_TRACE_SHADERS','VICTORY_D3D12_DRED',
  'VICTORY_D3D12_DEBUG','VICTORY_D3D12_GBV','VICTORY_WAIT_VALUE',
  'VICTORY_PEEK','VICTORY_PEEK_SHADOW') | ForEach-Object {
    Remove-Item "Env:$_" -ErrorAction SilentlyContinue
}
$env:VICTORY_CAPTURE = $dir
$env:VICTORY_CAPTURE_LISTED_ONLY = '1'
$env:VICTORY_CAPTURE_WORLD = '1'
$env:VICTORY_DUMP_TARGETS = '1'
$env:VICTORY_LOG = 'info,victory_gpu::trace=debug'
```

If using `run-victory.ps1` after integrating/building the candidate into the approved `up` tree, pass `-Build up -Console series-x -Diag -NoBuild -NoDebugLayer -NoDred` plus the previously agreed session settings. Otherwise use the approved launcher pointing to the exact candidate binary and preserve stdout/stderr and launch settings with the captures. Use `-Log $env:VICTORY_LOG` if the launcher would otherwise override the environment. Stop after the selected present PNG and dump records are written; no need to capture a second world frame. Recording ahead means command notes and a few probes can miss commands recorded before the arm; submission dumps use the actual submitted lists. If needed, use a confirmed absolute number via `VICTORY_CAPTURE_FRAMES` **instead of** the relative selector (unset `VICTORY_CAPTURE_WORLD`). Do not assume 4228 repeats.

`WORLD=1` picks the next present after the first present with at least 500 host draws. Verify the `world's first frame` log after the main fade and evidence of world geometry; the draw threshold is a heuristic. In the historical log ResetExposure bound in the 4227 interval and CalculateExposure in 4228. Capture offset 2 in a separate authorized run only if offset 1 has no steady-world exposure. `LISTED_ONLY=1` excludes the upstream power-of-two PNGs and avoids indirect readbacks recorded outside the selected interval; it also enables CPU command notes only while recording the selected frame, without enabling DRED. `DUMP_TARGETS=1` still tracks target IDs and may add many readbacks within that one frame; compare memory headroom and avoid enabling broad debug, GBV, per-list timing or `DUMP_WRITERS` for this run.

## Trace backwards from Present

Join the actual title present resource to its last writer, then match consumer SRV/UAV/RTV by **resource/subresource, slot and submission order** (also log the backing VA for aliases). `frame-N.png` is the host color image; `frame-N-sSUB-*` is the contents after each submission, and other `frame-N-*` files are end-of-frame targets. Images are downsampled/normalized; `-by-` is a recorded writer label, not necessarily the last writer. Check raw `tiny target` dwords/floats and RGB, not alpha, PNG brightness, or a missing image. Some targets/formats or views may not be dumped; compare at pass boundaries if producers share one submission. The following anchors are from the historical log and Dev Shaders' censo, not a proven execution order:

| Edge | Evidence to collect |
| --- | --- |
| G-buffer | f28 albedo/normals plus depth; prove geometry and the consumer's IDs. `f0c91b4d2578f5bf` generates material masks, not the G-buffer. |
| HDR light | Probe outputs of `ab619d0b7b7684d2` (Scene_Lights) and `6a62009bc6a4146e` (Scene_Emitters) as candidates; find the actual color SRV edge, not just f10/f26 dimensions. If zero, inspect prior tile/shadow producers and indirect counts. |
| Luminance and exposure | `23d507cbb6a22e86` / `b0095eb50222e808`; min/max `5e436c09d90ed218` / `fc377576bf880c30` / `92298455bb30ed0e`; histogram/reduce `214cae4f6e1ef745` / `e622b718b6d9b057`. Distinguish ResetExposure `42ac930cbc57650f` from CalculateExposure PS `00fae01469d3945d` (VS `414c1700b5d7fb36`); read the exposure actually bound at Composite, not merely the reset value. Match copy src/dst to that chain before linking the missing copy to exposure. |
| TAA | `e535cfd7baac93b8` PrepareFrameStats and **TAA2** `df11dcd012cd7382` Resolve; check input color, depth, motion, history and output UAV, with raw values. TAA2U Resolve `22f4971293b8fdcc` was not observed in this run. |
| Upscale | `0309a446e1a48352` / `4f4ec7b16673362f` SpatialUpscaleWithDithering; compare actual SRV, RTV and dimensions. |
| Composite to present | VS `7affbd5bbaa830f3`, PS `5970ee6dbc47ec61` / `21da915f3071c7a5`: compare color and exposure inputs, output RTV, any later fullscreen work, and last copy to host present. |

### Check the zero-fill hypothesis separately

Once a *buffer-backed* exposure/light input is identified and its title VA is known **in the run being measured**, set `VICTORY_PEEK_SHADOW=<hex VA>:16,<hex VA>:16` at launch for a controlled capture where these addresses are known/reproducible. Probes are limited to 64 bytes per range and to the listed frame; the address and length must fit an existing shadow flush range. Compare consecutive `shadow refresh probe` before/after for the same shadow pointer, VA and refresh span under the submission lock. `readable=false` with nonzero host bytes turning to zero is evidence for that shadow range, not by itself proof the pass consumed it. Use `VICTORY_PEEK` separately for title memory after submissions. Do not recycle absolute VAs from an earlier launch without confirming them. Without a stable VA, first use the selected frame to identify the resource and design a targeted logical-resource probe; no matching line is not evidence against zero-fill. Texture backing is not a Shadow: use producer/consumer readbacks or a targeted staging probe for that case.

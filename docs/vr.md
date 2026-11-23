# VR mode: OpenXR and the virtual cinema

With `LEONIDA_VR=openxr` the D3D12 layer shows every frame it presents on a flat screen in a VR headset, while the
desktop window keeps showing it too. This is the first step towards VR. It is not stereo: the game's camera is
untouched, and the headset shows the same 2D frame as the window, on a screen in front of you. It works before the
game boots, so it can be tried now with a test card (`openxr_cinema_probe`, below).

It is optional, off by default, and needs a build with `-DLEONIDA_OPENXR=ON`. Without that option nothing of it is
built and the runtime is unchanged, except that a `LEONIDA_VR` left set is answered with one log line saying the
build has no OpenXR. Windows only, like `gta6rt`.

## Building

`-DLEONIDA_OPENXR=ON` links the Khronos [OpenXR loader](https://github.com/KhronosGroup/OpenXR-SDK) (Apache-2.0)
statically into the D3D12 layer. The build downloads nothing unless asked. The loader comes from one of these:

- **An installed OpenXR SDK** that `find_package(OpenXR)` finds: set `OpenXR_DIR` or `CMAKE_PREFIX_PATH`.
- **`-DLEONIDA_OPENXR_FETCH=ON`**: downloads the pinned OpenXR-SDK release 1.1.65 from GitHub once at configure time,
  checks its SHA-256, and builds only its loader (static, with its bundled jsoncpp; no Python needed).
- **A local copy of that archive**: extract it and pass `-DFETCHCONTENT_SOURCE_DIR_OPENXR=<path>`; it is built the
  same way, without a download.

With the option on and none of these, configure stops and says so.

```powershell
cmake -S runtime -B runtime/build -DLEONIDA_OPENXR=ON -DLEONIDA_OPENXR_FETCH=ON
cmake --build runtime/build --config RelWithDebInfo
```

## Running

| Variable | Meaning | Default |
|---|---|---|
| `LEONIDA_VR` | `openxr` turns VR mode on. Unset, empty, `0` or `off`: off. Any other value is reported and ignored. | off |
| `LEONIDA_VR_SCREEN_DISTANCE` | Metres from your head to the screen's centre (0.5 to 20). | 3.0 |
| `LEONIDA_VR_SCREEN_WIDTH` | The screen's width in metres (0.25 to 40). Its height follows the frame's aspect ratio. | 3.2 |
| `LEONIDA_VR_SCREEN_HEIGHT` | How far the screen's centre is above your eyes, in metres (-10 to 10; negative is below). | 0 |

The defaults put a 16:9 frame on a 3.2 × 1.8 m screen 3 m away, about 56° wide. A value that is not a number in its
range is logged and the default is kept. The screen is placed where your head was when the headset session started,
straight ahead; use the runtime's recenter if it ends up behind you.

The OpenXR runtime is the one Windows has set as active (Virtual Desktop's VDXR, SteamVR, Meta Horizon Link, ...).
`XR_RUNTIME_JSON` overrides it, as for any OpenXR application.

VR mode never stops the game. With no runtime installed, no headset connected, or a runtime that lacks something
(below), the log says why once, and the window works as without VR:

```
[rt] vr: VR unavailable: no headset connected (XR_ERROR_FORM_FACTOR_UNAVAILABLE); showing the window only
```

SteamVR with no headset fails one step earlier, as `xrCreateInstance failed (XR_ERROR_RUNTIME_FAILURE)`, after the
OpenXR loader's own `Error [GENERAL | xrCreateInstance | OpenXR-Loader]` lines.

When it works the log shows the runtime, the headset and the screen:

```
[rt] vr: OpenXR runtime extensions: vulkan_enable2 yes, EXT_eye_gaze_interaction yes, FB_foveation no, ...
[rt] vr: OpenXR runtime: <runtime name> <version>
[rt] vr: headset: <system name> (swapchains up to <width>x<height>)
[rt] vr: OpenXR session on queue family 0; frames go to the headset and the window
[rt] vr: virtual cinema: 1920x1080 frame on a 3.20 x 1.80 m screen 3.00 m away (56 degrees wide)
[rt] vr: headset session running
```

Please include the `extensions` line when you report a VR result: it says what your headset offers for the next
steps (eye tracking and foveated rendering, below).

## How it works

`runtime/shims/openxr_session.h` has the details. In short:

- **Instance and device.** OpenXR's Vulkan binding (`XR_KHR_vulkan_enable2`) wants the runtime to see the Vulkan
  instance and device being made. `create_vulkan_device_standalone` (`d3d12_vulkan_device.cpp`) makes them through
  `xrCreateVulkanInstanceKHR` and `xrCreateVulkanDeviceKHR` with the layer's own extension and feature lists, and
  takes the GPU the runtime names (`xrGetVulkanGraphicsDevice2KHR`) first. If it is not usable, or
  `LEONIDA_VK_DEVICE` excludes it, VR is turned off and the layer picks as before.
- **Frame loop.** A thread of its own runs `xrWaitFrame`, `xrBeginFrame` and `xrEndFrame` at the headset's refresh
  rate, apart from the game's presents, so the game's pacing (`FramePacer`, the window's vsync) is unchanged. It
  uses the layer's first graphics queue under the layer's submit mutex, as OpenXR requires for that queue.
- **Virtual cinema.** `VulkanPresenter` hands each frame it shows to a `PresentMirror` once the frame is in the
  window's swapchain image, so a plain blit, a composed PresentX frame and a blank frame all reach the headset alike.
  The mirror copies the image into a headset swapchain of the same size, in the same command buffer, and the frame
  loop shows the last copied image as an `XrCompositionLayerQuad` in the `LOCAL` space. The copy is byte for byte
  into the sRGB format with the window's channel order, so the colours match the window.

Not yet:

- **HDR10.** A window swapchain in HDR10 (a PresentX plane that asks for it, on an HDR display) is not mirrored; the
  log says so once. SDR frames are.
- **Resolution.** The headset gets the window's image, at the window's size: a small window gives a blurry screen.
  Make the window larger (or full screen) for a sharper one.
- **A window larger than the headset's largest swapchain** (`swapchains up to` in the log) is not mirrored until it
  is smaller.
- **Stereo, head tracking in the game, controllers**: they need the game's camera (`docs/modding.md`, RT-5). The
  session is built for them: later sources of layers (projection layers per eye) plug into the same frame loop.
- **Eye-tracked foveated rendering** (for example Steam Frame, Quest Pro, Varjo): the extensions it would use are
  only listed in the log for now: `XR_EXT_eye_gaze_interaction` for the gaze, and `XR_FB_foveation` /
  `XR_FB_foveation_vulkan` / `XR_META_foveation_eye_tracked` or `XR_VARJO_quad_views` /
  `XR_VARJO_foveated_rendering` with `VK_KHR_fragment_shading_rate` for the foveation itself. None is enabled.

## Testing

Without a headset (CI-safe, no runtime needed):

- `vr_config_test` (ctest `vr_config`, every platform, built without `LEONIDA_OPENXR` too): `LEONIDA_VR` and the
  screen variables, the screen's pose and size, the extension line, and the headset swapchain format.
- `openxr_fallback_test` (ctest `openxr_fallback`, with `LEONIDA_OPENXR=ON`): CTest points `XR_RUNTIME_JSON` at a
  file that does not exist, so there is no runtime whatever is installed. VR must be reported unavailable once, and
  a device and window presenter made with `LEONIDA_VR=openxr` must present as without VR.

With a headset, by hand: `openxr_cinema_probe [seconds]` (built with `LEONIDA_OPENXR=ON`, default 60 seconds) opens a
1280×720 window with a test card: red in the top-left corner, green in the top-right, blue in the bottom-left, and a
white bar sweeping left to right once a second. The headset must show the same card on its screen: colours as named,
corners in their places (not mirrored, not upside down), and the bar moving smoothly.

```powershell
$env:LEONIDA_VR = "openxr"
runtime\build\openxr_cinema_probe.exe 60
```

### Quest 3 through Virtual Desktop (VDXR)

1. Connect the Quest 3 to the PC from the Virtual Desktop app in the headset.
2. In Virtual Desktop's streaming settings, set the OpenXR runtime to VDXR. While the headset is connected, VDXR is
   then the runtime OpenXR applications get.
3. Start `openxr_cinema_probe` (or `gta6rt` with `LEONIDA_VR=openxr`) on the PC, for example from Virtual Desktop's
   view of the desktop. The headset switches from the desktop view to the screen in front of you.

### Quest 3 through SteamVR

1. In SteamVR's settings, OpenXR, set SteamVR as the active OpenXR runtime.
2. Connect the Quest 3 to SteamVR (Steam Link, or Quest Link / Air Link with SteamVR started from it).
3. Start the probe or `gta6rt` with `LEONIDA_VR=openxr` as above.

Meta Horizon Link's own runtime should work the same way once it is set as the active OpenXR runtime in the Meta
Horizon Link app; it has not been tried yet.

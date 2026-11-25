# Progress and compatibility

The overall bar gives equal weight to verified library behavior and shader translation. The library percentage counts the reviewed behaviors Frosty can perform. The shader percentage counts entries in its surveyed archives that produced compiler output. Each library behavior counts once, including paths that use packaged or host DLLs. Each verified behavior needs an observable result from the game or a focused check. The percentage describes this workload and moves as games expose new requirements. [Progress map](progress.svg) · [Library audit](LIBRARIES.md) · [Shader survey](SHADERS.md)

Project Scarlett shader preparation compiles 12,087 of 16,868 captured requests (71.7%), with 4,781 unsupported and no compiler failures. The pass took 87 seconds because 12,038 compiled shaders were reused from earlier translator builds. Of the mesh requests translated at an earlier checkpoint, all 74 created D3D12 pipelines on an RTX 4070 Ti SUPER under stock Proton Experimental. That GPU check was not repeated for the later translations. The captured requests come from recorded runs of this build. [Preparation results](../evidence/projectscarlett-shader-preparation.json)

With the packaged Xbox graphics settings on stock Proton, Frosty rendered its main menu and accepted Start Chase. The game played Wake Up and activated the first street corner of the chase intro. The NVIDIA driver reported a GPU memory fault before the shooting tutorial. Earlier runs with reduced settings reached the shooting tutorial and the city. [Original-settings menu](../evidence/matrix-awakens/packaged-settings-menu.webp) · [Wake Up transition](../evidence/matrix-awakens/packaged-settings-room.json) · [Earlier chase and city](MATRIX_AWAKENS.md)

## Games

| Title | Current result | What remains |
| --- | --- | --- |
| **Lanes** | Runs through TuxBox's Xbox graphics, memory and input providers. Its replay checks GPU output and game state. | Continue testing presentation, input and other providers across systems. [Lanes](../examples/lanes/README.md) |
| **The Matrix Awakens (Frosty)** | With the original Xbox graphics settings, a Linux run rendered the menu, played Wake Up and loaded the first street corner of the chase intro. Earlier reduced-settings runs reached the highway and city with missing and distorted surfaces. | Verify the shooting tutorial, image quality and frame times with the packaged settings. [Setup and results](MATRIX_AWAKENS.md) · [Recorded original-settings run](../evidence/matrix-awakens/packaged-settings-room.json) |
| **ProjectScarlett (GTA VI 1.0.2.1)** | An offline Linux run got into the story intro: the game reported `Game Running` at frame 10,146 after 9 m 41 s, then crashed. [Run record](../evidence/projectscarlett-gameplay.json) · [Startup record](../evidence/projectscarlett-startup.json) | Find the crash cause, support the remaining shaders, shorten large driver compiles, then check images, input and frame times. [Setup guide](PROJECTSCARLETT.md#what-to-expect) |
| **Fortnite** | One supplied build provided graphics API call-site evidence. | Test initialization and rendering in TuxBox for a specific build. [Device evidence](../evidence/fortnite-device.json) |

No other retail title is currently claimed playable. Record the build, settings, host, provider and observed faults for each run, and distinguish starting, rendering, input and sustained gameplay.

## Current priorities

### ProjectScarlett

1. Find the cause of the crash in the recorded gameplay run, including the GPU memory waits that never reach their target.
2. Make large translated compute shaders compile quickly enough to use. Most unsupported programs need this before translating them helps. [Shader coverage](SHADERS.md#projectscarlett-refusals)
3. Support the remaining refused shader forms, ranked with the [pipeline failure record](PROJECTSCARLETT.md#finding-why-pipelines-fail).

### The Matrix Awakens

1. Finish the five remaining non-ray-tracing pixel shaders. Two light hair strands and three cull distance-field objects. Check their output with the packaged bindings on both GPUs. [Shader survey](SHADERS.md#measured-on-the-matrix-awakens)
2. Identify the GPU command that faults while the first chase street corner loads, then verify the shooting tutorial with the packaged settings. [Retail result](../evidence/matrix-awakens/packaged-settings-room.json)
3. Check the menu, characters, hair, vehicles and city in the game. Test input and audio, then measure cold and warm frame times.
4. Repeat title and provider checks on additional hardware and games as their executables become available.

The default compiler still contains a TomBox-derived HLSL emitter. TuxBox's independent emitter is growing, but it does not yet replace the legacy path for all supported shaders. [Source attribution](REFERENCES.md#shader-compiler-attribution)

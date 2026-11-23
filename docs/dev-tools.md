# Development tools

From the Leonida checkout root, use Python 3 to list the development tools:

```powershell
python -I scripts/dev.py --help
python -I scripts/dev.py discover
python -I scripts/dev.py discover --json
```

From another directory, pass the path to `scripts/dev.py`, quoting it if needed.
The inventory always describes the checkout containing that script.

`boot-trace` points to the analyzer for startup JSONL and D3D12 stderr logs.
`guest-scene` points to the headless synthetic guest runner. Run a listed help
command from the checkout root to see the producer's options. The inventory
checks for each producer script and lists its prerequisites without checking
those prerequisites or running the producer. Live attach, pause, step, input,
and screenshots are unavailable through this entry point.

## JSON output

`discover --json` writes one object to stdout:

- `schema_version` identifies the format. The current version is `1`.
- `tools` lists entries by `id`. Each `path` uses forward slashes and is
  relative to the checkout root. `present` indicates whether the script is
  a file. `help_argv` is a list of arguments for a command run from the checkout
  root, or `null` when the script is missing. `requirements` lists prerequisites
  for using the producer.
- `live_controls` contains an `available` boolean and a `reason`.

Exit code `0` indicates that the inventory was emitted, including when a
producer script is missing. Invalid CLI arguments exit with code `2`.

## Run a synthetic guest scene

Use the existing headless runner with a loader and guest built on Windows
with Vulkan, static SDL3 and `LEONIDA_GUEST_SCENE=ON`. Supply the synthetic
M1.5 shader fixture root containing `spirv/`. See the
[runtime instructions](../runtime/README.md#a-600-frame-headless-guest-scene)
for the build, fixture and Wine setup.

```powershell
python -I scripts/dev.py guest-scene --help
python -I scripts/dev.py guest-scene --loader "build/gta6rt.exe" --guest "build/tiny_guest_draw.exe" --shaders "m15 shaders" --out "scene run" --negative-controls
```

Pass `--out PATH` or `--out=PATH` with a new or empty directory. Relative
paths are interpreted from the directory where you run the CLI, including
when you invoke `scripts/dev.py` by its full path from another directory.

The runner uses exit code `0` for a successful scene and `1` for a scene
check failure. Exit code `2` indicates a usage or prerequisite error. The
runner prints the path to its structured `report.json` when it writes the
report. Open it in the output directory for the result and per-run details.
The directory also holds the logs and any frame records, PPM images and
other outputs produced before the run finished. A rejected run may leave
no report. If the runner rejects an occupied directory, any report already
there belongs to an earlier run.

You can also invoke `python -I scripts/run_guest_scene.py` directly.

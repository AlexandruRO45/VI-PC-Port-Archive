# Local viewer catalog

Point the builder at a local directory containing `rpf9dump models` exports, existing GLBs, or both.
Use a writable catalog directory outside the checkout and separate from the package and export roots.

```powershell
python -I tools/viewer_catalog.py --exports "<work>/exports" --out "<work>/catalog" --work "<work>"
```

`--exports` is repeatable. Each root can hold `data/models/manifest.json`, `packs/models/manifest.json`,
other archive exports in the same layout, or self-contained GLBs anywhere below it. The builder also
looks in `<work>/assets` when it is separate from the catalog. It reads GLBs already in the catalog
directory in place, including the `props/<archive>/<stem>.glb` layout, and imports files from other
roots. A `<stem>.shader.png` sibling becomes the optional native preview. Paths in `catalog.json` are
relative to its folder with forward slashes and URL-encoded segments, such as `props/data/Crate%20%2B%23.glb`.

```powershell
python -I viewer/serve.py --catalog "<work>/catalog" --world "<work>/world"
```

Open **Assets** from `http://127.0.0.1:8765/` to browse ready models as the catalog builds.
Omit `--world` to use the test terrain.

To fill in missing exports automatically, use the documented local `rpf9dump` build and add `--game`.
Point `--rpf9dump` at the local executable or DLL if the builder cannot find it.

```powershell
python -I tools/viewer_catalog.py --game "<package>" --exports "<work>/exports" --out "<work>/catalog" --work "<work>"
```

This first indexes reusable exports, then works through the ten model archive locations documented
in [Model recovery](models.md), including story, shared minigame, rest-of-world and levels archives.
Repeat `--archive` with documented package-relative paths to select a subset. New OBJ exports go
under `<work>/rpf9dump-models`. The existing exporter handles those inputs. Use `--batch-size` to
adjust catalog checkpoint frequency and `--max-new` to pause after a number of successful additions.
Rerunning the command reuses valid output files. The default free-space reserve is 4 GiB. Archive
preflights require additional headroom for their OBJ exports, including 24 GiB for `packs.rpf` and
200 GiB for `levels.rpf`. An export also stops if free space falls below the reserve during the run.

### Windowed import of large archives

A whole `levels.rpf` (about 71 GiB) needs more than 200 GiB of temporary OBJ. Add `--window-mib N` to export
it, and any archive named with a repeatable `--window-archive` (default `levels.rpf`), as consecutive
`rpf9dump models <src> <out> --start BYTES --length BYTES` windows. The flag is opt-in and needs `--game`.
Archives not larger than the window, and every run without the flag, use the whole-file path above.

```powershell
python -I tools/viewer_catalog.py --game "<package>" --archive levels.rpf --window-mib 256 --out "<work>/catalog" --work "<work>"
```

A 256 MiB window of `levels.rpf` recovered 1833 records in about 9 s and wrote about 1.0 GiB of OBJ, so
each window preflights four times its length plus the reserve. The windows run in order. Each one is
converted into the catalog, which is published after every window. Its staging folder under
`<work>/rpf9dump-models/.../windows/<archive>/<start>-<length>` is deleted only once all its rows converted without a
failure, the catalog is written and the checkpoint `<out>/.window-state.json` records the window. The window folder name goes
into each converted GLB's path, so identical filenames in different windows stay apart.

Anything else leaves the window partial: a full disk, a `--max-new` pause, an exporter failure or a row that
failed. The ready models stay in the catalog, the exit code is nonzero unless the run only paused, and the status
message says `levels windows complete: N/M`. The next run exports every incomplete window again from scratch and
reuses GLBs it already wrote, so an interruption at any point is safe. If a crash happens between the checkpoint and
the cleanup, the leftover staging folder is removed on the next run. Changing `--window-mib` or the archive size
discards the checkpoint. A window that cannot be exported is reported and the others still run.

Coverage limits: a window is a raw slice of the archive, so a resource that straddles a window edge may not be
recovered. Those candidates are counted in the status message as `window candidates not recovered` and are not
treated as failures, so `complete` means that every window finished, not that no resource was lost at an edge.
This slice only decodes ordinary static models; placing them in the world needs joins outside it. After a window's
OBJ is released, its models keep the name and category of the previous `catalog.json`.

The builder uses `packs.rpf` provenance and recognized part-name markers for Ped parts. It uses
existing `vehicles/` directories or verified names from an export root's `vehicle-names.txt` to
identify Vehicles. The file takes one verified model name per line and can also be supplied via
`--vehicle-names`. Uncertain names appear under Props. Known fragment bounds grids and empty
manifest rows are omitted. Other decoded fragment geometry may appear in raw bind space.

The catalog covers ready static geometry, textures already embedded in existing GLBs, and saved
previews. OBJ conversion produces geometry without reconstructing game materials. Fragment children,
complete vehicle assemblies and animated pedestrian assembly remain outside the catalog's coverage.
Status names archives whose manifests are unavailable and counts known decode, recovery and file
failures. `complete` describes the selected run's scope. A pause or an error leaves it false and
preserves the ready entries. A deliberate pause exits zero. A reported build failure exits one.

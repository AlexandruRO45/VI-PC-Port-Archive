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
relative to its folder with forward slashes and URL-encoded segments.

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

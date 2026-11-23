# Local viewer parts index (ASSET-375)

`tools/viewer_parts_index.py` builds a reusable SQLite database from existing exports. It reads manifests and entity JSONL, checks exported paths for availability, and never opens source RPF or TTPF files. It keeps drawable, placement and material references for later assembly.

Use uv-managed CPython 3.12.13 for the example below. Run `uv python find 3.12.13` to locate the interpreter if it is not the default `python` in your shell.

```powershell
python -I tools/viewer_parts_index.py `
  --out D:/local-index/parts.sqlite `
  --models D:/exports/sample/models/manifest.json `
  --window D:/work/window-manifests/region/00000000000000000000-00000000000001048576/catalog-map.json `
  --catalog-root D:/served-catalog `
  --world-window D:/placement-export/entities.jsonl D:/placement-export/report.json `
  --textures D:/textures/sample-pool/manifest.json `
  --texture-dictionaries D:/exports/texture-archive/textures/manifest.json
```

Supply any combination of inputs. Repeat `--models`, `--window`, `--world-window`, `--placements` and `--textures` for additional files. `--placements` accepts an entities JSONL on its own with unknown scan bounds. `--region MIN_X MIN_Y MAX_X MAX_Y` selects placements within inclusive XY bounds in world metres. All exported LOD tiers remain eligible. The output goes outside the repository and input directories. Use `--source-root` to also keep it outside a larger input tree.

OBJ and DDS paths are manifest-relative and must stay within the manifest's directory by default. `--asset-root` names a containing export root when assets are linked within a larger local tree. The index records absolute local paths. For ordinary model manifests an existing same-stem GLB is recorded, or the manifest's `Glb` path when supplied. For retained windows, pass the version 1 `catalog-map.json` commit marker from the catalog export. Its manifest digest, source metadata and scan window must agree with the retained rpf9dump manifest. The marker joins by exact `File` and contributes an existing GLB only for an unambiguous `ready` entry. GLB URLs are decoded within `--catalog-root`. A missing GLB or a skipped, failed or ambiguous mapping still leaves the drawable's metadata available.

## Coverage and joins

- `inputs` records paths, size, SHA-256, mtime, producer coverage and row counts. `archives` records source paths and fingerprints supplied by the exports. Use `scan_windows` for distinct reported spans. Unknown bounds and exporter recovery or decode gaps stay visible. A reported whole-archive scan can still have gaps.
- `models` has one row per normalized source archive path, resource offset and `Drawable` pointer (`-1` represents the default drawable). `model_exports` records provenance and `model_assets` lists distinct existing paths per drawable with a count of the identities sharing each path. Model identities survive repeated or overlapping scans of the same archive path. Moving or renaming a source archive in its producer metadata changes its identity.
- Join `placements.key_id` to `placement_matches.key_id` and `placement_candidates.key_id`. The candidate set uses the lower-cased Latin-1 JOaat and the seed-zero MurmurHash3 x64 low 64 bits of the original-case UTF-8 model stem, following `world_build.py`. A parsed placement key must agree for a candidate to be eligible. Without a usable key, the JOaat candidates are unverified. Match statuses distinguish verified, unverified, ambiguous, key mismatch and unresolved archetype. The `entity` JSON keeps the exported transform and map/LOD fields.
- `materials.model_id` and `texture_refs.material_id` connect shader slots and parameter references to a drawable. `texture_candidates.ref_id` lists distinct DDS paths by priority, from embedded index to embedded asset key to embedded name hash to external TTPF asset key. A shared embedded path has ambiguous ownership and stays in `embedded_textures` for inspection. `texture_matches` counts candidates at the highest available priority. A missing DDS, an unexported DDS or a path outside its root cannot become a candidate. `ttpf_textures` keeps those file statuses. A model with no material references has no entries in `texture_matches`.
- `--texture-dictionaries` inventories ordinary `rpf9dump textures` exports separately. `dictionary_textures` keeps the dictionary offset, texture index, supplied hashes, descriptor metadata and DDS path candidates under each exported `Dir`. This inventory is excluded from material matching.

Exact placement file copies with the same reported scan scope are imported once. With bare `--placements` inputs, byte-identical files share an unknown scope and are also imported once. Distinct overlapping windows can contain repeated placements because the JSONL omits resource offsets. Supply disjoint placement windows when unique counts matter. The tool keeps repeated entities within one file.

Rerunning the command rebuilds the index from its supplied inputs and atomically replaces the previous database on success. Structural failures, conflicting model or source metadata, and stale commit markers exit with status 2 and leave an existing database intact. On Windows, an open SQLite reader can prevent replacement. Close connections to the output database and rerun if replacement is denied. Skipped invalid rows and unavailable paths are counted in `inputs.counts` and the printed summary. File availability reflects the build-time exports. Keep the index and logs local.

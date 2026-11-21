# Optional skeleton metadata export

This integration depends on **unmerged rpf9-explorer PR #3**, commit
`374f9a74df49ea22d8ac6e095196f56a03994de3`. The submodule pin is explicit;
do not merge this dependency change without maintainer review.

Normal OBJ export is unchanged. For local research, a caller who has independently
identified the correct external drawable may opt into JSON sidecars:

```powershell
rpf9dump models <archive.rpf> <output> --skeleton-system <system-page-file> --skeleton-drawable-offset <decimal-offset>
```

The skeleton file must be the decoded **system page**, not an RPF archive or a
combined system/graphics file. The offset is the drawable's byte offset in that
page. The tool reuses the dependency's validated parser. Selection is explicit;
hash resolution alone is not evidence that this is the original character's rig.
This mode is intended for a matching model collection, not a heterogeneous archive.
Meshes without palettes or with missing/ambiguous bindings fail rather than receive
guessed assignments. In this mode a failed resource makes the command exit nonzero.

Each successful model receives `<stem>.skeleton.json` with joint hashes, nullable
parents, inverse bind matrices, and per-mesh palette hashes and joint indices in
original slot order. Matrices are flattened row-major System.Numerics row-vector
matrices; translation is at indices 12–14. Coordinates are unchanged. Sidecars
explicitly report `HasVertexSkinning=false` and `HasAnimation=false`.

This is **metadata preservation**, not glTF, weighted skinning, facial animation,
automatic skeleton discovery, or native clip decoding. No vertex weights are
invented. Sidecars and all other extracted content stay local under CONTRIBUTING.md.
Existing OBJ files are not rewritten. Failed resources do not remove outputs from
earlier runs; check exit status and use a fresh output directory for validation.

Synthetic CPU checks:

```powershell
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
dotnet run --project tests/Rpf9DumpSkeletonExport -c Release
dotnet run --project tests/Rpf9DumpDiagnostics -c Release
dotnet run --project third_party/rpf9-explorer/Tests/SkeletonSmoke -c Release
```

The synthetic tests check serialized hierarchy, matrix elements against an
independent bind-space coordinate calculation, repeated/reordered palettes,
missing and ambiguous bindings, malformed skeletons, explicit file/offset loading,
invalid offsets, whole-model rejection, CLI opt-out, and OBJ
non-mutation. They do not demonstrate a deformed real character.

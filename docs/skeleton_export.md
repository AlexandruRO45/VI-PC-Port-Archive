# Optional skeleton metadata export

This integration depends on **unmerged rpf9-explorer PR #3**, commit
`374f9a74df49ea22d8ac6e095196f56a03994de3`. The submodule pin is explicit;
do not merge this dependency change without maintainer review.

## Compatibility with Leonida PR #162

ASSET-10 is coordinated in Leonida issue #156. PR #96 is already merged and
indexes animation/rig metadata; this exporter uses the separate dependency API,
so it neither replaces nor duplicates that research.

PR #162 (reviewed at `65d927b61a88640d65317918923995aa39e1cd5f`) moves `Util` to
its own file and adds `Util.WriteAllText(string path, string text)`, backed by
`WriteAtomic(string path, Action<string> write)`. Once that PR is integrated,
replace **only the sidecar's** `File.WriteAllText` call in `Program.cs` with
`Util.WriteAllText`. Preserve #162's OBJ/DDS/audio calls and its helper/tests.
Then rerun all three smoke suites below and #162's updated diagnostics. This
is a prepared compatibility change, not a claim that atomic writes are currently
integrated or tested in this branch. No duplicate atomic writer is added here.
Same-file overlap is known; actual textual conflicts require a fetched upstream
and #162 branch before they can be assessed. Stopped sidecar writes can leave a
partial final file until the helper is adopted; use fresh outputs and check status.

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

The supported layout is the pinned dependency's external-drawable wrapper lookup
and a valid model hash-palette table. A directly referenced skeleton header that
the convenience lookup does not recognize, or a model for which the palette API
returns no table, remains unsupported and is rejected. Equal joint counts do not
justify assuming that vertex indices address the skeleton directly. This exporter
does not add a second parser or an implicit identity-mapping fallback.

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

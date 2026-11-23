"""Build a local SQLite index from existing model, placement and texture exports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path, PureWindowsPath
import posixpath
import re
import sqlite3
import stat
import sys
import tempfile
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viewer_catalog import ManifestReader
from world_names import hash64, joaat


SCHEMA = """
PRAGMA user_version = 1;
PRAGMA application_id = 1448104536;
PRAGMA foreign_keys = ON;
CREATE TABLE settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE archives (
    archive TEXT PRIMARY KEY, size INTEGER, sha256 TEXT, mtime_ns INTEGER
);
CREATE TABLE inputs (
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, path TEXT NOT NULL, root TEXT NOT NULL,
    size INTEGER NOT NULL, sha256 TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
    archive TEXT REFERENCES archives, scan_start INTEGER, scan_length INTEGER,
    coverage TEXT NOT NULL DEFAULT 'unknown', duplicate_of INTEGER REFERENCES inputs,
    metadata TEXT NOT NULL DEFAULT '{}', counts TEXT NOT NULL DEFAULT '{}'
);
CREATE VIEW scan_windows AS
    SELECT CASE WHEN kind = 'window_models' THEN 'models' ELSE kind END AS export_kind,
           archive, scan_start, scan_length, coverage, count(*) AS input_copies
    FROM inputs WHERE coverage != 'unknown'
    GROUP BY export_kind, archive, scan_start, scan_length, coverage;
CREATE TABLE models (
    id INTEGER PRIMARY KEY, archive TEXT NOT NULL REFERENCES archives,
    offset INTEGER NOT NULL, drawable INTEGER NOT NULL, name TEXT, joaat INTEGER, murmur64 TEXT,
    metadata TEXT NOT NULL, UNIQUE (archive, offset, drawable)
);
CREATE INDEX model_names ON models (joaat, murmur64);
CREATE TABLE model_exports (
    input_id INTEGER NOT NULL REFERENCES inputs, model_id INTEGER NOT NULL REFERENCES models,
    file TEXT NOT NULL, obj_path TEXT, glb_path TEXT, glb_status TEXT NOT NULL,
    PRIMARY KEY (input_id, model_id, file)
);
CREATE INDEX model_export_identity ON model_exports (model_id);
CREATE VIEW model_assets AS
    WITH files AS (
        SELECT DISTINCT model_id, path FROM (
            SELECT model_id, obj_path AS path FROM model_exports WHERE obj_path IS NOT NULL
            UNION ALL
            SELECT model_id, glb_path AS path FROM model_exports WHERE glb_path IS NOT NULL
        )
    )
    SELECT model_id, path, count(*) OVER (PARTITION BY path) AS models_sharing_path FROM files;
CREATE TABLE materials (
    id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL REFERENCES models,
    ordinal INTEGER NOT NULL, shader_index INTEGER NOT NULL, shader TEXT, preset TEXT, render_bucket INTEGER,
    UNIQUE (model_id, ordinal)
);
CREATE TABLE texture_refs (
    id INTEGER PRIMARY KEY, material_id INTEGER NOT NULL REFERENCES materials,
    ordinal INTEGER NOT NULL, param TEXT, texture_index INTEGER, name_hash TEXT, asset_key TEXT, tag TEXT,
    UNIQUE (material_id, ordinal)
);
CREATE TABLE embedded_textures (
    input_id INTEGER NOT NULL REFERENCES inputs, model_id INTEGER NOT NULL REFERENCES models,
    file TEXT NOT NULL, texture_index INTEGER NOT NULL, name_hash TEXT, asset_key TEXT,
    dds_path TEXT, metadata TEXT NOT NULL
);
CREATE INDEX embedded_model ON embedded_textures (model_id, texture_index);
CREATE INDEX embedded_path ON embedded_textures (dds_path) WHERE dds_path IS NOT NULL;
CREATE VIEW usable_embedded AS
    SELECT e.* FROM embedded_textures e JOIN (
        SELECT dds_path FROM embedded_textures WHERE dds_path IS NOT NULL
        GROUP BY dds_path HAVING count(DISTINCT model_id) = 1
    ) paths USING (dds_path);
CREATE TABLE ttpf_textures (
    input_id INTEGER NOT NULL REFERENCES inputs, asset_key TEXT NOT NULL, file TEXT,
    dds_path TEXT, status TEXT NOT NULL, metadata TEXT NOT NULL
);
CREATE INDEX ttpf_key ON ttpf_textures (asset_key) WHERE dds_path IS NOT NULL;
CREATE TABLE dictionary_textures (
    input_id INTEGER NOT NULL REFERENCES inputs, offset INTEGER NOT NULL,
    directory TEXT NOT NULL, texture_index INTEGER NOT NULL, name_hash TEXT, asset_key TEXT,
    dds_path TEXT, status TEXT NOT NULL, metadata TEXT NOT NULL
);
CREATE VIEW texture_candidates AS
    SELECT r.id AS ref_id, e.dds_path, 1 AS priority
    FROM texture_refs r JOIN materials m ON m.id = r.material_id
    JOIN usable_embedded e ON e.model_id = m.model_id AND e.texture_index = r.texture_index
    UNION
    SELECT r.id, e.dds_path, 2 FROM texture_refs r JOIN materials m ON m.id = r.material_id
    JOIN usable_embedded e ON e.model_id = m.model_id AND e.asset_key = r.asset_key
    UNION
    SELECT r.id, e.dds_path, 3 FROM texture_refs r JOIN materials m ON m.id = r.material_id
    JOIN usable_embedded e ON e.model_id = m.model_id AND e.name_hash = r.name_hash
    UNION
    SELECT r.id, t.dds_path, 4 FROM texture_refs r JOIN ttpf_textures t ON t.asset_key = r.asset_key
    WHERE t.dds_path IS NOT NULL;
CREATE TABLE texture_matches (
    ref_id INTEGER PRIMARY KEY REFERENCES texture_refs, priority INTEGER,
    candidates INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE placement_keys (
    id INTEGER PRIMARY KEY, joaat INTEGER NOT NULL, murmur64 TEXT NOT NULL,
    UNIQUE (joaat, murmur64)
);
CREATE TABLE placements (
    input_id INTEGER NOT NULL REFERENCES inputs, line INTEGER NOT NULL,
    key_id INTEGER NOT NULL REFERENCES placement_keys, key_status TEXT NOT NULL,
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL, entity TEXT NOT NULL,
    PRIMARY KEY (input_id, line)
);
CREATE INDEX placement_region ON placements (x, y);
CREATE INDEX placement_key ON placements (key_id);
CREATE VIEW placement_candidates AS
    SELECT k.id AS key_id, m.id AS model_id,
           CASE WHEN k.murmur64 = '' THEN NULL ELSE k.murmur64 = m.murmur64 END AS verified
    FROM placement_keys k JOIN models m ON m.joaat = k.joaat;
CREATE TABLE placement_matches (
    key_id INTEGER PRIMARY KEY REFERENCES placement_keys, joaat_candidates INTEGER NOT NULL,
    eligible_candidates INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE summary (metric TEXT PRIMARY KEY, count INTEGER NOT NULL);
"""


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value, minimum=0):
    if type(value) is int and minimum <= value < 2**63:
        return value
    return None


def hex_key(value, digits, *, zero=False):
    if isinstance(value, str):
        try:
            value = int(value, 16)
        except ValueError:
            return None
    if type(value) is int and int(not zero) <= value < 16**digits:
        return f"{value:0{digits}X}"
    return None


def archive_key(value):
    # Normalize producer paths lexically, including Windows manifests consumed on Linux.
    if isinstance(value, str) and "\0" not in value and PureWindowsPath(value).is_absolute():
        return ntpath.normcase(ntpath.normpath(value)).replace("\\", "/")
    if isinstance(value, str) and value.startswith("/") and "\0" not in value:
        return posixpath.normpath(value)
    raise ValueError("expected an absolute archive path in the export metadata")


def archive_record(db, archive, size=None, sha256=None, mtime_ns=None):
    prior = db.execute("SELECT size, sha256, mtime_ns FROM archives WHERE archive = ?", (archive,)).fetchone()
    values = (size, sha256, mtime_ns)
    if prior is not None:
        if any(old is not None and new is not None and old != new for old, new in zip(prior, values)):
            raise ValueError(f"conflicting archive fingerprints for {archive}")
        values = tuple(old if old is not None else new for old, new in zip(prior, values))
    db.execute("INSERT INTO archives VALUES (?, ?, ?, ?) ON CONFLICT(archive) DO UPDATE SET "
               "size=excluded.size, sha256=excluded.sha256, mtime_ns=excluded.mtime_ns", (archive, *values))


def fingerprint(path):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"expected an export file: {path}")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    check_unchanged(path, before)
    return before, digest


def check_unchanged(path, before):
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"input changed during indexing: {path}")


def add_input(db, kind, path, root, snapshots):
    info, digest = fingerprint(path)
    snapshots.append((path, info))
    input_id = db.execute("INSERT INTO inputs (kind,path,root,size,sha256,mtime_ns) VALUES (?,?,?,?,?,?)",
                          (kind, str(path), str(root), info.st_size, digest, info.st_mtime_ns)).lastrowid
    return input_id, digest


def finish_input(db, input_id, metadata, counts, archive=None, start=None, length=None,
                 coverage="unknown", duplicate_of=None):
    db.execute("UPDATE inputs SET archive=?, scan_start=?, scan_length=?, coverage=?, duplicate_of=?, "
               "metadata=?, counts=? WHERE id=?",
               (archive, start, length, coverage, duplicate_of, compact(metadata), compact(counts), input_id))


def rows(path, field, metadata):
    """Reuse the bounded JSON tokenizer for each producer's top-level row array."""
    with path.open(encoding="utf-8-sig") as handle:
        reader = ManifestReader(handle)
        reader.take("{")
        found = False
        for _ in reader.members("}"):
            name = reader.value()
            reader.take(":")
            if name == field and not found:
                found = True
                reader.take("[")
                for _ in reader.members("]"):
                    yield reader.value()
            elif not isinstance(name, str) or name == field or name in metadata:
                raise ValueError(f"invalid or duplicate manifest field: {name!r}")
            else:
                metadata[name] = reader.value()
        if not found or reader.peek():
            raise ValueError(f"expected {field} array and no trailing content")

def asset_path(base, name, root, suffix=None):
    """Return the contained path even when the export has been removed."""
    if not isinstance(name, str) or not name or any(ord(c) < 32 for c in name):
        raise ValueError("invalid asset filename")
    parts = name.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError("expected a relative asset filename")
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(root) or (suffix and path.suffix.lower() != suffix):
        raise ValueError("asset filename escapes its root or has the wrong extension")
    return path


def existing(path):
    return os.path.normcase(str(path)) if path.is_file() else None


def export_root(path, roots):
    return max((root for root in roots if path.is_relative_to(root)), default=path.parent,
               key=lambda root: len(root.parts))


def list_field(row, field, counts):
    value = row.get(field)
    if isinstance(value, list):
        return value
    if value is not None:
        counts["invalid_metadata"] += 1
    return []


def add_materials(db, model_id, row, counts):
    for ordinal, mat in enumerate(list_field(row, "Materials", counts)):
        shader_index = number(mat.get("ShaderIndex")) if isinstance(mat, dict) else None
        if shader_index is None:
            counts["invalid_materials"] += 1
            continue
        material_id = db.execute("INSERT INTO materials VALUES (NULL,?,?,?,?,?,?)", (
            model_id, ordinal, shader_index, hex_key(mat.get("Shader"), 8, zero=True),
            hex_key(mat.get("Preset"), 8, zero=True), number(mat.get("RenderBucket")),
        )).lastrowid
        for ordinal_ref, ref in enumerate(list_field(mat, "Textures", counts)):
            if not isinstance(ref, dict):
                counts["invalid_texture_refs"] += 1
                continue
            db.execute("INSERT INTO texture_refs VALUES (NULL,?,?,?,?,?,?,?)", (
                material_id, ordinal_ref, hex_key(ref.get("Param"), 8, zero=True),
                number(ref.get("TextureIndex")), hex_key(ref.get("Name"), 8),
                hex_key(ref.get("Id64"), 16), hex_key(ref.get("Tag"), 8, zero=True),
            ))


def dds_inventory(folder, root):
    inventory = defaultdict(list)
    if folder is not None and folder.is_dir() and folder.resolve().is_relative_to(root):
        for path in sorted(folder.iterdir()):
            prefix, sep, _ = path.name.partition("_")
            if sep and re.fullmatch(r"[0-9]{1,18}", prefix) and path.suffix.lower() == ".dds":
                resolved = path.resolve()
                if resolved.is_relative_to(root) and (dds := existing(resolved)):
                    inventory[int(prefix)].append(dds)
    return inventory


def add_embedded(db, input_id, model_id, file, obj, root, row, counts):
    folder = obj.with_name(obj.stem + "_tex") if obj is not None else None
    inventory = dds_inventory(folder, root)
    described = set()
    for rec in list_field(row, "Embedded", counts):
        index = number(rec.get("Index")) if isinstance(rec, dict) else None
        if index is None:
            counts["invalid_embedded"] += 1
            continue
        described.add(index)
        for path in inventory.get(index, [None]):
            db.execute("INSERT INTO embedded_textures VALUES (?,?,?,?,?,?,?,?)", (
                input_id, model_id, file, index, hex_key(rec.get("Hash"), 8), hex_key(rec.get("Key"), 16),
                path, compact(rec),
            ))
    for index in sorted(inventory.keys() - described):
        db.executemany("INSERT INTO embedded_textures VALUES (?,?,?,?,NULL,NULL,?,'{}')",
                       ((input_id, model_id, file, index, path) for path in inventory[index]))


def scan_span(start, length, size):
    if number(start) is None or number(length) is None or (size is not None and start + length > size):
        raise ValueError("invalid scan window")
    return start, length


def model_window(metadata):
    scan = metadata.get("Scan", {})
    if not isinstance(scan, dict):
        raise ValueError("expected Scan object")
    size = number(metadata.get("ArchiveSize"))
    if "Start" in scan or "Length" in scan:
        return (*scan_span(scan.get("Start"), scan.get("Length"), size), "window")
    return (0, size, "whole") if "Scan" in metadata else (None, None, "unknown")


def add_models(db, input_id, path, root, sidecar=None, catalog_root=None):
    counts, metadata = Counter(), {}
    filenames = Counter()
    archive = None
    for row in rows(path, "Items", metadata):
        if archive is None:
            archive = register_model_archive(db, metadata, sidecar)
        counts["rows"] += 1
        if sidecar and isinstance(row, dict) and isinstance(row.get("File"), str):
            filenames[row["File"]] += 1
        offset = number(row.get("Offset")) if isinstance(row, dict) else None
        drawable = row.get("Drawable") if offset is not None else None
        if offset is None or (drawable is not None and number(drawable) is None):
            counts["invalid_models"] += 1
            continue
        start, length, _ = model_window(metadata)
        size = sidecar["source"]["size"] if sidecar else number(metadata.get("ArchiveSize"))
        if ((size is not None and offset >= size) or
                (length is not None and not start <= offset < start + length)):
            counts["out_of_window"] += 1
            continue
        name = row.get("Name")
        stem = name.split(".#", 1)[0].strip() if isinstance(name, str) else ""
        if not stem:
            counts["unnamed_models"] += 1
        identity = (archive, offset, -1 if drawable is None else drawable)
        geometry = {key: value for key, value in row.items()
                    if key not in ("Offset", "Drawable", "File", "Name", "Glb", "Embedded", "Materials")}
        # Keep authoritative material/embedded descriptors with the drawable for consistency checks.
        geometry.update(Materials=row.get("Materials"), Embedded=row.get("Embedded"))
        encoded = compact(geometry)
        result = db.execute("INSERT INTO models VALUES (NULL,?,?,?,?,?,?,?) "
                            "ON CONFLICT(archive,offset,drawable) DO NOTHING RETURNING id", (
            *identity, name if isinstance(name, str) else None, joaat(stem) if stem else None,
            f"{hash64(stem):016X}" if stem else None, encoded,
        )).fetchone()
        if result is not None:
            model_id = result[0]
            add_materials(db, model_id, row, counts)
        else:
            model_id, previous_name, previous = db.execute(
                "SELECT id,name,metadata FROM models WHERE archive=? AND offset=? AND drawable=?", identity,
            ).fetchone()
            if previous != encoded or previous_name != (name if isinstance(name, str) else None):
                raise ValueError(f"conflicting model metadata for {archive} offset {offset}, drawable {drawable}")
            counts["duplicate_models"] += 1

        file = row.get("File")
        if not isinstance(file, str):
            counts["invalid_model_paths"] += 1
            continue
        obj_path, glb_path, glb_status = None, None, "unavailable"
        try:
            if Path(file).suffix.lower() != ".obj":
                raise ValueError("expected an OBJ filename")
            obj = asset_path(path.parent, file, root, ".obj")
            if sidecar:
                glb_path, glb_status = window_glb(sidecar, file, catalog_root)
            else:
                obj_path = existing(obj)
                glb = row.get("Glb") or str(Path(file).with_suffix(".glb"))
                glb_path = existing(asset_path(path.parent, glb, root, ".glb"))
                glb_status = "ready" if glb_path else "unavailable"
        except ValueError:
            counts["invalid_model_paths"] += 1
            obj = None
            glb_status = "invalid"
        if db.execute("INSERT INTO model_exports VALUES (?,?,?,?,?,?) ON CONFLICT DO NOTHING",
                      (input_id, model_id, file, obj_path, glb_path, glb_status)).rowcount:
            add_embedded(db, input_id, model_id, file, None if sidecar else obj, root, row, counts)

    if archive is None:
        archive = register_model_archive(db, metadata, sidecar)
    start, length, coverage = model_window(metadata)
    if sidecar:
        if (start, length) != (sidecar["window"]["start"], sidecar["window"]["length"]):
            raise ValueError("catalog map window differs from model manifest Scan")
        db.executemany("UPDATE model_exports SET glb_path=NULL, glb_status='ambiguous' WHERE input_id=? AND file=?",
                       ((input_id, file) for file, occurrences in filenames.items() if occurrences > 1))
        for status, total in db.execute("SELECT glb_status,count(*) FROM model_exports WHERE input_id=? "
                                        "GROUP BY glb_status", (input_id,)):
            counts[f"catalog_{status}"] = total
        counts["catalog_unmatched"] = sum(len(sidecar["by_file"][file]) for file in sidecar["by_file"].keys()
                                           - sidecar["seen"])
    counts["model_files_unavailable"] = db.execute("SELECT count(*) FROM model_exports WHERE input_id=? "
                                                  "AND obj_path IS NULL AND glb_path IS NULL", (input_id,)).fetchone()[0]
    count = number(metadata.get("Count"))
    if count is not None and count < counts["rows"]:
        raise ValueError("model manifest contains more rows than Count")
    if count is not None:
        counts["unexported_models"] = count - counts["rows"]
    finish_input(db, input_id, metadata, counts, archive, start, length, coverage)


def register_model_archive(db, metadata, sidecar):
    archive = archive_key(metadata.get("src"))
    size = number(metadata.get("ArchiveSize"))
    sha = metadata.get("ArchiveSha256")
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
        raise ValueError("invalid ArchiveSha256")
    mtime = number(metadata.get("ArchiveMtimeNs"))
    archive_record(db, archive, size, sha.lower() if sha else None, mtime)
    if sidecar:
        source = sidecar["source"]
        relative = source["archive"].replace("\\", "/")
        compare = relative.lower() if PureWindowsPath(metadata["src"]).is_absolute() else relative
        if not archive.endswith("/" + compare):
            raise ValueError("catalog map archive differs from model manifest src")
        archive_record(db, archive, source["size"], mtime_ns=source["mtime_ns"])
    return archive


def window_map(marker):
    with marker.open(encoding="utf-8-sig") as handle:
        doc = json.load(handle)
    if (not isinstance(doc, dict) or type(doc.get("version")) is not int or doc["version"] != 1
            or doc.get("kind") != "leonida-window-parts"):
        raise ValueError("expected a version 1 leonida-window-parts catalog map")
    source, window, manifest, models = (doc.get(key) for key in ("source", "window", "manifest", "models"))
    if not all(isinstance(item, dict) for item in (source, window, manifest)) or not isinstance(models, list):
        raise ValueError("missing catalog map metadata")
    archive = source.get("archive")
    if (not isinstance(archive, str) or any(p in ("", ".", "..") or ":" in p
                                           for p in archive.replace("\\", "/").split("/"))
            or number(source.get("size")) is None or number(source.get("mtime_ns")) is None):
        raise ValueError("invalid catalog map source")
    scan_span(window.get("start"), window.get("length"), source["size"])
    if (manifest.get("file") != "manifest.json" or not isinstance(manifest.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest["sha256"])):
        raise ValueError("invalid catalog map manifest record")
    by_file = defaultdict(list)
    for model in models:
        if (not isinstance(model, dict) or not isinstance(model.get("file"), str)
                or model.get("status") not in ("ready", "skipped", "duplicate", "failed")
                or (model["status"] == "ready" and not isinstance(model.get("glb"), str))):
            raise ValueError("invalid catalog map model record")
        by_file[model["file"]].append(model)
    doc.update(by_file=by_file, seen=set())
    return doc


def window_glb(sidecar, file, catalog_root):
    sidecar["seen"].add(file)
    matches = sidecar["by_file"].get(file, ())
    if not matches:
        return None, "unmapped"
    if len(matches) != 1:
        return None, "ambiguous"
    model = matches[0]
    if model["status"] != "ready":
        return None, model["status"]
    # Decode each URL segment once. Encoded separators cannot become path separators.
    parts = [unquote(part, errors="strict") for part in model["glb"].split("/")]
    if any("/" in part or "\\" in part for part in parts):
        raise ValueError("invalid encoded catalog path")
    path = existing(asset_path(catalog_root, "/".join(parts), catalog_root, ".glb"))
    return path, "ready" if path else "missing"


def add_textures(db, input_id, path, root):
    metadata, counts = {}, Counter()
    for row in rows(path, "Textures", metadata):
        counts["rows"] += 1
        key = hex_key(row.get("AssetKey"), 16) if isinstance(row, dict) else None
        if key is None:
            counts["invalid_texture_keys"] += 1
            continue
        file = row.get("File")
        dds = None
        try:
            if file is None:
                status = "unexported"
            else:
                dds = existing(asset_path(path.parent, file, root, ".dds"))
                status = "ready" if dds else "missing"
        except ValueError:
            status = "invalid"
        counts[f"texture_files_{status}"] += 1
        db.execute("INSERT INTO ttpf_textures VALUES (?,?,?,?,?,?)", (
            input_id, key, file if isinstance(file, str) else None, dds, status, compact(row),
        ))
    count = number(metadata.get("Count"))
    if count is not None and count != counts["rows"]:
        raise ValueError("texture Count differs from Textures length")
    finish_input(db, input_id, metadata, counts)


def add_dictionaries(db, input_id, path, root):
    metadata, counts = {}, Counter()
    offsets = set()
    for row in rows(path, "Items", metadata):
        counts["rows"] += 1
        offset = number(row.get("Offset")) if isinstance(row, dict) else None
        if offset is None:
            counts["invalid_dictionaries"] += 1
            continue
        start, length, _ = model_window(metadata)
        if length is not None and not start <= offset < start + length:
            counts["out_of_window"] += 1
            continue
        if offset in offsets:
            raise ValueError(f"duplicate texture dictionary offset {offset}")
        offsets.add(offset)
        directory = row.get("Dir")
        try:
            inventory = dds_inventory(asset_path(path.parent, directory, root), root)
        except ValueError:
            counts["invalid_dictionary_paths"] += 1
            continue
        for record in list_field(row, "Textures", counts):
            index = number(record.get("Index")) if isinstance(record, dict) else None
            if index is None:
                counts["invalid_dictionary_textures"] += 1
                continue
            candidates = inventory.get(index, ())
            status = "missing" if not candidates else "ready" if len(candidates) == 1 else "ambiguous"
            counts[f"dictionary_textures_{status}"] += 1
            db.executemany("INSERT INTO dictionary_textures VALUES (?,?,?,?,?,?,?,?,?)", (
                (input_id, offset, directory, index, hex_key(record.get("Hash"), 8),
                 hex_key(record.get("Key"), 16), dds, status, compact(record))
                for dds in candidates or (None,)
            ))
    count = number(metadata.get("Count"))
    if count is not None:
        if count < counts["rows"]:
            raise ValueError("dictionary manifest contains more rows than Count")
        counts["unexported_dictionaries"] = count - counts["rows"]
    archive = register_model_archive(db, metadata, None)
    finish_input(db, input_id, metadata, counts, archive, *model_window(metadata))


def add_placements(db, input_id, path, digest, report, region):
    archive, start, length, coverage, metadata = None, None, None, "unknown", {}
    if report:
        for _ in rows(report, "resources", metadata):
            pass
        metadata = metadata.get("summary")
        if not isinstance(metadata, dict):
            raise ValueError("missing world_index summary")
        archive = archive_key(metadata.get("path"))
        size = number(metadata.get("bytes"))
        start, end = number(metadata.get("scan_start")), number(metadata.get("scan_end"))
        if size is None or start is None or end is None:
            raise ValueError("expected an archive world_index report with byte coverage")
        start, length = scan_span(start, end - start, size)
        coverage = "window"
        archive_record(db, archive, size)
    prior = db.execute("SELECT id FROM inputs WHERE kind='placements' AND id < ? AND sha256=? "
                       "AND archive IS ? AND scan_start IS ? AND scan_length IS ? ORDER BY id LIMIT 1",
                       (input_id, digest, archive, start, length)).fetchone()
    counts = Counter()
    if not prior:
        with path.open(encoding="utf-8-sig") as handle:
            for line, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                counts["rows"] += 1
                try:
                    row = json.loads(raw)
                    arch = row["archetype"]
                    if isinstance(arch, str):
                        arch = int(arch, 16)
                    if type(arch) is not int:
                        raise ValueError("invalid archetype")
                    pos = vector(row["position"], 3)
                    vector(row["rotation"], 4)
                    entity = compact(row)
                except (ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError):
                    counts["invalid_placements"] += 1
                    continue
                if region and not (region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]):
                    counts["outside_region"] += 1
                    continue
                raw_key = row.get("key")
                try:
                    key = f"{int(raw_key, 16):016X}" if isinstance(raw_key, str) else None
                except ValueError:
                    key = None
                status = "present" if key is not None else "absent" if raw_key is None else "invalid"
                counts[f"placement_key_{status}"] += 1
                pair = (arch & 0xffffffff, key or "")
                db.execute("INSERT INTO placement_keys VALUES (NULL,?,?) ON CONFLICT DO NOTHING", pair)
                db.execute("INSERT INTO placements SELECT ?,?,id,?,?,?,?,? FROM placement_keys "
                           "WHERE joaat=? AND murmur64=?", (input_id, line, status, *pos, entity, *pair))
    finish_input(db, input_id, metadata, counts, archive, start, length, coverage, prior[0] if prior else None)


def vector(value, length):
    if isinstance(value, list) and len(value) == length:
        parts = tuple(float(part) for part in value)
        if all(math.isfinite(part) for part in parts):
            return parts
    raise ValueError("invalid placement vector")


def finish_matches(db):
    db.execute("""
        INSERT INTO placement_matches
        WITH counts AS (
            SELECT k.id, k.murmur64, count(c.model_id) AS candidates,
                   sum(CASE WHEN c.model_id IS NOT NULL AND c.verified IS NOT 0 THEN 1 ELSE 0 END) AS eligible
            FROM placement_keys k LEFT JOIN placement_candidates c ON c.key_id = k.id GROUP BY k.id
        )
        SELECT id, candidates, eligible, CASE
            WHEN candidates = 0 THEN 'unresolved_archetype'
            WHEN eligible = 0 THEN 'key_mismatch'
            WHEN eligible > 1 THEN 'ambiguous'
            WHEN murmur64 = '' THEN 'unverified'
            ELSE 'verified' END
        FROM counts
    """)
    db.execute("""
        INSERT INTO texture_matches
        WITH tiers AS (
            SELECT ref_id, priority, count(*) AS candidates FROM texture_candidates GROUP BY ref_id, priority
        ), ranked AS (
            SELECT *, row_number() OVER (PARTITION BY ref_id ORDER BY priority) AS rank FROM tiers
        )
        SELECT r.id, c.priority, coalesce(c.candidates, 0), CASE
            WHEN c.candidates IS NULL THEN 'missing'
            WHEN c.candidates > 1 THEN 'ambiguous'
            ELSE 'available' END
        FROM texture_refs r LEFT JOIN ranked c ON c.ref_id = r.id AND c.rank = 1
    """)
    totals = Counter()
    for table in ("models", "materials", "texture_refs", "placements", "archives"):
        totals[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    for status, count in db.execute("""
        SELECT m.status, count(*) FROM placements p JOIN placement_matches m ON p.key_id=m.key_id GROUP BY m.status
    """):
        totals[f"placements_{status}"] = count
    for status, count in db.execute("SELECT status,count(*) FROM texture_matches GROUP BY status"):
        totals[f"texture_refs_{status}"] = count
    totals["models_without_asset_files"] = db.execute("""
        SELECT count(*) FROM models m WHERE NOT EXISTS (
            SELECT 1 FROM model_exports e WHERE e.model_id=m.id AND (e.obj_path IS NOT NULL OR e.glb_path IS NOT NULL))
    """).fetchone()[0]
    totals["shared_model_paths"] = db.execute("""
        SELECT count(DISTINCT path) FROM model_assets WHERE models_sharing_path > 1
    """).fetchone()[0]
    totals["duplicate_placement_inputs"] = db.execute("SELECT count(*) FROM inputs WHERE duplicate_of IS NOT NULL").fetchone()[0]
    for (counts,) in db.execute("SELECT counts FROM inputs"):
        totals.update({"input_" + key: value for key, value in json.loads(counts).items()})
    db.executemany("INSERT INTO summary VALUES (?,?)", sorted(totals.items()))
    return totals


def prepare(args):
    output = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    roots = [path.resolve() for path in args.asset_root]
    catalog = args.catalog_root.resolve() if args.catalog_root else None
    sources = {path.resolve() for path in args.source_root} | set(roots) | {repo}
    if catalog:
        sources.add(catalog)
    models = {path.resolve() for path in args.models}
    if any(path.name == "catalog-map.json" or any(p.lower() == "window-manifests" for p in path.parts)
           or path.with_name("catalog-map.json").exists() for path in models):
        raise ValueError("retained model windows require --window with the catalog-map.json commit marker")
    windows = {path.resolve() for path in args.window}
    textures = {path.resolve() for path in args.textures}
    dictionaries = {path.resolve() for path in args.texture_dictionaries}
    placements = {path.resolve(): None for path in args.placements}
    for file, report in args.world_window:
        file, report = file.resolve(), report.resolve()
        if placements.get(file) not in (None, report):
            raise ValueError(f"multiple world_index reports for {file}")
        placements[file] = report
    consumed = models | windows | textures | dictionaries | placements.keys()
    consumed |= {path for path in placements.values() if path is not None}
    consumed |= {path.with_name("manifest.json").resolve() for path in windows}
    sources.update(path.parent for path in consumed)
    if (any(output.is_relative_to(root) for root in sources)
            or any((parent / ".git").exists() for parent in output.parents)):
        raise ValueError("--out must be outside the repository and input directories")
    return output, roots, catalog, models, windows, textures, dictionaries, placements


def build(args):
    output, roots, catalog, models, windows, textures, dictionaries, placements = prepare(args)
    if output.exists():
        with closing(sqlite3.connect(output.as_uri() + "?mode=ro", uri=True)) as current:
            if current.execute("PRAGMA application_id").fetchone() != (1448104536,):
                raise ValueError("existing --out is not a viewer parts index")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parts-index-", dir=output.parent) as staging:
        staged = Path(staging) / "index.sqlite"
        snapshots = []
        with closing(sqlite3.connect(staged)) as db, db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO settings VALUES ('region',?)", (compact(args.region),))
            for path in sorted(models):
                root = export_root(path, roots)
                input_id, _ = add_input(db, "models", path, root, snapshots)
                add_models(db, input_id, path, root)
            for marker in sorted(windows):
                marker_id, _ = add_input(db, "catalog_map", marker, catalog, snapshots)
                sidecar = window_map(marker)
                path = marker.with_name("manifest.json")
                input_id, digest = add_input(db, "window_models", path, path.parent, snapshots)
                if digest != sidecar["manifest"]["sha256"].lower():
                    raise ValueError(f"stale catalog map, manifest SHA-256 differs: {marker}")
                add_models(db, input_id, path, path.parent, sidecar, catalog)
                finish_input(db, marker_id, {key: sidecar[key] for key in ("source", "window", "manifest")},
                             {"catalog_map_rows": len(sidecar["models"])})
            for path in sorted(textures):
                root = export_root(path, roots)
                input_id, _ = add_input(db, "ttpf", path, root, snapshots)
                add_textures(db, input_id, path, root)
            for path in sorted(dictionaries):
                root = export_root(path, roots)
                input_id, _ = add_input(db, "texture_dictionaries", path, root, snapshots)
                add_dictionaries(db, input_id, path, root)
            for path, report in sorted(placements.items()):
                input_id, digest = add_input(db, "placements", path, path.parent, snapshots)
                if report:
                    report_id, _ = add_input(db, "world_report", report, report.parent, snapshots)
                    db.execute("INSERT INTO settings VALUES (?,?)", (f"placement_report/{input_id}", str(report_id)))
                add_placements(db, input_id, path, digest, report, args.region)
            totals = finish_matches(db)
        for path, info in snapshots:
            check_unchanged(path, info)
        os.replace(staged, output)
    return totals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="SQLite output outside the repository and input roots")
    ap.add_argument("--models", type=Path, action="append", default=[], metavar="MANIFEST")
    ap.add_argument("--window", type=Path, action="append", default=[], metavar="CATALOG_MAP",
                    help="retained window catalog-map.json commit marker")
    ap.add_argument("--catalog-root", type=Path, help="served catalog root for window GLB URLs")
    ap.add_argument("--textures", type=Path, action="append", default=[], metavar="TTPF_MANIFEST")
    ap.add_argument("--texture-dictionaries", type=Path, action="append", default=[], metavar="MANIFEST",
                    help="inventory an ordinary rpf9dump textures manifest")
    ap.add_argument("--placements", type=Path, action="append", default=[], metavar="ENTITIES_JSONL")
    ap.add_argument("--world-window", type=Path, action="append", default=[], nargs=2,
                    metavar=("ENTITIES_JSONL", "WORLD_REPORT"), help="placements paired with their world_index report")
    ap.add_argument("--asset-root", type=Path, action="append", default=[], metavar="DIR",
                    help="allow manifest-relative asset paths within this export root")
    ap.add_argument("--source-root", type=Path, action="append", default=[], metavar="DIR",
                    help="additional input tree to keep the index outside")
    ap.add_argument("--region", type=float, nargs=4, metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
                    help="inclusive XY placement bounds in world metres")
    args = ap.parse_args(argv)
    if not any((args.models, args.window, args.textures, args.texture_dictionaries, args.placements,
                args.world_window)):
        ap.error("provide at least one export input")
    if args.window and not args.catalog_root:
        ap.error("--window requires --catalog-root")
    if args.region and (not all(math.isfinite(v) for v in args.region)
                        or args.region[0] > args.region[2] or args.region[1] > args.region[3]):
        ap.error("--region requires finite, ordered XY bounds")
    try:
        totals = build(args)
    except (OSError, ValueError, sqlite3.Error, RecursionError) as exc:
        print(f"viewer_parts_index: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(dict(sorted(totals.items())), indent=2))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())

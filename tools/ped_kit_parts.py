"""Look up decoded components and external TTPF textures in a local viewer parts index."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world_names import hash64


def _one(db: sqlite3.Connection, sql: str, params: tuple, label: str):
    try:
        rows = db.execute(sql, params).fetchmany(2)
    except sqlite3.Error as exc:
        raise ValueError(f"incompatible parts index while reading {label}: {exc}") from exc
    if not rows:
        raise ValueError(f"{label}: no match in the parts index")
    if len(rows) > 1:
        raise ValueError(f"{label}: ambiguous matches in the parts index")
    return rows[0]


def _existing_export(value: str, suffix: str, label: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{label}: expected an absolute export path on this machine")
    path = Path(value)
    if path.suffix.lower() != suffix or not path.is_file():
        raise ValueError(f"{label}: indexed {suffix} file is unavailable")
    return path


def component_path(db: sqlite3.Connection, drawable_hash: int, *, textured: bool = False) -> Path:
    """Return the decoded model.json for one uniquely identified, skinned GLB export."""
    key = f"{drawable_hash:016X}"
    label = f"drawable {key}"
    model_id, indexed_name = _one(db, "SELECT id, name FROM models WHERE murmur64 = ?", (key,), label)
    glb_file, = _one(db, """
        SELECT DISTINCT glb_path FROM model_exports
        WHERE model_id = ? AND glb_status = 'ready' AND glb_path IS NOT NULL
    """, (model_id,), f"{label} ready GLB")
    owners, = _one(db, """
        SELECT models_sharing_path FROM model_assets WHERE model_id = ? AND path = ?
    """, (model_id, glb_file), f"{label} GLB ownership")
    if owners != 1:
        raise ValueError(f"{label}: ambiguous GLB path shared by multiple models")
    if textured:
        slots, = _one(db, """
            SELECT count(DISTINCT shader_index) FROM materials WHERE model_id = ?
        """, (model_id,), f"{label} material slots")
        if slots > 1:
            raise ValueError(f"{label}: texture override is ambiguous across multiple material slots")

    glb = _existing_export(glb_file, ".glb", label)
    sidecar = glb.with_name("model.json")
    if not sidecar.is_file():
        raise ValueError(f"{label}: indexed GLB has no decoded model.json skin sidecar")
    with sidecar.open(encoding="utf-8") as stream:
        model = json.load(stream)
    if not isinstance(model, dict):
        raise ValueError(f"{label}: decoded model.json must contain an object")
    glb_name = model.get("glb", "model.glb")
    if not isinstance(glb_name, str) or not glb_name or (sidecar.parent / glb_name).resolve() != glb.resolve():
        raise ValueError(f"{label}: decoded model.json refers to a different GLB")
    drawable = model.get("drawable")
    name = drawable.get("name") if isinstance(drawable, dict) else None
    stem = name.split(".#", 1)[0].strip() if isinstance(name, str) else ""
    indexed_stem = indexed_name.split(".#", 1)[0].strip() if isinstance(indexed_name, str) else ""
    if not stem or stem != indexed_stem or hash64(stem) != drawable_hash:
        raise ValueError(f"{label}: decoded drawable name does not match the indexed name and hash")
    meshes = model.get("meshes")
    if not isinstance(meshes, list) or not any(
        isinstance(mesh, dict) and mesh.get("skinned") and mesh.get("skin_influences") for mesh in meshes
    ):
        raise ValueError(f"{label}: decoded model.json contains no skinned mesh")
    return sidecar


def texture_path(db: sqlite3.Connection, asset_key: int) -> Path:
    """Return the unique ready external TTPF DDS for the exact 64-bit asset key."""
    key = f"{asset_key:016X}"
    label = f"texture {key}"
    value, = _one(db, """
        SELECT DISTINCT dds_path FROM ttpf_textures
        WHERE asset_key = ? AND status = 'ready' AND dds_path IS NOT NULL
    """, (key,), f"{label} ready TTPF DDS")
    return _existing_export(value, ".dds", label)

"""Build a local viewer catalog from exported drawables and existing GLBs."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.parse import quote

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import glb_writer

GIB = 1024 ** 3
# make_game.MODEL_ARCHIVES, with conservative free-space estimates for staged OBJ exports.
ARCHIVES = {
    "common.rpf": 1, "textures.rpf": 1, "data.rpf": 4,
    "pack/story000_restofworld/story000_restofworld.rpf": 1,
    "pack/shared_minigame000/shared_minigame000.rpf": 8,
    "pack/core_restofworld/core_restofworld.rpf": 2,
    "pack/core_restofworld001/core_restofworld001.rpf": 1,
    "packs.rpf": 24, "pack/story000/story000.rpf": 16, "levels.rpf": 200,
}
CATEGORY_DIRS = {"props", "ped-parts", "vehicles"}
PED_TOKEN = re.compile(
    r"(?:^|[._-])(?:player_story|head(?:fr|cs|_)|head_|eyes?_|eyecap|hand_|feet_|"
    r"hair|pants|jacket|tshirt|shirt|hoodie|boots|bikini|undies|bodygen|bodynipples|"
    r"uppr_|lowr_|legacypedcs)"
)
GLB_HEADER = struct.Struct("<4sIII4s")
COVERAGE = (
    "Static drawable geometry and existing self-contained GLBs. "
    "Fragment children, full vehicle coverage, ped assembly and game material reconstruction are outside this build."
)


class SpaceError(OSError):
    """A write or export would use the free-space reserve."""


@dataclass(frozen=True)
class Model:
    source: Path
    destination: Path
    name: str
    category: str


def linked(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (junction() if junction else os.name == "nt" and path.resolve() != path)


def overlap(a: Path, b: Path) -> bool:
    return a.is_relative_to(b) or b.is_relative_to(a)


def segment(name: str) -> bool:
    return bool(name and name not in (".", "..") and not any(c in name for c in "/\\:")
                and not any(ord(c) < 32 or ord(c) == 127 for c in name))


def relative_url(path: Path, root: Path) -> str:
    return "/".join(quote(part, safe="") for part in path.relative_to(root).parts)


def key(path: Path) -> str:
    return hashlib.sha256(os.fsencode(path)).hexdigest()[:16]


def safe_destination(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved != path or linked(path):
        raise OSError(f"refusing linked destination {path}")


def free_space(folder: Path, reserve: int, extra: int = 0) -> None:
    if shutil.disk_usage(folder).free < reserve + extra:
        raise SpaceError(f"{folder} needs {(reserve + extra) / GIB:.2f} GiB free to continue")


@contextmanager
def catalog_lock(root: Path):
    """The file may survive a crash, but the OS releases its lock with the process."""
    lock = root / ".catalog.lock"
    safe_destination(lock, root)
    with lock.open("a+b") as f:
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OSError(f"another catalog build is using {root}") from exc
        yield


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".catalog-", delete=False) as f:
            staged = Path(f.name)
            write(f)
        # The viewer can briefly hold catalog.json open while it polls on Windows.
        glb_writer._replace(str(staged), str(path))
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def copy_file(source: Path, destination: Path) -> None:
    def copy(f):
        with source.open("rb") as data:
            shutil.copyfileobj(data, f)

    atomic_write(destination, copy)


def maybe_glb(path: Path) -> bool:
    """Check enough of an existing GLB to reuse it without reading the BIN chunk."""
    try:
        with path.open("rb") as f:
            magic, version, length, json_size, kind = GLB_HEADER.unpack(f.read(GLB_HEADER.size))
            if (magic != b"glTF" or version != 2 or length != os.fstat(f.fileno()).st_size
                    or json_size % 4
                    or kind != b"JSON" or not 0 < json_size <= min(16 * 1024 ** 2, length - GLB_HEADER.size)):
                return False
            document = json.loads(f.read(json_size))
            while f.tell() < length:
                chunk_size, _ = struct.unpack("<I4s", f.read(8))
                if chunk_size % 4 or chunk_size > length - f.tell():
                    return False
                f.seek(chunk_size, os.SEEK_CUR)
            return (
                isinstance(document, dict)
                and any(mesh.get("primitives") for mesh in document.get("meshes", []))
                and all("uri" not in entry or str(entry["uri"]).startswith("data:")
                        for field in ("buffers", "images") for entry in document.get(field, []))
            )
    except (OSError, ValueError, struct.error, TypeError, AttributeError, RecursionError):
        return False


def png_sidecar(model: Path) -> Path:
    return model.with_name(model.stem + ".shader.png")


def is_png(path: Path) -> bool:
    if linked(path) or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def normalized_name(name: str) -> str:
    return re.sub(r"(?:\.#(?:dr|ft)|\.(?:yft|ydr|ydd))$", "", name.casefold()).removesuffix("_hi")


def category(name: str, path: Path, archive: str, vehicles: set[str]) -> str:
    if normalized_name(name) in vehicles or normalized_name(path.stem) in vehicles:
        return "vehicles"
    for part in path.parent.parts:
        if part.casefold() in CATEGORY_DIRS:
            return part.casefold()
    if archive == "packs" and (PED_TOKEN.search(name.casefold()) or PED_TOKEN.search(path.stem.casefold())):
        return "ped-parts"
    return "props"


def placeholder(name: str) -> bool:
    return "bipedbounds" in name.casefold()


def local_exporter(explicit: Path | None, work: Path) -> list[str]:
    binary = "rpf9dump.exe" if os.name == "nt" else "rpf9dump"
    built = TOOLS / "rpf9dump" / "bin" / "Release" / "net8.0"
    installed = work / "tools" / "rpf9dump"
    options = [explicit] if explicit else [
        installed / binary, installed / "rpf9dump.dll", built / binary, built / "rpf9dump.dll",
    ]
    for path in options:
        if path.is_file():
            return ["dotnet", str(path.resolve())] if path.suffix.casefold() == ".dll" else [str(path.resolve())]
    raise OSError("local rpf9dump not found. Build tools/rpf9dump or provide --rpf9dump")


def export_models(command: list[str], source: Path, exports: Path, reserve: int) -> None:
    """Use the documented exporter; stop the child if free space falls below the reserve."""
    with subprocess.Popen([*command, "models", str(source), str(exports)]) as child:
        try:
            while True:
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    free_space(exports, reserve)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    if code != 0:
        raise OSError(f"rpf9dump exited {code} for {source.name}")


class ManifestReader:
    """Stream the export manifest's Items array without retaining material metadata for every row."""

    def __init__(self, file):
        self.file = file
        self.buffer = ""
        self.pos = 0
        self.eof = False
        self.decoder = json.JSONDecoder()
        self.metadata = {}

    def refill(self) -> None:
        chunk = self.file.read(65536)
        self.eof = not chunk
        self.buffer = self.buffer[self.pos:] + chunk
        self.pos = 0

    def peek(self) -> str:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos] in " \t\r\n":
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ""
            self.refill()

    def take(self, token: str) -> None:
        if self.peek() != token:
            raise ValueError(f"expected {token!r} in model manifest")
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                if self.eof or (end < len(self.buffer) and self.buffer[end] in " \t\r\n,]}:"):
                    self.pos = end
                    return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
            if len(self.buffer) - self.pos >= 16 * 1024 ** 2:
                raise ValueError("model manifest contains an invalid or oversized entry")
            self.refill()

    def members(self, closing: str):
        if self.peek() != closing:
            while True:
                yield
                if self.peek() == closing:
                    break
                self.take(",")
        self.take(closing)

    def rows(self):
        self.take("{")
        got_items = False
        for _ in self.members("}"):
            field = self.value()
            if not isinstance(field, str):
                raise ValueError("model manifest field names must be strings")
            self.take(":")
            if field != "Items":
                self.metadata[field] = self.value()
                continue
            if got_items:
                raise ValueError("model manifest has duplicate Items fields")
            got_items = True
            self.take("[")
            for _ in self.members("]"):
                yield self.value()
        if not got_items or self.peek():
            raise ValueError("model manifest requires Items and no trailing content")


class Catalog:
    def __init__(self, args, out: Path):
        self.args = args
        self.out = out
        self.models: dict[Path, Model] = {}
        self.entries: dict[Path, dict] = {}
        self.blocked: set[Path] = set()
        self.pending: dict[Path, Model] = {}
        self.scanned: set[Path] = set()
        self.archives: set[str] = set()
        self.vehicles: set[str] = set()
        self.notes: Counter[str] = Counter()
        self.errors = 0
        self.failed: set[Path] = set()
        self.new = 0
        self.attempts = 0

    def issue(self, kind: str, detail, *, failed: bool = True, count: int = 1) -> None:
        if self.notes[kind] < 5:
            print(f"{kind}: {detail}", file=sys.stderr, flush=True)
        self.notes[kind] += count
        if failed:
            self.errors += count

    def publish(self, activity: str, *, complete: bool = False) -> None:
        counts = Counter(entry["category"] for entry in self.entries.values())
        summary = ", ".join(f"{counts[name]} {name}" for name in sorted(CATEGORY_DIRS))
        notes = ", ".join(f"{count} {kind}" for kind, count in sorted(self.notes.items()) if count)
        unscanned = [Path(path).stem for path in ARCHIVES if Path(path).stem not in self.archives]
        coverage = f" Export manifests unavailable for {', '.join(unscanned)}." if unscanned else ""
        message = f"{activity} {summary}. {notes + '. ' if notes else ''}{COVERAGE}{coverage}"
        processed = len((self.entries.keys() | self.failed) - self.pending.keys())
        total = processed + len(self.pending)
        doc = {
            "version": 1,
            "assets": [self.entries[path] for path in sorted(self.entries, key=lambda p: p.as_posix())],
            "status": {
                "complete": complete,
                "processed": processed,
                "total": total,
                "message": message,
            },
        }
        target = self.out / "catalog.json"
        safe_destination(target, self.out)
        atomic_write(target, lambda f: f.write((json.dumps(doc, ensure_ascii=True, separators=(",", ":")) + "\n").encode()))
        print(f"{processed}/{total} {activity} ({summary})", flush=True)

    def names(self, roots: list[Path]) -> None:
        files = [root / "vehicle-names.txt" for root in roots]
        if self.args.vehicle_names:
            files.append(self.args.vehicle_names)
        for file in files:
            if not file.is_file() and file != self.args.vehicle_names:
                continue
            try:
                if linked(file):
                    raise OSError(f"refusing linked vehicle name list {file}")
                with file.open(encoding="utf-8-sig") as f:
                    for line in f:
                        name = line.strip()
                        if name and not name.startswith("#"):
                            self.vehicles.add(normalized_name(name))
            except (OSError, ValueError) as exc:
                self.issue("vehicle name lists unreadable", exc)

    def scan(self, root: Path, *, in_place: bool = False) -> None:
        if root in self.scanned or not root.is_dir():
            return
        self.scanned.add(root)
        for directory, subdirs, names in os.walk(
            root, onerror=lambda exc: self.issue("unreadable folders", exc),
        ):
            folder = Path(directory)
            subdirs[:] = sorted(name for name in subdirs if segment(name) and not linked(folder / name))
            if not in_place and folder.name.casefold() == "models" and "manifest.json" in names:
                self.manifest(folder / "manifest.json", root)
            for name in sorted(names):
                if not name.casefold().endswith(".glb") or not segment(name):
                    continue
                source = folder / name
                if linked(source) or not source.is_file():
                    continue
                rel = source.relative_to(root)
                if placeholder(source.stem):
                    self.blocked.add(source if in_place else self.out / "library" / key(root) / rel.with_suffix(".glb"))
                    self.notes["fragment placeholders skipped"] += 1
                    continue
                destination = source if in_place else self.out / "library" / key(root) / rel.with_suffix(".glb")
                if destination.suffix != ".glb":
                    self.issue("unsupported GLB filenames", source)
                    continue
                archive = "packs" if "packs" in (part.casefold() for part in rel.parts[:-1]) else ""
                self.models.setdefault(destination, Model(
                    source, destination, source.stem, category(source.stem, rel, archive, self.vehicles),
                ))

    def manifest(self, path: Path, root: Path) -> None:
        if linked(path):
            self.issue("linked manifests skipped", path)
            return
        archive = path.parent.parent.name.casefold()
        seen: set[str] = set()
        count_rows = 0
        try:
            with path.open(encoding="utf-8-sig") as f:
                reader = ManifestReader(f)
                for row in reader.rows():
                    self.manifest_row(row, path, root, archive, seen)
                    count_rows += 1
            metadata = reader.metadata
            scan = metadata.get("Scan") or {}
            if not isinstance(scan, dict):
                raise ValueError("Scan must be an object")
        except (OSError, ValueError, RecursionError) as exc:
            self.issue("unreadable model manifests", f"{path}: {exc}")
            return
        whole = not any(field in scan for field in ("Start", "Length"))
        if whole:
            self.archives.add(archive)
        unavailable = scan.get("NotRecovered")
        if type(unavailable) is int and unavailable > 0:
            self.issue("unrecovered candidates", path, count=unavailable)
        elif unavailable is None:
            self.issue("manifests with unknown scan coverage", path, failed=False)
        count = metadata.get("Count")
        missing = count - count_rows if type(count) is int else 0
        failures = metadata.get("ModelDecodeFailures")
        if isinstance(failures, dict):
            missing = max(missing, sum(n for n in failures.values() if type(n) is int and n > 0))
        if missing > 0:
            self.issue("export rows missing or undecodable", path, count=missing)

    def manifest_row(self, row, path: Path, root: Path, archive: str, seen: set[str]) -> None:
        filename = row.get("File") if isinstance(row, dict) else None
        if not isinstance(filename, str) or not segment(filename) or Path(filename).suffix.casefold() != ".obj":
            self.issue("unusable manifest rows", path)
            return
        source = path.parent / filename
        name = row.get("Name") or source.stem
        if not isinstance(name, str):
            name = source.stem
        glb = source.with_suffix(".glb")
        rel = glb.relative_to(root)
        converted = self.out / "models" / archive / key(path.parent) / glb.name
        imported = self.out / "library" / key(root) / rel
        group = category(name, rel, archive, self.vehicles)
        legacy_paths = [self.out / candidate / archive / glb.name for candidate in (group, "props")]
        bounds_grid = row.get("IsFragment") and (row.get("Verts"), row.get("Tris")) == (25, 32)
        if bounds_grid or placeholder(name) or row.get("Verts") == 0 or row.get("Tris") == 0:
            self.blocked.update((converted, imported, *legacy_paths))
            self.notes["fragment or empty rows skipped"] += 1
            return
        model_key = os.path.normcase(filename)
        if model_key in seen:
            self.notes["rows sharing an exported filename"] += 1
            return
        seen.add(model_key)
        legacy = next((candidate for candidate in legacy_paths if candidate in self.models and maybe_glb(candidate)), None)
        if legacy is not None:
            self.models[legacy] = Model(source, legacy, name, group)
            self.blocked.update((converted, imported))
        elif glb.is_file() and not linked(glb) and maybe_glb(glb):
            self.models[imported] = Model(glb, imported, name, group)
            self.blocked.add(converted)
        else:
            self.models[converted] = Model(source, converted, name, group)
            if glb.is_file():
                self.blocked.add(imported)

    def register(self, model: Model) -> None:
        entry = {
            "name": model.name, "category": model.category,
            "model": relative_url(model.destination, self.out),
        }
        preview = png_sidecar(model.destination)
        if is_png(preview):
            entry["preview"] = relative_url(preview, self.out)
        self.entries[model.destination] = entry

    def discover_ready(self) -> None:
        for path in self.blocked:
            self.entries.pop(path, None)
            self.pending.pop(path, None)
        for path, model in self.models.items():
            if path in self.blocked:
                continue
            ready = path in self.entries or (not linked(path) and maybe_glb(path))
            if ready:
                self.register(model)
            preview = png_sidecar(model.source)
            if not ready or (model.source != path and is_png(preview) and not is_png(png_sidecar(path))):
                self.pending[path] = model
        self.models.clear()

    def drained(self) -> bool:
        for model in list(self.pending.values()):
            if self.args.max_new is not None and self.new >= self.args.max_new:
                return False
            try:
                self.build_one(model)
            except SpaceError as exc:
                self.issue("storage limits reached", exc)
                return False
            if self.attempts % self.args.batch_size == 0:
                self.publish("Building catalog.")
        return True

    def build_one(self, model: Model) -> None:
        source, destination = model.source, model.destination
        try:
            safe_destination(destination, self.out)
            if linked(source) or not source.is_file():
                raise OSError("source file is missing or linked")
            if destination not in self.entries:
                is_glb = source.suffix.casefold() == ".glb"
                estimate = source.stat().st_size * (1 if is_glb else 4)
                free_space(self.out, self.args.reserve_gib * GIB, estimate)
                if is_glb:
                    if not maybe_glb(source):
                        raise ValueError("expected a self-contained GLB with geometry")
                    copy_file(source, destination)
                else:
                    primitives = glb_writer.parse_obj(str(source), is_path=True, up_axis="z")
                    if not primitives:
                        raise ValueError("OBJ has no drawable geometry")
                    glb_writer.write_glb(str(destination), primitives)
                self.register(model)
            preview = png_sidecar(source)
            if source != destination and not is_png(png_sidecar(destination)) and is_png(preview):
                target = png_sidecar(destination)
                safe_destination(target, self.out)
                free_space(self.out, self.args.reserve_gib * GIB, preview.stat().st_size)
                copy_file(preview, target)
            self.register(model)
            self.new += 1
        except SpaceError:
            raise
        except (OSError, ValueError, OverflowError, struct.error) as exc:
            self.issue("models failed", f"{source}: {exc}")
            self.failed.add(destination)
        del self.pending[destination]
        self.attempts += 1


def build(args) -> int:
    out = args.out.resolve()
    work = args.work.resolve() if args.work else out.with_name(out.name + ".work")
    game = args.game.resolve(strict=True) if args.game else None
    managed = work / "rpf9dump-models" / key(out)
    exports = managed / key(game) if game else managed
    roots = list(dict.fromkeys(path.resolve(strict=True) for path in args.preexported))
    standard = work / "assets"
    if standard.is_dir() and not linked(standard) and not overlap(standard, out) and standard not in roots:
        roots.append(standard)
    if any(not root.is_dir() for root in [*roots, *([game] if game else [])]):
        raise ValueError("--game and --preexported must name directories")
    for writable in (out, managed):
        for source in (TOOLS.parent, *roots, *([game] if game else [])):
            if overlap(writable, source):
                raise ValueError(f"output {writable} overlaps source {source}")
    if overlap(out, managed):
        raise ValueError("--out must be separate from the exporter staging directory")
    safe_destination(exports, work)
    out.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(args, out)
    with catalog_lock(out):
        if not (out / "catalog.json").is_file():
            catalog.publish("Discovering local models.")
        catalog.names([out, *roots])
        catalog.scan(out, in_place=True)
        for root in [*roots, exports]:
            catalog.scan(root)
        catalog.discover_ready()
        catalog.publish("Discovered local models.")
        exhausted = catalog.drained()
        if game and exhausted:
            for relative in args.archive or ARCHIVES:
                archive = Path(relative).stem
                if archive in catalog.archives:
                    continue
                if args.max_new is not None and catalog.new >= args.max_new:
                    exhausted = False
                    break
                source = game / relative
                catalog.publish(f"Exporting {relative}.")
                try:
                    if source.resolve() != source or linked(source) or not source.is_file():
                        raise OSError(f"{relative} is missing or linked")
                    command = local_exporter(args.rpf9dump, work)
                    exports.mkdir(parents=True, exist_ok=True)
                    free_space(exports, args.reserve_gib * GIB, ARCHIVES[relative] * GIB)
                    staging = exports / archive
                    safe_destination(staging, exports)
                    for directory, subdirs, files in os.walk(staging):
                        for name in subdirs + files:
                            if linked(Path(directory) / name):
                                raise OSError(f"refusing linked entry in exporter staging: {name}")
                    export_models(command, source, exports, args.reserve_gib * GIB)
                    manifest = exports / archive / "models" / "manifest.json"
                    if not manifest.is_file():
                        raise OSError(f"rpf9dump did not publish a manifest for {source.name}")
                    catalog.manifest(manifest, exports)
                except SpaceError as exc:
                    catalog.issue("archives deferred for free space", f"{relative}: {exc}")
                    continue
                except OSError as exc:
                    catalog.issue("archives unavailable or export failed", f"{source.name}: {exc}")
                    continue
                catalog.discover_ready()
                if not catalog.drained():
                    exhausted = False
                    break
        if not catalog.entries and exhausted and not catalog.errors:
            catalog.issue("no ready models", "provide a local model export or valid GLBs")
        complete = exhausted and catalog.errors == 0
        catalog.publish("Catalog ready." if complete else "Catalog incomplete.", complete=complete)
    return int(bool(catalog.errors))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="viewer catalog and ready GLB directory")
    ap.add_argument("--work", type=Path, help="scratch root (default: <out>.work beside --out)")
    ap.add_argument("--exports", "--preexported", dest="preexported", type=Path, action="append", default=[],
                    help="existing export root; repeatable")
    ap.add_argument("--game", type=Path, help="local package root for missing documented model exports")
    ap.add_argument("--archive", choices=ARCHIVES, action="append",
                    help="documented package-relative archive for --game; repeatable, default: all")
    ap.add_argument("--rpf9dump", type=Path, help="local rpf9dump executable or DLL")
    ap.add_argument("--vehicle-names", type=Path, help="optional local list of verified vehicle names, one per line")
    ap.add_argument("--batch-size", type=int, default=256, help="publish a checkpoint every N attempted models")
    ap.add_argument("--max-new", type=int, help="pause after N successful conversions/imports")
    ap.add_argument("--reserve-gib", type=int, default=4, help="keep at least this much disk free")
    args = ap.parse_args(argv)
    if args.archive and not args.game:
        ap.error("--archive requires --game")
    if args.batch_size < 1 or (args.max_new is not None and args.max_new < 1) or args.reserve_gib < 0:
        ap.error("--batch-size and --max-new must be positive and --reserve-gib must be nonnegative")
    try:
        return build(args)
    except (OSError, ValueError) as exc:
        print(f"catalog: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

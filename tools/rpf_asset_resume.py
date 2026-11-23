"""Completion inventory for rpf9dump's model and texture exports."""

import codecs
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from replace_retry import replace


INVENTORY = "rpf9dump.outputs.json"
MANIFEST = "manifest.json"
SCHEMA = 1
_STRUCTURAL = re.compile(r'["{}\[\]\\]')
_STRING_END = re.compile(r'["\\]')
_NUMBER = re.compile(r"[-+0-9.eE]+")
_CLOSING = {"{": "}", "[": "]"}


def linked(path):
    return os.path.islink(path) or os.path.isjunction(path)


class LinkedExport(ValueError):
    """A link where rpf9dump would expect one of its own files or directories."""


def check_unlinked(path):
    if linked(path):
        raise LinkedExport(f"RPF export path is a link or junction: {path}")


def check_directories(folder, mode):
    """rpf9dump can write through existing directories when resuming an incomplete export."""
    check_unlinked(folder)
    for marker in (MANIFEST, MANIFEST + ".partial", INVENTORY):
        check_unlinked(Path(folder, marker))
    with os.scandir(folder) as entries:
        for entry in entries:
            if mode == "textures" or entry.name.endswith("_tex"):
                if linked(entry.path) and (entry.is_dir() or not os.path.exists(entry.path)):
                    check_unlinked(entry.path)


class ManifestItems:
    """Stream rpf9dump's JSON object one Items row at a time and hash the same bytes."""

    def __init__(self, stream):
        self.stream = stream
        self.text = ""
        self.pos = 0
        self.eof = False
        self.decode = codecs.getincrementaldecoder("utf-8-sig")()
        self.digest = hashlib.sha256()

    def read_more(self):
        data = self.stream.read(64 * 1024)
        self.eof = not data
        self.digest.update(data)
        self.text = self.decode.decode(data, final=self.eof)
        self.pos = 0

    def peek(self):
        while True:
            while self.pos < len(self.text):
                char = self.text[self.pos]
                if char not in " \t\r\n":
                    return char
                self.pos += 1
            if self.eof:
                return ""
            self.read_more()

    def take(self, char):
        if self.peek() != char:
            raise ValueError(f"expected {char!r} in RPF export manifest")
        self.pos += 1

    def value(self):
        initial = self.peek()
        if initial in _CLOSING or initial == '"':
            return self.structured_value(initial)
        keyword = {"t": "true", "f": "false", "n": "null"}.get(initial)
        if keyword:
            for char in keyword:
                while self.pos == len(self.text) and not self.eof:
                    self.read_more()
                if self.text[self.pos:self.pos + 1] != char:
                    raise ValueError("invalid literal in RPF export manifest")
                self.pos += 1
            return json.loads(keyword)
        if not initial or initial not in "-0123456789":
            raise ValueError(f"invalid JSON token {initial!r} in RPF export manifest")
        fragments = []
        while True:
            match = _NUMBER.match(self.text, self.pos)
            if match:
                fragments.append(match[0])
                self.pos = match.end()
            if self.eof or self.pos < len(self.text):
                return json.loads("".join(fragments))
            self.read_more()

    def structured_value(self, initial):
        """Frame one string, object or array before asking the standard decoder to validate it."""
        fragments = []
        closing = [_CLOSING[initial]] if initial in _CLOSING else []
        quoted, escaped = initial == '"', False
        start = self.pos
        self.pos += 1
        while quoted or closing:
            if self.pos == len(self.text):
                if self.eof:
                    raise ValueError("unterminated value in RPF export manifest")
                fragments.append(self.text[start:])
                self.read_more()
                start = 0
                continue
            if escaped:
                self.pos += 1
                escaped = False
                continue
            match = (_STRING_END if quoted else _STRUCTURAL).search(self.text, self.pos)
            if not match:
                self.pos = len(self.text)
                continue
            char, self.pos = match[0], match.end()
            if quoted:
                escaped = char == "\\"
                quoted = escaped
            elif char == '"':
                quoted = True
            elif char in _CLOSING:
                closing.append(_CLOSING[char])
            elif char != closing.pop():
                raise ValueError("mismatched delimiters in RPF export manifest")
        fragments.append(self.text[start:self.pos])
        return json.loads("".join(fragments))

    def __iter__(self):
        self.take("{")
        found = False
        while self.peek() != "}":
            key = self.value()
            if not isinstance(key, str):
                raise ValueError("expected a property name in RPF export manifest")
            self.take(":")
            if key == "Items":
                if found:
                    raise ValueError("duplicate Items in RPF export manifest")
                found = True
                self.take("[")
                if self.peek() != "]":
                    yield self.value()
                    while self.peek() == ",":
                        self.take(",")
                        yield self.value()
                self.take("]")
            else:
                self.value()
            if self.peek() != ",":
                break
            self.take(",")
            if self.peek() == "}":
                raise ValueError("trailing comma in RPF export manifest")
        self.take("}")
        if not found or self.peek():
            raise ValueError("expected a single RPF export manifest with Items")


def leaf(name):
    if (not isinstance(name, str) or name in ("", ".", "..") or "\0" in name or
            os.path.basename(name) != name or os.path.splitdrive(name)[0] or
            (os.name == "nt" and (":" in name or name.endswith((" ", "."))))):
        raise ValueError(f"invalid RPF export file or directory: {name!r}")
    return name


def texture_file(row):
    name, index = row.get("Name"), row.get("Index")
    if not isinstance(name, str) or type(index) is not int or index < 0:
        raise ValueError("invalid texture name or index in RPF export manifest")
    # Match rpf9dump Util.Safe, including C#'s 120 UTF-16 code unit limit.
    special = '/\\<>:"|?*' if os.name == "nt" else "/"
    name = "".join("_" if ch in special or ord(ch) < 32 or 127 <= ord(ch) <= 159 else ch for ch in name)
    name = name.strip(" .") or "x"
    name = name.encode("utf-16-le", "surrogatepass")[:240].decode("utf-16-le", "surrogatepass")
    return leaf(f"{index:03d}_{name}.dds")


def output_paths(row, mode):
    if not isinstance(row, dict):
        raise ValueError("expected an RPF export object in Items")
    if mode == "models":
        model = leaf(row.get("File"))
        if not model.endswith(".obj"):
            raise ValueError(f"expected an OBJ export, got {model!r}")
        yield (model,), True
        directory, textures = model[:-4] + "_tex", row.get("Embedded")
        if textures is None:
            return
    else:
        directory, textures = leaf(row.get("Dir")), row.get("Textures")
    if not isinstance(textures, list) or not all(isinstance(item, dict) for item in textures):
        raise ValueError(f"invalid texture list for {directory!r} in RPF export manifest")
    for texture in textures:
        yield (directory, texture_file(texture)), False


def present(folder, parts):
    path = folder
    for part in parts:
        path /= leaf(part)
        check_unlinked(path)
    return path.is_file()


def collect(folder, mode, require_all):
    path = folder / MANIFEST
    check_unlinked(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    files = set()
    with path.open("rb") as stream:
        manifest = ManifestItems(stream)
        for row in manifest:
            for parts, required in output_paths(row, mode):
                if present(folder, parts):
                    files.add(parts)
                elif require_all:
                    return None
                elif required:
                    raise ValueError(f"missing exported model: {folder.joinpath(*parts)}")
    return {"version": SCHEMA, "manifest_sha256": manifest.digest.hexdigest(),
            "file_count": len(files), "files": sorted(files)}


def publish(folder, inventory):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", delete=False,
                                         dir=folder, prefix=".rpf-outputs-", suffix=".partial") as stream:
            temporary = Path(stream.name)
            json.dump(inventory, stream, separators=(",", ":"))
            stream.write("\n")
        replace(temporary, folder / INVENTORY)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def remember(folder, mode):
    """Record a successful exporter run. Missing DDS candidates may be permanent decode skips."""
    folder = Path(folder)
    publish(folder, collect(folder, mode, require_all=False))


def complete(folder, mode):
    """Check the inventory or migrate a legacy manifest whose candidates all exist."""
    folder = Path(folder)
    check_unlinked(folder)
    index = folder / INVENTORY
    manifest = folder / MANIFEST
    check_unlinked(index)
    check_unlinked(manifest)
    if not index.exists():
        try:
            inventory = collect(folder, mode, require_all=True)
        except LinkedExport:
            raise
        except (OSError, ValueError, RecursionError):
            return False
        if inventory is None:
            return False
        publish(folder, inventory)
        return True
    try:
        if not index.is_file() or not manifest.is_file():
            return False
        with index.open(encoding="utf-8") as stream:
            inventory = json.load(stream)
        if not isinstance(inventory, dict):
            return False
        version = inventory.get("version")
        if type(version) is not int or version != SCHEMA:
            return False
        files = inventory.get("files")
        if not isinstance(files, list) or not all(isinstance(parts, list) and 1 <= len(parts) <= 2 for parts in files):
            return False
        if inventory.get("file_count") != len(files):
            return False
        with manifest.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return digest == inventory.get("manifest_sha256") and all(present(folder, parts) for parts in files)
    except LinkedExport:
        raise
    except (OSError, ValueError, RecursionError):
        return False

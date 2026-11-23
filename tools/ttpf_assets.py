"""Complete the rpf9dump texture-pool manifest during the assets build."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

# `python -I` implies `-P`, so the script's directory is not on sys.path; the sibling modules are next to it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bcpack
from file_mode import published_mode
import ttpf_mips as ttpf


def load_profile(work, rust_exporter=None):
    path = Path(work, "bcpack-calibration.json")
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return "v1-strict", None
    profile = json.loads(raw)
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise ValueError(f"Unsupported calibration version in {path}")
    if rust_exporter is None:
        check_fitted_reference()
    return "v1-" + hashlib.sha256(raw).hexdigest(), profile


def check_fitted_reference():
    try:
        from calibrated_texture import native
        native()
    except (OSError, ImportError) as exc:
        raise RuntimeError("Calibrated reference export requires NumPy and the release build of "
                           "tools/bcpack_calibration (see its README.md)") from exc


def batch_executable():
    configured = os.environ.get("LEONIDA_BCPACK_DDS")
    if configured:
        return Path(configured).resolve(strict=True)
    name = "bcpack_dds.exe" if os.name == "nt" else "bcpack_dds"
    path = Path(__file__).with_name("bcpack_dds") / "target" / "release" / name
    return path if path.is_file() else None


def publish(path, data):
    f = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    tmp = Path(f.name)
    try:
        with f:
            f.write(data)
        if os.name != "nt":
            tmp.chmod(published_mode(path))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def candidate(row, profile):
    if row.get("DxgiFormat") not in ttpf.BC_FORMATS:
        return False
    if row.get("BCPackDecoder"):
        return profile is not None or row["BCPackDecoder"].endswith("-strict")
    return (row["File"] is None or row.get("BaseMipLevel", 0) > 0
            or row.get("Mips", 0) < row.get("SourceMips", 0))


def complete_pool(source, directory, manifest, settings, rust_exporter=None, jobs=1, *, report=False):
    """Publish full chains next to ordinary DDS files, then switch the catalog to them."""
    stamp, profile = settings
    pass_stamp = stamp if rust_exporter is None else f"{stamp}-rust-batch-v2"
    if manifest.get("BCPackPass") == pass_stamp:
        return 0, 0
    rows = [row for row in manifest["Textures"] if candidate(row, profile)]
    if not rows:
        return 0, 0
    directory = Path(directory)
    entries = {row["Index"]: {"index": row["Index"], "rust": None, "publisher": None,
                               "outcome": "skipped"} for row in rows} if report else {}

    def finish():
        if report:
            summary = {"pass": pass_stamp, "entries": list(entries.values())}
            publish(directory / "bcpack-export-report.json", json.dumps(summary, indent=2).encode("utf-8"))
        manifest["BCPackPass"] = pass_stamp
        publish(directory / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
        return written, blocked

    written = blocked = 0
    if rust_exporter is not None:
        with tempfile.TemporaryDirectory(prefix=".bcpack-", dir=directory) as temp:
            staging = Path(temp)
            selection = staging / "selection.json"
            selection.write_text(json.dumps({
                "indices": [row["Index"] for row in rows],
                "bcpack_only": True, "prefer_fitted": profile is not None,
            }), encoding="utf-8", newline="\n")
            command = [str(rust_exporter), str(source), str(staging), "--jobs",
                       str(max(1, min(jobs, 256))), "--selection", str(selection)]
            if profile is not None:
                profile_path = staging / "profile.json"
                profile_path.write_text(json.dumps(profile), encoding="utf-8", newline="\n")
                command += ["--calibration", str(profile_path)]
            result = subprocess.run(command, capture_output=True, encoding="utf-8")
            if result.returncode:
                for item in entries.values():
                    item["rust"] = {"status": "batch_failed", "exit_code": result.returncode,
                                    "reason": result.stderr.strip()}
                print(f"Rust BCPack export failed for {source}: {result.stderr.strip()}", file=sys.stderr)
            else:
                exported = {row["index"]: row for row in json.loads((staging / "manifest.json").read_bytes())}
                remaining = []
                for row in rows:
                    index = row["Index"]
                    entry = exported.get(index, {})
                    if report:
                        attempt = {"status": entry.get("status", "missing")}
                        if "reason" in entry:
                            attempt["reason"] = entry["reason"]
                        entries[index]["rust"] = attempt
                    if entry.get("status") == "skipped":
                        continue
                    if entry.get("status") != "success":
                        remaining.append(row)
                        continue
                    method = stamp if entry["mode"] == "fitted" else "v1-strict"
                    name = f"{index:06d}_bcpack_{method[:19]}.dds"
                    staged = staging / f"tex_{index:05}.dds"
                    if os.name != "nt":
                        staged.chmod(published_mode(directory / name))
                    os.replace(staged, directory / name)
                    row.update(File=name, Mips=entry["mips"], BaseMipLevel=0, Limitation=None, BCPackDecoder=method)
                    row.pop("BCPackLimitation", None)
                    if report:
                        entries[index].update(publisher="rust", outcome="published", decoder=method)
                    written += 1
                rows = remaining
    if not rows:
        return finish()

    with closing(ttpf.Pool(source)) as pool:
        decoder = None
        try:
            for row in rows:
                index = row["Index"]
                desc = pool.descriptor(index)
                base = desc["base"]
                if not pool.chunks or base < pool.chunks[0][0] or base >= pool.virtual_size:
                    continue
                span = pool.span(index, base)
                if not span:
                    continue
                try:
                    if not pool.codecs(base, span) & bcpack.SUPPORTED_REBUILD_CODECS:
                        continue
                    if profile is not None and desc["dimension"] == 1 and desc["format"] in ttpf.BC_FORMATS:
                        if decoder is None:
                            check_fitted_reference()
                            from calibrated_texture import LocalDecoder
                            decoder = LocalDecoder(pool, profile)
                    if decoder and desc["dimension"] == 1 and desc["format"] in decoder.formats:
                        method = stamp
                        blob = decoder.dds(index)
                    else:
                        method = "v1-strict"
                        analysis = ttpf.analyze(pool, index)
                        blob = ttpf.texture_dds(pool, analysis)
                except (ValueError, KeyError, IndexError, ttpf.BcpackRequired,
                        bcpack.BcpackUnsupportedCodec, zlib.error) as exc:
                    row["BCPackLimitation"] = str(exc)
                    if report:
                        entries[index].update(outcome="blocked", reason=str(exc))
                    blocked += 1
                    continue
                # A distinct name keeps an existing lower-mip DDS usable until the manifest switches.
                name = f"{index:06d}_bcpack_{method[:19]}.dds"
                publish(directory / name, blob)
                row.update(File=name, Mips=desc["mips"], BaseMipLevel=0, Limitation=None, BCPackDecoder=method)
                row.pop("BCPackLimitation", None)
                if report:
                    entries[index].update(publisher="python", outcome="published", decoder=method)
                written += 1
        finally:
            if decoder:
                decoder.read_chunk.cache_clear()
    return finish()

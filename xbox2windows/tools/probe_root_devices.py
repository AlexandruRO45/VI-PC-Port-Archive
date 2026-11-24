"""Test source-built root signatures and optionally measure layouts in a local call inventory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

import pefile
from call_evidence import NAMES
from probe_roots import PROJECT, parts


def measure(command, blob):
    result = subprocess.run(command, input=blob, stdout=subprocess.PIPE, check=True, timeout=30)
    return [json.loads(line) for line in result.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, nargs="?", help="Omit to run just the two source-built controls")
    parser.add_argument("--warp-only", action="store_true")
    parser.add_argument("--bin", type=Path, default=PROJECT / "out/build/dev")
    args = parser.parse_args()
    command = [args.bin / "root_device_probe.exe"]
    backends = ["warp"]
    if args.warp_only:
        command.append("--warp-only")
    else:
        backends.append("hardware")
    result = {"windows": sys.getwindowsversion().build, "controls": {}, "images": {}}
    for name in ("compute_root", "compute_tables"):
        rows = measure(command, (args.bin / (name + ".cso")).read_bytes())
        if (len(rows) != 1 or rows[0]["cleared"] != 0 or rows[0]["remaining_flags"] != 0 or
                any(rows[0].get(key) != "0x00000000" for key in ["serialize", *backends])):
            raise ValueError(f"{name} control failed: {rows}")
        result["controls"][name] = rows
    if args.build is None:
        return

    dest = PROJECT / "out/calls" / ("root-devices-warp.json" if args.warp_only else "root-devices.json")
    if dest.resolve().is_relative_to(args.build.resolve()):
        parser.error("Choose an input directory separate from the report directory")
    inventory = json.loads((dest.parent / "six-apis.json").read_bytes())
    for name in NAMES:
        entry = inventory[name]
        if not entry["containers"]:
            continue
        image = (args.build / name).read_bytes()
        if hashlib.sha256(image).hexdigest() != entry["sha256"]:
            raise ValueError(f"Regenerate six-apis.json for {name}")
        layouts = {}
        with pefile.PE(data=image, fast_load=True) as pe:
            for item in entry["containers"]:
                blob = pe.get_data(int(item["rva"], 16), item["bytes"])
                key = hashlib.sha256(parts(blob)[b"RTS0"]).hexdigest()
                if key not in layouts:
                    layouts[key] = {"rvas": [], "outcomes": measure(command, blob)}
                layouts[key]["rvas"].append(item["rva"])
        result["images"][name] = {"sha256": entry["sha256"], "layouts": layouts}
        print(f"{name}: {len(layouts)} distinct layouts")
    with NamedTemporaryFile(mode="w", dir=dest.parent, delete=False, encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    Path(stream.name).replace(dest)
    print(dest)


if __name__ == "__main__":
    main()

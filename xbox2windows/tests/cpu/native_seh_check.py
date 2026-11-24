"""Raise and catch SEH exceptions inside a mapped image, compared with the Windows-loaded original."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys

from direct_check import SENTINELS, register_probe
from native_map_check import Mapper, STATUS

Word = ctypes.c_uint64
FAULT_VALUE = 0x0fa017fa017fa017
LOADED = 0x1122334455667788
ILLEGAL_INSTRUCTION = 0xc000001d


def call(probe, fn, address, mode):
    trace = (Word * 2)(0x5a5a5a5a5a5a5a5a, 0xa5a5a5a5a5a5a5a5)
    regs = (Word * 2)()
    try:
        result = probe(fn, address, mode, trace, regs)
    except OSError as exc:  # ctypes reports an exception no frame handled as OSError
        code = exc.winerror & 0xffffffff if exc.winerror else str(exc)
        return {"exception": code, "trace": list(trace)}
    return {"result": result, "trace": list(trace), "regs": list(regs)}


def run(target, image, mapper_path, probe_path):
    """Child process per target: a broken unwind crashes the child and fails the check, not the harness."""
    probe = register_probe(probe_path)
    base = None
    if target == "native":
        fn = ctypes.cast(ctypes.CDLL(image).guarded_load, ctypes.c_void_p).value
    else:
        mapper = Mapper(mapper_path)
        base, status, detail, _ = mapper(Path(image), {})
        if status != STATUS["ok"]:
            raise AssertionError(f"map failed: {status} {detail}")
        fn = mapper.export(base, b"guarded_load")
    loaded = Word(LOADED)
    address = ctypes.addressof(loaded)
    cases = {
        "load": call(probe, fn, address, 0),
        "caught_access_violation": call(probe, fn, 0, 0),
        "declined_illegal_instruction": call(probe, fn, address, 1),
        "after_exception": call(probe, fn, address, 0),
    }
    if base is not None:
        mapper.release(base)
    print(json.dumps({"address": address, "cases": cases}))


def check(image, mapper_path, probe_path):
    results = {}
    for target in ("native", "mapped"):
        child = subprocess.run([sys.executable, "-B", __file__, "--run", target, image, mapper_path, probe_path],
                               capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise AssertionError(f"{target} child exited with {child.returncode:#x}: {child.stderr.strip()}")
        results[target] = json.loads(child.stdout)
    for target, observed in results.items():
        address, cases = observed["address"], observed["cases"]
        expected = {
            "load": {"result": LOADED, "trace": [address, LOADED], "regs": list(SENTINELS)},
            "caught_access_violation": {"result": FAULT_VALUE, "trace": [0, FAULT_VALUE], "regs": list(SENTINELS)},
            # trace[0] shows guarded_load ran up to ud2; trace[1] keeps its sentinel.
            "declined_illegal_instruction": {"exception": ILLEGAL_INSTRUCTION,
                                             "trace": [address, 0xa5a5a5a5a5a5a5a5]},
            "after_exception": {"result": LOADED, "trace": [address, LOADED], "regs": list(SENTINELS)},
        }
        if cases != expected:
            raise AssertionError(f"{target}: {cases} != {expected}")
    print("native and mapped: load returns the value, the image's own handler resumes an access violation "
          "with RBX/RSI restored, an illegal instruction unwinds through the mapped frame to the caller")


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("image", "mapper", "probe"):
            parser.add_argument(arg)
        args = parser.parse_args()
        check(args.image, args.mapper, args.probe)

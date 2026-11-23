"""Check the two traces written by tiny_boot_trace_guest under `gta6rt --run`.

Standard library only, no game data. The guest prints its own marker; this script checks that gta6rt's startup trace
and the shim's D3D12 trace agree with it, using the same parsers as tools/boot_trace.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import boot_trace  # noqa: E402

EXPECTED_METHODS = [
    "D3D12XboxCreateDevice",
    "ID3D12Device::CreateCommandQueue",
    "ID3D12Device::CreateCommandAllocator",
    "ID3D12Device::CreateCommandList",
    "ID3D12GraphicsCommandList::Close",
    "XboxID3D12CommandQueue::ExecuteCommandLists",
    "ID3D12CommandQueue::ExecuteCommandLists",
    "ID3D12Device::CreateFence",
    "XboxID3D12CommandQueue::Signal",
    "ID3D12CommandQueue::Signal",
    "XboxID3D12CommandQueue::Wait",
    "ID3D12CommandQueue::Wait",
    "XboxID3D12Fence::SetEventOnCompletion",
]
# The Xbox-ABI queue and fence proxies record a `called` line before forwarding these calls.
CALLED_METHODS = {
    "XboxID3D12CommandQueue::ExecuteCommandLists",
    "XboxID3D12CommandQueue::Signal",
    "XboxID3D12CommandQueue::Wait",
    "XboxID3D12Fence::SetEventOnCompletion",
}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_boot_trace_guest.py boot.jsonl d3d12.log", file=sys.stderr)
        return 2
    startup = boot_trace.parse_startup(sys.argv[1], limit=10)
    d3d12 = boot_trace.parse_d3d12(sys.argv[2], limit=10)

    errors = []
    # An EXE guest has no DllMain, so there is no dllmain event in its startup trace.
    for kind in ("trace_open", "map", "exit"):
        if not startup["kinds"].get(kind):
            errors.append(f"startup trace has no {kind} event")
    if startup["bad_lines"]:
        errors.append(f"{startup['bad_lines']} bad startup line(s)")
    if startup["exception_count"]:
        errors.append(f"{startup['exception_count']} exception(s)")
    if startup["exit"] != 0:
        errors.append(f"guest exit is {startup['exit']}, not 0")

    if d3d12["methods"] != EXPECTED_METHODS:
        errors.append(f"D3D12 methods are {d3d12['methods']}, not {EXPECTED_METHODS}")
    if d3d12["calls"] != len(EXPECTED_METHODS):
        errors.append(f"{d3d12['calls']} D3D12 call(s), not {len(EXPECTED_METHODS)}")
    if d3d12["last_seq"] != d3d12["calls"]:
        errors.append(f"last sequence is {d3d12['last_seq']}, not {d3d12['calls']}")
    for outcome, count in sorted(d3d12["outcomes"].items()):
        if outcome not in ("ok", "called"):
            errors.append(f"{count} D3D12 call(s) ended as {outcome}")
    if d3d12["outcomes"].get("ok") != len(EXPECTED_METHODS) - len(CALLED_METHODS):
        errors.append(f"D3D12 ok outcomes are {d3d12['outcomes'].get('ok')}, not "
                      f"{len(EXPECTED_METHODS) - len(CALLED_METHODS)}")
    if d3d12["outcomes"].get("called") != len(CALLED_METHODS):
        errors.append(f"D3D12 called outcomes are {d3d12['outcomes'].get('called')}, not {len(CALLED_METHODS)}")
    for bucket in ("failed", "unsupported", "refused"):
        if d3d12[bucket]:
            errors.append(f"{len(d3d12[bucket])} {bucket} call(s)")

    if errors:
        print("boot trace check failed: " + "; ".join(errors), file=sys.stderr)
        return 1
    print(
        f"boot trace verified: {startup['events']} startup event(s), {d3d12['calls']} D3D12 call(s), "
        "0 failed, 0 unsupported, 0 refused, exit 0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

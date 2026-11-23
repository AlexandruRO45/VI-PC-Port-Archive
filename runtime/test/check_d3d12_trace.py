"""Check the trace on the CPU command-list fixture, or on d3d12_com_test with --queue."""

import argparse
import re
import subprocess


RUNNER = []  # the command that runs a Windows executable here: empty on Windows, ["wine"] for a cross build
ROOT_CONSTANTS_DETAIL = "[rt] SetGraphicsRoot32BitConstants #"


def run_fixture(exe, *args):
    # A first start of Wine in a fresh prefix can take tens of seconds, hence the longer limit through a runner.
    process = subprocess.run([*RUNNER, exe, *args], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=120 if RUNNER else 30)
    if process.returncode:
        raise SystemExit(f"{exe} exited with {process.returncode}\n{process.stdout}{process.stderr}")
    return process.stderr.splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    parser.add_argument("--queue", action="store_true", help="use the Vulkan-backed d3d12_com_test fixture")
    parser.add_argument("--runner", action="append", default=[],
                        help="run the fixture through this command (repeat for its arguments), e.g. wine")
    options = parser.parse_args()
    RUNNER[:] = options.runner

    if any(line.startswith(("[d3d12-trace]", ROOT_CONSTANTS_DETAIL)) for line in run_fixture(options.exe)):
        raise SystemExit("D3D12 trace appeared with tracing disabled")

    stderr = run_fixture(options.exe, "--trace-d3d12")
    trace = [line for line in stderr if line.startswith("[d3d12-trace]")]
    entries = []
    for number, line in enumerate(trace, 1):
        if "synthetic-marker-argument" in line:
            raise SystemExit("D3D12 trace leaked the synthetic marker payload")
        match = re.fullmatch(r"\[d3d12-trace\] #(\d+) ([A-Za-z_][\w:]* (?:called|ok|failed|unsupported|ignored))", line)
        if not match or int(match[1]) != number:
            raise SystemExit(f"Invalid trace sequence at entry {number}\n{line}")
        entries.append(match[2])

    if options.queue:
        expected = [
            "ID3D12CommandQueue::ExecuteCommandLists failed",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList::Close failed",
            "ID3D12CommandQueue::Wait ok",
            "ID3D12CommandQueue::ExecuteCommandLists ok",
            "ID3D12CommandQueue::Signal ok",
        ]
    else:
        root_constants = "ID3D12GraphicsCommandList::SetGraphicsRoot32BitConstants called"
        expected = [
            "ID3D12GraphicsCommandList::SetPipelineState called",
            "ID3D12GraphicsCommandList::SetGraphicsRootSignature called",
            "ID3D12GraphicsCommandList::SetMarker called",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList3::SetProtectedResourceSession unsupported",
            "ID3D12GraphicsCommandList::Close ok",
            "ID3D12GraphicsCommandList::Close failed",
            "ID3D12GraphicsCommandList::Reset ok",
            "ID3D12GraphicsCommandList::SetDescriptorHeaps ignored",
            "ID3D12GraphicsCommandList::SetGraphicsRootDescriptorTable called",
            root_constants,
            "ID3D12GraphicsCommandList::SetComputeRootSignature called",
            "ID3D12GraphicsCommandList::Dispatch called",
            "ID3D12GraphicsCommandList::DrawInstanced called",
            "ID3D12GraphicsCommandList::DrawIndexedInstanced called",
            "ID3D12GraphicsCommandList::Close failed",
        ]

    observed = iter(entries)
    for entry in expected:
        if entry not in observed:
            raise SystemExit(f"Missing or out-of-order trace entry {entry!r}\n" + "\n".join(trace))
    if not options.queue:
        if entries.count(root_constants) != 1:
            raise SystemExit("Expected exactly one graphics root constants call")
        number = entries.index(root_constants) + 1  # trace numbers are contiguous and start at 1
        details = [line for line in stderr if line.startswith(ROOT_CONSTANTS_DETAIL)]
        if details != [f"{ROOT_CONSTANTS_DETAIL}{number} index=3 count=2 offset=5"]:
            raise SystemExit(f"Expected only index, count and offset for graphics root constants call #{number}")
    print(f"D3D12 trace verified with {len(entries)} ordered entries")


if __name__ == "__main__":
    main()

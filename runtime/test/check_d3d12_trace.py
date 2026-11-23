"""Check the trace on the CPU command-list fixture, or on d3d12_com_test with --queue."""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script's folder out
import wine_session  # noqa: E402


RUNNER = []  # the command that runs a Windows executable here: empty on Windows, ["wine"] for a cross build


def run_fixture(exe, *args):
    # A first start of Wine in a fresh prefix can take tens of seconds, hence the longer limit through a runner.
    # wine_session.run: output through a file, the wait on the process, and with an xvfb-run runner the prefix's Wine
    # stopped around the run, since Wine left on another X server can hang the next Vulkan call.
    timeout = 120 if RUNNER else 30
    code, output = wine_session.run([*RUNNER, exe, *args], timeout)
    if code is None:
        raise SystemExit(f"{exe} did not exit within {timeout} s\n{output}")
    if code:
        raise SystemExit(f"{exe} exited with {code}\n{output}")
    return [line for line in output.splitlines() if line.startswith("[d3d12-trace]")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe")
    parser.add_argument("--queue", action="store_true", help="use the Vulkan-backed d3d12_com_test fixture")
    parser.add_argument("--runner", action="append", default=[],
                        help="run the fixture through this command (repeat for its arguments), e.g. wine")
    options = parser.parse_args()
    RUNNER[:] = options.runner

    if run_fixture(options.exe):
        raise SystemExit("D3D12 trace appeared with tracing disabled")

    trace = run_fixture(options.exe, "--trace-d3d12")
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
    print(f"D3D12 trace verified with {len(entries)} ordered entries")


if __name__ == "__main__":
    main()

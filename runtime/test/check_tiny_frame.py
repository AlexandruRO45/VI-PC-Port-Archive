"""Run tiny_frame under `gta6rt --run --trace-d3d12` on the Vulkan D3D12 device and check the run from outside.

    python -I check_tiny_frame.py <gta6rt.exe> <tiny_frame.exe> [--probe d3d12_com_test.exe] [--runner wine ...]
                                  [--validation]

Checks the guest's verdict and exit code, that the loader's report binds d3d12_xs.dll!D3D12XboxCreateDevice to the
shim, and that the D3D12 trace is a gapless numbered sequence in which the calls a first frame needs appear in
order. Exits 77 (ctest: skipped) when --probe, one of the Vulkan device tests, says
there is no usable Vulkan device on this machine.

--validation runs it with LEONIDA_VK_VALIDATION=1 and VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation (unless already
set): the device must still be created, the runtime must say whether it enabled the layer itself or why it did not (a
loader that lists no such layer: Wine's vulkan-1.dll, or Windows without the Vulkan SDK), and no validation error may
be printed. Under Wine only the environment variable reaches the layer, through the host's loader.
"""

import argparse
import os
import re
import subprocess
import sys

SKIP = 77

# What the runtime prints when LEONIDA_VK_VALIDATION=1: it enabled the layer, or the loader does not list it.
VALIDATION_ENABLED = "[rt] d3d12: LEONIDA_VK_VALIDATION: VK_LAYER_KHRONOS_validation enabled"
VALIDATION_UNLISTED = "[rt] d3d12: LEONIDA_VK_VALIDATION: the Vulkan loader does not list VK_LAYER_KHRONOS_validation"

# The first frame, in the order the fixture makes the calls, as the trace words them (`called` for a method that
# returns nothing). Other traced calls, such as the queue and fence proxies' own entries, may sit between them.
EXPECTED = [
    "D3D12XboxCreateDevice ok",
    "ID3D12Device::CreateCommandQueue ok",
    "ID3D12Device::CreateCommandAllocator ok",
    "ID3D12Device::CreateCommandList ok",
    "ID3D12Device::CreateFence ok",
    "ID3D12Device::CreateDescriptorHeap ok",
    "ID3D12Device::CreateCommittedResource ok",
    "ID3D12Device::CreateCommittedResource ok",
    "ID3D12Device::CreateRenderTargetView called",
    "ID3D12GraphicsCommandList::ClearRenderTargetView called",
    "ID3D12GraphicsCommandList::ResourceBarrier called",
    "ID3D12GraphicsCommandList::CopyTextureRegion called",
    "ID3D12GraphicsCommandList::Close ok",
    "ID3D12CommandQueue::ExecuteCommandLists ok",
    "ID3D12CommandQueue::Signal ok",
]


def ordered_subsequence(expected, entries):
    """The first expected entry that does not appear, in order, in entries; None when all do."""
    position = 0
    for want in expected:
        while position < len(entries) and entries[position] != want:
            position += 1
        if position == len(entries):
            return want
        position += 1
    return None


def parse_trace(lines):
    """[(method, outcome)] from the `[d3d12-trace] #N method outcome` lines; raises ValueError on a gap."""
    entries = []
    for line in lines:
        if not line.startswith("[d3d12-trace]"):
            continue
        match = re.fullmatch(r"\[d3d12-trace\] #(\d+) (.+) (called|ok|failed|unsupported|ignored|refused)", line)
        if not match or int(match[1]) != len(entries) + 1:
            raise ValueError(f"invalid trace sequence at entry {len(entries) + 1}: {line}")
        entries.append((match[2], match[3]))
    return entries


def check_output(code, output, validation=False):
    """A list of problems with one run's exit code and combined output; empty when it is what the test expects."""
    problems = []
    lines = output.replace("\r\n", "\n").split("\n")
    if validation:
        if not any(line.startswith((VALIDATION_ENABLED, VALIDATION_UNLISTED)) for line in lines):
            problems.append("LEONIDA_VK_VALIDATION=1, but the runtime said neither that it enabled the layer "
                            "nor why not")
        errors = [line for line in lines if "Validation Error" in line]
        if errors:
            problems.append(f"{len(errors)} validation error(s), the first: {errors[0]}")
    if code != 0:
        problems.append(f"exit code {code}")
    if "tiny_frame: OK" not in lines:
        problems.append("the guest did not print `tiny_frame: OK`")
    if any(line.startswith("tiny_frame: FAIL") for line in lines):
        problems.append("the guest reported a failed check")
    if "game exited with code 0" not in lines:
        problems.append("gta6rt did not report `game exited with code 0`")
    if not any(re.fullmatch(r" +shim +d3d12_xs\.dll!D3D12XboxCreateDevice", line) for line in lines):
        problems.append("d3d12_xs.dll!D3D12XboxCreateDevice was not bound to the shim")
    try:
        entries = parse_trace(lines)
    except ValueError as error:
        return problems + [str(error)]
    bad = [f"{method} {outcome}" for method, outcome in entries if outcome in ("failed", "unsupported", "refused")]
    if bad:
        problems.append("traced calls that did not succeed: " + ", ".join(bad))
    missing = ordered_subsequence(EXPECTED, [f"{method} {outcome}" for method, outcome in entries])
    if missing:
        problems.append(f"the trace has no `{missing}` at its place in the first-frame order")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("loader")
    parser.add_argument("guest")
    parser.add_argument("--probe", help="a Vulkan device test; its exit code 77 skips this test")
    parser.add_argument("--runner", action="append", default=[],
                        help="run Windows executables through this command (repeat for its arguments), e.g. wine")
    parser.add_argument("--validation", action="store_true",
                        help="run under VK_LAYER_KHRONOS_validation (LEONIDA_VK_VALIDATION=1); "
                             "fail on a validation error")
    options = parser.parse_args()
    timeout = 180 if options.runner else 90  # a first start of Wine in a fresh prefix can take tens of seconds

    if options.probe:
        probe = subprocess.run([*options.runner, options.probe], capture_output=True, timeout=timeout)
        if probe.returncode == SKIP:
            print("SKIP: no usable Vulkan device (the probe test skipped)")
            return SKIP

    environment = dict(os.environ, LEONIDA_D3D12="vulkan")
    if options.validation:
        environment["LEONIDA_VK_VALIDATION"] = "1"
        environment.setdefault("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
    run = subprocess.run([*options.runner, options.loader, "--run", os.path.basename(options.guest), "--trace-d3d12"],
                         cwd=os.path.dirname(os.path.abspath(options.guest)), env=environment, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=timeout)
    output = run.stdout + run.stderr
    problems = check_output(run.returncode, output, options.validation)
    if problems:
        print(output)
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    suffix = " with validation" if options.validation else ""
    print(f"tiny_frame under gta6rt --run on the Vulkan D3D12 device{suffix}: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

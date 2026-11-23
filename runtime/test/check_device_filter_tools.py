"""LEONIDA_VK_DEVICE in vk_descriptor_bench and vk_sandbox (TEST-1141).

With a filter that matches no device, a tool must not choose another GPU. It names the filter and fails the run, as
the other harnesses do (runtime/test/vk_device_pick.h, docs/testing.md "Choosing the GPU"). Before, both tools
ignored the filter and ran on the best-ranked GPU.

    python -I check_device_filter_tools.py <vk_descriptor_bench> <vk_sandbox> <work dir>

Exit 0 when both tools refuse the filter, 1 when one does not, and 77 (skip) on a machine with no Vulkan device,
where no tool can choose one. Standard library only.
"""
import os
import struct
import subprocess
import sys

FILTER = "leonida-no-such-gpu"
UNSUPPORTED = 77


def run(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=120)
    return p.returncode, p.stdout + p.stderr


def probe_status(code, output):
    if code == 0 and "device probe available:" in output:
        return "available"
    if code == UNSUPPORTED and "device probe unavailable:" in output:
        return "unavailable"
    return "failure"


def self_test_probe_status():
    # Exercise the subprocess boundary: a real unavailable result skips, but arbitrary nonzero child failures do not.
    child = "import sys; print(sys.argv[2]); raise SystemExit(int(sys.argv[1]))\n"
    cases = ((0, "device probe available: 2 Vulkan physical device(s)", "available"),
             (UNSUPPORTED, "device probe unavailable: no Vulkan physical devices found", "unavailable"),
             (23, "synthetic unexpected enumeration error", "failure"),
             (UNSUPPORTED, "synthetic unexpected enumeration error", "failure"))
    for code, message, want in cases:
        actual_code, output = run([sys.executable, "-c", child, str(code), message])
        if actual_code != code or probe_status(actual_code, output) != want:
            print(f"FAIL: child status {code} with {message!r}: got {actual_code}, {output.strip()}")
            return 1
    print("ok: device probe child distinguishes available, explicit unavailable, and unexpected failures")
    return 0


def main(argv):
    if argv[1:] == ["--self-test-probe-status"]:
        return self_test_probe_status()
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    bench, sandbox, work = argv[1:]
    os.makedirs(work, exist_ok=True)

    # Probe ignores the filter and distinguishes no device from startup/enumeration errors.
    plain = {k: v for k, v in os.environ.items() if k != "LEONIDA_VK_DEVICE"}
    code, out = run([bench, "--probe-device"], env=plain)
    status = probe_status(code, out)
    if status == "unavailable":
        print(f"SKIP: {out.strip()}")
        return UNSUPPORTED
    if status != "available":
        print(f"FAIL: {bench} --probe-device exited {code} without a recognized status:\n{out.strip()}")
        return 1

    # The smallest scene vk_sandbox reads: magic, version 1, no textures, meshes or draws, and a centre.
    scene = os.path.join(work, "empty_scene.bin")
    with open(scene, "wb") as f:
        f.write(b"LSB1" + struct.pack("<4I", 1, 0, 0, 0) + struct.pack("<3f", 0.0, 0.0, 0.0))
    missing = os.path.join(work, "never_read.spv")   # the device is chosen before the shaders are read

    env = dict(plain, LEONIDA_VK_DEVICE=FILTER)
    failures = []
    for name, cmd in (("vk_descriptor_bench", [bench]),
                      ("vk_sandbox", [sandbox, scene, missing, missing, "--out", os.path.join(work, "out.ppm")])):
        code, out = run(cmd, env=env)
        named = f"LEONIDA_VK_DEVICE={FILTER}" in out
        if code == 0 or not named or "device:" in out:
            failures.append(f"{name}: exit {code}, {'names' if named else 'does not name'} the filter:\n{out.strip()}")
        else:
            print(f"ok: {name} refuses LEONIDA_VK_DEVICE={FILTER} (exit {code})")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""LEONIDA_VK_VALIDATION=1 never yields a clean result from a validation layer that is not there.

    python -I check_validation_probe.py <d3d12_com_test.exe> [--runner wine ...]

vk_test_device.h asks for VK_LAYER_KHRONOS_validation when the loader lists it, and otherwise (Wine's loader lists
none) relies on a layer enabled outside the program. Either way it makes one deliberate invalid call after creating
the device and fails with "validation is not active" when that call is not reported. Two runs of a GPU test check it:

- with the layer available (VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation, the way under Wine): the probe is
  reported and the test passes;
- with the layer disabled in the loader (VK_LOADER_LAYERS_DISABLE): the test fails and says that validation is not
  active, instead of reporting 0 validation errors.

Exits 77 (ctest: skipped) when the test has no Vulkan device, or the layer is not installed for the first run (the
loader does not list it and nothing reports the probe, or vkCreateInstance says VK_ERROR_LAYER_NOT_PRESENT).
"""

import argparse
import os
import subprocess
import sys

SKIP = 77
LAYER = "VK_LAYER_KHRONOS_validation"
ACTIVE = "validation probe: reported, validation is active"
# What the device says when no layer watches it: the probe went unreported, or (on a loader that refuses a disabled
# layer the application names) the instance could not be created for that reason, VK_ERROR_LAYER_NOT_PRESENT (-6).
# Any other vkCreateInstance failure is a failure.
NOT_ACTIVE = ("validation is not active", "vkCreateInstance failed (VkResult -6)")
# vk_test_device.h's line when the loader does not list the layer: only then can "not active" mean "not installed".
UNLISTED = f"the Vulkan loader does not list {LAYER}"


def run(command, environment, timeout):
    """(exit code, combined output) of one run."""
    done = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)
    return done.returncode, done.stdout + done.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", help="a GPU test built on vk_test_device.h, e.g. d3d12_com_test")
    parser.add_argument("--runner", action="append", default=[],
                        help="run Windows executables through this command (repeat for its arguments), e.g. wine")
    options = parser.parse_args()
    timeout = 180 if options.runner else 90
    command = [*options.runner, options.test]
    base = {k: v for k, v in os.environ.items() if k not in ("VK_INSTANCE_LAYERS", "VK_LOADER_LAYERS_DISABLE")}
    base["LEONIDA_VK_VALIDATION"] = "1"

    code, output = run(command, dict(base, VK_INSTANCE_LAYERS=LAYER), timeout)
    if code == SKIP:
        print("SKIP: no usable Vulkan device")
        return SKIP
    if ACTIVE not in output:
        print(output)
        # Not installed: the loader does not list the layer (or refuses it) and nothing reported the probe. A layer the
        # loader lists that then reports nothing, or an instance that fails for another reason, is a failure.
        if (UNLISTED in output and NOT_ACTIVE[0] in output) or NOT_ACTIVE[1] in output:
            print(f"SKIP: {LAYER} is not installed here, so the probe cannot be shown to report")
            return SKIP
        print("FAIL: with the layer available, the device did not report its validation probe")
        return 1
    if code != 0:
        print(output)
        print(f"FAIL: with the layer available, the test exited with {code}")
        return 1

    code, output = run(command, dict(base, VK_LOADER_LAYERS_DISABLE=LAYER), timeout)
    if code in (0, SKIP) or not any(text in output for text in NOT_ACTIVE):
        print(output)
        print(f"FAIL: with {LAYER} disabled the test exited with {code} and did not say that validation is not active")
        return 1
    print("validation probe: reported with the layer, and 'validation is not active' without it: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

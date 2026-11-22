"""Device-dependent translator modes, chosen the same way by every entry point that translates or judges shaders:
make_game.py, scripts/run_gpu_tests.py, runtime/test/shaders/build.py and check.py, tools/crosslane_fuzz.py and
tools/valu_fuzz.py.

The order is: the mode's flag on the entry point's own command line (where it has one) turns it on; otherwise its
environment variable decides (1/true/yes/on or 0/false/no/off, case and surrounding space ignored); otherwise the
device does, from what tools/vk_subgroup_info.py reports for the GPU the runtime or the tests use (the same choice
of device: LEONIDA_VK_DEVICE, LEONIDA_VULKAN_LOADER, VK_DRIVER_FILES). A device that can't be queried leaves the mode
off. The translator records a mode it ran with in each index.json entry, so check.py can tell a build made for
another device.

    subgroup_temps  --subgroup-temps  LEONIDA_SUBGROUP_TEMPS  driver id 14 (MoltenVK): its SPIRV-Cross moves a
                                                              subgroup operation into a short-circuiting && / ?:

Standard library only.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VK_DRIVER_ID_MOLTENVK = 14

# name -> (command-line flag, environment variable, rule on tools/vk_subgroup_info.py's answer)
MODES = {
    "subgroup_temps": ("--subgroup-temps", "LEONIDA_SUBGROUP_TEMPS",
                       lambda info: info.get("driver_id") == VK_DRIVER_ID_MOLTENVK),
}

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
_INFO = []


def env_choice(var, environ=None):
    """True or False when the variable is set to a yes or no word, None when it is unset, empty or anything else."""
    v = ((os.environ if environ is None else environ).get(var) or "").strip().lower()
    return True if v in _TRUE else False if v in _FALSE else None


def device_info():
    """tools/vk_subgroup_info.py's answer for the device (cached for the process); {} when it can't tell."""
    if not _INFO:
        answer = {}
        try:
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "vk_subgroup_info.py")],
                               capture_output=True, text=True, timeout=60)
            answer = json.loads(r.stdout.strip().splitlines()[-1])
            if not isinstance(answer, dict) or "error" in answer:
                answer = {}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        _INFO.append(answer)
    return _INFO[0]


def resolve(name, cli=False, environ=None, info=None):
    """Whether mode `name` is on: `cli` (its flag was given) wins, then its variable, then the device. `info` is
    vk_subgroup_info.py's answer, or a function returning it (called only when the device decides); by default
    device_info()."""
    _flag, var, rule = MODES[name]
    if cli:
        return True
    choice = env_choice(var, environ)
    if choice is not None:
        return choice
    if info is None:
        info = device_info
    answer = info() if callable(info) else info
    return bool(rule(answer or {}))


def flag(name):
    """The translator flag of mode `name`."""
    return MODES[name][0]

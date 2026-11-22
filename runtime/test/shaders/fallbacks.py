"""Which exact fallbacks the GPU tests translate and run with, on this machine's GPU.

build.py, check.py and scripts/run_gpu_tests.py all ask here, so a shader is built and judged under the same choice.
Each fallback has an environment variable: 1/true/yes/on forces it on, 0/false/no/off forces it off, and unset leaves
the choice to the device, as tools/vk_subgroup_info.py describes it (the device the runtime and the test programs
pick, LEONIDA_VK_DEVICE and LEONIDA_VULKAN_LOADER included):

- LEONIDA_W64_EMULATE: --wave64-emulate where the compute stage can't run a 64-wide subgroup (Apple, NVIDIA, Intel,
  llvmpipe). Wave64 compute and mesh shaders then run exactly on 32-wide (or narrower) subgroups.
- LEONIDA_MESH_COMPUTE: --mesh-compute where there is no VK_EXT_mesh_shader (MoltenVK, KosmicKrisp, NVIDIA Pascal):
  mesh shaders are dispatched as compute and drawn by a generated vertex shader. Also on llvmpipe under
  --wave64-emulate, whose emulated mesh shaders llvmpipe runs wrong (_mesh_compute_needed).
- LEONIDA_GS_COMPUTE: geometry shaders run as a compute capture pass (tools/spirv_gs_to_compute.py) where there is no
  geometryShader feature (MoltenVK, KosmicKrisp).
- LEONIDA_F16_INTERP_FALLBACK: --f16-interp-fallback where PerVertexKHR inputs can't be compiled (MoltenVK, and
  devices without fragment barycentrics).

When the device can't be asked (no Vulkan, no usable device), every unset fallback stays off, as before.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBGROUP_INFO = os.path.join(HERE, "..", "..", "..", "tools", "vk_subgroup_info.py")
ON = ("1", "true", "yes", "on")
OFF = ("0", "false", "no", "off")

_DEVICE = []


def device() -> dict:
    """tools/vk_subgroup_info.py's answer for the device the tests run on (asked once), or {} when there is none."""
    if not _DEVICE:
        answer = {}
        try:
            r = subprocess.run([sys.executable, "-I", SUBGROUP_INFO], capture_output=True, text=True, timeout=60)
            answer = json.loads(r.stdout.strip().splitlines()[-1])
            if "error" in answer:
                answer = {}
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
        _DEVICE.append(answer)
    return _DEVICE[0]


def compute_native64(info: dict):
    """Whether the compute stage runs 64 lanes per subgroup: a required size of 64 or a default of 64 (None if unknown).
    The compute half of vk_subgroup_info's native64, which also wants the mesh stage."""
    if "subgroup_size" not in info:
        return None
    required = (info.get("subgroup_size_control") and info.get("compute_in_required_stages")
                and info.get("min_subgroup_size", 0) <= 64 <= info.get("max_subgroup_size", 0))
    return bool(info.get("compute_in_supported_stages") and (required or info["subgroup_size"] == 64))


VK_DRIVER_ID_MESA_LLVMPIPE = 13


def _lacks(has, reason):
    """`reason` when the device lacks the feature, False when it has it, None when that is unknown."""
    return None if has is None else reason if not has else False


def _mesh_compute_needed():
    info = device()
    if info.get("mesh_shader") is False:
        return "no VK_EXT_mesh_shader"
    # llvmpipe (Mesa 26.3.1, LLVM 23) draws nothing for ms_tri and ms_clip translated with --wave64-emulate and run as
    # mesh shaders: a load of both output counts from Workgroup memory after the rounds' last barrier, then
    # OpSetMeshOutputsEXT with them, gives no primitives (one more OpControlBarrier, or either count a constant, and the
    # frame is exact). The same modules are exact on NVIDIA and AMD, and through --mesh-compute on llvmpipe, so the
    # tests take that path there (docs/testing.md, "Fallbacks the tests choose by themselves").
    if info.get("driver_id") == VK_DRIVER_ID_MESA_LLVMPIPE and enabled("w64_emulate"):
        return "llvmpipe runs mesh shaders made with --wave64-emulate wrong"
    return _lacks(info.get("mesh_shader"), "no VK_EXT_mesh_shader")


# name: (environment variable, translator flag or None, needed()): needed() is why the device needs the fallback
# (a reason, which is true), False where it doesn't, None where that is unknown.
FALLBACKS = {
    "w64_emulate": ("LEONIDA_W64_EMULATE", "--wave64-emulate",
                    lambda: _lacks(compute_native64(device()), "no 64-wide compute subgroup")),
    "mesh_compute": ("LEONIDA_MESH_COMPUTE", "--mesh-compute", _mesh_compute_needed),
    "gs_compute": ("LEONIDA_GS_COMPUTE", None, lambda: _lacks(device().get("geometry_shader"), "no geometryShader")),
    "f16_interp": ("LEONIDA_F16_INTERP_FALLBACK", "--f16-interp-fallback",
                   lambda: _lacks(device().get("per_vertex_inputs"), "no PerVertexKHR inputs")),
}


def forced(name):
    """True or False when the fallback's variable forces it, None when it is unset (or not a yes/no word)."""
    value = (os.environ.get(FALLBACKS[name][0]) or "").strip().lower()
    return True if value in ON else False if value in OFF else None


def enabled(name) -> bool:
    """Whether fallback `name` (a FALLBACKS key) is on: forced by its variable, else where the device lacks the feature."""
    choice = forced(name)
    return choice if choice is not None else bool(FALLBACKS[name][2]())


def describe(names=FALLBACKS) -> str:
    """One line naming the fallbacks that are on and why, for the runners to print ("" when none is)."""
    parts = []
    for name in names:
        if enabled(name):
            var, flag, needed = FALLBACKS[name]
            what = flag or name.replace("_", " ")
            parts.append(f"{what} ({var}=1)" if forced(name) else f"{what} ({needed()}; {var}=0 turns it off)")
    return "fallbacks: " + ", ".join(parts) if parts else ""

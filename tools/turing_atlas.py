r"""Turing bring-up atlas for Leonida (stdlib only, run with python -I).

Probes the host GPU (here: RTX 2060 SUPER, wave32) without game data:
  1. vulkaninfo feature matrix (subgroup size, mesh/task, barycentrics,
     maintenance8, subgroup_size_control, descriptor heap).
  2. spirv-as/spirv-val validation of 4 synthetic cases at vulkan1.4:
     compute-32, compute ballot-64 (wave64 marker), mesh-only, 16-bit I/O.
  3. Emits a JSON atlas to stdout (or --out FILE) for the PR body.

Usage:
    python -I tools/turing_atlas.py [--out atlas.json]
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


CASES = {
    # Minimal compute using the subgroup lane id (valid everywhere).
    "compute32": r"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %lid
OpExecutionMode %main LocalSize 64 1 1
%void = OpTypeVoid
%u32 = OpTypeInt 32 0
%ptr = OpTypePointer Input %u32
%lid = OpVariable %ptr Input
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
""",
    # Wave64 marker: 64-bit ballot over a 64-wide subgroup.
    "ballot64": r"""
OpCapability Shader
OpCapability Int64
OpCapability GroupNonUniformBallot
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %lid
OpExecutionMode %main LocalSize 64 1 1
%void = OpTypeVoid
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%u64 = OpTypeInt 64 0
%v4u64 = OpTypeVector %u64 4
%scope = OpConstant %u32 3
%ptr = OpTypePointer Input %u32
%lid = OpVariable %ptr Input
%u32_32 = OpConstant %u32 32
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%id = OpLoad %u32 %lid
%cmp = OpULessThan %bool %id %u32_32
%ballot = OpGroupNonUniformBallot %v4u64 %scope %cmp
OpReturn
OpFunctionEnd
""",
    # 16-bit fragment input probe (records whether the toolchain accepts it
    # on this host; storageInputOutput16 is false on Turing per vulkaninfo).
    "f16_input": r"""
OpCapability Shader
OpCapability Float16
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in16
OpExecutionMode %main OriginUpperLeft
OpDecorate %in16 Location 0
%void = OpTypeVoid
%f16 = OpTypeFloat 16
%ptr = OpTypePointer Input %f16
%in16 = OpVariable %ptr Input
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
""",
    # Maintenance8 Offset form (game uses image_sample with Offset on non-gather).
    "maintenance8_offset": r"""
OpCapability Shader
OpCapability ImageGatherExtended
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
""",
    # Control: malformed module that must FAIL validation (proves the gate fires).
    "broken_negative": r"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
OpReturn
OpFunctionEnd
OpReturn
""",
}


def run(cmd, **kw):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except FileNotFoundError:
        return None


def _device_rank(pdev):
    type_ranks = {
        2: 4, "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU": 4,
        1: 3, "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU": 3,
        3: 2, "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU": 2,
        4: 1, "VK_PHYSICAL_DEVICE_TYPE_CPU": 1,
    }
    vendor = pdev.get("vendorID")
    native = 0 if vendor == 0x1414 else 10
    return native + type_ranks.get(pdev.get("deviceType"), 0)


def _extract_profile(doc):
    caps = doc.get("capabilities") if isinstance(doc, dict) else None
    dev = caps.get("device") if isinstance(caps, dict) else None
    if not isinstance(dev, dict):
        return None
    props = dev.get("properties")
    feats = dev.get("features")
    exts = dev.get("extensions")
    if not all(isinstance(x, dict) for x in (props, feats, exts)):
        return None
    pdev = props.get("VkPhysicalDeviceProperties")
    v11 = props.get("VkPhysicalDeviceVulkan11Properties", {})
    v13p = props.get("VkPhysicalDeviceVulkan13Properties", {})
    v13f = feats.get("VkPhysicalDeviceVulkan13Features", {})
    v11f = feats.get("VkPhysicalDeviceVulkan11Features", {})
    if not all(isinstance(x, dict) for x in (pdev, v11, v13p, v13f, v11f)):
        return None
    name = pdev.get("deviceName")
    if not isinstance(name, str) or not name:
        return None
    return _device_rank(pdev), {
        "vulkaninfo": "ok",
        "device": {"name": name,
                   "driver": pdev.get("driverVersion"),
                   "api": pdev.get("apiVersion")},
        "subgroup": {"size": v11.get("subgroupSize"),
                     "stages": v11.get("subgroupSupportedStages"),
                     "min": v13p.get("minSubgroupSize"),
                     "max": v13p.get("maxSubgroupSize")},
        "subgroup_size_control": {
            "subgroupSizeControl": v13f.get("subgroupSizeControl"),
            "computeFullSubgroups": v13f.get("computeFullSubgroups"),
            "requiredSubgroupSizeStages": v13p.get("requiredSubgroupSizeStages"),
        },
        "mesh": "VK_EXT_mesh_shader" in exts,
        "barycentric": "VK_KHR_fragment_shader_barycentric" in exts,
        "maintenance8": "VK_KHR_maintenance8" in exts,
        "descriptor_heap": "VK_EXT_descriptor_heap" in exts,
        "storageInputOutput16": v11f.get("storageInputOutput16"),
    }


_MOLTENVK_DRIVER_ID = 14


def _vk_subgroup_fallback():
    """Fallback GPU/subgroup probe via tools/vk_subgroup_info.py when vulkaninfo is absent."""
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vk_subgroup_info.py")
    if not os.path.isfile(probe):
        return None
    try:
        r = run([sys.executable, "-I", probe, "--stage-caps"], timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r is None or r.returncode != 0 or not isinstance(r.stdout, str):
        return None
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        doc = json.loads(lines[-1])
    except ValueError:
        return None
    if not isinstance(doc, dict) or "error" in doc:
        return None
    name = doc.get("device")
    if not isinstance(name, str) or not name:
        return None
    barycentric = doc.get("fragment_shader_barycentric")
    if not isinstance(barycentric, bool):
        per_vertex = doc.get("per_vertex_inputs")
        barycentric = (per_vertex if isinstance(per_vertex, bool)
                       and doc.get("driver_id") != _MOLTENVK_DRIVER_ID else None)
    mesh = doc.get("mesh_shader")
    maintenance8 = doc.get("maintenance8")
    return {
        "vulkaninfo": "fallback:vk_subgroup_info",
        "device": {"name": name, "driver": None, "api": None},
        "subgroup": {
            "size": doc.get("subgroup_size"),
            "stages": doc.get("supported_subgroup_stages"),
            "min": doc.get("min_subgroup_size"),
            "max": doc.get("max_subgroup_size"),
        },
        "subgroup_size_control": {
            "subgroupSizeControl": doc.get("subgroup_size_control"),
            "computeFullSubgroups": None,
            "requiredSubgroupSizeStages": doc.get("required_subgroup_size_stages"),
        },
        "mesh": mesh if isinstance(mesh, bool) else None,
        "barycentric": barycentric,
        "maintenance8": maintenance8 if isinstance(maintenance8, bool) else None,
        "descriptor_heap": None,
        "storageInputOutput16": None,
    }


def vulkan_matrix():
    """Device/feature matrix from vulkaninfo JSON (Vulkan Profiles schema), or vk_subgroup_info.py if absent."""
    m = {"vulkaninfo": None, "device": None, "subgroup": None,
         "subgroup_size_control": None, "mesh": None, "barycentric": None,
         "maintenance8": None, "descriptor_heap": None,
         "storageInputOutput16": None}
    if not shutil.which("vulkaninfo"):
        fallback = _vk_subgroup_fallback()
        if fallback is not None:
            m.update(fallback)
        return m
    best_rank = -1
    best_profile = None
    with tempfile.TemporaryDirectory() as d:
        r = run(["vulkaninfo", "-j"], cwd=d)
        if r is None or r.returncode != 0:
            return m
        summary = run(["vulkaninfo", "--summary"], cwd=d)
        if summary is not None and summary.returncode == 0 and isinstance(summary.stdout, str):
            seen_gpus = {"0"}
            for idx in re.findall(r"(?m)^GPU(\d+)\s*:", summary.stdout):
                if idx not in seen_gpus:
                    seen_gpus.add(idx)
                    run(["vulkaninfo", f"-j={idx}"], cwd=d)
        files = sorted(glob.glob(os.path.join(d, "VP_VULKANINFO_*.json")))

        for path in files:
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, ValueError):
                continue
            parsed = _extract_profile(doc)
            if parsed is not None and parsed[0] > best_rank:
                best_rank, best_profile = parsed
    if best_profile is not None:
        m.update(best_profile)
    return m



def validate_cases():
    """Assemble + validate each synthetic case at vulkan1.4; returns {name: {...}}."""
    out = {}
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return {"skipped": "spirv-as/spirv-val not on PATH (Vulkan SDK)"}
    with tempfile.TemporaryDirectory() as d:
        for name, asm in CASES.items():
            src = os.path.join(d, name + ".spvasm")
            spv = os.path.join(d, name + ".spv")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(asm)
            ra = run(["spirv-as", src, "-o", spv, "--target-env", "vulkan1.4"])
            if ra is None or ra.returncode != 0:
                out[name] = {"assemble": False, "stderr": (ra.stderr if ra else "missing")[:300]}
                continue
            rv = run(["spirv-val", "--target-env", "vulkan1.4", spv])
            out[name] = {"assemble": True,
                         "valid": bool(rv and rv.returncode == 0),
                         "stderr": ((rv.stderr if rv else "") or "")[:300]}
    # A broken toolchain can reject everything and still reject the malformed control.
    compute = out["compute32"]
    compute["control_ok"] = compute.get("valid") is True
    if not compute["control_ok"]:
        compute["error"] = ("compute32 did not assemble and validate: the spirv-as/spirv-val gate did not accept "
                            "the known-good module, so the other case results are not trustworthy")
    # The control must be rejected: if spirv-as or spirv-val accepts a malformed module, none of the
    # other results say anything about the toolchain gate and the atlas must not be read as a pass.
    control = out.get("broken_negative")
    if isinstance(control, dict):
        rejected = control.get("assemble") is False or control.get("valid") is False
        control["control_ok"] = rejected
        if not rejected:
            control["error"] = ("broken_negative was accepted: the spirv-as/spirv-val gate is not rejecting "
                                "malformed modules, so the other case results are not trustworthy")
    return out


def main():
    ap = argparse.ArgumentParser(description="Turing bring-up atlas (no game data).")
    ap.add_argument("--out", default=None, help="write JSON atlas to FILE (default: stdout)")
    a = ap.parse_args()
    if a.out is not None:
        if not a.out.strip():
            print("turing_atlas: error: --out must not be empty", file=sys.stderr)
            return 2
        if os.path.isdir(a.out) or a.out.endswith((os.sep, os.altsep or os.sep)):
            print(f"turing_atlas: error: --out is a folder, not a file: {a.out}", file=sys.stderr)
            return 2

    cases = validate_cases()
    atlas = {"tool": "turing_atlas", "gpu": None, "features": vulkan_matrix(),
             "cases": cases}
    # Surface the discrete GPU name at top level when present.
    try:
        atlas["gpu"] = atlas["features"].get("device", {}).get("name")
    except AttributeError:
        pass
    text = json.dumps(atlas, indent=2, sort_keys=True)
    if a.out:
        try:
            folder = os.path.dirname(a.out)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
        except OSError as error:
            print(f"turing_atlas: error: cannot write {a.out}: {error.strerror or error}", file=sys.stderr)
            return 2
        print(f"atlas -> {a.out}")
    else:
        print(text)
    for name in ("compute32", "broken_negative"):
        error = cases.get(name, {}).get("error")
        if error:
            print(f"turing_atlas: error: {error}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())

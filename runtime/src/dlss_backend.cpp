// DLSS backend: the NVIDIA NGX path behind the FFX front the game's FSR4 dispatch drives. The game calls the
// FidelityFX five functions; when LEONIDA_DLSS=1 the shim's upscale dispatch reaches this backend instead, which
// translates the FFX dispatch description to NGX (the two are ML upscalers with isomorphic inputs: color, depth,
// motion vectors, jitter, exposure; see docs/d3d12_vulkan.md for the FFX side and the DLSS Programming Guide
// 310.6 sections 5.2-5.4 for this side) and runs it through the driver's NGX Core, which is loaded by name and
// never shipped with Leonida: nvngx.dll from the NVIDIA driver distribution, nvngx_dlss.dll next to the exe.
//
// Backend contract (src-level, matches what fsr_dispatch_test measures against the game's DLL):
//   init(device)                 once, before any create; also queries capability (SuperSampling.Available)
//   create(...)                  once per (in, out) size pair on quality/flag change; releases the old handle
//   eval_from_ffx(...)           per dispatch: fills the parameter bag from the FFX description and records
//   shutdown(device)             releases the feature, parameters and the NGX instance
// The caller owns state preservation: NGX rewrites the command list state (guide 5.4 IMPORTANT), so the shim
// wraps the game's list and restores the game's state after the records.
#include "dlss_backend.h"

#include <d3d12.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "../test/nvngx_api_min.h"

namespace leonida::dlss {

namespace {

struct Core {
    nvngx::PfnInitWithProjectId init_project_id = nullptr;
    nvngx::PfnAllocateParameters allocate = nullptr;
    nvngx::PfnGetCapabilityParameters capability = nullptr;
    nvngx::PfnCreateFeature create_feature = nullptr;
    nvngx::PfnEvaluateFeature evaluate = nullptr;
    nvngx::PfnReleaseFeature release_feature = nullptr;
    nvngx::PfnDestroyParameters destroy = nullptr;
    nvngx::PfnShutdown1 shutdown1 = nullptr;
    bool enabled = false;
};

Core& core() {
    static Core c;
    return c;
}

bool g_initiated = false;

void* find_export(HMODULE module, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(module, name));
}

// Loads the NGX Core and binds exports. Env LEONIDA_NGX_DLL names a full nvngx.dll path; without it, the
// loader search finds nvngx.dll the way NGX itself does (application folder, then the driver distribution).
bool load_core() {
    if (core().enabled) return true;
    HMODULE module = nullptr;
    if (const char* path = std::getenv("LEONIDA_NGX_DLL"); path && *path) {
        module = LoadLibraryA(path);
    } else {
        module = GetModuleHandleA("nvngx.dll");  // already in the process (the game loads it), else below
        if (!module) module = LoadLibraryA("nvngx.dll");
    }
    if (!module) {
        std::printf("dlss: no NGX Core (set LEONIDA_NGX_DLL=<path to nvngx.dll>, or install the NVIDIA driver)\n");
        return false;
    }
    Core& c = core();
    c.init_project_id = reinterpret_cast<nvngx::PfnInitWithProjectId>(find_export(module, "NVSDK_NGX_D3D12_Init_ProjectID"));
    c.allocate = reinterpret_cast<nvngx::PfnAllocateParameters>(find_export(module, "NVSDK_NGX_D3D12_AllocateParameters"));
    c.capability = reinterpret_cast<nvngx::PfnGetCapabilityParameters>(find_export(module, "NVSDK_NGX_D3D12_GetCapabilityParameters"));
    c.create_feature = reinterpret_cast<nvngx::PfnCreateFeature>(find_export(module, "NVSDK_NGX_D3D12_CreateFeature"));
    c.evaluate = reinterpret_cast<nvngx::PfnEvaluateFeature>(find_export(module, "NVSDK_NGX_D3D12_EvaluateFeature"));
    c.release_feature = reinterpret_cast<nvngx::PfnReleaseFeature>(find_export(module, "NVSDK_NGX_D3D12_ReleaseFeature"));
    c.destroy = reinterpret_cast<nvngx::PfnDestroyParameters>(find_export(module, "NVSDK_NGX_D3D12_DestroyParameters"));
    c.shutdown1 = reinterpret_cast<nvngx::PfnShutdown1>(find_export(module, "NVSDK_NGX_D3D12_Shutdown1"));
    if (!(c.init_project_id && c.allocate && c.capability && c.create_feature && c.evaluate && c.release_feature &&
          c.destroy && c.shutdown1)) {
        std::printf("dlss: NGX Core exports incomplete\n");
        return false;
    }
    c.enabled = true;
    return true;
}

}  // namespace

bool init(ID3D12Device* device, const wchar_t* data_path) {
    if (!load_core()) return false;
    if (g_initiated) return true;
    // The NGX Core dereferences the info record unconditionally: a null here ends the process (observed),
    // a zeroed one is what the SDK's own init path passes.
    nvngx::FeatureCommonInfo info{};
    const nvngx::Result rc = core().init_project_id(
        "7c2f18e5-3bda-41e6-9f0d-2a4b8c6e5d17", nvngx::ENGINE_CUSTOM, "leonida 0.1", data_path, device, &info,
        nvngx::SDK_VERSION);
    if (rc != nvngx::RESULT_SUCCESS) {
        std::printf("dlss: init_project_id -> 0x%x\n", static_cast<unsigned>(rc));
        return false;
    }
    // capability: SuperSampling.Available + driver gate, as guide 5.2.6 prescribes
    nvngx::Parameter* caps = nullptr;
    if (core().capability(&caps) == nvngx::RESULT_SUCCESS && caps) {
        int available = 0, needs_newer = 0;
        caps->Get(nvngx::SUPER_AVAILABLE, &available);
        caps->Get(nvngx::SUPER_NEEDS_UPDATED_DRIVER, &needs_newer);
        core().destroy(caps);
        g_initiated = available != 0 && needs_newer == 0;
        return g_initiated;
    }
    return false;
}

bool initiated() {
    return g_initiated;
}

void shutdown(ID3D12Device* device) {
    if (core().enabled) core().shutdown1(device);
}

Feature create(ID3D12GraphicsCommandList* list, const Description& desc, ID3D12Device* device,
               unsigned int creation_node_mask) {
    (void)device;
    if (!core().enabled) return Feature{};
    nvngx::Parameter* params = nullptr;
    if (core().allocate(&params) != nvngx::RESULT_SUCCESS || !params) return Feature{};
    // Quality: the FFX mode's ratio names the DLSS slot (the game's own ffxQuery feeds desc.upscale_ratio).
    // The two collisions in the shipped enum resolve conservatively: 3.0x keeps MaxPerf, 1.7237x keeps Balanced.
    nvngx::PerfQualityValue quality = nvngx::PERF_VALUE_MAX_QUALITY;
    if (desc.upscale_ratio >= 3.0f) {
        quality = nvngx::PERF_VALUE_MAX_PERF;
    } else if (desc.upscale_ratio > 1.71f) {
        quality = nvngx::PERF_VALUE_BALANCED;   // 1.7237x
    } else if (desc.upscale_ratio > 1.6f) {
        quality = nvngx::PERF_VALUE_ULTRA_QUALITY;  // 1.7x
    } else if (desc.upscale_ratio > 1.05f) {
        quality = nvngx::PERF_VALUE_MAX_QUALITY;    // 1.5x
    } else {
        quality = nvngx::PERF_VALUE_DLAA;
    }
    int flags = 0;
    if (desc.hdr) flags |= nvngx::DLSS_FLAG_IS_HDR;
    if (desc.mv_low_res) flags |= nvngx::DLSS_FLAG_MV_LOW_RES;
    if (desc.mv_jittered) flags |= nvngx::DLSS_FLAG_MV_JITTERED;
    if (desc.depth_inverted) flags |= nvngx::DLSS_FLAG_DEPTH_INVERTED;
    if (desc.auto_exposure) flags |= nvngx::DLSS_FLAG_AUTO_EXPOSURE;
    params->Set(nvngx::CREATION_NODE_MASK, creation_node_mask);
    params->Set(nvngx::VISIBILITY_NODE_MASK, creation_node_mask);
    params->Set(nvngx::WIDTH, desc.render_width);
    params->Set(nvngx::HEIGHT, desc.render_height);
    params->Set(nvngx::OUT_WIDTH, desc.upscale_width);
    params->Set(nvngx::OUT_HEIGHT, desc.upscale_height);
    params->Set(nvngx::PERF_QUALITY_VALUE, static_cast<int>(quality));
    params->Set(nvngx::DLSS_CREATE_FLAGS, flags);
    params->Set(nvngx::DLSS_ENABLE_OUTPUT_SUBRECTS, 0);
    Feature f;
    f.params = params;
    nvngx::Handle* handle = nullptr;
    f.rc_create = core().create_feature(list, nvngx::FEATURE_SUPERSAMPLING, params, &handle);
    f.ngx = handle;
    f.quality = quality;
    f.flags = flags;
    return f;
}

bool eval_from_ffx(Feature& f, ID3D12GraphicsCommandList* list, const FfxDescription& fx) {
    if (!f.ngx || !f.params) return false;
    f.params->Set(nvngx::COLOR, fx.color);
    f.params->Set(nvngx::OUTPUT, fx.output);
    if (fx.depth) f.params->Set(nvngx::DEPTH, fx.depth);
    if (fx.motion_vectors) f.params->Set(nvngx::MOTION_VECTORS, fx.motion_vectors);
    f.params->Set(nvngx::JITTER_OFFSET_X, fx.jitter_x);
    f.params->Set(nvngx::JITTER_OFFSET_Y, fx.jitter_y);
    f.params->Set(nvngx::MV_SCALE_X, fx.mv_scale_x == 0.0f ? 1.0f : fx.mv_scale_x);
    f.params->Set(nvngx::MV_SCALE_Y, fx.mv_scale_y == 0.0f ? 1.0f : fx.mv_scale_y);
    f.params->Set(nvngx::SHARPNESS, fx.sharpness);
    f.params->Set(nvngx::RESET, fx.reset ? 1 : 0);
    if (fx.exposure) f.params->Set(nvngx::EXPOSURE_TEXTURE, fx.exposure);
    if (fx.transparency_mask) f.params->Set(nvngx::TRANSPARENCY_MASK, fx.transparency_mask);
    return core().evaluate(list, f.ngx, f.params, nullptr) == nvngx::RESULT_SUCCESS;
}

void destroy(Feature& f) {
    if (f.ngx && core().enabled) core().release_feature(f.ngx);
    f.ngx = nullptr;
    if (f.params && core().enabled) core().destroy(f.params);
    f.params = nullptr;
}

}  // namespace leonida::dlss

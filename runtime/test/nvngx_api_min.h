// The subset of the NVIDIA NGX interface that dlss_dispatch_test.cpp and src/dlss_backend.cpp call, written out
// from the ABI as shipped:
//   - the export names, from a PE export-table dump of the driver's NGX Core (nvngx.dll, 310.x line, D3D12 section
//     read here: NVSDK_NGX_D3D12_{Init_ProjectID, Init_Ext, Init, AllocateParameters, GetCapabilityParameters,
//     GetParameters, CreateFeature, EvaluateFeature, ReleaseFeature, DestroyParameters, GetScratchBufferSize,
//     Shutdown, Shutdown1});
//   - the 17-virtual-slot parameter object, as the SDK ships it (single-inheritance MSVC vtable; the order below
//     exercised end to end by a probe that runs Init/Capability/Create/Evaluate on real hardware);
//   - the semantics of the create/evaluate parameters, documented in the public DLSS Programming Guide (310.6,
//     sections 5.2 to 5.4, key strings and flag bits included).
// Nothing from the NVIDIA DLSS SDK is copied here: those files carry the NVIDIA RTX SDK license, which forbids
// making them subject to an open-source license; nvngx_dlss.dll itself is a user-side component (the driver's
// NGX Core loads it, and the game-side copy lives next to the executable per the guide's distribution section).
#pragma once

#include <cstdint>

typedef struct ID3D11Resource ID3D11Resource;
typedef struct ID3D12Resource ID3D12Resource;

namespace nvngx {

using Result = int;
using Version = int;
using EngineType = int;
using Feature = int;  // NVSDK_NGX_Feature_SuperSampling = 1

constexpr Result RESULT_SUCCESS = 0x1;
constexpr Result RESULT_FAIL = static_cast<int>(0xBAD00000u);
constexpr Result RESULT_FAIL_FEATURE_NOT_SUPPORTED = RESULT_FAIL | 1;
constexpr Result RESULT_FAIL_PLATFORM_ERROR = RESULT_FAIL | 2;
constexpr Result RESULT_FAIL_FEATURE_ALREADY_EXISTS = RESULT_FAIL | 3;
constexpr Result RESULT_FAIL_FEATURE_NOT_FOUND = RESULT_FAIL | 4;
constexpr Result RESULT_FAIL_INVALID_PARAMETER = RESULT_FAIL | 5;
constexpr Result RESULT_FAIL_NOT_INITIALIZED = RESULT_FAIL | 7;
constexpr Result RESULT_FAIL_OUT_OF_DATE = RESULT_FAIL | 8;  // an older driver: the deprecated GetParameters fallback

constexpr Version SDK_VERSION = 0x15;  // NVSDK_NGX_Version_API of the 310.6 release (NVNGX version 1.5.0)
constexpr EngineType ENGINE_CUSTOM = 0;
constexpr Feature FEATURE_SUPERSAMPLING = 1;

// DLSS create flags, as the guide documents them (section 5.3) and the runtime checks them.
constexpr int DLSS_FLAG_IS_HDR = 1 << 0;
constexpr int DLSS_FLAG_MV_LOW_RES = 1 << 1;
constexpr int DLSS_FLAG_MV_JITTERED = 1 << 2;
constexpr int DLSS_FLAG_DEPTH_INVERTED = 1 << 3;
constexpr int DLSS_FLAG_AUTO_EXPOSURE = 1 << 6;

// PerfQualityValue, the shipped enum order (guide section 5.2.8.1): MaxPerf=MaxPerformance, MaxQuality=Quality.
enum PerfQualityValue {
    PERF_VALUE_MAX_PERF = 0,         // 3.0x
    PERF_VALUE_BALANCED = 1,         // 1.7237x
    PERF_VALUE_MAX_QUALITY = 2,      // 1.5x
    PERF_VALUE_ULTRA_PERFORMANCE = 3,  // 3.0x, extended slot
    PERF_VALUE_ULTRA_QUALITY = 4,    // 1.7x, extended slot
    PERF_VALUE_DLAA = 5,             // 1.0x
};

// The opaque object Allocate/Get(Capability)Parameters hands out: 17 virtual slots, single-inheritance vtable,
// declared in this exact order (verified at runtime; changing the order changes the ABI).
struct Parameter {
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;
    virtual Result Get(const char* name, unsigned long long* out) const = 0;
    virtual Result Get(const char* name, float* out) const = 0;
    virtual Result Get(const char* name, double* out) const = 0;
    virtual Result Get(const char* name, unsigned int* out) const = 0;
    virtual Result Get(const char* name, int* out) const = 0;
    virtual Result Get(const char* name, ID3D11Resource** out) const = 0;
    virtual Result Get(const char* name, ID3D12Resource** out) const = 0;
    virtual Result Get(const char* name, void** out) const = 0;
    virtual void Reset() = 0;
};

// Where the NGX Core looks for feature dlls beyond the application folder (guide section 5.1).
struct PathListInfo {
    const wchar_t* const* path;
    unsigned int length;
};
struct FeatureCommonInfo {
    PathListInfo pathList;
    void* internal;  // engine-managed
    void* logging;   // NGX_LoggingInfo, may be null
};

typedef struct Handle Handle;  // opaque, from CreateFeature, consumed by Evaluate/Release

// Function signatures exactly as the NGX Core exports them (the short Init name; the SDK's longer
// NVSDK_NGX_D3D12_Init_with_ProjectID is a wrapper around it).
using PfnInitWithProjectId = Result (*)(const char* project_id, EngineType engine, const char* engine_version,
                                        const wchar_t* data_path, ID3D12Device* device,
                                        const FeatureCommonInfo* info, Version version);
using PfnAllocateParameters = Result (*)(Parameter**);
using PfnGetCapabilityParameters = Result (*)(Parameter**);
using PfnCreateFeature = Result (*)(ID3D12GraphicsCommandList*, Feature, Parameter*, Handle**);
using PfnEvaluateFeature = Result (*)(ID3D12GraphicsCommandList*, const Handle*, Parameter*, void* callback);
using PfnReleaseFeature = Result (*)(Handle*);
using PfnDestroyParameters = Result (*)(Parameter*);
using PfnShutdown1 = Result (*)(void* device);

// The parameter keys this integration reads or writes (string values documented in the guide 5.2-5.5 and
// exercised by the runtime).
constexpr const char* CREATION_NODE_MASK = "CreationNodeMask";
constexpr const char* VISIBILITY_NODE_MASK = "VisibilityNodeMask";
constexpr const char* WIDTH = "Width";
constexpr const char* HEIGHT = "Height";
constexpr const char* OUT_WIDTH = "OutWidth";
constexpr const char* OUT_HEIGHT = "OutHeight";
constexpr const char* PERF_QUALITY_VALUE = "PerfQualityValue";
constexpr const char* DLSS_CREATE_FLAGS = "DLSS.Feature.Create.Flags";
constexpr const char* DLSS_ENABLE_OUTPUT_SUBRECTS = "DLSS.Enable.Output.Subrects";
constexpr const char* COLOR = "Color";
constexpr const char* OUTPUT = "Output";
constexpr const char* DEPTH = "Depth";
constexpr const char* MOTION_VECTORS = "MotionVectors";
constexpr const char* JITTER_OFFSET_X = "Jitter.Offset.X";
constexpr const char* JITTER_OFFSET_Y = "Jitter.Offset.Y";
constexpr const char* MV_SCALE_X = "MV.Scale.X";
constexpr const char* MV_SCALE_Y = "MV.Scale.Y";
constexpr const char* RESET = "Reset";
constexpr const char* SHARPNESS = "Sharpness";
constexpr const char* EXPOSURE_TEXTURE = "ExposureTexture";
constexpr const char* TRANSPARENCY_MASK = "TransparencyMask";
constexpr const char* SUPER_AVAILABLE = "SuperSampling.Available";
constexpr const char* SUPER_NEEDS_UPDATED_DRIVER = "SuperSampling.NeedsUpdatedDriver";
constexpr const char* SUPER_MIN_DRIVER_VERSION_MAJOR = "SuperSampling.MinDriverVersionMajor";
constexpr const char* SUPER_MIN_DRIVER_VERSION_MINOR = "SuperSampling.MinDriverVersionMinor";
constexpr const char* SUPER_FEATURE_INIT_RESULT = "SuperSampling.FeatureInitResult";

}  // namespace nvngx

// Pipelines one session creates are found in the pipeline cache of the next. A D3D12 device is made on a Vulkan device
// several times in one process, each with LEONIDA_PIPELINE_CACHE naming the same scratch file, and creates the same
// compute PSOs; VK_EXT_pipeline_creation_feedback (rt::d3d12::pipeline_cache_stats) says how many came from the cache:
//   1. no file: every pipeline is compiled, and the device saves its cache when it is destroyed;
//   2. the saved file: every pipeline is found in the cache;
//   2b. a session that made nothing new leaves the file alone; over LEONIDA_PIPELINE_CACHE_MAX_MB nothing is saved;
//       LEONIDA_PIPELINE_CACHE=off saves and loads nothing;
//   3. a file that is not a pipeline cache, then a path that cannot be written: the device still creates every
//      pipeline, and the first is replaced by a real cache.
// The shaders are hand-written SPIR-V no-ops that differ in their workgroup size, so each PSO is a pipeline of its own.
// No game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace fs = std::filesystem;

constexpr unsigned pipeline_count = 24;

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

using SerializeFn = HRESULT(WINAPI*)(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC*, ID3DBlob**, ID3DBlob**);

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// zlib with one stored block (RFC 1950/1951), enough to wrap a test program.
std::vector<std::uint8_t> zlib_stored(const std::vector<std::uint8_t>& data) {
    std::vector<std::uint8_t> out = {0x78, 0x01, 0x01};
    const auto length = std::uint16_t(data.size());
    out.insert(out.end(), {std::uint8_t(length), std::uint8_t(length >> 8), std::uint8_t(~length),
                           std::uint8_t(~length >> 8)});
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (const std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    out.insert(out.end(), {std::uint8_t(adler >> 24), std::uint8_t(adler >> 16), std::uint8_t(adler >> 8),
                           std::uint8_t(adler)});
    return out;
}

// DXBC { DXIL { stub, XDAT { 0x0D: XBCH + zlib(fssm) } } }, the layout tools/fssm_unpack.py reads.
std::vector<std::uint8_t> shader_container(const std::vector<std::uint8_t>& fssm) {
    const auto stream = zlib_stored(fssm);
    std::vector<std::uint8_t> xbch = {'X', 'B', 'C', 'H'};
    for (const std::uint32_t v : {1u, std::uint32_t(stream.size()), 0u, std::uint32_t(fssm.size()), 0u}) put32(xbch, v);
    xbch.insert(xbch.end(), stream.begin(), stream.end());
    std::vector<std::uint8_t> dxil(16, 0);
    dxil.insert(dxil.end(), {'X', 'D', 'A', 'T'});
    put32(dxil, 1);
    put32(dxil, 12);
    put32(dxil, 0x0d);
    put32(dxil, 16 + 12 + 12);
    put32(dxil, std::uint32_t(xbch.size()));
    dxil.insert(dxil.end(), xbch.begin(), xbch.end());
    std::vector<std::uint8_t> out = {'D', 'X', 'B', 'C'};
    out.resize(28, 0);
    put32(out, 1);
    put32(out, 36);
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(out, std::uint32_t(dxil.size()));
    out.insert(out.end(), dxil.begin(), dxil.end());
    return out;
}

std::vector<std::uint8_t> serialize(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto fn = d3d12 ? reinterpret_cast<SerializeFn>(
                                reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeVersionedRootSignature")))
                          : nullptr;
    std::vector<std::uint8_t> out;
    ID3DBlob* blob = nullptr;
    if (fn && SUCCEEDED(fn(&desc, &blob, nullptr)) && blob) {
        const auto* p = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        out.assign(p, p + blob->GetBufferSize());
        blob->Release();
    }
    return out;
}

// void main() {} with a workgroup of x * y invocations: a compute shader of no use, valid, and one pipeline per size.
std::vector<std::uint32_t> noop_compute_spirv(std::uint32_t x, std::uint32_t y) {
    return {0x07230203, 0x00010000, 0, 5, 0,                                // header, ids 1..4
            0x00020011, 1,                                                  // OpCapability Shader
            0x0003000e, 0, 1,                                               // OpMemoryModel Logical GLSL450
            0x0005000f, 5, 1, 0x6e69616d, 0,                                // OpEntryPoint GLCompute %1 "main"
            0x00060010, 1, 17, x, y, 1,                                     // OpExecutionMode %1 LocalSize x y 1
            0x00020013, 2,                                                  // %2 = OpTypeVoid
            0x00030021, 3, 2,                                               // %3 = OpTypeFunction %2
            0x00050036, 2, 1, 0, 3,                                         // %1 = OpFunction %2 None %3
            0x000200f8, 4,                                                  // %4 = OpLabel
            0x000100fd,                                                     // OpReturn
            0x00010038};                                                    // OpFunctionEnd
}

std::uint32_t workgroup_x(unsigned i) { return 32u * (1 + i % 4); }
std::uint32_t workgroup_y(unsigned i) { return 1 + i / 4; }

// The program the PSO carries: its SHA-256 is what the translator index finds the module by.
std::vector<std::uint8_t> fssm_of(unsigned i) {
    std::vector<std::uint8_t> fssm = {'F', 'S', 'S', 'M'};
    put32(fssm, 5000 + i);
    for (std::uint32_t k = 0; k < 14; ++k) put32(fssm, 0xbf810000u + k);
    return fssm;
}

bool write_library(const fs::path& dir) {
    std::string index = "{";
    for (unsigned i = 0; i < pipeline_count; ++i) {
        const std::string name = "psocache_" + std::to_string(i);
        const auto words = noop_compute_spirv(workgroup_x(i), workgroup_y(i));
        std::ofstream spv(dir / (name + ".spv"), std::ios::binary);
        spv.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        if (!spv) return false;
        index += std::string(i ? ",\n " : "") + "\"" + name + "\": {\"stage\": \"cs\", \"fssm_sha256\": \"" +
                 leonida::sha256_hex(fssm_of(i)) + "\", \"bindings\": [], \"push_constant_bytes\": 144, " +
                 "\"local_size\": [" + std::to_string(workgroup_x(i)) + ", " + std::to_string(workgroup_y(i)) +
                 ", 1], \"uses_code_base\": true, \"wave64\": false}";
    }
    std::ofstream out(dir / "index.json", std::ios::binary);
    out << index << "}";
    return bool(out);
}

struct Session {
    bool ok = false;
    rt::d3d12::PipelineCacheStats stats;
    double wall_ms = 0;  // vkCreateComputePipelines and the rest of CreateComputePipelineState, as the caller sees
};

// One device, `pipeline_count` PSOs, then everything released: the device saves its cache while it is destroyed.
Session run_session(vktest::Device& d, const fs::path& library, const std::vector<std::uint8_t>& root_blob) {
    Session out;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.pipeline_creation_feedback = d.core_1_3();
    info.shader_dir = library.string();
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return out;
    Ref<ID3D12RootSignature> root;
    if (FAILED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p)))) return out;
    std::vector<Ref<ID3D12PipelineState>> psos(pipeline_count);
    LARGE_INTEGER frequency, begin, end;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    for (unsigned i = 0; i < pipeline_count; ++i) {
        const auto container = shader_container(fssm_of(i));
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root.p;
        desc.CS = {container.data(), container.size()};
        if (FAILED(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&psos[i].p)))) return out;
    }
    QueryPerformanceCounter(&end);
    out.wall_ms = 1000.0 * double(end.QuadPart - begin.QuadPart) / double(frequency.QuadPart);
    out.stats = rt::d3d12::pipeline_cache_stats(device.p);
    out.ok = true;
    return out;  // the PSOs, the root signature and the device are released here, in that order
}

void print(const char* label, const Session& s) {
    std::printf("%s: %u pipelines, %u from the pipeline cache, %.2f ms creating them (%.2f ms through the API), "
                "%llu bytes loaded\n",
                label, s.stats.pipelines, s.stats.cache_hits, double(s.stats.create_nanoseconds) / 1e6, s.wall_ms,
                static_cast<unsigned long long>(s.stats.loaded_bytes));
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);
    if (!d.core_1_3()) {
        std::fprintf(stderr, "SKIP: no VK_EXT_pipeline_creation_feedback on a Vulkan 1.2 device\n");
        return 77;
    }

    const testfiles::ScratchDir scratch("leonida_pso_cache_test");
    const auto& dir = scratch.path();
    EXPECT(write_library(dir));
    const fs::path session_file = dir / "session_cache.bin";
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE", session_file.c_str()));

    // The console compute layout of d3d12_compute_test: two root constants, a root UAV and a one-CBV table.
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1, 0, 0};
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_blob = serialize(root_desc);
    EXPECT(!root_blob.empty());

    const Session first = run_session(d, dir, root_blob);
    EXPECT(first.ok);
    print("session 1, no cache file", first);
    if (first.stats.pipelines != pipeline_count) {
        std::fprintf(stderr, "SKIP: the driver gave no pipeline creation feedback\n");
        return 77;
    }
    EXPECT(fs::is_regular_file(session_file));

    const auto saved_at = fs::last_write_time(session_file);
    const auto saved_size = fs::file_size(session_file);
    const Session second = run_session(d, dir, root_blob);
    EXPECT(second.ok);
    print("session 2, cache saved by session 1", second);
    EXPECT(second.stats.loaded_bytes > 0);
    EXPECT(second.stats.pipelines == pipeline_count);
    EXPECT(second.stats.cache_hits == pipeline_count);
    // Every pipeline came from the cache, so there was nothing to save: the file is not rewritten.
    EXPECT(fs::last_write_time(session_file) == saved_at && fs::file_size(session_file) == saved_size);

    // A file that is not a pipeline cache is ignored and then replaced by a real one.
    {
        std::ofstream junk(session_file, std::ios::binary | std::ios::trunc);
        junk << std::string(100, 'x');
    }
    const Session third = run_session(d, dir, root_blob);
    EXPECT(third.ok);
    print("session 3, junk in the cache file", third);
    EXPECT(third.stats.loaded_bytes == 0);
    EXPECT(third.stats.pipelines == pipeline_count && third.stats.cache_hits == 0);
    EXPECT(fs::file_size(session_file) > 100);
    const Session fourth = run_session(d, dir, root_blob);
    EXPECT(fourth.ok && fourth.stats.cache_hits == pipeline_count);

    // A cache over LEONIDA_PIPELINE_CACHE_MAX_MB is not saved, and "off" saves and loads nothing.
    const fs::path limited = dir / "limited.bin";
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE", limited.c_str()));
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_MAX_MB", L"0"));
    const Session sixth = run_session(d, dir, root_blob);
    EXPECT(sixth.ok && sixth.stats.pipelines == pipeline_count);
    EXPECT(!fs::exists(limited));
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE_MAX_MB", nullptr));
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE", L"off"));
    const Session seventh = run_session(d, dir, root_blob);
    EXPECT(seventh.ok && seventh.stats.pipelines == pipeline_count && seventh.stats.loaded_bytes == 0);
    EXPECT(!fs::exists(dir / "off"));

    // A path nothing can be saved to costs the session nothing but the save.
    const fs::path unwritable = dir / "a_directory";
    EXPECT(fs::create_directory(unwritable));
    EXPECT(SetEnvironmentVariableW(L"LEONIDA_PIPELINE_CACHE", unwritable.c_str()));
    const Session fifth = run_session(d, dir, root_blob);
    EXPECT(fifth.ok && fifth.stats.pipelines == pipeline_count);
    EXPECT(fs::is_directory(unwritable));
    std::size_t entries = 0;
    for (const auto& entry : fs::directory_iterator(dir)) entries += entry.path().filename().string().rfind("a_directory", 0) == 0;
    EXPECT(entries == 1);  // no half-written staging file left beside it
    std::printf("pipeline cache persistence: ok\n");
    return 0;
}

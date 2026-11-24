// Wave64 emulation (src/wave64_emulate.hlsli) against a CPU reference. The fixture runs at the adapter's
// native width and, where SM 6.6 and the lane range allow it, at a forced width of 64, the setting in
// which WARP 1.0.22's own WaveReadLaneAt misroutes (wave_shuffle, issue #145). The skewed build is the
// negative control and must be reported on every affine read.
#include "bvh_fixture.h"
#include "Wave64Emulate.h"
#include "Wave64Emulate64.h"
#include "Wave64EmulateSkew.h"
#include <utility>

using namespace bvh_fixture;

namespace {
constexpr Word threads = 64, fields = 8, checked = 7;

Word value(Word t) { return t * 2654435761u + 12345u; }

// Returns {wrong values, wrong affine reads}.
std::pair<unsigned, unsigned> run_variant(ID3D12Device* device, const char* label, D3D12_SHADER_BYTECODE code)
{
    constexpr UINT bytes = threads * fields * sizeof(Word);
    Kernel shader(device, code);
    auto out = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands cmd(device, shader.pipeline.Get());
    auto* list = cmd.list.Get();
    transition(list, out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(shader.root.Get());
    list->SetComputeRootUnorderedAccessView(0, out->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(), 0, out.Get(), 0, bytes);
    cmd.submit_and_wait(device);
    const auto r = read(back.Get(), threads * fields);

    Word sum = 0, ballot[2] = {};
    for (Word t = 0; t < threads; ++t) {
        sum += value(t);
        if (t % 3 == 0) ballot[t / 32] |= 1u << (t % 32);
    }
    unsigned wrong = 0, affine = 0;
    for (Word t = 0; t < threads; ++t) {
        const Word* rec = r.data() + t * fields;
        const Word expected[checked] = {value((t * 7 + 3) % 64), value((t + 1) % 64), value(63 - t), value(0),
                                        sum, ballot[0], ballot[1]};
        for (Word i = 0; i < checked; ++i)
            if (rec[i] != expected[i]) {
                ++wrong;
                if (i == 0) ++affine;
            }
    }
    std::printf("wave64_emulate %s wrong=%u of %u affine_wrong=%u\n", label, wrong, threads * checked, affine);
    return {wrong, affine};
}
}

void check_wave64_emulate(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 caps{};
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS1, &caps, sizeof(caps)), "Query wave range");
    check(check_feature(device, D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)), "Query wave shader model");
    // The native build has no wave intrinsics, so it cannot observe its width; report the adapter's range.
    std::printf("wave64_emulate native_lane_range=%u..%u\n", caps.WaveLaneCountMin, caps.WaveLaneCountMax);

    unsigned errors = run_variant(device, "native", {Wave64Emulate, sizeof(Wave64Emulate)}).first;
    const auto skew = run_variant(device, "skew", {Wave64EmulateSkew, sizeof(Wave64EmulateSkew)});
    if (skew.first != threads || skew.second != threads)
        throw std::runtime_error("Negative control: the skewed affine read was not reported on every thread");

    if (caps.WaveOps && model.HighestShaderModel >= D3D_SHADER_MODEL_6_6 && caps.WaveLaneCountMin <= 64 &&
        caps.WaveLaneCountMax >= 64)
        errors += run_variant(device, "forced64", {Wave64Emulate64, sizeof(Wave64Emulate64)}).first;
    else
        std::printf("wave64_emulate forced width=64 skipped\n");
    if (errors) throw std::runtime_error("Wave64 emulation differs from the CPU reference");
}

#include "bvh_fixture.h"
#include "WaveShuffle.h"
#include "WaveShuffle16.h"
#include "WaveShuffle32.h"
#include "WaveShuffle64.h"
#include <map>
#include <utility>

using namespace bvh_fixture;

namespace {
constexpr Word threads = 64, fields = 8;

Word affine(Word lane, Word width) { return (lane * 7 + 3) % width; }
Word rotation(Word lane, Word width) { return (lane + 1) % width; }

// The lane a read returns when an adapter runs a wave as blocks of `block` lanes and picks the source
// block with the request of the wrong lane: the lane of the reader's block that sits at the requested
// position, instead of the reader. It equals the request whenever all lanes of a block read from one
// source block, so uniform reads, the mirror and a single-block wave are unaffected. WARP 1.0.22
// matches this at forced widths 16, 32 and 64 with blocks of its native 8 lanes (issue #145). For
// the affine shuffle it coincides with the inverse permutation.
Word block_misrouted(Word (*request)(Word, Word), Word lane, Word width, Word block)
{
    const Word position = request(lane, width) % block;
    return request(lane - lane % block + position, width) / block * block + position;
}

// Runs one shader variant and names the lane every shuffle read came from. `block` is the adapter's
// native width, or 0 for the native variant itself. Returns the number of wrong reads and sets
// `width` to the width the variant ran at.
unsigned run_variant(ID3D12Device* device, const char* label, D3D12_SHADER_BYTECODE code, Word forced,
                     Word block, Word& width)
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

    width = r[1];
    if (!width || width > threads || threads % width) throw std::runtime_error("Unexpected wave width");
    if (forced && width != forced) throw std::runtime_error("Forced wave width was not applied");
    if (!block || block > width || width % block) block = width;  // one block: the model is the request
    std::map<std::pair<Word, Word>, Word> thread_at;  // (wave, lane) -> group thread
    for (Word t = 0; t < threads; ++t) {
        const Word* rec = r.data() + t * fields;
        if (rec[1] != width || rec[6] != width) throw std::runtime_error("Wave not full or width varies");
        if (!thread_at.emplace(std::make_pair(rec[2], rec[0]), t).second)
            throw std::runtime_error("Duplicate native lane");
    }
    auto source = [&](Word wave, Word lane) {
        auto it = thread_at.find({wave, lane});
        return it == thread_at.end() ? UINT32_MAX : it->second;
    };
    const struct { const char* name; Word (*request)(Word, Word); Word field; } shuffles[] = {
        {"affine", affine, 3}, {"rotation", rotation, 7}};
    unsigned wrong = 0, controls = 0;
    for (const auto& shuffle : shuffles) {
        unsigned correct = 0, misrouted = 0, other = 0;
        for (Word t = 0; t < threads; ++t) {
            const Word* rec = r.data() + t * fields;
            const Word lane = rec[0], wave = rec[2], got = rec[shuffle.field];
            if (got == source(wave, shuffle.request(lane, width))) ++correct;
            else ++(got == source(wave, block_misrouted(shuffle.request, lane, width, block)) ? misrouted : other);
        }
        std::printf("wave_shuffle %s width=%u %s correct=%u block_misrouted=%u other=%u block=%u\n",
                    label, width, shuffle.name, correct, misrouted, other, block);
        wrong += threads - correct;
    }
    for (Word t = 0; t < threads; ++t) {
        const Word* rec = r.data() + t * fields;
        if (rec[4] != source(rec[2], width - 1 - rec[0]) || rec[5] != source(rec[2], 3 % width)) ++controls;
    }
    std::printf("wave_shuffle %s width=%u control_errors=%u\n", label, width, controls);
    return wrong + controls;
}
}

bool check_wave_shuffle(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 caps{};
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS1, &caps, sizeof(caps)), "Query wave range");
    check(check_feature(device, D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model)), "Query wave shader model");
    std::printf("native_lane_range=%u..%u shader_model_capped_at_6_6=0x%x\n",
                caps.WaveLaneCountMin, caps.WaveLaneCountMax, model.HighestShaderModel);
    // The native variant is cs_6_5 like the other dynamic-shuffle fixtures. The Windows 10 in-box WARP
    // (10.0.19041, SM 6.2) removes the device for any WaveReadLaneAt whose lane index is not a literal.
    if (!caps.WaveOps || model.HighestShaderModel < D3D_SHADER_MODEL_6_5) return false;
    Word native = 0, width = 0;
    unsigned errors = run_variant(device, "native", {WaveShuffle, sizeof(WaveShuffle)}, 0, 0, native);
    const struct { Word width; D3D12_SHADER_BYTECODE code; } forced[] = {
        {16, {WaveShuffle16, sizeof(WaveShuffle16)}},
        {32, {WaveShuffle32, sizeof(WaveShuffle32)}},
        {64, {WaveShuffle64, sizeof(WaveShuffle64)}},
    };
    bool ran_forced = false;
    for (const auto& f : forced) {
        if (model.HighestShaderModel < D3D_SHADER_MODEL_6_6 || caps.WaveLaneCountMin > f.width ||
            caps.WaveLaneCountMax < f.width) {
            std::printf("wave_shuffle forced width=%u skipped\n", f.width);
            continue;
        }
        errors += run_variant(device, "forced", f.code, f.width, native, width);
        ran_forced = true;
    }
    if (errors) throw std::runtime_error("WaveReadLaneAt returned a value from a lane other than the requested one");
    return ran_forced;
}

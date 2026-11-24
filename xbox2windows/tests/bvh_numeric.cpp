#define NOMINMAX
#include "bvh_fixture.h"
#include "NumericBounds.h"
#include <algorithm>
#include <cmath>

using namespace bvh_fixture;

namespace {
Word finite_half(Word code)
{
    code &= 0xffff;
    if ((code & 0x7fff) == 0 || (code & 0x7c00) == 0x7c00) return 0;
    return code;
}

double half_value(Word half)
{
    const int exponent = (half >> 10) & 31;
    const int fraction = half & 1023;
    const double value = std::ldexp(exponent ? 1024 + fraction : fraction, exponent ? exponent - 25 : -24);
    return half & 0x8000 ? -value : value;
}

float value(Word bits)
{
    float number;
    std::memcpy(&number, &bits, sizeof(number));
    return number;
}
}

void check_bvh_numeric(ID3D12Device* device)
{
    constexpr Word count = 65536;
    constexpr std::array<Word, 13> floats{0xff800000,0xff7fffff,0xbf800000,0,0x0effffff,0x0f000000,0x0f000001,
                                          0x3f7ff971,0x3f7ff972,0x3f7ff973,0x3f800000,0x7f7fffff,0x7f800000};
    std::vector<std::array<Word, 8>> inputs(count);
    std::vector<std::array<Word, 10>> expected(count);
    for (Word i = 0; i < count; ++i) {
        const Word a = finite_half(i), b = finite_half(40503 * i + 19);
        const Word c = finite_half(65535 - i), d = finite_half(i ^ 0xad51u);
        constexpr auto n = floats.size();
        std::array<Word, 3> f{floats[i % n], floats[(i / n) % n], floats[(i / (n * n)) % n]};
        std::array<Word, 3> u{i * 2654435761u, ~i, i % 2 ? i * 2654435761u : i << 16};
        inputs[i] = {a | (c << 16), b | (d << 16), f[0], f[1], f[2], u[0], u[1], u[2]};
        const Word x = f[0];
        std::sort(f.begin(), f.end(), [](Word l, Word r) { return value(l) < value(r); });
        std::sort(u.begin(), u.end());
        const auto bounded = [x](Word low, Word high) {
            if (value(x) < value(low)) return low;
            if (value(x) > value(high)) return high;
            return x;
        };
        const bool ab = half_value(a) < half_value(b), cd = half_value(c) < half_value(d);
        expected[i] = {(ab ? a : b) | ((cd ? c : d) << 16), (ab ? b : a) | ((cd ? d : c) << 16),
                       f.front(), f.back(), bounded(0xff7fffff,0x7f7fffff), bounded(0x0f000000,0x7f7fffff),
                       bounded(0,0x3f7ff972), u[0], u[1], u[2]};
    }
    Kernel kernel(device, {NumericBounds, sizeof(NumericBounds)});
    auto input = upload(device, inputs);
    const size_t bytes = expected.size() * sizeof(expected[0]);
    auto out = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands cmd(device, kernel.pipeline.Get());
    auto* list = cmd.list.Get();
    transition(list, out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(kernel.root.Get());
    list->SetComputeRootShaderResourceView(0, input->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
    list->Dispatch(count / 64, 1, 1);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(), 0, out.Get(), 0, bytes);
    cmd.submit_and_wait(device);
    const auto actual = read(back.Get(), bytes / sizeof(Word));
    for (Word i = 0; i < count; ++i)
        for (Word field = 0; field < expected[i].size(); ++field)
            equal(actual[i * expected[i].size() + field], expected[i][field], i, field);
    std::printf("numeric_bounds rows=%u words=%zu mismatches=0\n", count, actual.size());
}

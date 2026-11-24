#define NOMINMAX
#include "bvh_fixture.h"
#include "AtomicBounds.h"
#include "InitBounds.h"
#include "QueryBounds.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace bvh_fixture;

namespace {
constexpr Word buckets = 7, fields = 8, capacity = 1025;
constexpr Word firstAdd = 0xfffffff0u, firstSub = 16u, guard = 0x13579bdf;
constexpr Word ticketStart = buckets * fields;

Word bits(float value)
{
    Word result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

float as_float(Word value)
{
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::vector<Row> make_points()
{
    std::vector<Row> rows;
    for (Word i = 0; i < capacity; ++i) {
        const Word bucket = i % 8 < 4 ? 0 : i % 8 - 3;
        const float a = static_cast<float>(static_cast<int>((i * 37) % 1009) - 503) / 8;
        Row point{bits(a), bits(-a * 2), bits(a + 3), bucket};
        if (bucket == 1 || bucket == 2) {
            const float sign = bucket == 1 ? 1.f : -1.f;
            for (Word axis = 0; axis < 3; ++axis)
                point[axis] = bits(sign * (1 + std::abs(as_float(point[axis]))));
        } else if (bucket == 3) {
            const Word zero = (i & 8) ? 0x80000000u : 0u;
            point = {zero, zero ^ 0x80000000u, 0u, bucket};
        } else if (bucket == 4) {
            point[(i / 8) % 3] = (i & 8) ? 0x7fc00001u : 0xff800000u;
        }
        rows.push_back(point);
    }
    rows[0] = {0xff7fffffu, 0x00000001u, 0x80000001u, 5};
    rows[1] = {0x7f7fffffu, 0x80000001u, 0x00000001u, 5};
    return rows;
}

std::vector<Row> make_rays()
{
    const float origins[][3]{{0,0,0}, {1,1,1}, {-1,-1,-1}, {12,8,4}, {-12,-8,-4},
                             {62,0,0}, {0,180,0}, {0,0,-180}, {1000,1000,1000}};
    std::vector<Row> rays;
    for (const auto& origin : origins)
        for (Word direction = 0; direction < 6; ++direction)
            rays.push_back({bits(origin[0]), bits(origin[1]), bits(origin[2]), direction});
    return rays;
}

// Project the original accepted points on each axis independently of the GPU bounds.
Word ray_mask(const Row& ray, const std::vector<Row>& points, const std::vector<bool>& valid)
{
    std::array<Word, buckets> planes{};
    for (size_t i = 0; i < points.size(); ++i) {
        if (!valid[i]) continue;
        const auto& point = points[i];
        for (Word axis = 0; axis < 3; ++axis) {
            if (as_float(point[axis]) <= as_float(ray[axis])) planes[point[3]] |= 1u << axis;
            if (as_float(point[axis]) >= as_float(ray[axis])) planes[point[3]] |= 8u << axis;
        }
    }
    const Word required = 63u ^ (1u << ray[3]);
    Word mask = 0;
    for (Word b = 0; b < buckets; ++b)
        if ((planes[b] & required) == required) mask |= 1u << b;
    return mask;
}

void run(ID3D12Device* device, const Kernel& init, const Kernel& kernel, const Kernel& consumer,
         const std::vector<Row>& rows, Word count)
{
    const auto rays = make_rays();
    constexpr Word rayStart = ticketStart + 2 * capacity;
    std::vector<Word> initial(rayStart + rays.size() + 16, guard);
    auto expected = initial;
    for (Word bucket = 0; bucket < buckets; ++bucket) {
        auto* record = expected.data() + bucket * fields;
        std::fill_n(record, 3, bits(std::numeric_limits<float>::max()));
        std::fill_n(record + 3, 3, bits(std::numeric_limits<float>::lowest()));
        record[6] = initial[bucket * fields + 6] = firstAdd;
        record[7] = initial[bucket * fields + 7] = firstSub;
    }
    std::array<Word, buckets> accepted{};
    std::vector<bool> valid(capacity);
    for (Word i = 0; i < count; ++i) {
        const Row& point = rows[i];
        if (!std::all_of(point.begin(), point.begin() + 3, [](Word v) { return std::isfinite(as_float(v)); }))
            continue;
        valid[i] = true;
        ++accepted[point[3]];
        auto* record = expected.data() + point[3] * fields;
        for (Word axis = 0; axis < 3; ++axis) {
            const float v = as_float(point[axis]);
            const float low = as_float(record[axis]), high = as_float(record[axis + 3]);
            if (v < low || (v == low && std::signbit(v))) record[axis] = point[axis];
            if (v > high || (v == high && !std::signbit(v))) record[axis + 3] = point[axis];
        }
        ++record[6];
        --record[7];
    }
    const size_t bytes = initial.size() * sizeof(Word);
    auto input = upload(device, rows);
    auto rayInput = upload(device, rays);
    auto seed = upload(device, initial);
    auto out = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto back = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands cmd(device, init.pipeline.Get());
    auto* list = cmd.list.Get();
    transition(list, out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(out.Get(), 0, seed.Get(), 0, bytes);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // The two kernels share an identical serialized root layout.
    list->SetComputeRootSignature(init.root.Get());
    list->SetComputeRootShaderResourceView(0, input->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
    const Word args[]{count, ticketStart * sizeof(Word), buckets};
    list->SetComputeRoot32BitConstants(2, 3, args, 0);
    list->Dispatch((buckets + 31) / 32, 1, 1);
    D3D12_RESOURCE_BARRIER order{};
    order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    order.UAV.pResource = out.Get();
    list->ResourceBarrier(1, &order);
    list->SetPipelineState(kernel.pipeline.Get());
    list->Dispatch((capacity + 31) / 32, 1, 1);
    list->ResourceBarrier(1, &order);
    list->SetPipelineState(consumer.pipeline.Get());
    list->SetComputeRootSignature(consumer.root.Get());
    list->SetComputeRootShaderResourceView(0, rayInput->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress());
    const Word rayArgs[]{static_cast<Word>(rays.size()), rayStart * sizeof(Word), buckets};
    list->SetComputeRoot32BitConstants(2, 3, rayArgs, 0);
    list->Dispatch((rayArgs[0] + 31) / 32, 1, 1);
    transition(list, out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(), 0, out.Get(), 0, bytes);
    cmd.submit_and_wait(device);
    const auto actual = read(back.Get(), initial.size());
    for (Word i = 0; i < ticketStart; ++i) equal(actual[i], expected[i], i / fields, i % fields);
    std::array<std::array<std::vector<Word>, 2>, buckets> tickets;
    for (Word i = 0; i < capacity; ++i) {
        const Word* pair = actual.data() + ticketStart + 2 * i;
        if (valid[i]) {
            tickets[rows[i][3]][0].push_back(pair[0] - firstAdd);
            tickets[rows[i][3]][1].push_back(firstSub - pair[1]);
        } else {
            equal(pair[0], guard, i, 0);
            equal(pair[1], guard, i, 1);
        }
    }
    for (Word b = 0; b < buckets; ++b) {
        for (auto& order : tickets[b]) {
            std::sort(order.begin(), order.end());
            for (Word rank = 0; rank < accepted[b]; ++rank) equal(order.at(rank), rank, b, rank);
        }
    }
    for (Word r = 0; r < rays.size(); ++r) equal(actual[rayStart + r], ray_mask(rays[r], rows, valid), r, 0);
    for (size_t i = rayStart + rays.size(); i < actual.size(); ++i) equal(actual[i], guard, static_cast<Word>(i), 0);
    std::printf("atomic_bounds inputs=%u accepted=%zu buckets=%u ray_masks=%zu mismatches=0\n",
                count, static_cast<size_t>(std::count(valid.begin(), valid.end(), true)), buckets, rays.size());
}
}

void check_bvh_atomics(ID3D12Device* device)
{
    Kernel init(device, {InitBounds, sizeof(InitBounds)});
    Kernel kernel(device, {AtomicBounds, sizeof(AtomicBounds)});
    Kernel consumer(device, {QueryBounds, sizeof(QueryBounds)});
    auto rows = make_points();
    for (Word count : {0u, 31u, 33u, capacity}) run(device, init, kernel, consumer, rows, count);
    std::reverse(rows.begin(), rows.end());
    run(device, init, kernel, consumer, rows, capacity);
}

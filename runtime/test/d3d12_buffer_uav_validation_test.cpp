// Buffer UAV descriptor validation in ID3D12Device::CreateUnorderedAccessView (write_buffer_uav):
//   - Invalid descriptors (unknown Buffer.Flags bits, structured+RAW, raw+StructureByteStride,
//     typed+StructureByteStride, NumElements == 0, FirstElement+NumElements 64-bit overflow,
//     out-of-bounds ranges, structured with zero stride, typed+RAW) must leave the slot zeroed
//     (null descriptor).
//   - Valid raw, structured, typed, and counter-backed structured buffer UAVs must encode expected
//     GFX10 V# descriptors and clear/readback cleanly on the GPU.
//
// No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace g = leonida::gfx10;

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

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return b;
}

bool slot_is_zero(const std::byte* slot, UINT step) {
    for (UINT i = 0; i < step; ++i) {
        if (slot[i] != std::byte{0}) return false;
    }
    return true;
}

g::BufferWords read_vsharp(const std::byte* slot, std::size_t byte_offset = 0) {
    g::BufferWords words{};
    std::memcpy(words.data(), slot + byte_offset, sizeof(words));
    return words;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;

        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12CommandQueue> queue;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 32};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        auto* slots = reinterpret_cast<std::byte*>(heap->GetCPUDescriptorHandleForHeapStart().ptr);
        const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        EXPECT(step >= sizeof(g::BufferWords) * 2);

        auto slot_ptr = [&](UINT i) -> std::byte* { return slots + std::size_t(i) * step; };
        auto slot_handle = [&](UINT i) -> D3D12_CPU_DESCRIPTOR_HANDLE {
            return {reinterpret_cast<SIZE_T>(slot_ptr(i))};
        };

        auto committed = [&](UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_FLAGS flags) {
            Ref<ID3D12Resource> out;
            const D3D12_HEAP_PROPERTIES hp{type};
            const D3D12_RESOURCE_DESC rd = buffer_desc(bytes, flags);
            const D3D12_RESOURCE_STATES initial = type == D3D12_HEAP_TYPE_READBACK
                                                      ? D3D12_RESOURCE_STATE_COPY_DEST
                                                      : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, initial, nullptr, IID_PPV_ARGS(&out.p));
            return out;
        };

        constexpr UINT64 kBufBytes = 256;
        constexpr UINT64 kCounterBytes = 8192;
        Ref<ID3D12Resource> buf = committed(kBufBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> counter = committed(kCounterBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Ref<ID3D12Resource> readback = committed(kBufBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(buf.p && counter.p && readback.p);

        const UINT64 buf_va = buf->GetGPUVirtualAddress();
        const UINT64 counter_va = counter->GetGPUVirtualAddress();
        EXPECT(buf_va != 0 && counter_va != 0);

        auto check_invalid = [&](const char* label, ID3D12Resource* c, const D3D12_UNORDERED_ACCESS_VIEW_DESC& desc) {
            // Poison slot first to prove CreateUnorderedAccessView clears it to a null descriptor.
            std::memset(slot_ptr(0), 0xA5, step);
            device->CreateUnorderedAccessView(buf.p, c, &desc, slot_handle(0));
            if (!slot_is_zero(slot_ptr(0), step)) {
                std::fprintf(stderr, "expected null descriptor for invalid UAV case: %s\n", label);
                return false;
            }
            return true;
        };

        // 1. Unknown view.Flags bits
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = 16;
            desc.Buffer.Flags = static_cast<D3D12_BUFFER_UAV_FLAGS>(0x2);
            EXPECT(check_invalid("unknown_flags_0x2_r32_typeless", nullptr, desc));

            desc.Buffer.Flags = static_cast<D3D12_BUFFER_UAV_FLAGS>(D3D12_BUFFER_UAV_FLAG_RAW | 0x2);
            EXPECT(check_invalid("unknown_flags_0x3_raw_plus_bit1", nullptr, desc));

            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = 16;
            desc.Buffer.Flags = static_cast<D3D12_BUFFER_UAV_FLAGS>(0x4);
            EXPECT(check_invalid("unknown_flags_0x4_structured", nullptr, desc));

            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = static_cast<D3D12_BUFFER_UAV_FLAGS>(0x8);
            EXPECT(check_invalid("unknown_flags_0x8_typed", nullptr, desc));
        }

        // 2. Structured UAV with D3D12_BUFFER_UAV_FLAG_RAW
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = 8;
            desc.Buffer.StructureByteStride = 16;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            EXPECT(check_invalid("structured_with_raw_flag", nullptr, desc));
        }

        // 3. Raw UAV with StructureByteStride != 0
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = 16;
            desc.Buffer.StructureByteStride = 4;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            EXPECT(check_invalid("raw_with_nonzero_stride", nullptr, desc));
        }

        // 4. Typed UAV with StructureByteStride != 0 (and typed with RAW)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = 16;
            desc.Buffer.StructureByteStride = 4;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            EXPECT(check_invalid("typed_with_nonzero_stride", nullptr, desc));

            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            EXPECT(check_invalid("typed_with_raw_flag", nullptr, desc));
        }

        // 5. NumElements == 0 (raw, structured, typed)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = 0;

            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            EXPECT(check_invalid("raw_num_elements_0", nullptr, desc));

            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = 16;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            EXPECT(check_invalid("structured_num_elements_0", nullptr, desc));

            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            EXPECT(check_invalid("typed_num_elements_0", nullptr, desc));
        }

        // 6. FirstElement + NumElements 64-bit unsigned overflow & out-of-bounds
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = (std::numeric_limits<UINT64>::max)();
            desc.Buffer.NumElements = 1;

            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            EXPECT(check_invalid("raw_overflow_first_uint64max", nullptr, desc));

            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Buffer.StructureByteStride = 16;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            EXPECT(check_invalid("structured_overflow_first_uint64max", nullptr, desc));

            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.Buffer.StructureByteStride = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
            EXPECT(check_invalid("typed_overflow_first_uint64max", nullptr, desc));

            // Out-of-bounds without overflow
            desc.Buffer.FirstElement = 60;
            desc.Buffer.NumElements = 5;  // 65 * 4 = 260 > 256
            EXPECT(check_invalid("typed_oob_past_end", nullptr, desc));
        }

        // 7. Valid raw, structured, typed, and counter-backed structured buffer UAVs
        // Slot 1: Raw UAV [16..48) (FirstElement=4, NumElements=8)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 4;
            desc.Buffer.NumElements = 8;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device->CreateUnorderedAccessView(buf.p, nullptr, &desc, slot_handle(1));

            g::Buffer decoded{};
            EXPECT(g::decode(read_vsharp(slot_ptr(1)), decoded));
            EXPECT(decoded.address == buf_va + 16);
            EXPECT(decoded.stride == 0);
            EXPECT(decoded.num_records == 32);
            EXPECT(decoded.oob_select == g::OobSelect::raw);
        }

        // Slot 2: Structured UAV [64..128) (FirstElement=4, NumElements=4, Stride=16)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 4;
            desc.Buffer.NumElements = 4;
            desc.Buffer.StructureByteStride = 16;
            device->CreateUnorderedAccessView(buf.p, nullptr, &desc, slot_handle(2));

            g::Buffer decoded{};
            EXPECT(g::decode(read_vsharp(slot_ptr(2)), decoded));
            EXPECT(decoded.address == buf_va + 64);
            EXPECT(decoded.stride == 16);
            EXPECT(decoded.num_records == 4);
            EXPECT(decoded.oob_select == g::OobSelect::structured_with_offset);
        }

        // Slot 3: Typed UAV [128..160) (FirstElement=32, NumElements=8, R32_UINT)
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 32;
            desc.Buffer.NumElements = 8;
            device->CreateUnorderedAccessView(buf.p, nullptr, &desc, slot_handle(3));

            g::Buffer decoded{};
            EXPECT(g::decode(read_vsharp(slot_ptr(3)), decoded));
            EXPECT(decoded.address == buf_va + 128);
            EXPECT(decoded.stride == 4);
            EXPECT(decoded.num_records == 8);
            EXPECT(decoded.format == g::format_from_vulkan(VK_FORMAT_R32_UINT));
            EXPECT(decoded.oob_select == g::OobSelect::structured);
        }

        // Slot 4: Structured UAV with counter at offset 4096
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 1;
            desc.Buffer.NumElements = 4;
            desc.Buffer.StructureByteStride = 16;
            desc.Buffer.CounterOffsetInBytes = 4096;
            device->CreateUnorderedAccessView(buf.p, counter.p, &desc, slot_handle(4));

            g::Buffer primary{};
            EXPECT(g::decode(read_vsharp(slot_ptr(4), 0), primary));
            EXPECT(primary.address == buf_va + 16);
            EXPECT(primary.stride == 16);
            EXPECT(primary.num_records == 4);
            EXPECT(primary.oob_select == g::OobSelect::structured_with_offset);

            g::Buffer cnt{};
            EXPECT(g::decode(read_vsharp(slot_ptr(4), sizeof(g::BufferWords)), cnt));
            EXPECT(cnt.address == counter_va + 4096);
            EXPECT(cnt.stride == 0);
            EXPECT(cnt.num_records == 4);
            EXPECT(cnt.oob_select == g::OobSelect::raw);
        }

        // Slot 5: Full-buffer raw UAV to zero-initialize before testing partial clears
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_R32_TYPELESS;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = UINT(kBufBytes / 4);
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device->CreateUnorderedAccessView(buf.p, nullptr, &desc, slot_handle(5));
        }

        // 8. GPU clear & readback using the created raw (slot 1), structured (slot 2), and typed (slot 3) UAVs
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));

        const UINT zero_vals[4] = {0, 0, 0, 0};
        const UINT raw_vals[4] = {0x11223344u, 0, 0, 0};
        const UINT struct_vals[4] = {0x55667788u, 0, 0, 0};
        const UINT typed_vals[4] = {0x99AABBCCu, 0, 0, 0};

        D3D12_RESOURCE_BARRIER uav_barrier{};
        uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav_barrier.UAV.pResource = buf.p;

        list->ClearUnorderedAccessViewUint({0}, slot_handle(5), buf.p, zero_vals, 0, nullptr);
        list->ResourceBarrier(1, &uav_barrier);
        list->ClearUnorderedAccessViewUint({0}, slot_handle(1), buf.p, raw_vals, 0, nullptr);
        list->ResourceBarrier(1, &uav_barrier);
        list->ClearUnorderedAccessViewUint({0}, slot_handle(2), buf.p, struct_vals, 0, nullptr);
        list->ResourceBarrier(1, &uav_barrier);
        list->ClearUnorderedAccessViewUint({0}, slot_handle(3), buf.p, typed_vals, 0, nullptr);
        list->ResourceBarrier(1, &uav_barrier);

        const auto to_copy = transition(buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(readback.p, 0, buf.p, 0, kBufBytes);
        EXPECT(SUCCEEDED(list->Close()));

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        queue->Signal(fence.p, 1);
        fence->SetEventOnCompletion(1, nullptr);

        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        const auto* words = static_cast<const std::uint32_t*>(mapped);
        for (UINT i = 0; i < kBufBytes / 4; ++i) {
            std::uint32_t want = 0;
            if (i >= 4 && i < 12) want = 0x11223344u;        // raw [16..48)
            else if (i >= 16 && i < 32) want = 0x55667788u;  // structured [64..128)
            else if (i >= 32 && i < 40) want = 0x99AABBCCu;  // typed [128..160)
            if (words[i] != want) {
                std::fprintf(stderr, "dword %u: got 0x%08x, want 0x%08x\n", i, words[i], want);
                return 1;
            }
        }
        readback->Unmap(0, nullptr);
    }

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_buffer_uav_validation: all invalid cases produced null descriptors and valid raw/structured/typed/counter UAVs passed\n");
    return 0;
}

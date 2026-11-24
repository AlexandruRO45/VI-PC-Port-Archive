#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <cstdint>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

#include "startup_stream_draw.h"

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT XGCreateTextureComputer(const D3D12_RESOURCE_DESC*, void***, UINT64, UINT64);

// The components under test receive the guest ABI's vtable slots. The observation
// queue and its readback buffer are desktop objects owned only by the fixture.
using Word = std::uint64_t;
template<class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0,
            Word e = 0, Word f = 0, Word g = 0, Word h = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g, h);
}

int draw_logical_tables(ID3D12Device* desktop, void* guest, void* list)
{
    constexpr UINT count = 1245184;
    auto accepts = [&](UINT n) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, n,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> probe;
        return SUCCEEDED(desktop->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&probe)));
    };
    if (accepts(count)) return 77;
    UINT boundary = 0, upper = 1048577;
    while (upper - boundary > 1) {
        const UINT n = boundary + (upper - boundary) / 2;
        if (accepts(n)) boundary = n; else upper = n;
    }
    if (!boundary) return 34;
    std::printf("logical views: physical capacity %u, logical count %u\n", boundary, count);
    const UINT stride = static_cast<UINT>(invoke(guest, 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    if (stride < 32) return 34;
    struct Extended {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word cpu, bytes;
    } info{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
    void* heaps[2]{};
    Word handles[2]{};
    Extended staging_info{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2}, 0, 0};
    void* staging = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&staging_info), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&staging))))) return 34;
    const Word staging_cpu = invoke(staging, 9);
    D3D12_SHADER_RESOURCE_VIEW_DESC null_srv{};
    null_srv.Format = DXGI_FORMAT_R32_TYPELESS;
    null_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    null_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    null_srv.Buffer.NumElements = 1;
    null_srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    alignas(256) static std::array<std::array<UINT, 64>, 6> words{};
    for (UINT h = 0; h < 2; ++h) {
        if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&info), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heaps[h]))))) return 34;
        const Word cpu = invoke(heaps[h], 9);
        handles[h] = invoke(heaps[h], 10);
        if (!cpu || !handles[h]) return 34;
        const UINT slots[] = {boundary - 1, boundary, count - 1};
        for (UINT i = 0; i < 3; ++i) {
            auto& data = words[3 * h + i];
            data[0] = 16 + 96 * h + 16 * i;
            const D3D12_CONSTANT_BUFFER_VIEW_DESC cb{arg(data.data()), static_cast<UINT>(sizeof(data))};
            // Initialize a real native descriptor as well as the fixture's guest view token.
            const Word destination = i < 2 ? staging_cpu + Word{i} * stride : cpu + Word{slots[i]} * stride;
            invoke(guest, 18, 0, arg(&null_srv), destination);
            invoke(guest, 17, arg(&cb), destination);
        }
        const Word destination = cpu + Word{boundary - 1} * stride;
        const UINT two = 2;
        if (h == 0) {
            invoke(guest, 24, two, destination, staging_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        } else {
            invoke(guest, 23, 1, arg(&destination), arg(&two), 1, arg(&staging_cpu), arg(&two), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        // Reject the complete copy: even its first entry must leave the final sentinel intact.
        invoke(guest, 24, two, cpu + Word{count - 1} * stride, staging_cpu, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
    const auto bind = [&](UINT h) { invoke(list, 32, 1, arg(&heaps[h])); };
    const auto table = [&](UINT h, UINT i) { invoke(list, 36, 0, handles[h] + Word{i} * stride); };
    const auto draw = [&](LONG x) {
        const D3D12_RECT strip{x, 0, x + 1, 4};
        invoke(list, 22, 1, arg(&strip));
        invoke(list, 12, 3, 1);
    };
    bind(0);
    table(0, boundary - 1); draw(0);
    table(0, boundary); draw(1);
    table(0, count - 1); draw(2);
    table(0, count); draw(3);
    bind(1);
    table(1, boundary); draw(4);
    bind(0); draw(5);                   // Changing heaps invalidates its previous table.
    table(1, boundary); draw(6);        // A handle belonging to the other heap.
    table(0, boundary); draw(7);
    return 0;
}

namespace {
struct Bc7Binding {
    std::vector<unsigned char> backing, views;
    UINT64 table = 0;
};

bool populate_tiled_bc7(const StartupBc7Input& input, const D3D12_RESOURCE_DESC& desc,
                        const std::vector<unsigned char>& blocks, UINT row_bytes, Bc7Binding& binding)
{
    void** computer = nullptr;
    if (FAILED(XGCreateTextureComputer(&desc, &computer, 0, 0)) || !computer) return false;
    const auto table = static_cast<void**>(computer[0]);
    using Query = HRESULT(*)(void*, unsigned char*);
    using Write = HRESULT(*)(void*, unsigned char*, UINT, UINT, const unsigned char*, UINT, UINT);
    std::array<unsigned char, 0x18 + 4 * 0x6c8 + 8> layout{};
    UINT64 span = 0;
    HRESULT hr = reinterpret_cast<Query>(table[2])(computer, layout.data());
    if (SUCCEEDED(hr)) std::memcpy(&span, layout.data(), sizeof span);
    const UINT64 expected = input.width == 64 ? 4096 : 16384;
    if (span != expected) hr = E_FAIL;
    if (SUCCEEDED(hr)) {
        binding.backing.resize(static_cast<std::size_t>(span), 0xa5);
        hr = reinterpret_cast<Write>(table[8])(computer, binding.backing.data(), 0, 0, blocks.data(),
                                                row_bytes, static_cast<UINT>(blocks.size()));
    }
    reinterpret_cast<UINT64(*)(void*)>(table[1])(computer);
    return SUCCEEDED(hr);
}

bool bind_bc7(ID3D12Device* desktop, void* guest, const StartupBc7Input& input, Bc7Binding& binding)
{
    std::ifstream file(input.blocks, std::ios::binary);
    const std::vector<unsigned char> blocks{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>{}};
    const UINT columns = (input.width + 3) / 4, rows = (input.height + 3) / 4, row_bytes = columns * 16;
    if (blocks.size() != static_cast<std::size_t>(row_bytes) * rows) return false;
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, input.width, input.height, 1, 1,
        DXGI_FORMAT_BC7_UNORM, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
    if (input.tiled) {
        if (!populate_tiled_bc7(input, desc, blocks, row_bytes, binding)) return false;
    } else {
        auto aligned = desc;
        aligned.Width = columns * 4;
        aligned.Height = rows * 4;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 size = 0;
        desktop->GetCopyableFootprints(&aligned, 0, 1, 0, &footprint, nullptr, nullptr, &size);
        if (!size || size == UINT64_MAX || footprint.Footprint.RowPitch < row_bytes) return false;
        binding.backing.resize(static_cast<std::size_t>(size));
        for (UINT y = 0; y < rows; ++y)
            std::memcpy(binding.backing.data() + footprint.Offset + y * footprint.Footprint.RowPitch,
                        blocks.data() + y * row_bytes, row_bytes);
    }

    void* texture = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 133, arg(binding.backing.data()), arg(&desc),
        D3D12_RESOURCE_STATE_COMMON, 0, arg(&__uuidof(ID3D12Resource)), arg(&texture))))) return false;

    const UINT increment = desktop->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT entries = (32 + increment - 1) / increment;
    binding.views.resize(static_cast<std::size_t>(entries) * increment);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        Word backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, entries, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE},
                arg(binding.views.data()), binding.views.size()};
    void* heap = nullptr;
    if (FAILED(static_cast<HRESULT>(invoke(guest, 14, arg(&heap_desc), arg(&__uuidof(ID3D12DescriptorHeap)), arg(&heap)))))
        return false;
    const auto cpu = invoke(heap, 9);
    if (cpu != arg(binding.views.data())) return false;
    invoke(guest, 18, arg(texture), 0, cpu);
    binding.table = invoke(heap, 10);
    return binding.table != 0;
}
}  // namespace

int check_stream_draw(IDXGIAdapter* adapter, void* guest, void* pso, const char* pixels, const StartupBc7Input* bc7, DXGI_FORMAT format)
{
    const bool bc = GetEnvironmentVariableW(L"X2W_TEST_BC_VIEW", nullptr, 0) != 0;
    const bool bc_chain = GetEnvironmentVariableW(L"X2W_TEST_BC_CHAIN", nullptr, 0) != 0;
    const UINT width = bc_chain ? 384 : bc7 ? bc7->width : bc ? 4 * 128 : 8;
    const UINT height = bc_chain ? 32 : bc7 ? bc7->height : bc ? 128 : 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 20;
    Bc7Binding binding;
    if (bc7 && !bind_bc7(desktop.Get(), guest, *bc7, binding)) return 34;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    const bool rgba = format == DXGI_FORMAT_R8G8B8A8_UNORM;
    const D3D12_CLEAR_VALUE clear{format, {0, 0, 1, rgba ? 1.0f : 0.0f}};
    ComPtr<ID3D12Resource> target;
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&target)))) return 21;
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    if (FAILED(desktop->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)))) return 22;
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    invoke(guest, 20, arg(target.Get()), 0, rtv.ptr);

    void* list = nullptr;
    void* queue = nullptr;
    void* allocator = nullptr;
    void* fence = nullptr;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (FAILED(static_cast<HRESULT>(invoke(guest, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&queue)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list)))) ||
        FAILED(static_cast<HRESULT>(invoke(guest, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&fence))))) return 23;

    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const D3D12_RECT rect{0, 0, static_cast<LONG>(bc7 || bc_chain ? width : width / 2), static_cast<LONG>(height)};
    invoke(list, 139, arg(allocator));
    invoke(list, 52, rtv.ptr, arg(clear.Color));
    invoke(list, 50, 1, arg(&rtv));
    invoke(list, 21, 1, arg(&viewport));
    invoke(list, 22, 1, arg(&rect));
    invoke(list, 20, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    invoke(list, 25, arg(pso));
    if (GetEnvironmentVariableW(L"X2W_TEST_LOGICAL_VIEW", nullptr, 0)) {
        if (int result = draw_logical_tables(desktop.Get(), guest, list)) return result;
    } else if (bc_chain) {
        extern int draw_bc_chain(void*, void*);
        if (const int result = draw_bc_chain(guest, list)) return result;
    } else if (GetEnvironmentVariableW(L"X2W_TEST_ADDRESS_SRV", nullptr, 0)) {
        extern int draw_address_srvs(ID3D12Device*, void*, void*);
        if (const int result = draw_address_srvs(desktop.Get(), guest, list)) return result;
    } else {
        if (bc7) invoke(list, 36, 0, binding.table);
        invoke(list, 12, 3, 1);
    }
    if (bc) {
        extern int draw_bc_address_srvs(void*, void*);
        if (const int result = draw_bc_address_srvs(guest, list)) return result;
    }
    invoke(list, 9);
    invoke(queue, 10, 1, arg(&list));
    if (FAILED(static_cast<HRESULT>(invoke(queue, 14, arg(fence), 1)))) return 24;
    const auto deadline = GetTickCount64() + 10000;
    while (invoke(fence, 8) == 0 && GetTickCount64() < deadline) Sleep(1);
    if (invoke(fence, 8) != 1) return 25;

    ComPtr<ID3D12CommandQueue> observer;
    ComPtr<ID3D12CommandAllocator> copy_allocator;
    ComPtr<ID3D12GraphicsCommandList> copy;
    ComPtr<ID3D12Fence> complete;
    if (FAILED(desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&observer))) ||
        FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copy_allocator))) ||
        FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copy_allocator.Get(), nullptr, IID_PPV_ARGS(&copy))) ||
        FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&complete)))) return 26;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    desktop->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, total, 1, 1, 1,
                               DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    ComPtr<ID3D12Resource> readback;
    if (FAILED(desktop->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return 27;
    D3D12_TEXTURE_COPY_LOCATION destination{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    destination.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION source{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    copy->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    if (FAILED(copy->Close())) return 28;
    ID3D12CommandList* submit[] = {copy.Get()};
    observer->ExecuteCommandLists(1, submit);
    if (FAILED(observer->Signal(complete.Get(), 1))) return 29;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 30;
    const bool completed = SUCCEEDED(complete->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 10000) == WAIT_OBJECT_0 && complete->GetCompletedValue() == 1;
    CloseHandle(event);
    if (!completed) return 31;
    void* memory = nullptr;
    if (FAILED(readback->Map(0, nullptr, &memory))) return 32;
    std::ofstream out(pixels, std::ios::binary);
    for (UINT y = 0; y < height; ++y)
        out.write(static_cast<const char*>(memory) + y * footprint.Footprint.RowPitch, width * (rgba ? 4 : 1));
    readback->Unmap(0, nullptr);
    return out ? 0 : 33;
}

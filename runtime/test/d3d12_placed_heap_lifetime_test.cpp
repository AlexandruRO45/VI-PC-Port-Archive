// Placed buffers must keep their backing heap's GPU arena allocation alive after ID3D12Heap::Release().
// In D3D12, CreatePlacedResource holds an internal reference to the heap so releasing the application's heap
// handle while placed buffers are still alive cannot free or reuse the underlying memory.
//
// Checks on the Vulkan-backed D3D12 device (and against Windows WARP when not under Wine):
//   1. Two placed UPLOAD buffers in one 128 KiB heap: release the heap, allocate new DEFAULT buffers and heaps,
//      confirm no VA overlaps either live placed buffer and their mapped bytes stay intact; release the first
//      buffer and confirm the second still keeps the heap's arena range pinned and copies cleanly on the GPU.
//   2. A placed DEFAULT buffer in a 64 KiB heap: upload a non-zero pattern via CopyBufferRegion, release the heap,
//      create committed DEFAULT buffers and heaps (which zero-fill freshly allocated arena ranges), confirm no VA
//      overlap and GPU readback of the placed buffer still returns the original pattern.
//   3. Arena reclamation: once all placed buffers and their heap are released, repeated create-place-release-heap-
//      then-release-buffer cycles succeed without leaking arena space (64 x 1 MiB) and reuse freed arena offsets.
//
// No game data or shaders. LEONIDA_VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation.
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

constexpr UINT64 k64K = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
constexpr UINT64 k1M = 16 * k64K;

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_HEAP_DESC buffer_heap_desc(D3D12_HEAP_TYPE type, UINT64 bytes) {
    D3D12_HEAP_DESC d{};
    d.SizeInBytes = bytes;
    d.Properties.Type = type;
    d.Alignment = k64K;
    d.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    return d;
}

bool ranges_overlap(UINT64 a, UINT64 a_size, UINT64 b, UINT64 b_size) {
    return a < b + b_size && b < a + a_size;
}

bool copy_buffer_sync(ID3D12Device* dev, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& fence_val,
                      ID3D12Resource* dst, ID3D12Resource* src, UINT64 bytes) {
    Ref<ID3D12CommandAllocator> alloc;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr, IID_PPV_ARGS(&list.p))))
        return false;
    list->CopyBufferRegion(dst, 0, src, 0, bytes);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    const UINT64 target = ++fence_val;
    if (FAILED(queue->Signal(fence, target))) return false;
    if (fence->GetCompletedValue() < target) {
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!ev) return false;
        const HRESULT hr = fence->SetEventOnCompletion(target, ev);
        if (SUCCEEDED(hr)) WaitForSingleObject(ev, 10000);
        CloseHandle(ev);
        if (FAILED(hr) || fence->GetCompletedValue() < target) return false;
    }
    return true;
}

bool verify_bytes(const char* label, const void* ptr, UINT64 bytes, std::uint8_t expected) {
    const auto* p = static_cast<const std::uint8_t*>(ptr);
    for (UINT64 i = 0; i < bytes; ++i) {
        if (p[i] != expected) {
            std::fprintf(stderr, "FAIL [%s]: byte %llu = 0x%02x, expected 0x%02x\n",
                         label, static_cast<unsigned long long>(i), unsigned(p[i]), unsigned(expected));
            return false;
        }
    }
    return true;
}

unsigned run_suite(const char* tag, ID3D12Device* dev, bool check_arena_reclaim) {
    unsigned failures = 0;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) {
        std::fprintf(stderr, "FAIL [%s]: failed to create queue/fence\n", tag);
        return 1;
    }
    UINT64 fence_val = 0;

    // 1. Two placed UPLOAD buffers in one 128 KiB heap survive heap->Release() and partial buffer release.
    {
        Ref<ID3D12Heap> heap;
        const D3D12_HEAP_DESC hd = buffer_heap_desc(D3D12_HEAP_TYPE_UPLOAD, 2 * k64K);
        if (FAILED(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap.p)))) {
            std::fprintf(stderr, "FAIL [%s/upload]: CreateHeap failed\n", tag);
            ++failures;
        } else {
            const D3D12_RESOURCE_DESC bd = buffer_desc(k64K);
            Ref<ID3D12Resource> buf_a, buf_b;
            if (FAILED(dev->CreatePlacedResource(heap.p, 0, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&buf_a.p))) ||
                FAILED(dev->CreatePlacedResource(heap.p, k64K, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&buf_b.p)))) {
                std::fprintf(stderr, "FAIL [%s/upload]: CreatePlacedResource failed\n", tag);
                ++failures;
            } else {
                const UINT64 va_a = buf_a->GetGPUVirtualAddress();
                const UINT64 va_b = buf_b->GetGPUVirtualAddress();
                void* map_a = nullptr;
                void* map_b = nullptr;
                if (FAILED(buf_a->Map(0, nullptr, &map_a)) || FAILED(buf_b->Map(0, nullptr, &map_b))) {
                    std::fprintf(stderr, "FAIL [%s/upload]: Map failed\n", tag);
                    ++failures;
                } else {
                    std::memset(map_a, 0xA5, static_cast<std::size_t>(k64K));
                    std::memset(map_b, 0x5A, static_cast<std::size_t>(k64K));

                    // Drop the application's heap reference while both placed buffers are still alive.
                    heap.reset();

                    // Allocate committed DEFAULT buffers and new heaps to provoke arena reuse if freed early.
                    const D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
                    std::vector<Ref<ID3D12Resource>> probes(4);
                    for (std::size_t i = 0; i < probes.size(); ++i) {
                        if (FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                IID_PPV_ARGS(&probes[i].p)))) {
                            std::fprintf(stderr, "FAIL [%s/upload]: probe CreateCommittedResource failed\n", tag);
                            ++failures;
                            break;
                        }
                        const UINT64 pva = probes[i]->GetGPUVirtualAddress();
                        if (ranges_overlap(pva, k64K, va_a, k64K) || ranges_overlap(pva, k64K, va_b, k64K)) {
                            std::fprintf(stderr,
                                         "FAIL [%s/upload]: probe %zu VA 0x%llx overlaps live placed buffer "
                                         "(A=0x%llx, B=0x%llx) after heap->Release()\n",
                                         tag, i, static_cast<unsigned long long>(pva),
                                         static_cast<unsigned long long>(va_a),
                                         static_cast<unsigned long long>(va_b));
                            ++failures;
                        }
                    }

                    if (!verify_bytes("upload buf_a after heap release", map_a, k64K, 0xA5)) ++failures;
                    if (!verify_bytes("upload buf_b after heap release", map_b, k64K, 0x5A)) ++failures;

                    // Release buf_a; buf_b must still keep the heap's arena allocation alive.
                    buf_a->Unmap(0, nullptr);
                    buf_a.reset();

                    Ref<ID3D12Resource> probe_after_a;
                    if (SUCCEEDED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                               D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                               IID_PPV_ARGS(&probe_after_a.p)))) {
                        const UINT64 pva = probe_after_a->GetGPUVirtualAddress();
                        if (ranges_overlap(pva, k64K, va_b, k64K)) {
                            std::fprintf(stderr,
                                         "FAIL [%s/upload]: probe VA 0x%llx overlaps live buf_b 0x%llx after buf_a release\n",
                                         tag, static_cast<unsigned long long>(pva),
                                         static_cast<unsigned long long>(va_b));
                            ++failures;
                        }
                    }
                    if (!verify_bytes("upload buf_b after buf_a release", map_b, k64K, 0x5A)) ++failures;

                    // GPU copy from surviving placed UPLOAD buffer B into a readback buffer.
                    const D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
                    Ref<ID3D12Resource> readback;
                    if (FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&readback.p))) ||
                        !copy_buffer_sync(dev, queue.p, fence.p, fence_val, readback.p, buf_b.p, k64K)) {
                        std::fprintf(stderr, "FAIL [%s/upload]: GPU readback of buf_b failed\n", tag);
                        ++failures;
                    } else {
                        void* rb_ptr = nullptr;
                        if (SUCCEEDED(readback->Map(0, nullptr, &rb_ptr))) {
                            if (!verify_bytes("upload buf_b GPU readback", rb_ptr, k64K, 0x5A)) ++failures;
                            readback->Unmap(0, nullptr);
                        } else {
                            std::fprintf(stderr, "FAIL [%s/upload]: readback Map failed\n", tag);
                            ++failures;
                        }
                    }
                    buf_b->Unmap(0, nullptr);
                }
            }
        }
    }

    // 2. Placed DEFAULT buffer survives heap->Release() with GPU data intact.
    {
        Ref<ID3D12Heap> heap;
        const D3D12_HEAP_DESC hd = buffer_heap_desc(D3D12_HEAP_TYPE_DEFAULT, k64K);
        const D3D12_RESOURCE_DESC bd = buffer_desc(k64K);
        const D3D12_HEAP_PROPERTIES up_hp{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
        const D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
        Ref<ID3D12Resource> placed, upload, readback;
        if (FAILED(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap.p))) ||
            FAILED(dev->CreatePlacedResource(heap.p, 0, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                             IID_PPV_ARGS(&placed.p))) ||
            FAILED(dev->CreateCommittedResource(&up_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&upload.p))) ||
            FAILED(dev->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback.p)))) {
            std::fprintf(stderr, "FAIL [%s/default]: setup failed\n", tag);
            ++failures;
        } else {
            const UINT64 va_placed = placed->GetGPUVirtualAddress();
            void* up_ptr = nullptr;
            if (SUCCEEDED(upload->Map(0, nullptr, &up_ptr))) {
                std::memset(up_ptr, 0xC3, static_cast<std::size_t>(k64K));
                upload->Unmap(0, nullptr);
            }
            if (!copy_buffer_sync(dev, queue.p, fence.p, fence_val, placed.p, upload.p, k64K)) {
                std::fprintf(stderr, "FAIL [%s/default]: upload copy failed\n", tag);
                ++failures;
            } else {
                // Release the heap while the placed DEFAULT buffer is still alive.
                heap.reset();

                // Allocate new DEFAULT buffers and heaps; on main, one reuses va_placed and zeroes it.
                std::vector<Ref<ID3D12Resource>> probes(4);
                std::vector<Ref<ID3D12Heap>> heap_probes(2);
                for (std::size_t i = 0; i < probes.size(); ++i) {
                    if (FAILED(dev->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &bd,
                                                            D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&probes[i].p)))) {
                        std::fprintf(stderr, "FAIL [%s/default]: probe buffer failed\n", tag);
                        ++failures;
                        break;
                    }
                    const UINT64 pva = probes[i]->GetGPUVirtualAddress();
                    if (ranges_overlap(pva, k64K, va_placed, k64K)) {
                        std::fprintf(stderr,
                                     "FAIL [%s/default]: probe %zu VA 0x%llx overlaps live placed DEFAULT buffer 0x%llx\n",
                                     tag, i, static_cast<unsigned long long>(pva),
                                     static_cast<unsigned long long>(va_placed));
                        ++failures;
                    }
                }
                for (std::size_t i = 0; i < heap_probes.size(); ++i) {
                    if (FAILED(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap_probes[i].p)))) {
                        std::fprintf(stderr, "FAIL [%s/default]: probe heap failed\n", tag);
                        ++failures;
                        break;
                    }
                }

                if (!copy_buffer_sync(dev, queue.p, fence.p, fence_val, readback.p, placed.p, k64K)) {
                    std::fprintf(stderr, "FAIL [%s/default]: readback copy failed\n", tag);
                    ++failures;
                } else {
                    void* rb_ptr = nullptr;
                    if (SUCCEEDED(readback->Map(0, nullptr, &rb_ptr))) {
                        if (!verify_bytes("default placed GPU readback after heap release", rb_ptr, k64K, 0xC3))
                            ++failures;
                        readback->Unmap(0, nullptr);
                    } else {
                        std::fprintf(stderr, "FAIL [%s/default]: readback Map failed\n", tag);
                        ++failures;
                    }
                }
            }
        }
    }

    // 3. No arena leak after both heap and placed buffer are released.
    if (check_arena_reclaim) {
        const D3D12_HEAP_DESC hd = buffer_heap_desc(D3D12_HEAP_TYPE_DEFAULT, k1M);
        const D3D12_RESOURCE_DESC bd = buffer_desc(k1M);
        UINT64 first_va = 0;
        bool reused = false;
        for (int iter = 0; iter < 64; ++iter) {
            Ref<ID3D12Heap> heap;
            Ref<ID3D12Resource> placed;
            if (FAILED(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap.p))) ||
                FAILED(dev->CreatePlacedResource(heap.p, 0, &bd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&placed.p)))) {
                std::fprintf(stderr, "FAIL [%s/reclaim]: iteration %d failed (possible arena leak)\n", tag, iter);
                ++failures;
                break;
            }
            const UINT64 va = placed->GetGPUVirtualAddress();
            if (iter == 0) first_va = va;
            else if (va == first_va) reused = true;
            // Release heap BEFORE placed buffer, then release placed buffer.
            heap.reset();
            placed.reset();
        }
        if (!reused) {
            std::fprintf(stderr, "FAIL [%s/reclaim]: arena offset 0x%llx was never reclaimed after releasing both heap and buffer\n",
                         tag, static_cast<unsigned long long>(first_va));
            ++failures;
        }
    }

    std::printf("placed heap lifetime [%s]: %u failure(s)\n", tag, failures);
    return failures;
}

}  // namespace

int main() {
    Ref<ID3D12Device> ours;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }

    unsigned failures = run_suite("leonida", ours.p, true);

    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        Ref<ID3D12Device> host;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) &&
            SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) &&
            SUCCEEDED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            failures += run_suite("WARP", host.p, false);
        }
    }

    if (failures) {
        std::fprintf(stderr, "d3d12_placed_heap_lifetime_test: FAIL (%u failures)\n", failures);
        return 1;
    }
    std::printf("d3d12_placed_heap_lifetime_test: PASS\n");
    return 0;
}

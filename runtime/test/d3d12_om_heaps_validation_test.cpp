// OMSetRenderTargets and SetDescriptorHeaps validation against Windows' own D3D12 (default adapter and WARP,
// without the debug layer) plus layer-only null-pointer/segfault guards and a positive GPU control.
// No game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

#define REQUIRE(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

struct Env {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12Resource> rt0, rt1;
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    Ref<ID3D12DescriptorHeap> cbv_sv0, cbv_sv1, sampler_sv0, sampler_sv1;
    Ref<ID3D12DescriptorHeap> cbv_cpu, sampler_cpu;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[2]{};
    UINT64 fence_value = 0;

    bool init(ID3D12Device* d) {
        device = d;
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = tex.Height = 16;
        tex.DepthOrArraySize = tex.MipLevels = 1;
        tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tex.SampleDesc.Count = 1;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                        IID_PPV_ARGS(&rt0.p))));
        CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                        IID_PPV_ARGS(&rt1.p))));

        const auto make_heap = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count, D3D12_DESCRIPTOR_HEAP_FLAGS flags,
                                   Ref<ID3D12DescriptorHeap>& out) {
            const D3D12_DESCRIPTOR_HEAP_DESC desc{type, count, flags, 0};
            return SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&out.p)));
        };
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, rtv_heap));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, dsv_heap));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, cbv_sv0));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, cbv_sv1));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, sampler_sv0));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, sampler_sv1));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, cbv_cpu));
        CHECK(make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, sampler_cpu));

        const UINT rtv_stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        rtv[0] = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        rtv[1] = {rtv[0].ptr + rtv_stride};
        device->CreateRenderTargetView(rt0.p, nullptr, rtv[0]);
        device->CreateRenderTargetView(rt1.p, nullptr, rtv[1]);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }
};

struct Case {
    const char* name;
    bool closed_first;
    std::function<void(Env&, ID3D12GraphicsCommandList*)> record;
};

std::vector<Case> parity_cases() {
    return {
        {"OM: 0 targets", false,
         [](Env&, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(0, nullptr, FALSE, nullptr); }},
        {"OM: 1 target", false,
         [](Env& e, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(1, &e.rtv[0], FALSE, nullptr); }},
        {"OM: 2 targets array", false,
         [](Env& e, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(2, e.rtv, FALSE, nullptr); }},
        {"OM: 2 targets range", false,
         [](Env& e, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(2, &e.rtv[0], TRUE, nullptr); }},
        {"OM: count=9", false,
         [](Env& e, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(9, &e.rtv[0], TRUE, nullptr); }},
        {"OM: closed list", true,
         [](Env& e, ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(1, &e.rtv[0], FALSE, nullptr); }},

        {"Heaps: num=0", false,
         [](Env&, ID3D12GraphicsCommandList* l) { l->SetDescriptorHeaps(0, nullptr); }},
        {"Heaps: 1 CBV_SRV_UAV", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_sv0.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: 1 SAMPLER", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.sampler_sv0.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: CBV + SAMPLER", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_sv0.p, e.sampler_sv0.p};
             l->SetDescriptorHeaps(2, h);
         }},
        {"Heaps: num=3", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_sv0.p, e.sampler_sv0.p, e.cbv_sv1.p};
             l->SetDescriptorHeaps(3, h);
         }},
        {"Heaps: RTV heap", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.rtv_heap.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: DSV heap", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.dsv_heap.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: CPU CBV", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_cpu.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: CPU SAMPLER", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.sampler_cpu.p};
             l->SetDescriptorHeaps(1, h);
         }},
        {"Heaps: dup CBV", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_sv0.p, e.cbv_sv1.p};
             l->SetDescriptorHeaps(2, h);
         }},
        {"Heaps: dup SAMPLER", false,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.sampler_sv0.p, e.sampler_sv1.p};
             l->SetDescriptorHeaps(2, h);
         }},
        {"Heaps: closed list", true,
         [](Env& e, ID3D12GraphicsCommandList* l) {
             ID3D12DescriptorHeap* h[] = {e.cbv_sv0.p};
             l->SetDescriptorHeaps(1, h);
         }},
    };
}

bool run_case(Env& env, const Case& c, HRESULT& out_close) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                  IID_PPV_ARGS(&list.p))));
    if (c.closed_first) CHECK(SUCCEEDED(list->Close()));
    c.record(env, list.p);
    out_close = list->Close();
    return true;
}

bool run_layer_only_guards(Env& env) {
    const auto expect_invalid = [&](const char* label, const std::function<void(ID3D12GraphicsCommandList*)>& fn) {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                      IID_PPV_ARGS(&list.p))));
        fn(list.p);
        const HRESULT hr = list->Close();
        if (hr != E_INVALIDARG) {
            std::fprintf(stderr, "layer-only guard '%s': expected E_INVALIDARG, got 0x%08lx\n",
                         label, static_cast<unsigned long>(hr));
            return false;
        }
        return true;
    };

    // Segfault regression: single_range == TRUE with targets[0].ptr == 0 and count >= 2 previously computed
    // handle = 0 + 1 * 32 = 0x20 and dereferenced 0x00000020.
    const D3D12_CPU_DESCRIPTOR_HANDLE zero_handle{0};
    CHECK(expect_invalid("OM single_range=TRUE zero handle count=2",
                         [&](ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(2, &zero_handle, TRUE, nullptr); }));
    CHECK(expect_invalid("OM single_range=TRUE zero handle count=1",
                         [&](ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(1, &zero_handle, TRUE, nullptr); }));
    CHECK(expect_invalid("OM count=1 null targets FALSE",
                         [](ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(1, nullptr, FALSE, nullptr); }));
    CHECK(expect_invalid("OM count=2 null targets TRUE",
                         [](ID3D12GraphicsCommandList* l) { l->OMSetRenderTargets(2, nullptr, TRUE, nullptr); }));
    CHECK(expect_invalid("Heaps num=1 null array",
                         [](ID3D12GraphicsCommandList* l) { l->SetDescriptorHeaps(1, nullptr); }));
    CHECK(expect_invalid("Heaps num=1 null element",
                         [](ID3D12GraphicsCommandList* l) {
                             ID3D12DescriptorHeap* h[] = {nullptr};
                             l->SetDescriptorHeaps(1, h);
                         }));
    CHECK(expect_invalid("Heaps num=2 second null",
                         [&](ID3D12GraphicsCommandList* l) {
                             ID3D12DescriptorHeap* h[] = {env.cbv_sv0.p, nullptr};
                             l->SetDescriptorHeaps(2, h);
                         }));

    // Closed list guard: calling OMSetRenderTargets / SetDescriptorHeaps on a closed list does not crash,
    // subsequent Close returns E_FAIL, and Reset recovers cleanly.
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                      IID_PPV_ARGS(&list.p))));
        CHECK(list->Close() == S_OK);
        list->OMSetRenderTargets(2, &zero_handle, TRUE, nullptr);
        list->SetDescriptorHeaps(1, nullptr);
        CHECK(list->Close() == E_FAIL);
        CHECK(list->Reset(alloc.p, nullptr) == S_OK);
        ID3D12DescriptorHeap* valid_heaps[] = {env.cbv_sv0.p, env.sampler_sv0.p};
        list->SetDescriptorHeaps(2, valid_heaps);
        list->OMSetRenderTargets(2, &env.rtv[0], TRUE, nullptr);
        CHECK(list->Close() == S_OK);
    }

    // Positive GPU control: bind valid descriptor heaps and render targets, clear both targets, execute, and sync.
    {
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        CHECK(SUCCEEDED(env.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        CHECK(SUCCEEDED(env.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                      IID_PPV_ARGS(&list.p))));
        ID3D12DescriptorHeap* valid_heaps[] = {env.cbv_sv0.p, env.sampler_sv0.p};
        list->SetDescriptorHeaps(2, valid_heaps);
        list->OMSetRenderTargets(2, &env.rtv[0], TRUE, nullptr);
        const FLOAT color0[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        const FLOAT color1[4] = {0.75f, 0.25f, 0.5f, 1.0f};
        list->ClearRenderTargetView(env.rtv[0], color0, 0, nullptr);
        list->ClearRenderTargetView(env.rtv[1], color1, 0, nullptr);
        CHECK(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        env.queue->ExecuteCommandLists(1, lists);
        CHECK(rt::d3d12::last_execute_result(env.queue.p) == leonida::ReplayResult::success);
        CHECK(SUCCEEDED(env.queue->Signal(env.fence.p, ++env.fence_value)));
        CHECK(SUCCEEDED(env.fence->SetEventOnCompletion(env.fence_value, nullptr)));
    }
    return true;
}

struct Run {
    const char* name = "";
    bool ran = false;
    std::vector<HRESULT> results;
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::fprintf(stderr, "SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }
    REQUIRE(device_status == vktest::DeviceStatus::ready);
    const unsigned errors_before = vk.validation_errors();
    const auto cases = parity_cases();

    Run layer;
    layer.name = "layer";
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        Ref<ID3D12Device> ours;
        REQUIRE(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p))));
        Env env;
        REQUIRE(env.init(ours.p));
        REQUIRE(run_layer_only_guards(env));
        for (const Case& c : cases) {
            HRESULT hr = E_FAIL;
            REQUIRE(run_case(env, c, hr));
            layer.results.push_back(hr);
        }
        layer.ran = true;
    }

    Run hosts[2];
    hosts[0].name = "default adapter";
    hosts[1].name = "WARP";
    if (!hostwine::host_is_wine()) {
        Ref<IDXGIFactory4> factory;
        REQUIRE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))));
        for (int i = 0; i < 2; ++i) {
            bool usable = true;
            Ref<ID3D12Device> host;
            Env env;
            const auto recreate = [&]() -> bool {
                env = Env{};
                host = Ref<ID3D12Device>{};
                Ref<IDXGIAdapter> adapter;
                if (i == 1 && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) return false;
                if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return false;
                return env.init(host.p);
            };
            if (!recreate()) {
                usable = false;
            } else {
                for (const Case& c : cases) {
                    if (host->GetDeviceRemovedReason() != S_OK) REQUIRE(recreate());
                    HRESULT hr = E_FAIL;
                    REQUIRE(run_case(env, c, hr));
                    hosts[i].results.push_back(hr);
                }
            }
            hosts[i].ran = usable;
        }
    }

    const auto text = [](HRESULT hr) {
        char out[16];
        std::snprintf(out, sizeof(out), "0x%08lx", static_cast<unsigned long>(hr));
        return std::string(out);
    };
    unsigned failures = 0, compared = 0, disagreed = 0;
    std::printf("%-24s | %-12s | %-15s | %-12s\n", "case", "layer", hosts[0].name, hosts[1].name);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        std::printf("%-24s | %-12s", cases[i].name, text(layer.results[i]).c_str());
        for (int h = 0; h < 2; ++h) {
            if (!hosts[h].ran)
                std::printf(" | %-12s", "(not run)");
            else
                std::printf(" | %-12s", text(hosts[h].results[i]).c_str());
        }
        std::printf("\n");

        if (hosts[0].ran && hosts[1].ran) {
            const bool h0_ok = SUCCEEDED(hosts[0].results[i]);
            const bool h1_ok = SUCCEEDED(hosts[1].results[i]);
            if (h0_ok != h1_ok) {
                ++disagreed;
                continue;
            }
            ++compared;
            if (SUCCEEDED(layer.results[i]) != h0_ok) {
                std::fprintf(stderr, "MISMATCH on '%s': layer=%s default=%s warp=%s\n",
                             cases[i].name, text(layer.results[i]).c_str(),
                             text(hosts[0].results[i]).c_str(), text(hosts[1].results[i]).c_str());
                ++failures;
            }
        } else if (hosts[0].ran || hosts[1].ran) {
            const Run& active = hosts[0].ran ? hosts[0] : hosts[1];
            ++compared;
            if (SUCCEEDED(layer.results[i]) != SUCCEEDED(active.results[i])) {
                std::fprintf(stderr, "MISMATCH on '%s': layer=%s %s=%s\n",
                             cases[i].name, text(layer.results[i]).c_str(),
                             active.name, text(active.results[i]).c_str());
                ++failures;
            }
        }
    }
    std::printf("Summary: %u cases compared, %u host-disagreed, %u mismatches, %u validation errors\n",
                compared, disagreed, failures, vk.validation_errors() - errors_before);
    REQUIRE(failures == 0);
    REQUIRE(vk.validation_errors() == errors_before);
    return 0;
}

// Milestone M0 of docs/d3d12_vulkan.md section 6, without the game: a window; the layer's DXGI factory and its one
// adapter (the Vulkan device); an IDXGISwapChain3 created by the factory on one of the layer's ID3D12CommandQueues,
// as a game creates it; 60 frames of PRESENT -> RENDER_TARGET, clear, RENDER_TARGET -> PRESENT and Present through
// the D3D12 COM surface. The last frame's back buffer is read back
// before it is presented and must hold the clear colour. No shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails the test.
// Needs a desktop session: it shows a window for about a second.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/dxgi_factory.h"
#include "../shims/dxgi_swapchain.h"

#include <array>
#include <cstdint>
#include <cstdlib>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

HWND make_window(int width, int height) {
    WNDCLASSEXW window_class{sizeof(window_class)};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW, which is narrow without UNICODE
    window_class.lpszClassName = L"LeonidaPresentTest";
    RegisterClassExW(&window_class);
    RECT rect{0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida M0: clear + present", WS_OVERLAPPEDWINDOW,
                                  CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left, rect.bottom - rect.top,
                                  nullptr, nullptr, window_class.hInstance, nullptr);
    if (window) ShowWindow(window, SW_SHOWNORMAL);
    return window;
}

void pump_messages() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

}  // namespace

int main() {
    using namespace leonida;
    constexpr UINT frames = 60, buffers = 3;
    // Exactly representable in UNORM8, so the read-back has one right answer: R 51, G 153, B 255, A 255.
    const FLOAT colour[4] = {0.2f, 0.6f, 1.0f, 1.0f};
    const std::array<std::uint8_t, 4> want_bgra{255, 153, 51, 255};

    vktest::Device d;
    EXPECT(d.create(true));
    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());
    HWND window = make_window(640, 360);
    EXPECT(window);
    pump_messages();
    const rt::dxgi::Host host{&context, d.instance_proc, d.instance, d.physical};

    unsigned mismatches = 0, pixels = 0;
    {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList7> list;
        Ref<ID3D12Fence1> fence;
        EXPECT(SUCCEEDED(rt::d3d12::create_command_queue(context, d.queue, d.family, queue_desc,
                                                          IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_allocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_command_list(D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                                         IID_PPV_ARGS(&list.p))));
        EXPECT(list->Close() == S_OK);  // created open
        EXPECT(SUCCEEDED(rt::d3d12::create_fence(context, 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = 640;
        desc.Height = 360;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = buffers;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

        // The factory: one adapter, the Vulkan device, described from Vulkan.
        Ref<IDXGIFactory5> factory;
        EXPECT(SUCCEEDED(rt::dxgi::create_factory(host, 0, IID_PPV_ARGS(&factory.p))));
        {
            Ref<IDXGIAdapter1> adapter, second, by_luid;
            EXPECT(SUCCEEDED(factory->EnumAdapters1(0, &adapter.p)));
            EXPECT(factory->EnumAdapters1(1, &second.p) == DXGI_ERROR_NOT_FOUND && !second.p);
            DXGI_ADAPTER_DESC1 adapter_desc{};
            EXPECT(SUCCEEDED(adapter->GetDesc1(&adapter_desc)));
            VkPhysicalDeviceProperties vk_properties{};
            d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &vk_properties);
            EXPECT(adapter_desc.VendorId == vk_properties.vendorID && adapter_desc.DeviceId == vk_properties.deviceID);
            EXPECT(adapter_desc.Description[0] != 0 && adapter_desc.DedicatedVideoMemory > 0);
            std::printf("adapter: %ls, %zu MB dedicated\n", adapter_desc.Description,
                        size_t(adapter_desc.DedicatedVideoMemory >> 20));
            EXPECT(SUCCEEDED(factory->EnumAdapterByLuid(adapter_desc.AdapterLuid, IID_PPV_ARGS(&by_luid.p))));
            Ref<IDXGIFactory5> parent;
            EXPECT(SUCCEEDED(adapter->GetParent(IID_PPV_ARGS(&parent.p))) && parent.p == factory.p);
            BOOL tearing = TRUE;
            EXPECT(SUCCEEDED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing,
                                                          sizeof(tearing))) && !tearing);
        }

        // What DXGI refuses for D3D12 is refused here too, before any Vulkan object is made.
        {
            Ref<IDXGISwapChain1> none;
            DXGI_SWAP_CHAIN_DESC1 bad = desc;
            bad.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;  // blt model
            EXPECT(factory->CreateSwapChainForHwnd(queue.p, window, &bad, nullptr, nullptr, &none.p) ==
                       DXGI_ERROR_INVALID_CALL && !none.p);
            bad = desc;
            bad.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;  // flip model has no sRGB back buffers
            EXPECT(factory->CreateSwapChainForHwnd(queue.p, window, &bad, nullptr, nullptr, &none.p) ==
                       DXGI_ERROR_INVALID_CALL && !none.p);
            // For D3D12 the "device" must be a command queue.
            EXPECT(factory->CreateSwapChainForHwnd(fence.p, window, &desc, nullptr, nullptr, &none.p) ==
                       DXGI_ERROR_INVALID_CALL && !none.p);
        }

        Ref<IDXGISwapChain3> swapchain;
        {
            Ref<IDXGISwapChain1> created;
            EXPECT(SUCCEEDED(factory->CreateSwapChainForHwnd(queue.p, window, &desc, nullptr, nullptr, &created.p)));
            EXPECT(SUCCEEDED(created->QueryInterface(IID_PPV_ARGS(&swapchain.p))));
            Ref<IDXGIFactory5> parent;
            EXPECT(SUCCEEDED(swapchain->GetParent(IID_PPV_ARGS(&parent.p))) && parent.p == factory.p);
        }
        DXGI_SWAP_CHAIN_DESC1 got{};
        EXPECT(SUCCEEDED(swapchain->GetDesc1(&got)));
        EXPECT(got.BufferCount == buffers && got.Format == desc.Format && got.Width && got.Height);
        RECT client{};
        GetClientRect(window, &client);
        EXPECT(got.Width == UINT(client.right - client.left) && got.Height == UINT(client.bottom - client.top));
        {
            Ref<ID3D12CommandQueue> device;
            EXPECT(SUCCEEDED(swapchain->GetDevice(IID_PPV_ARGS(&device.p))) && device.p == queue.p);
            HWND hwnd = nullptr;
            EXPECT(SUCCEEDED(swapchain->GetHwnd(&hwnd)) && hwnd == window);
            Ref<ID3D12Resource> none;
            EXPECT(swapchain->GetBuffer(buffers, IID_PPV_ARGS(&none.p)) == DXGI_ERROR_INVALID_CALL && !none.p);
        }

        Ref<ID3D12Resource> back[buffers];
        TargetView views[buffers]{};
        D3D12_CPU_DESCRIPTOR_HANDLE rtv[buffers]{};
        for (UINT i = 0; i < buffers; ++i) {
            EXPECT(SUCCEEDED(swapchain->GetBuffer(i, IID_PPV_ARGS(&back[i].p))));
            const D3D12_RESOURCE_DESC resource = back[i]->GetDesc();
            EXPECT(resource.Width == got.Width && resource.Height == got.Height && resource.Format == desc.Format);
            EXPECT(rt::dxgi::vulkan_image(swapchain.p, i) != VK_NULL_HANDLE);
            views[i] = {rt::d3d12::resource_id(back[i].p), 0, 0, 1};
            rtv[i].ptr = reinterpret_cast<SIZE_T>(&views[i]);
        }

        const VkDeviceSize bytes = VkDeviceSize(got.Width) * got.Height * 4;
        void* mapped = nullptr;
        const VkBuffer readback = d.buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
        EXPECT(readback);

        ID3D12CommandList* lists[] = {list.p};
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event);
        UINT64 fence_value = 0;
        for (UINT frame = 0; frame < frames; ++frame) {
            pump_messages();
            const UINT index = swapchain->GetCurrentBackBufferIndex();
            EXPECT(index < buffers);
            EXPECT(SUCCEEDED(allocator->Reset()));
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            transition(list.p, back[index].p, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ClearRenderTargetView(rtv[index], colour, 0, nullptr);
            transition(list.p, back[index].p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));

            if (frame == frames - 1) {
                // Before it goes to the screen: wait for the frame, then copy the back buffer out.
                EXPECT(SUCCEEDED(fence->SetEventOnCompletion(fence_value, event)));
                EXPECT(WaitForSingleObject(event, 5000) == WAIT_OBJECT_0);
                const ResourceId id = rt::d3d12::resource_id(back[index].p);
                EXPECT(context.layout(id, 0) == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
                const VkImage image = rt::dxgi::vulkan_image(swapchain.p, index);
                const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
                EXPECT(d.run([&](VkCommandBuffer cb) {
                    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.image = image;
                    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    b.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                            nullptr, 1, &b);
                    d.read_back(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, got.Width,
                                got.Height, 1, readback, 0);
                    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    b.dstAccessMask = 0;
                    barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                            nullptr, 1, &b);
                    return true;
                }));
                const auto* data = static_cast<const std::uint8_t*>(mapped);
                pixels = got.Width * got.Height;
                for (unsigned i = 0; i < pixels; ++i) {
                    if (std::equal(want_bgra.begin(), want_bgra.end(), data + i * 4)) continue;
                    if (mismatches++ < 4)
                        std::fprintf(stderr, "pixel %u: %u,%u,%u,%u\n", i, unsigned(data[i * 4]),
                                     unsigned(data[i * 4 + 1]), unsigned(data[i * 4 + 2]), unsigned(data[i * 4 + 3]));
                }
            }
            EXPECT(swapchain->Present(1, 0) == S_OK);
        }
        CloseHandle(event);

        UINT presented = 0;
        EXPECT(SUCCEEDED(swapchain->GetLastPresentCount(&presented)) && presented == frames);
        EXPECT(swapchain->Present(1, DXGI_PRESENT_TEST) == S_OK);
        EXPECT(SUCCEEDED(swapchain->GetDesc1(&got)) && got.BufferCount == buffers);
    }
    DestroyWindow(window);

    EXPECT(pixels && !mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("M0: factory, adapter and swapchain made; %u frames cleared and presented; the last back buffer "
                "(%u pixels) held the clear colour\n", frames, pixels);
    return 0;
}

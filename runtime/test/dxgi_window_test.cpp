// DXGI + PresentX over shared-device WSI (PR C, API compatibility): a real window whose frame is recorded through
// the D3D12 COM surface and presented by the DXGI swapchain and by ID3D12CommandQueue::PresentX.
//
// CreateDXGIFactory1 -> EnumAdapters1 (and EnumOutputs, which answers with the primary monitor's modes) ->
// IDXGIFactory2::CreateSwapChainForHwnd -> GetBuffer(0) gives back a D3D12 committed render target; the list records
// OMSetRenderTargets/IASetVertexBuffers/SetPipelineState/DrawInstanced and ExecuteCommandLists runs it. Present()
// hands that image to the shared-device presenter, which copies it on the GPU into the swapchain image and presents
// it. PresentX() composes: the triangle's render target goes into the left half of the swapchain and a second target
// cleared to blue into the right half, through the same presenter. Mid-run ResizeBuffers() recreates the swapchain at
// 320x240, and the end toggles SetFullscreenState on and off and presents after each.
//
// Two captured swapchain images (the first Present and the PresentX composite) prove the pixels that reached the
// screen; the first is also written to dxgi_window.bmp.
//
// No game data, no keys. Exit 77 means no Vulkan device or no window. Set LEONIDA_VK_VALIDATION=1 for validation.

#include "wsi_presenter.h"  // first: Vulkan prototypes (d3d12_com.h disables them)

#include "d3d12_draw_scene.h"
#include "d3d12_present.h"
#include "dxgi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace d3d12_scene;

int failures = 0;

#define EXPECT(condition)                                                \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                  \
        }                                                                \
    } while (0)

constexpr ULONGLONG kRunMilliseconds = 2500;
constexpr ULONGLONG kResizeAtMilliseconds = 1200;

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    switch (message) {
        case WM_DESTROY: PostQuitMessage(0); return 0;
        case WM_CLOSE: DestroyWindow(window); return 0;
        default: return DefWindowProcW(window, message, wparam, lparam);
    }
}

void pump_messages(bool& running) {
    MSG msg{};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) running = false;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

struct CaptureBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

bool create_capture(const rt::d3d12::VulkanDeviceInfo& info, VkDeviceSize size, CaptureBuffer& out) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(info.device, &buffer_info, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(info.device, out.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(info.physical, &memory);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    bool found = false;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags wanted =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
            allocate.memoryTypeIndex = i;
            found = true;
            break;
        }
    }
    if (!found || vkAllocateMemory(info.device, &allocate, nullptr, &out.memory) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(info.device, out.buffer, out.memory, 0) != VK_SUCCESS) return false;
    return vkMapMemory(info.device, out.memory, 0, size, 0, &out.mapped) == VK_SUCCESS;
}

void destroy_capture(const rt::d3d12::VulkanDeviceInfo& info, CaptureBuffer& capture) {
    if (capture.memory) {
        vkUnmapMemory(info.device, capture.memory);
        vkFreeMemory(info.device, capture.memory, nullptr);
    }
    if (capture.buffer) vkDestroyBuffer(info.device, capture.buffer, nullptr);
    capture = CaptureBuffer{};
}

struct PixelCounts {
    std::size_t orange = 0;
    std::size_t blue = 0;
    std::size_t other = 0;
};

// Counts pixels in a rectangle of a captured RGBA8 image (the swapchain format is R8G8B8A8 here).
PixelCounts count_pixels(const void* mapped, VkExtent2D extent, VkOffset2D from, VkExtent2D size) {
    PixelCounts counts{};
    if (!mapped) return counts;
    const auto* pixels = static_cast<const std::uint8_t*>(mapped);
    for (std::uint32_t y = from.y; y < from.y + size.height && y < extent.height; ++y) {
        const std::uint8_t* line = pixels + std::size_t(y) * extent.width * 4;
        for (std::uint32_t x = from.x; x < from.x + size.width && x < extent.width; ++x) {
            const std::uint8_t r = line[x * 4], g = line[x * 4 + 1], b = line[x * 4 + 2];
            if (r > 200 && g > 30 && g < 120 && b < 40) ++counts.orange;
            else if (b > 200 && r < 40 && g < 40) ++counts.blue;
            else if (r || g || b) ++counts.other;
        }
    }
    return counts;
}

bool write_bmp(const char* path, const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height) {
    const std::uint32_t row_bytes = (width * 3 + 3) & ~3u;
    const std::uint32_t image_bytes = row_bytes * height;
    std::vector<std::uint8_t> file(54 + image_bytes, 0);
    file[0] = 'B';
    file[1] = 'M';
    const std::uint32_t total = 54 + image_bytes;
    std::memcpy(file.data() + 2, &total, 4);
    const std::uint32_t offset = 54;
    std::memcpy(file.data() + 10, &offset, 4);
    const std::uint32_t header = 40;
    std::memcpy(file.data() + 14, &header, 4);
    const std::int32_t w = std::int32_t(width), h = std::int32_t(height);
    std::memcpy(file.data() + 18, &w, 4);
    std::memcpy(file.data() + 22, &h, 4);
    const std::uint16_t planes = 1, bits = 24;
    std::memcpy(file.data() + 26, &planes, 2);
    std::memcpy(file.data() + 28, &bits, 2);
    std::memcpy(file.data() + 34, &image_bytes, 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        std::uint8_t* out = file.data() + 54 + std::size_t(height - 1 - y) * row_bytes;
        const std::uint8_t* line = pixels + std::size_t(y) * width * 4;
        for (std::uint32_t x = 0; x < width; ++x) {
            out[x * 3 + 0] = line[x * 4 + 2];  // B
            out[x * 3 + 1] = line[x * 4 + 1];  // G
            out[x * 3 + 2] = line[x * 4 + 0];  // R
        }
    }
    std::FILE* out_file = std::fopen(path, "wb");
    if (!out_file) return false;
    const bool ok = std::fwrite(file.data(), 1, file.size(), out_file) == file.size();
    std::fclose(out_file);
    return ok;
}

}  // namespace

int main() {
    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_OWNDC;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
    window_class.lpszClassName = L"LeonidaDxgiWindowClass";
    if (!RegisterClassExW(&window_class)) {
        std::printf("dxgi_window_test: cannot register the window class\n");
        return 77;
    }
    RECT rect{0, 0, 640, 480};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida - DXGI + PresentX over shared-device WSI",
                                  WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left,
                                  rect.bottom - rect.top, nullptr, nullptr, window_class.hInstance, nullptr);
    if (!window) {
        std::printf("dxgi_window_test: cannot create the window\n");
        return 77;
    }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), device.as_void(), true))) {
        std::printf("dxgi_window_test: no Vulkan device\n");
        DestroyWindow(window);
        return 77;
    }
    const rt::d3d12::VulkanDeviceInfo info = rt::d3d12::vulkan_device_info(device.p);

    // The DXGI surface: factory, adapter, its output (the primary monitor), swapchain for the window.
    Ref<IDXGIFactory2> factory;
    EXPECT(SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory2), factory.as_void())));
    Ref<IDXGIAdapter1> adapter;
    if (factory.p) EXPECT(SUCCEEDED(factory->EnumAdapters1(0, adapter.out())));
    {
        Ref<IDXGIAdapter1> none;  // only one adapter: the host GPU
        EXPECT(factory.p && factory->EnumAdapters1(1, none.out()) == DXGI_ERROR_NOT_FOUND);
    }
    if (adapter.p) {
        DXGI_ADAPTER_DESC1 desc{};
        EXPECT(SUCCEEDED(adapter->GetDesc1(&desc)));
        std::printf("dxgi_window_test: adapter: %S\n", desc.Description);

        // The display: a real monitor answers with a name and the modes it supports.
        Ref<IDXGIOutput> output;
        const HRESULT enumerated = adapter->EnumOutputs(0, output.out());
        if (SUCCEEDED(enumerated)) {
            DXGI_OUTPUT_DESC output_desc{};
            EXPECT(SUCCEEDED(output->GetDesc(&output_desc)));
            EXPECT(output_desc.AttachedToDesktop);
            UINT mode_count = 0;
            EXPECT(SUCCEEDED(output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &mode_count, nullptr)));
            EXPECT(mode_count > 0);
            std::vector<DXGI_MODE_DESC> modes(mode_count);
            EXPECT(SUCCEEDED(output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &mode_count, modes.data())));
            EXPECT(mode_count > 0 && modes[0].Width > 0 && modes[0].Height > 0);
            DXGI_MODE_DESC wanted{};
            wanted.Width = 640;
            wanted.Height = 480;
            wanted.RefreshRate = {60, 1};
            wanted.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            DXGI_MODE_DESC closest{};
            EXPECT(SUCCEEDED(output->FindClosestMatchingMode(&wanted, &closest, nullptr)));
            EXPECT(closest.Width > 0 && closest.Height > 0);
            EXPECT(SUCCEEDED(output->WaitForVBlank()));
            std::printf("dxgi_window_test: output %S, %u modes, closest to 640x480 is %ux%u\n", output_desc.DeviceName,
                        mode_count, closest.Width, closest.Height);
        } else {
            std::printf("dxgi_window_test: no display output (headless); the output checks are skipped\n");
        }
        Ref<IDXGIOutput> extra;
        EXPECT(adapter->EnumOutputs(1, extra.out()) == DXGI_ERROR_NOT_FOUND);
    }

    RECT client{};
    GetClientRect(window, &client);
    DXGI_SWAP_CHAIN_DESC1 swap_desc{};
    swap_desc.Width = std::uint32_t(client.right);
    swap_desc.Height = std::uint32_t(client.bottom);
    swap_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swap_desc.SampleDesc = {1, 0};
    swap_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_desc.BufferCount = 2;
    swap_desc.Scaling = DXGI_SCALING_STRETCH;
    swap_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swap_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    Ref<IDXGISwapChain1> swapchain;
    EXPECT(SUCCEEDED(factory->CreateSwapChainForHwnd(reinterpret_cast<IUnknown*>(device.p), window, &swap_desc,
                                                     nullptr, nullptr, swapchain.out())));
    if (!swapchain.p) {
        DestroyWindow(window);
        return 1;
    }
    DXGI_SWAP_CHAIN_DESC1 reported{};
    EXPECT(SUCCEEDED(swapchain->GetDesc1(&reported)));
    EXPECT(reported.Width == swap_desc.Width && reported.Height == swap_desc.Height);
    HWND reported_window = nullptr;
    EXPECT(SUCCEEDED(swapchain->GetHwnd(&reported_window)) && reported_window == window);

    // Back buffer, views, pipeline and vertex buffer: the frame the game would record.
    Ref<ID3D12Resource> backbuffer;
    EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 2;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, __uuidof(ID3D12DescriptorHeap), rtv_heap.as_void())));
    const UINT rtv_stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv2 = D3D12_CPU_DESCRIPTOR_HANDLE{rtv.ptr + rtv_stride};
    if (backbuffer.p && rtv_heap.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);

    // The second render target of the composite: a plain blue plane covering the right half of the window.
    const D3D12_RESOURCE_DESC overlay_desc =
        texture_desc(swap_desc.Width, swap_desc.Height, DXGI_FORMAT_R8G8B8A8_UNORM,
                     D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> overlay;
    EXPECT(SUCCEEDED(create_texture(device.p, overlay_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, overlay.out())));
    if (overlay.p && rtv_heap.p) device->CreateRenderTargetView(overlay.p, nullptr, rtv2);

    Ref<ID3D12PipelineState> pipeline;
    EXPECT(SUCCEEDED(create_pipeline(device.p, DXGI_FORMAT_R8G8B8A8_UNORM, pipeline.out())));
    Ref<ID3D12Resource> vertex_buffer;
    EXPECT(SUCCEEDED(create_buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, sizeof(kTriangle), vertex_buffer.out())));
    if (vertex_buffer.p) {
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(vertex_buffer->Map(0, nullptr, &mapped)));
        if (mapped) std::memcpy(mapped, kTriangle, sizeof(kTriangle));
        vertex_buffer->Unmap(0, nullptr);
    }
    D3D12_VERTEX_BUFFER_VIEW vertex_view{};
    if (vertex_buffer.p) {
        vertex_view.BufferLocation = vertex_buffer->GetGPUVirtualAddress();
        vertex_view.SizeInBytes = sizeof(kTriangle);
        vertex_view.StrideInBytes = 2 * sizeof(float);
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, __uuidof(ID3D12CommandQueue), queue.as_void())));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                    __uuidof(ID3D12CommandAllocator), allocator.as_void())));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               __uuidof(ID3D12GraphicsCommandList), list.as_void())));
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), fence.as_void())));
    UINT64 fence_value = 0;
    bool list_open = true;  // a list starts open; Reset only applies from the second recording on

    // The captures: the first Present and the PresentX composite (the test's only CPU readback).
    CaptureBuffer capture, composite;
    EXPECT(create_capture(info, VkDeviceSize(swap_desc.Width) * swap_desc.Height * 4, capture));
    EXPECT(create_capture(info, VkDeviceSize(swap_desc.Width) * swap_desc.Height * 4, composite));
    leonida::wsi::Presenter* presenter = rt::d3d12::presentation_target();
    EXPECT(presenter != nullptr);

    // One recorded frame: the triangle into the swapchain's back buffer.
    const auto record_triangle = [&] {
        if (!list_open) {
            EXPECT(SUCCEEDED(allocator->Reset()));
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        }
        list_open = false;
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->IASetVertexBuffers(0, 1, &vertex_view);
        list->SetPipelineState(pipeline.p);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == S_OK);
        list_open = false;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        return wait_for(queue.p, fence.p, ++fence_value);
    };
    // The overlay plane: cleared to blue, no draw.
    const auto record_overlay = [&] {
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        const FLOAT blue[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        list->ClearRenderTargetView(rtv2, blue, 0, nullptr);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        return wait_for(queue.p, fence.p, ++fence_value);
    };

    // 1. A plain Present, captured: the triangle covers part of the zero-filled back buffer.
    EXPECT(record_triangle());
    if (presenter && capture.buffer) presenter->capture_next_frame(capture.buffer);
    EXPECT(SUCCEEDED(swapchain->Present(0, 0)));
    vkDeviceWaitIdle(info.device);
    const PixelCounts whole = count_pixels(capture.mapped, presenter ? presenter->extent() : VkExtent2D{},
                                           {0, 0}, presenter ? presenter->extent() : VkExtent2D{});
    std::printf("dxgi_window_test: first Present: %zu orange, %zu other of %u\n", whole.orange, whole.other,
                swap_desc.Width * swap_desc.Height);
    EXPECT(whole.orange > 0);
    EXPECT(whole.other == 0);
    if (capture.mapped && whole.orange)
        write_bmp("dxgi_window.bmp", static_cast<const std::uint8_t*>(capture.mapped), swap_desc.Width,
                  swap_desc.Height);

    // 2. The PresentX composite: triangle on the left half, blue overlay on the right half, one call.
    EXPECT(record_overlay());
    if (presenter && composite.buffer) presenter->capture_next_frame(composite.buffer);
    const std::uint32_t half = swap_desc.Width / 2;
    const rt::d3d12::PresentPlane planes[] = {
        {backbuffer.p, {}, {0, 0, LONG(half), LONG(swap_desc.Height)}},
        {overlay.p, {}, {LONG(half), 0, LONG(swap_desc.Width), LONG(swap_desc.Height)}},
    };
    EXPECT(SUCCEEDED(rt::d3d12::present_x(queue.p, planes)));
    vkDeviceWaitIdle(info.device);
    const VkExtent2D extent = presenter ? presenter->extent() : VkExtent2D{};
    const PixelCounts left = count_pixels(composite.mapped, extent, {0, 0}, {half, swap_desc.Height});
    const PixelCounts right =
        count_pixels(composite.mapped, extent, {LONG(half), 0}, {swap_desc.Width - half, swap_desc.Height});
    std::printf("dxgi_window_test: PresentX composite: left %zu orange / %zu other, right %zu blue / %zu orange / "
                "%zu other\n",
                left.orange, left.other, right.blue, right.orange, right.other);
    EXPECT(left.orange > 0);      // the triangle reached the left half
    EXPECT(left.other == 0);
    EXPECT(right.blue > 0);       // the overlay covers the right half
    EXPECT(right.orange == 0);    // and the triangle did not spill into it
    EXPECT(right.other == 0);

    // 3. Frames for a while: Present normally, and five frames through PresentX (a single plane, the old path).
    int frames = 0, present_x_frames = 0;
    bool resized = false;
    const ULONGLONG start = GetTickCount64();
    bool running = true;
    while (running) {
        pump_messages(running);
        const ULONGLONG elapsed = GetTickCount64() - start;
        if (!running || elapsed > kRunMilliseconds) break;

        // Resize once, half-way: the window's client area becomes 320x240 (as a game does on WM_SIZE), then the
        // swapchain is recreated at that size and the back buffer is re-acquired.
        if (!resized && elapsed > kResizeAtMilliseconds) {
            resized = true;
            RECT outer{0, 0, 320, 240};
            AdjustWindowRect(&outer, WS_OVERLAPPEDWINDOW, FALSE);
            SetWindowPos(window, nullptr, 0, 0, outer.right - outer.left, outer.bottom - outer.top,
                         SWP_NOMOVE | SWP_NOZORDER);
            pump_messages(running);
            EXPECT(SUCCEEDED(swapchain->ResizeBuffers(2, 320, 240, DXGI_FORMAT_UNKNOWN, 0)));
            DXGI_SWAP_CHAIN_DESC1 after{};
            EXPECT(SUCCEEDED(swapchain->GetDesc1(&after)));
            EXPECT(after.Width == 320 && after.Height == 240);
            backbuffer.p->Release();
            backbuffer.p = nullptr;
            EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
            if (backbuffer.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);
        }

        if (!record_triangle()) break;
        HRESULT presented = E_FAIL;
        if (frames >= 3 && frames < 8) {
            const rt::d3d12::PresentPlane single[] = {{backbuffer.p, {}, {}}};
            presented = rt::d3d12::present_x(queue.p, single);
            if (SUCCEEDED(presented)) ++present_x_frames;
        } else {
            presented = swapchain->Present(0, 0);
        }
        if (FAILED(presented)) {
            std::fprintf(stderr, "dxgi_window_test: present failed (0x%08lx)\n", static_cast<unsigned long>(presented));
            break;
        }
        ++frames;
    }
    pump_messages(running);
    vkDeviceWaitIdle(info.device);

    // 4. Fullscreen on and off: both recreate the swapchain at a new size, and a frame is presented after each.
    if (swapchain.p) {
        EXPECT(SUCCEEDED(swapchain->SetFullscreenState(TRUE, nullptr)));
        BOOL fullscreen = FALSE;
        EXPECT(SUCCEEDED(swapchain->GetFullscreenState(&fullscreen, nullptr)) && fullscreen);
        backbuffer.p->Release();
        backbuffer.p = nullptr;
        EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
        if (backbuffer.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);
        if (backbuffer.p) {
            record_triangle();
            EXPECT(SUCCEEDED(swapchain->Present(0, 0)));
        }
        EXPECT(SUCCEEDED(swapchain->SetFullscreenState(FALSE, nullptr)));
        EXPECT(SUCCEEDED(swapchain->GetFullscreenState(&fullscreen, nullptr)) && !fullscreen);
        backbuffer.p->Release();
        backbuffer.p = nullptr;
        EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
        if (backbuffer.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);
        if (backbuffer.p) {
            record_triangle();
            EXPECT(SUCCEEDED(swapchain->Present(0, 0)));
        }
        vkDeviceWaitIdle(info.device);
    }

    UINT last_present = 0;
    swapchain->GetLastPresentCount(&last_present);
    std::printf("dxgi_window_test: %d frames (%d via PresentX), last present count %u\n", frames, present_x_frames,
                last_present);

    std::FILE* log = std::fopen("dxgi_window.log", "w");
    if (log) {
        std::fprintf(log, "dxgi=1 present_x=%d frames=%d orange=%zu composite=%zu/%zu presents=%u\n",
                     present_x_frames, frames, whole.orange, left.orange, right.blue, last_present);
        std::fclose(log);
    }

    destroy_capture(info, capture);
    destroy_capture(info, composite);
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    EXPECT(frames > 0);
    EXPECT(present_x_frames == 5);
    EXPECT(last_present > 0);

    if (failures) {
        std::fprintf(stderr, "dxgi_window_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dxgi_window_test: all tests passed\n");
    return 0;
}

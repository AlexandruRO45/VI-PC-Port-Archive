// DXGI + PresentX over shared-device WSI (PR C, API compatibility): a real window whose frame is recorded through
// the D3D12 COM surface and presented by the DXGI swapchain and, for a few frames, by ID3D12CommandQueue::PresentX.
//
// CreateDXGIFactory1 -> EnumAdapters1 -> IDXGIFactory2::CreateSwapChainForHwnd -> GetBuffer(0) gives back a D3D12
// committed render target; the list records OMSetRenderTargets/IASetVertexBuffers/SetPipelineState/DrawInstanced and
// ExecuteCommandLists runs it. Present() hands that image to the shared-device presenter, which copies it on the GPU
// into the swapchain image and presents it. PresentX() does the same through the layer's Xbox present path. Mid-run
// ResizeBuffers() recreates the swapchain at 320x240 and the loop keeps drawing and presenting there.
//
// One captured swapchain image (the first present) proves the pixels that reached the screen: the triangle's orange
// on the zero-filled target.
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

constexpr ULONGLONG kRunMilliseconds = 3000;
constexpr ULONGLONG kResizeAtMilliseconds = 1500;

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

// The captured frame as a BGRA BMP, so the presented image can be looked at outside the test.
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
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida - DXGI swapchain over shared-device WSI",
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

    // The DXGI surface: factory, adapter, swapchain for the window.
    Ref<IDXGIFactory2> factory;
    EXPECT(SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory2), factory.as_void())));
    Ref<IDXGIAdapter1> adapter;
    if (factory.p) EXPECT(SUCCEEDED(factory->EnumAdapters1(0, adapter.out())));
    if (adapter.p) {
        DXGI_ADAPTER_DESC1 desc{};
        EXPECT(SUCCEEDED(adapter->GetDesc1(&desc)));
        std::printf("dxgi_window_test: adapter: %S\n", desc.Description);
    }
    {
        Ref<IDXGIAdapter1> none;  // only one adapter: the host GPU
        EXPECT(factory.p && factory->EnumAdapters1(1, none.out()) == DXGI_ERROR_NOT_FOUND);
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

    // Back buffer, view, pipeline, vertex buffer: the frame the game would record.
    Ref<ID3D12Resource> backbuffer;
    EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, __uuidof(ID3D12DescriptorHeap), rtv_heap.as_void())));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    if (backbuffer.p && rtv_heap.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);

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

    // The capture of the first presented frame (the test's only CPU readback).
    CaptureBuffer capture;
    EXPECT(create_capture(info, VkDeviceSize(swap_desc.Width) * swap_desc.Height * 4, capture));
    leonida::wsi::Presenter* presenter = rt::d3d12::presentation_target();
    EXPECT(presenter != nullptr);
    if (presenter && capture.buffer) presenter->capture_next_frame(capture.buffer);

    int frames = 0, present_x_frames = 0;
    bool resized = false;
    UINT64 fence_value = 0;
    const ULONGLONG start = GetTickCount64();
    bool running = true;
    while (running) {
        pump_messages(running);
        const ULONGLONG elapsed = GetTickCount64() - start;
        if (!running || elapsed > kRunMilliseconds) break;

        // Resize once, halfway: the swapchain recreates itself at 320x240 and the back buffer is re-acquired.
        if (!resized && elapsed > kResizeAtMilliseconds) {
            resized = true;
            EXPECT(SUCCEEDED(swapchain->ResizeBuffers(2, 320, 240, DXGI_FORMAT_UNKNOWN, 0)));
            DXGI_SWAP_CHAIN_DESC1 after{};
            EXPECT(SUCCEEDED(swapchain->GetDesc1(&after)));
            EXPECT(after.Width == 320 && after.Height == 240);
            backbuffer.p->Release();
            backbuffer.p = nullptr;
            EXPECT(SUCCEEDED(swapchain->GetBuffer(0, __uuidof(ID3D12Resource), backbuffer.as_void())));
            if (backbuffer.p) device->CreateRenderTargetView(backbuffer.p, nullptr, rtv);
        }

        // One frame of recording into the swapchain's back buffer.
        if (frames > 0) {
            EXPECT(SUCCEEDED(allocator->Reset()));
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        }
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->IASetVertexBuffers(0, 1, &vertex_view);
        list->SetPipelineState(pipeline.p);
        list->DrawInstanced(3, 1, 0, 0);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (!wait_for(queue.p, fence.p, ++fence_value)) break;

        // Frames 4..8 go through PresentX instead of the swapchain, to exercise the Xbox present path.
        const bool via_present_x = frames >= 4 && frames < 9;
        HRESULT presented = E_FAIL;
        if (via_present_x) {
            const rt::d3d12::PresentPlane planes[] = {{backbuffer.p, {}, {}}};
            presented = rt::d3d12::present_x(queue.p, planes);
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

    // What the first present put on the swapchain: the triangle on the zero-filled target.
    std::size_t orange = 0, lit = 0, other = 0;
    if (capture.mapped) {
        const auto* pixels = static_cast<const std::uint8_t*>(capture.mapped);
        for (std::uint32_t i = 0; i < swap_desc.Width * swap_desc.Height; ++i) {
            const std::uint8_t r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
            if (r || g || b) ++lit;
            if (r > 200 && g > 30 && g < 120 && b < 40) ++orange;
            else if (r || g || b) ++other;
        }
    }
    UINT last_present = 0;
    swapchain->GetLastPresentCount(&last_present);
    if (capture.mapped && orange)
        write_bmp("dxgi_window.bmp", static_cast<const std::uint8_t*>(capture.mapped), swap_desc.Width,
                  swap_desc.Height);
    std::printf("dxgi_window_test: %d frames (%d via PresentX), %zu/%u captured pixels lit (%zu orange), last "
                "present count %u\n",
                frames, present_x_frames, lit, swap_desc.Width * swap_desc.Height, orange, last_present);

    std::FILE* log = std::fopen("dxgi_window.log", "w");
    if (log) {
        std::fprintf(log, "dxgi=1 present_x=%d frames=%d orange=%zu lit=%zu presents=%u\n", present_x_frames, frames,
                     orange, lit, last_present);
        std::fclose(log);
    }

    destroy_capture(info, capture);
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    EXPECT(frames > 0);
    EXPECT(present_x_frames == 5);   // the PresentX frames all went through the layer's present path
    EXPECT(orange > 0);              // the DXGI present put the triangle on the swapchain image
    EXPECT(other == 0);
    EXPECT(last_present > 0);

    if (failures) {
        std::fprintf(stderr, "dxgi_window_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dxgi_window_test: all tests passed\n");
    return 0;
}

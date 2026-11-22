// A real window showing a frame the D3D12 layer rendered (PR C, step 5): shared-device WSI.
//
// The device is created presentable (VK_KHR_surface / VK_KHR_win32_surface on the instance, VK_KHR_swapchain on the
// device), an orange triangle is recorded and executed through the COM command list into a committed render target,
// and leonida::wsi::Presenter moves that image into the swapchain image on the GPU (vkCmdCopyImage, or
// vkCmdBlitImage when the surface's format differs) and calls vkQueuePresentKHR. Nothing travels through the host in
// the bridge; the one CPU readback in this test only exists to check the pixels that reached the swapchain, and the
// captured frame is written to window_draw.bmp as evidence.
//
// No game data, no keys. Exit 77 means the machine has no usable Vulkan device or window. Set
// LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.

#include "wsi_presenter.h"  // first: the Vulkan prototypes the presenter uses

#include "d3d12_draw_scene.h"

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

// Writes the captured (top-down) frame as a BGRA BMP, so the presented image can be looked at outside the test.
bool write_bmp(const char* path, const std::uint8_t* pixels, std::uint32_t width, std::uint32_t height,
               bool source_is_rgba) {
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
            const std::uint8_t r = line[x * 4], g = line[x * 4 + 1], b = line[x * 4 + 2];
            out[x * 3 + 0] = source_is_rgba ? b : line[x * 4 + 2];
            out[x * 3 + 1] = g;
            out[x * 3 + 2] = source_is_rgba ? r : line[x * 4 + 0];
        }
    }
    std::FILE* file_out = std::fopen(path, "wb");
    if (!file_out) return false;
    const bool ok = std::fwrite(file.data(), 1, file.size(), file_out) == file.size();
    std::fclose(file_out);
    return ok;
}

// A host-visible buffer the presenter copies a presented swapchain image into (the test's only CPU readback).
struct CaptureBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

bool create_capture(const leonida::wsi::Gpu& gpu, VkDeviceSize size, CaptureBuffer& out) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(gpu.device, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(gpu.device, out.buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(gpu.physical, &memory);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    bool found = false;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
            allocate.memoryTypeIndex = i;
            found = true;
            break;
        }
    }
    if (!found || vkAllocateMemory(gpu.device, &allocate, nullptr, &out.memory) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(gpu.device, out.buffer, out.memory, 0) != VK_SUCCESS) return false;
    out.size = size;
    return vkMapMemory(gpu.device, out.memory, 0, size, 0, &out.mapped) == VK_SUCCESS;
}

void destroy_capture(const leonida::wsi::Gpu& gpu, CaptureBuffer& capture) {
    if (capture.memory) {
        vkUnmapMemory(gpu.device, capture.memory);
        vkFreeMemory(gpu.device, capture.memory, nullptr);
    }
    if (capture.buffer) vkDestroyBuffer(gpu.device, capture.buffer, nullptr);
    capture = CaptureBuffer{};
}

}  // namespace

int main() {
    WNDCLASSEXW window_class{};
    window_class.cbSize = sizeof(window_class);
    window_class.style = CS_OWNDC;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
    window_class.lpszClassName = L"LeonidaWindowDrawClass";
    if (!RegisterClassExW(&window_class)) {
        std::printf("window_draw_test: cannot register the window class\n");
        return 77;
    }
    RECT rect{0, 0, 640, 480};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida - D3D12 COM frame over shared-device WSI",
                                  WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left,
                                  rect.bottom - rect.top, nullptr, nullptr, window_class.hInstance, nullptr);
    if (!window) {
        std::printf("window_draw_test: cannot create the window\n");
        return 77;
    }
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);

    // The rendering device, created presentable because the presenter shares it.
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), device.as_void(), true))) {
        std::printf("window_draw_test: no Vulkan device\n");
        DestroyWindow(window);
        return 77;
    }
    const rt::d3d12::VulkanDeviceInfo info = rt::d3d12::vulkan_device_info(device.p);
    const leonida::wsi::Gpu gpu{info.get_instance_proc, info.instance, info.physical, info.device, info.queue,
                                info.queue_family};
    EXPECT(gpu.device != VK_NULL_HANDLE);

    RECT client{};
    GetClientRect(window, &client);
    leonida::wsi::Presenter presenter;
    if (!presenter.create(gpu, window, VK_FORMAT_R8G8B8A8_UNORM,
                          VkExtent2D{std::uint32_t(client.right), std::uint32_t(client.bottom)})) {
        std::fprintf(stderr, "window_draw_test: the presenter could not be created\n");
        DestroyWindow(window);
        return 1;
    }
    const VkExtent2D extent = presenter.extent();
    const bool rgba_surface = presenter.format() == VK_FORMAT_R8G8B8A8_UNORM ||
                              presenter.format() == VK_FORMAT_R8G8B8A8_SRGB;
    std::printf("window_draw_test: swapchain %ux%u, %u images, format %d\n", extent.width, extent.height,
                presenter.image_count(), int(presenter.format()));

    // The render target the D3D12 list draws into, its view, the pipeline and the vertex buffer.
    const D3D12_RESOURCE_DESC rt_desc =
        texture_desc(extent.width, extent.height, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Ref<ID3D12Resource> render_target;
    EXPECT(SUCCEEDED(create_texture(device.p, rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, render_target.out())));
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
    rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_heap_desc.NumDescriptors = 1;
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, __uuidof(ID3D12DescriptorHeap), rtv_heap.as_void())));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    if (render_target.p && rtv_heap.p) device->CreateRenderTargetView(render_target.p, nullptr, rtv);

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

    CaptureBuffer capture;
    EXPECT(create_capture(gpu, VkDeviceSize(extent.width) * extent.height * 4, capture));

    int frames = 0;
    bool captured = false;
    std::size_t orange = 0, lit = 0, other = 0;
    UINT64 fence_value = 0;
    const ULONGLONG start = GetTickCount64();
    bool running = true;
    while (running) {
        pump_messages(running);
        if (!running || GetTickCount64() - start > kRunMilliseconds) break;

        // One frame of the game's recording: clear nothing, draw the triangle into the committed render target. A list
        // starts open, so the first frame records without Reset (D3D12's own rules).
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
        if (frames == 0) EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        if (!wait_for(queue.p, fence.p, ++fence_value)) break;

        // The bridge: the layer left the render target in the attachment layout, the presenter copies it into the
        // swapchain image on the GPU and presents it, then puts the layout back where the layer expects it.
        const VkImageLayout source_layout = rt::d3d12::vulkan_image_layout(device.p, render_target.p);
        if (frames == 0 && capture.buffer) presenter.capture_next_frame(capture.buffer);
        if (!presenter.present(rt::d3d12::vulkan_image(render_target.p), VK_FORMAT_R8G8B8A8_UNORM, extent,
                               source_layout))
            break;
        if (frames == 0) captured = true;
        ++frames;
    }
    pump_messages(running);
    vkDeviceWaitIdle(gpu.device);

    // What reached the swapchain: the pixels of the frame the presenter captured before presenting it.
    if (captured && capture.mapped) {
        const auto* pixels = static_cast<const std::uint8_t*>(capture.mapped);
        for (std::uint32_t y = 0; y < extent.height; ++y) {
            const std::uint8_t* line = pixels + std::size_t(y) * extent.width * 4;
            for (std::uint32_t x = 0; x < extent.width; ++x) {
                const std::uint8_t c0 = line[x * 4], g = line[x * 4 + 1], c2 = line[x * 4 + 2];
                const std::uint8_t r = rgba_surface ? c0 : c2, b = rgba_surface ? c2 : c0;
                if (r || g || b) ++lit;
                if (r > 200 && g > 30 && g < 120 && b < 40) ++orange;  // the shader's vec4(1.0, 0.25, 0.0, 1.0)
                else if (r || g || b) ++other;
            }
        }
        write_bmp("window_draw.bmp", pixels, extent.width, extent.height, rgba_surface);
    }
    std::printf("window_draw_test: %d frames presented, %zu/%u captured pixels lit, %zu orange, %zu other\n", frames,
                lit, extent.width * extent.height, orange, other);

    std::FILE* log = std::fopen("window_draw.log", "w");
    if (log) {
        std::fprintf(log, "wsi=shared-device frames=%d width=%u height=%u format=%d images=%u orange=%zu lit=%zu\n",
                     frames, extent.width, extent.height, int(presenter.format()), presenter.image_count(), orange,
                     lit);
        std::fclose(log);
    }

    destroy_capture(gpu, capture);
    DestroyWindow(window);
    UnregisterClassW(window_class.lpszClassName, window_class.hInstance);

    EXPECT(frames > 0);
    EXPECT(presenter.frames_presented() > 0);
    EXPECT(lit > 0);
    EXPECT(orange > 0);
    EXPECT(other == 0);  // every lit pixel is the triangle: the bridge copied the whole image, not garbage

    if (failures) {
        std::fprintf(stderr, "window_draw_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("window_draw_test: all tests passed\n");
    return 0;
}

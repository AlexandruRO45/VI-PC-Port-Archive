// Shim for the PE-loader hello test: d3d12.dll!LeonidaRenderTriangle draws the hand-written triangle through the
// Vulkan backend and writes hello_triangle.bmp. The shim registry makes the import resolve to this function, so the
// test proves: PE mapped by rt::Loader -> import bound through the shim registry -> Vulkan render -> visible output.

#include "vk_test_device.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "first_draw_shaders.h"
#include <windows.h>

#include "shims.h"
#include "d3d12_com.h"
#include <d3d12.h>
#include "../test/animated_shaders.h"

extern "C" int __cdecl leonida_run_animation(void* hwnd);

namespace {

constexpr std::uint32_t kSize = 64;

bool write_bmp(const std::filesystem::path& path, const std::uint8_t* rgba, std::uint32_t width,
               std::uint32_t height) {
    const std::uint32_t row_bytes = (width * 3 + 3) & ~3u;
    const std::uint32_t image_bytes = row_bytes * height;
    const std::uint32_t file_bytes = 54 + image_bytes;
    std::vector<std::uint8_t> data(file_bytes, 0);
    data[0] = 'B';
    data[1] = 'M';
    std::memcpy(&data[2], &file_bytes, 4);
    const std::uint32_t offset = 54;
    std::memcpy(&data[10], &offset, 4);
    const std::uint32_t header_size = 40;
    std::memcpy(&data[14], &header_size, 4);
    std::memcpy(&data[18], &width, 4);
    std::memcpy(&data[22], &height, 4);
    const std::uint16_t planes = 1, bpp = 24;
    std::memcpy(&data[26], &planes, 2);
    std::memcpy(&data[28], &bpp, 2);
    std::memcpy(&data[34], &image_bytes, 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        std::uint8_t* row = data.data() + offset + std::size_t(height - 1 - y) * row_bytes;
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint8_t* pixel = rgba + (std::size_t(y) * width + x) * 4;
            row[x * 3 + 0] = pixel[2];
            row[x * 3 + 1] = pixel[1];
            row[x * 3 + 2] = pixel[0];
        }
    }
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    return ok;
}

int render_triangle() {
    vktest::Device device;
    if (!device.create()) return 1;

    VkPhysicalDeviceMemoryProperties memory{};
    device.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(
        device.physical, &memory);

    const VkImage image = device.image(VK_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (!image) return 2;
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView view = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateImageView>("vkCreateImageView")(device.device, &view_info, nullptr, &view) != VK_SUCCESS)
        return 3;

    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
    module_info.codeSize = sizeof(first_draw::triangle_vert_spv);
    module_info.pCode = first_draw::triangle_vert_spv;
    if (device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &module_info, nullptr, &vert) !=
        VK_SUCCESS)
        return 4;
    module_info.codeSize = sizeof(first_draw::triangle_frag_spv);
    module_info.pCode = first_draw::triangle_frag_spv;
    if (device.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device.device, &module_info, nullptr, &frag) !=
        VK_SUCCESS)
        return 5;

    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device.device, &layout_info, nullptr,
                                                                        &layout) != VK_SUCCESS)
        return 6;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    VkVertexInputBindingDescription binding{0, 2 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attribute{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = 1;
    vertex_input.pVertexAttributeDescriptions = &attribute;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &color_format;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport_state;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (device.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(device.device, VK_NULL_HANDLE, 1, &info,
                                                                            nullptr, &pipeline) != VK_SUCCESS)
        return 7;

    const float vertices[6] = {-0.8f, -0.8f, 0.8f, -0.8f, 0.0f, 0.8f};
    void* vertex_data = nullptr;
    const VkBuffer vertex_buffer = device.buffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vertex_data);
    if (!vertex_buffer) return 8;
    std::memcpy(vertex_data, vertices, sizeof(vertices));

    void* mapped = nullptr;
    const VkBuffer readback = device.buffer(VkDeviceSize(kSize) * kSize * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
    if (!readback) return 9;

    const bool recorded = device.run([&](VkCommandBuffer command_buffer) {
        device.initial_layout(command_buffer, image, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.clearValue = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
        VkRenderingInfo render_info{VK_STRUCTURE_TYPE_RENDERING_INFO};
        render_info.renderArea = {{0, 0}, {kSize, kSize}};
        render_info.layerCount = 1;
        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &color;
        device.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering")(command_buffer, &render_info);
        device.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkViewport viewport{0.0f, 0.0f, float(kSize), float(kSize), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, {kSize, kSize}};
        device.fn<PFN_vkCmdSetViewport>("vkCmdSetViewport")(command_buffer, 0, 1, &viewport);
        device.fn<PFN_vkCmdSetScissor>("vkCmdSetScissor")(command_buffer, 0, 1, &scissor);
        const VkDeviceSize offset = 0;
        device.fn<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers")(command_buffer, 0, 1, &vertex_buffer, &offset);
        device.fn<PFN_vkCmdDraw>("vkCmdDraw")(command_buffer, 3, 1, 0, 0);
        device.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering")(command_buffer);
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        device.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
            command_buffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
            nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kSize, kSize, 1};
        device.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(
            command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &region);
        return true;
    });
    if (!recorded) return 10;
    if (!write_bmp("hello_triangle.bmp", static_cast<const std::uint8_t*>(mapped), kSize, kSize)) return 11;

    device.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device.device, pipeline, nullptr);
    device.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device.device, layout, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, vert, nullptr);
    device.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device.device, frag, nullptr);
    device.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(device.device, view, nullptr);
    return 0;
}

}  // namespace

// Loads the BMP the renderer wrote into a 32-bit pixel buffer (top-down RGBA).
bool load_bmp_pixels(const char* path, std::vector<std::uint8_t>& pixels, int& width, int& height) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::uint8_t header[54]{};
    if (std::fread(header, 1, 54, f) != 54) { std::fclose(f); return false; }
    std::memcpy(&width, header + 18, 4);
    std::memcpy(&height, header + 22, 4);
    const std::uint32_t offset = *reinterpret_cast<const std::uint32_t*>(header + 10);
    const std::uint32_t row_bytes = (std::uint32_t(width) * 3 + 3) & ~3u;
    std::fseek(f, long(offset), SEEK_SET);
    std::vector<std::uint8_t> bgr(std::size_t(row_bytes) * height);
    if (std::fread(bgr.data(), 1, bgr.size(), f) != bgr.size()) { std::fclose(f); return false; }
    std::fclose(f);
    pixels.resize(std::size_t(width) * height * 4);
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = bgr.data() + std::size_t(height - 1 - y) * row_bytes;
        for (int x = 0; x < width; ++x) {
            pixels[(std::size_t(y) * width + x) * 4 + 0] = row[x * 3 + 2];
            pixels[(std::size_t(y) * width + x) * 4 + 1] = row[x * 3 + 1];
            pixels[(std::size_t(y) * width + x) * 4 + 2] = row[x * 3 + 0];
            pixels[(std::size_t(y) * width + x) * 4 + 3] = 255;
        }
    }
    return true;
}

// Creates a real Win32 window, blits the rendered frame into it for a second (the message loop keeps it alive), and
// captures the window back into window_capture.bmp. No Vulkan swapchain yet: this is a GDI presenter over the same
// Vulkan-rendered pixels.
int present_window() {
    if (render_triangle() != 0) return 1;
    std::vector<std::uint8_t> pixels;
    int width = 0, height = 0;
    if (!load_bmp_pixels("hello_triangle.bmp", pixels, width, height)) return 2;

    WNDCLASSEXA wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.lpszClassName = "LeonidaHelloClass";
    if (!RegisterClassExA(&wc)) return 3;
    HWND hwnd = CreateWindowExA(0, wc.lpszClassName, "Leonida - Hello Triangle (PE loader)", WS_OVERLAPPEDWINDOW,
                                CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) return 4;
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    int frames = 0;
    const DWORD start = GetTickCount();
    while (GetTickCount() - start < 1500) {
        MSG msg{};
        while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        HDC dc = GetDC(hwnd);
        StretchDIBits(dc, 0, 0, 640, 480, 0, 0, width, height, pixels.data(), &bmi, DIB_RGB_COLORS, SRCCOPY);
        ReleaseDC(hwnd, dc);
        ++frames;
        Sleep(16);
    }

    // Capture the window's client area back to a BMP: the proof that the pixels reached the screen surface.
    HDC window_dc = GetDC(hwnd);
    HDC memory_dc = CreateCompatibleDC(window_dc);
    HBITMAP bitmap = CreateCompatibleBitmap(window_dc, 640, 480);
    HGDIOBJ old = SelectObject(memory_dc, bitmap);
    BitBlt(memory_dc, 0, 0, 640, 480, window_dc, 0, 0, SRCCOPY);
    BITMAPINFO capture_info = bmi;
    capture_info.bmiHeader.biWidth = 640;
    capture_info.bmiHeader.biHeight = 480;
    std::vector<std::uint8_t> captured(std::size_t(640) * 480 * 4);
    GetDIBits(memory_dc, bitmap, 0, 480, captured.data(), &capture_info, DIB_RGB_COLORS);
    SelectObject(memory_dc, old);
    DeleteObject(bitmap);
    DeleteDC(memory_dc);
    ReleaseDC(hwnd, window_dc);
    write_bmp("window_capture.bmp", captured.data(), 640, 480);

    DestroyWindow(hwnd);
    UnregisterClassA(wc.lpszClassName, wc.hInstance);
    std::FILE* log = std::fopen("leonida_window.log", "w");
    if (log) {
        std::fprintf(log, "frames=%d width=%d height=%d\n", frames, width, height);
        std::fclose(log);
    }
    return frames > 0 ? 0 : 5;
}

extern "C" int __cdecl LeonidaRenderTriangle(void) { return render_triangle(); }
extern "C" int __cdecl LeonidaPresentWindow(void) { return present_window(); }

extern "C" void* __stdcall shim_virtual_alloc(void*, std::size_t size, unsigned long, unsigned long) {
    return ::VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
extern "C" void __stdcall shim_sleep(unsigned long milliseconds) { ::Sleep(milliseconds); }
extern "C" void* __stdcall shim_create_file_w(const wchar_t* name, unsigned long access, unsigned long share,
                                              void* security, unsigned long creation, unsigned long flags, void* file) {
    return ::CreateFileW(name, access, share, static_cast<LPSECURITY_ATTRIBUTES>(security), creation, flags,
                         static_cast<HANDLE>(file));
}
extern "C" void* __stdcall shim_create_window_ex_w(unsigned long ex_style, const wchar_t* class_name,
                                                   const wchar_t* title, unsigned long style, int x, int y, int w,
                                                   int h, void* parent, void* menu, void* instance, void* param) {
    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.lpszClassName = L"LeonidaGameClass";
        RegisterClassExW(&wc);
        registered = true;
    }
    return ::CreateWindowExW(ex_style, class_name, title, style, x, y, w, h, static_cast<HWND>(parent),
                             static_cast<HMENU>(menu), static_cast<HINSTANCE>(instance), param);
}
extern "C" long __stdcall shim_d3d12_create_device(void*, int, const GUID&, void** out) {
    return rt::d3d12::create_vulkan_device_standalone(__uuidof(ID3D12Device), out);
}
extern "C" int __cdecl LeonidaRunAnimation(void* hwnd) { return leonida_run_animation(hwnd); }
// --- engine-facing system shims (threads, clocks, file I/O, D3D12 helpers) --------------------------------
extern "C" HANDLE __stdcall shim_create_thread(void* attributes, std::size_t stack, DWORD(WINAPI* start)(void*),
                                                void* parameter, DWORD flags, DWORD* id) {
    return ::CreateThread(static_cast<LPSECURITY_ATTRIBUTES>(attributes), stack, start, parameter, flags, id);
}
extern "C" DWORD __stdcall shim_wait_for_single_object(HANDLE handle, DWORD milliseconds) {
    return ::WaitForSingleObject(handle, milliseconds);
}
extern "C" BOOL __stdcall shim_query_performance_counter(LARGE_INTEGER* counter) {
    return ::QueryPerformanceCounter(counter);
}
extern "C" BOOL __stdcall shim_query_performance_frequency(LARGE_INTEGER* frequency) {
    return ::QueryPerformanceFrequency(frequency);
}
extern "C" ULONGLONG __stdcall shim_get_tick_count64() { return ::GetTickCount64(); }
extern "C" void* __stdcall shim_create_file_a(const char* name, unsigned long access, unsigned long share,
                                              void* security, unsigned long creation, unsigned long flags, void* file) {
    return ::CreateFileA(name, access, share, static_cast<LPSECURITY_ATTRIBUTES>(security), creation, flags,
                         static_cast<HANDLE>(file));
}
extern "C" BOOL __stdcall shim_read_file(HANDLE file, void* buffer, DWORD bytes, DWORD* read, void* overlapped) {
    return ::ReadFile(file, buffer, bytes, read, static_cast<LPOVERLAPPED>(overlapped));
}
extern "C" BOOL __stdcall shim_write_file(HANDLE file, const void* buffer, DWORD bytes, DWORD* written,
                                          void* overlapped) {
    return ::WriteFile(file, buffer, bytes, written, static_cast<LPOVERLAPPED>(overlapped));
}
extern "C" BOOL __stdcall shim_set_file_pointer_ex(HANDLE file, LARGE_INTEGER distance, LARGE_INTEGER* new_position,
                                                   DWORD method) {
    return ::SetFilePointerEx(file, distance, new_position, method);
}
extern "C" BOOL __stdcall shim_get_file_size_ex(HANDLE file, LARGE_INTEGER* size) {
    return ::GetFileSizeEx(file, size);
}
extern "C" BOOL __stdcall shim_close_handle(HANDLE handle) { return ::CloseHandle(handle); }

extern "C" HRESULT __stdcall shim_d3d12_create_root_signature_deserializer(const void*, SIZE_T, const GUID&, void** out) {
    if (out) *out = nullptr;
    return E_NOTIMPL;  // root signatures are not modelled as objects yet
}
extern "C" HRESULT __stdcall shim_map(void* resource, UINT subresource, const D3D12_RANGE* read_range, void** data) {
    return static_cast<ID3D12Resource*>(resource)->Map(subresource, read_range, data);
}
extern "C" void __stdcall shim_unmap(void* resource, UINT subresource, const D3D12_RANGE* written_range) {
    static_cast<ID3D12Resource*>(resource)->Unmap(subresource, written_range);
}
extern "C" void __stdcall shim_execute_command_lists(void* queue, UINT count, void* const* lists) {
    static_cast<ID3D12CommandQueue*>(queue)->ExecuteCommandLists(
        count, reinterpret_cast<ID3D12CommandList* const*>(lists));
}
extern "C" HRESULT __stdcall shim_create_committed_resource(void* device, const void* heap_properties,
                                                            D3D12_HEAP_FLAGS flags, const void* desc,
                                                            D3D12_RESOURCE_STATES state, const void* clear,
                                                            const GUID& riid, void** out) {
    return static_cast<ID3D12Device*>(device)->CreateCommittedResource(
        static_cast<const D3D12_HEAP_PROPERTIES*>(heap_properties), flags,
        static_cast<const D3D12_RESOURCE_DESC*>(desc), state, static_cast<const D3D12_CLEAR_VALUE*>(clear), riid, out);
}
SHIM_TABLE(d3d12, {"d3d12.dll", "LeonidaRunAnimation", reinterpret_cast<void*>(&LeonidaRunAnimation)},
                    {"kernel32.dll", "VirtualAlloc", reinterpret_cast<void*>(&shim_virtual_alloc)},
                    {"kernel32.dll", "Sleep", reinterpret_cast<void*>(&shim_sleep)},
                    {"kernel32.dll", "CreateFileW", reinterpret_cast<void*>(&shim_create_file_w)},
                    {"kernel32.dll", "CreateFileA", reinterpret_cast<void*>(&shim_create_file_a)},
                    {"kernel32.dll", "ReadFile", reinterpret_cast<void*>(&shim_read_file)},
                    {"kernel32.dll", "WriteFile", reinterpret_cast<void*>(&shim_write_file)},
                    {"kernel32.dll", "SetFilePointerEx", reinterpret_cast<void*>(&shim_set_file_pointer_ex)},
                    {"kernel32.dll", "GetFileSizeEx", reinterpret_cast<void*>(&shim_get_file_size_ex)},
                    {"kernel32.dll", "CloseHandle", reinterpret_cast<void*>(&shim_close_handle)},
                    {"kernel32.dll", "CreateThread", reinterpret_cast<void*>(&shim_create_thread)},
                    {"kernel32.dll", "WaitForSingleObject", reinterpret_cast<void*>(&shim_wait_for_single_object)},
                    {"kernel32.dll", "QueryPerformanceCounter", reinterpret_cast<void*>(&shim_query_performance_counter)},
                    {"kernel32.dll", "QueryPerformanceFrequency", reinterpret_cast<void*>(&shim_query_performance_frequency)},
                    {"kernel32.dll", "GetTickCount64", reinterpret_cast<void*>(&shim_get_tick_count64)},
                    {"d3d12.dll", "D3D12CreateRootSignatureDeserializer", reinterpret_cast<void*>(&shim_d3d12_create_root_signature_deserializer)},
                    {"d3d12.dll", "Map", reinterpret_cast<void*>(&shim_map)},
                    {"d3d12.dll", "Unmap", reinterpret_cast<void*>(&shim_unmap)},
                    {"d3d12.dll", "ExecuteCommandLists", reinterpret_cast<void*>(&shim_execute_command_lists)},
                    {"d3d12.dll", "CreateCommittedResource", reinterpret_cast<void*>(&shim_create_committed_resource)},
                    {"user32.dll", "CreateWindowExW", reinterpret_cast<void*>(&shim_create_window_ex_w)},
                    {"d3d12.dll", "LeonidaPresentWindow", reinterpret_cast<void*>(&LeonidaPresentWindow)},
                    {"d3d12.dll", "LeonidaRenderTriangle", reinterpret_cast<void*>(&LeonidaRenderTriangle)});

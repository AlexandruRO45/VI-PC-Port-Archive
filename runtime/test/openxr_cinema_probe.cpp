// The virtual cinema by hand, without the game (docs/vr.md): run with LEONIDA_VR=openxr and a headset connected
// through an OpenXR runtime. A 1280x720 window shows a test card that the headset shows on its screen too:
//   - red in the top-left corner, green in the top-right, blue in the bottom-left: the colours must match the names
//     (channel order) and the corners their places (the screen is neither mirrored nor upside down);
//   - a white bar sweeping left to right once a second: the screen follows the window's frames.
// Not a CTest. `openxr_cinema_probe [seconds]` (default 60); closing the window ends it early.
#include "../shims/d3d12_com.h"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {

bool g_closed = false;

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    if (message == WM_CLOSE) g_closed = true;
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

template <typename T>
void release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 60.0;
    constexpr UINT width = 1280, height = 720;

    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device))) || !device) {
        std::fprintf(stderr, "no Vulkan device\n");
        return 1;
    }
    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaOpenXrCinemaProbe";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    RegisterClassW(&window_class);
    RECT frame_rect{0, 0, LONG(width), LONG(height)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: virtual cinema probe", WS_OVERLAPPEDWINDOW,
                                  100, 100, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top,
                                  nullptr, nullptr, window_class.hInstance, nullptr);
    if (!window) {
        std::fprintf(stderr, "cannot create the window\n");
        return 1;
    }
    ShowWindow(window, SW_SHOWNORMAL);
    pump();
    auto presenter = rt::d3d12::create_window_presenter(device, window);
    if (!presenter) {
        std::fprintf(stderr, "no window presenter\n");
        return 1;
    }

    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    ID3D12Resource* back_buffer = nullptr;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) ||
        FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap))) ||
        FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_PRESENT,
                                               nullptr, IID_PPV_ARGS(&back_buffer)))) {
        std::fprintf(stderr, "cannot create the D3D12 objects\n");
        return 1;
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(back_buffer, nullptr, rtv);

    const auto start = std::chrono::steady_clock::now();
    UINT64 signalled = 0;
    std::uint64_t frames = 0;
    for (;;) {
        pump();
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (g_closed || t >= seconds) break;
        allocator->Reset();
        list->Reset(allocator, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = back_buffer;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        list->ResourceBarrier(1, &barrier);
        const FLOAT background[4] = {0.1f, 0.1f, 0.1f, 1.0f};
        const FLOAT red[4] = {0.9f, 0.1f, 0.1f, 1.0f}, green[4] = {0.1f, 0.9f, 0.1f, 1.0f},
                    blue[4] = {0.1f, 0.2f, 0.95f, 1.0f}, white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        list->ClearRenderTargetView(rtv, background, 0, nullptr);
        constexpr LONG corner = 160;
        const D3D12_RECT top_left{0, 0, corner, corner}, top_right{LONG(width) - corner, 0, LONG(width), corner},
            bottom_left{0, LONG(height) - corner, corner, LONG(height)};
        list->ClearRenderTargetView(rtv, red, 1, &top_left);
        list->ClearRenderTargetView(rtv, green, 1, &top_right);
        list->ClearRenderTargetView(rtv, blue, 1, &bottom_left);
        const LONG x = LONG((t - double(long long(t))) * double(width - 40));
        const D3D12_RECT bar{x, corner, x + 40, LONG(height) - corner};
        list->ClearRenderTargetView(rtv, white, 1, &bar);
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        list->ResourceBarrier(1, &barrier);
        list->Close();
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(presenter->present(back_buffer, 1))) {
            std::fprintf(stderr, "present failed\n");
            break;
        }
        queue->Signal(fence, ++signalled);
        while (fence->GetCompletedValue() < signalled) Sleep(0);
        ++frames;
    }
    std::printf("openxr_cinema_probe: %llu frames presented\n", static_cast<unsigned long long>(frames));
    release(back_buffer);
    release(rtv_heap);
    release(fence);
    release(list);
    release(allocator);
    release(queue);
    presenter.reset();
    device->Release();
    DestroyWindow(window);
    return 0;
}

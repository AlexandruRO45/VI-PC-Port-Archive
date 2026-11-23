// Regression tests for D3D-818: ~SwapChain() must destroy its VulkanSwapchain resources before releasing
// device_ (the command queue, which may hold the last reference to the D3D12 device, VulkanContext, VkDevice
// and VkInstance).
//
// Part 1 (synthetic, no GPU or window station required): drives the real rt::dxgi::create_factory / SwapChain
// COM implementation over a fake Vulkan function table and a mock ID3D12CommandQueue that owns the
// VulkanContext, verifying that every Vulkan resource is destroyed before device_->Release() drops the
// queue/device — covering direct swapchains, offscreen + waitable + output-restricted swapchains, and
// CreateSwapChainForHwnd partial-initialization failure paths.
//
// Part 2 (real GPU, skipped only when no interactive window station or Vulkan 1.3 GPU is available): creates a
// standalone Vulkan D3D12 device + direct queue + DXGI factory + IDXGISwapChain3, exercises live swapchain
// resources (plain present, offscreen + waitable + output-restricted present, and RTV clear + ExecuteCommandLists
// + present), releases factory/device/queue so the swapchain holds the last reference, and releases the swapchain.
//
// No shaders or game data.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "../shims/d3d12_com.h"
#include "../shims/dxgi_com.h"
#include "../src/vulkan_device_filter.h"
#include "window_station.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// ---------------------------------------------------------------------------
// Part 1: Synthetic COM lifetime tests (no GPU needed)
// ---------------------------------------------------------------------------

namespace synthetic {

constexpr std::uintptr_t kSurface = 0x5100;
constexpr std::uintptr_t kSwapchain = 0x5101;
constexpr std::uintptr_t kFenceBase = 0x5200;
constexpr std::uintptr_t kPool = 0x5103;
constexpr std::uintptr_t kCmd = 0x5104;
constexpr std::uintptr_t kNativeBase = 0x6000;
constexpr std::uintptr_t kOffscreenBase = 0x7000;

struct State {
    std::uint32_t min_image_count = 2;
    std::uint32_t returned_images = 2;
    bool fail_create_swapchain = false;
    bool fail_acquire_next_image = false;

    bool device_destroyed = false;
    bool vk_called_after_device_destroyed = false;
    bool queue_released_while_vk_live = false;

    int live_surfaces = 0;
    int live_swapchains = 0;
    int live_fences = 0;
    int live_pools = 0;
    int live_images = 0;
    int live_memories = 0;

    unsigned wait_idle_calls = 0;
    unsigned destroy_swapchain_calls = 0;
    unsigned destroy_surface_calls = 0;
    unsigned destroy_fence_calls = 0;
    unsigned destroy_pool_calls = 0;
    unsigned destroy_image_calls = 0;
    unsigned free_mem_calls = 0;

    std::uint32_t next_fence = 0;
    std::uint32_t next_offscreen = 0;
    std::uint32_t acquire_cursor = 0;

    int total_live_vk() const {
        return live_surfaces + live_swapchains + live_fences + live_pools + live_images + live_memories;
    }

    void note_vk_call() {
        if (device_destroyed) vk_called_after_device_destroyed = true;
    }
};

State g;

template <typename T>
PFN_vkVoidFunction as_fn(T fn) {
    return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

VKAPI_ATTR VkResult VKAPI_CALL create_win32_surface(VkInstance, const VkWin32SurfaceCreateInfoKHR*,
                                                    const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    g.note_vk_call();
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkSurfaceKHR>(kSurface);
    ++g.live_surfaces;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_surface(VkInstance, VkSurfaceKHR s, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (s) {
        --g.live_surfaces;
        ++g.destroy_surface_calls;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL surface_support(VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32* out) {
    g.note_vk_call();
    if (out) *out = VK_TRUE;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_caps(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* caps) {
    g.note_vk_call();
    if (!caps) return VK_ERROR_INITIALIZATION_FAILED;
    std::memset(caps, 0, sizeof(*caps));
    caps->minImageCount = g.min_image_count;
    caps->maxImageCount = 0;
    caps->currentExtent = {64, 64};
    caps->minImageExtent = {1, 1};
    caps->maxImageExtent = {4096, 4096};
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_formats(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                               VkSurfaceFormatKHR* formats) {
    g.note_vk_call();
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (formats && *count) {
        formats[0].format = VK_FORMAT_B8G8R8A8_UNORM;
        formats[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_present_modes(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                     VkPresentModeKHR* modes) {
    g.note_vk_call();
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (modes && *count) modes[0] = VK_PRESENT_MODE_FIFO_KHR;
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* props) {
    g.note_vk_call();
    std::memset(props, 0, sizeof(*props));
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = 1ull << 30;
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}
VKAPI_ATTR VkResult VKAPI_CALL create_swapchain(VkDevice, const VkSwapchainCreateInfoKHR*,
                                                const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    g.note_vk_call();
    if (g.fail_create_swapchain || !out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkSwapchainKHR>(kSwapchain);
    ++g.live_swapchains;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_swapchain(VkDevice, VkSwapchainKHR sc, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (sc) {
        --g.live_swapchains;
        ++g.destroy_swapchain_calls;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL get_images(VkDevice, VkSwapchainKHR, std::uint32_t* count, VkImage* images) {
    g.note_vk_call();
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (images) {
        for (std::uint32_t i = 0; i < g.returned_images && i < *count; ++i)
            images[i] = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kNativeBase + i));
    }
    *count = g.returned_images;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL acquire_next_image(VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore, VkFence,
                                                  std::uint32_t* out) {
    g.note_vk_call();
    if (g.fail_acquire_next_image || !out || g.returned_images == 0) return VK_ERROR_OUT_OF_DATE_KHR;
    *out = (g.acquire_cursor++) % g.returned_images;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue, const VkPresentInfoKHR*) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                            VkFence* out) {
    g.note_vk_call();
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkFence>(static_cast<std::uintptr_t>(kFenceBase + g.next_fence++));
    ++g.live_fences;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice, VkFence f, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (f) {
        --g.live_fences;
        ++g.destroy_fence_calls;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, std::uint32_t, const VkFence*) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL device_wait_idle(VkDevice) {
    g.note_vk_call();
    ++g.wait_idle_calls;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                                            VkImage* out) {
    g.note_vk_call();
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kOffscreenBase + g.next_offscreen++));
    ++g.live_images;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage img, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (img) {
        --g.live_images;
        ++g.destroy_image_calls;
    }
}
VKAPI_ATTR void VKAPI_CALL image_mem_req(VkDevice, VkImage, VkMemoryRequirements* req) {
    g.note_vk_call();
    std::memset(req, 0, sizeof(*req));
    req->size = 4096;
    req->alignment = 1;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_mem(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                         VkDeviceMemory* out) {
    g.note_vk_call();
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkDeviceMemory>(static_cast<std::uintptr_t>(kOffscreenBase + g.next_offscreen++));
    ++g.live_memories;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (mem) {
        --g.live_memories;
        ++g.free_mem_calls;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL bind_image_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_cmd_pool(VkDevice, const VkCommandPoolCreateInfo*,
                                               const VkAllocationCallbacks*, VkCommandPool* out) {
    g.note_vk_call();
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkCommandPool>(kPool);
    ++g.live_pools;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_cmd_pool(VkDevice, VkCommandPool p, const VkAllocationCallbacks*) {
    g.note_vk_call();
    if (p) {
        --g.live_pools;
        ++g.destroy_pool_calls;
    }
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_cmd_bufs(VkDevice, const VkCommandBufferAllocateInfo*,
                                              VkCommandBuffer* out) {
    g.note_vk_call();
    if (out) *out = reinterpret_cast<VkCommandBuffer>(kCmd);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_cmd_buf(VkCommandBuffer, VkCommandBufferResetFlags) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL begin_cmd_buf(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL end_cmd_buf(VkCommandBuffer) {
    g.note_vk_call();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, std::uint32_t,
                                       const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                       std::uint32_t, const VkImageMemoryBarrier*) {
    g.note_vk_call();
}
VKAPI_ATTR void VKAPI_CALL cmd_blit(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                    std::uint32_t, const VkImageBlit*, VkFilter) {
    g.note_vk_call();
}
VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    g.note_vk_call();
    return VK_SUCCESS;
}

struct Entry {
    const char* name;
    PFN_vkVoidFunction fn;
};

const Entry kEntries[] = {
    {"vkCreateWin32SurfaceKHR", as_fn(create_win32_surface)},
    {"vkDestroySurfaceKHR", as_fn(destroy_surface)},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", as_fn(surface_support)},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", as_fn(surface_caps)},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", as_fn(surface_formats)},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", as_fn(surface_present_modes)},
    {"vkGetPhysicalDeviceMemoryProperties", as_fn(memory_properties)},
    {"vkCreateSwapchainKHR", as_fn(create_swapchain)},
    {"vkDestroySwapchainKHR", as_fn(destroy_swapchain)},
    {"vkGetSwapchainImagesKHR", as_fn(get_images)},
    {"vkAcquireNextImageKHR", as_fn(acquire_next_image)},
    {"vkQueuePresentKHR", as_fn(queue_present)},
    {"vkCreateFence", as_fn(create_fence)},
    {"vkDestroyFence", as_fn(destroy_fence)},
    {"vkWaitForFences", as_fn(wait_for_fences)},
    {"vkResetFences", as_fn(reset_fences)},
    {"vkDeviceWaitIdle", as_fn(device_wait_idle)},
    {"vkCreateImage", as_fn(create_image)},
    {"vkDestroyImage", as_fn(destroy_image)},
    {"vkGetImageMemoryRequirements", as_fn(image_mem_req)},
    {"vkAllocateMemory", as_fn(alloc_mem)},
    {"vkFreeMemory", as_fn(free_mem)},
    {"vkBindImageMemory", as_fn(bind_image_mem)},
    {"vkCreateCommandPool", as_fn(create_cmd_pool)},
    {"vkDestroyCommandPool", as_fn(destroy_cmd_pool)},
    {"vkAllocateCommandBuffers", as_fn(alloc_cmd_bufs)},
    {"vkResetCommandBuffer", as_fn(reset_cmd_buf)},
    {"vkBeginCommandBuffer", as_fn(begin_cmd_buf)},
    {"vkEndCommandBuffer", as_fn(end_cmd_buf)},
    {"vkCmdPipelineBarrier", as_fn(cmd_barrier)},
    {"vkCmdBlitImage", as_fn(cmd_blit)},
    {"vkQueueSubmit", as_fn(queue_submit)},
};

PFN_vkVoidFunction lookup(const char* name) {
    for (const Entry& e : kEntries)
        if (name && std::strcmp(name, e.name) == 0) return e.fn;
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc(VkInstance, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice, const char* name) { return lookup(name); }

// Mock ID3D12CommandQueue that owns the VulkanContext, modelling the real CommandQueue -> Device -> VulkanContext
// ownership chain. Dropping the last reference destroys the VulkanContext and marks device_destroyed = true.
class FakeQueue final : public ID3D12CommandQueue {
public:
    explicit FakeQueue(std::unique_ptr<leonida::VulkanContext> ctx) : context_(std::move(ctx)) {}

    ~FakeQueue() {
        g.device_destroyed = true;
        context_.reset();
    }

    leonida::VulkanContext& context() { return *context_; }

    // Set while ~SwapChain() is expected to drop its queue reference so even non-final Release() calls
    // (e.g. CreateSwapChainForHwnd partial-init failure) verify that no VulkanSwapchain objects remain live.
    void set_check_vk_dead_on_release(bool on) { check_vk_dead_on_release_ = on; }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
            riid == __uuidof(ID3D12Pageable) || riid == __uuidof(ID3D12CommandQueue)) {
            *out = static_cast<ID3D12CommandQueue*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override {
        if (check_vk_dead_on_release_ && g.total_live_vk() > 0) g.queue_released_while_vk_live = true;
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }

    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }
    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource*, UINT, const D3D12_TILED_RESOURCE_COORDINATE*,
                                              const D3D12_TILE_REGION_SIZE*, ID3D12Heap*, UINT,
                                              const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*,
                                              D3D12_TILE_MAPPING_FLAGS) override {}
    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, ID3D12Resource*,
                                            const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*,
                                            D3D12_TILE_MAPPING_FLAGS) override {}
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT, ID3D12CommandList* const*) override {}
    void STDMETHODCALLTYPE SetMarker(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE BeginEvent(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE EndEvent() override {}
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence*, UINT64) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence*, UINT64) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64*, UINT64*) override { return E_NOTIMPL; }
#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override {
        return {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    }
#else
    D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* ret) override {
        if (ret) *ret = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        return ret;
    }
#endif

private:
    std::atomic<ULONG> refs_{1};
    std::unique_ptr<leonida::VulkanContext> context_;
    bool check_vk_dead_on_release_ = false;
};

int run_last_ref_case(const char* label, std::uint32_t returned_images, bool waitable, bool with_output) {
    g = State{};
    g.returned_images = returned_images;

    auto ctx = std::make_unique<leonida::VulkanContext>(get_device_proc, reinterpret_cast<VkDevice>(3), true);
    auto* queue = new FakeQueue(std::move(ctx));
    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::dxgi::create_factory(queue->context(), get_instance_proc, reinterpret_cast<VkInstance>(1),
                                        reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                        reinterpret_cast<VkQueue>(4), 0, IID_PPV_ARGS(&factory))) ||
        !factory) {
        queue->Release();
        return fail("synthetic: create_factory");
    }

    IDXGIOutput* restrict_output = nullptr;
    if (with_output) {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(factory->EnumAdapters(0, &adapter)) && adapter) {
            adapter->EnumOutputs(0, &restrict_output);
            adapter->Release();
        }
    }

    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = 64;
    desc.Height = 64;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    if (waitable) desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    const HWND fake_hwnd = reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1));
    IDXGISwapChain1* sc = nullptr;
    const HRESULT hr = factory->CreateSwapChainForHwnd(queue, fake_hwnd, &desc, nullptr, restrict_output, &sc);
    if (restrict_output) restrict_output->Release();
    if (FAILED(hr) || !sc) {
        factory->Release();
        queue->Release();
        return fail("synthetic: CreateSwapChainForHwnd");
    }

    if (FAILED(sc->Present(0, 0))) {
        sc->Release();
        factory->Release();
        queue->Release();
        return fail("synthetic: Present");
    }

    // Drop external references to factory and queue so `sc` holds the last reference to the queue/VulkanContext.
    if (factory->Release() == 0) {
        sc->Release();
        queue->Release();
        return fail("synthetic: factory released too early");
    }
    if (queue->Release() == 0) {
        sc->Release();
        return fail("synthetic: queue released too early");
    }
    if (g.device_destroyed) {
        sc->Release();
        return fail("synthetic: device destroyed while swapchain still alive");
    }

    queue->set_check_vk_dead_on_release(true);
    const ULONG left = sc->Release();
    if (left != 0) return fail("synthetic: swapchain still referenced after final Release");
    if (!g.device_destroyed) {
        std::fprintf(stderr, "FAIL: synthetic (%s): queue/device was not released by ~SwapChain\n", label);
        return 1;
    }
    if (g.vk_called_after_device_destroyed) {
        std::fprintf(stderr, "FAIL: synthetic (%s): VulkanSwapchain called Vulkan after device_ was released (D3D-818)\n",
                     label);
        return 1;
    }
    if (g.queue_released_while_vk_live) {
        std::fprintf(stderr, "FAIL: synthetic (%s): device_->Release() ran while VulkanSwapchain objects were still live (D3D-818)\n",
                     label);
        return 1;
    }
    if (g.total_live_vk() != 0) {
        std::fprintf(stderr, "FAIL: synthetic (%s): leaked %d VulkanSwapchain object(s)\n", label, g.total_live_vk());
        return 1;
    }
    if (g.destroy_swapchain_calls != 1 || g.destroy_surface_calls != 1 || g.destroy_pool_calls != 1 ||
        g.destroy_fence_calls != 2) {
        std::fprintf(stderr, "FAIL: synthetic (%s): unexpected destroy counts (sc=%u surf=%u pool=%u fence=%u)\n",
                     label, g.destroy_swapchain_calls, g.destroy_surface_calls, g.destroy_pool_calls,
                     g.destroy_fence_calls);
        return 1;
    }
    return 0;
}

int run_partial_init_case(const char* label, bool fail_swapchain, bool fail_acquire) {
    g = State{};
    g.returned_images = 4;  // forces offscreen backbuffers too when swapchain creation succeeds
    g.fail_create_swapchain = fail_swapchain;
    g.fail_acquire_next_image = fail_acquire;

    auto ctx = std::make_unique<leonida::VulkanContext>(get_device_proc, reinterpret_cast<VkDevice>(3), true);
    auto* queue = new FakeQueue(std::move(ctx));
    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::dxgi::create_factory(queue->context(), get_instance_proc, reinterpret_cast<VkInstance>(1),
                                        reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                        reinterpret_cast<VkQueue>(4), 0, IID_PPV_ARGS(&factory))) ||
        !factory) {
        queue->Release();
        return fail("synthetic partial-init: create_factory");
    }

    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = 64;
    desc.Height = 64;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    const HWND fake_hwnd = reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1));
    // Verify that when ~SwapChain() runs during init failure, all VulkanSwapchain handles are destroyed before
    // device_->Release() executes. Note check_hwnd_swap_chain also calls QueryInterface+Release before new SwapChain,
    // at which point total_live_vk() == 0, so enabling the check here only trips if ~SwapChain() releases device_
    // with Vulkan objects still alive.
    queue->set_check_vk_dead_on_release(true);
    IDXGISwapChain1* sc = reinterpret_cast<IDXGISwapChain1*>(static_cast<std::uintptr_t>(1));
    const HRESULT hr = factory->CreateSwapChainForHwnd(queue, fake_hwnd, &desc, nullptr, nullptr, &sc);
    queue->set_check_vk_dead_on_release(false);

    factory->Release();
    const ULONG q_left = queue->Release();
    if (hr != E_FAIL || sc != nullptr) {
        std::fprintf(stderr, "FAIL: synthetic partial-init (%s): expected E_FAIL and null swapchain\n", label);
        return 1;
    }
    if (q_left != 0 || !g.device_destroyed) {
        std::fprintf(stderr, "FAIL: synthetic partial-init (%s): queue leaked after failed init\n", label);
        return 1;
    }
    if (g.queue_released_while_vk_live || g.vk_called_after_device_destroyed) {
        std::fprintf(stderr, "FAIL: synthetic partial-init (%s): Vulkan objects still live when device_ released (D3D-818)\n",
                     label);
        return 1;
    }
    if (g.total_live_vk() != 0) {
        std::fprintf(stderr, "FAIL: synthetic partial-init (%s): leaked %d Vulkan object(s)\n", label, g.total_live_vk());
        return 1;
    }
    return 0;
}

int run_all() {
    if (const int r = run_last_ref_case("direct", 2, false, false)) return r;
    if (const int r = run_last_ref_case("offscreen+waitable+output", 4, true, true)) return r;
    if (const int r = run_partial_init_case("fail vkCreateSwapchainKHR", true, false)) return r;
    if (const int r = run_partial_init_case("fail vkAcquireNextImageKHR", false, true)) return r;
    std::printf("dxgi_swapchain_lifetime_test [synthetic]: PASS (4 cases: direct, offscreen+waitable+output, "
                "2 partial-init failures)\n");
    return 0;
}

}  // namespace synthetic

// ---------------------------------------------------------------------------
// Part 2: Real GPU standalone-device lifetime tests
// ---------------------------------------------------------------------------

namespace gpu {

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) { return DefWindowProcW(hwnd, msg, wp, lp); }

HWND create_test_window(int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaDxgiSwapchainLifetimeTestWnd";
    RegisterClassW(&wc);
    RECT rc{0, 0, width, height};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExW(0, wc.lpszClassName, L"Leonida DXGI Lifetime Test", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                           CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance,
                           nullptr);
}

D3D12_RESOURCE_BARRIER make_transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

int run_standalone_case(const char* label, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
                        bool with_output, bool render_clear) {
    ID3D12Device* dev = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev))) || !dev) {
        std::fprintf(stderr, "FAIL: GPU (%s): create_vulkan_device_standalone\n", label);
        return 1;
    }

    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::d3d12::create_dxgi_factory(dev, IID_PPV_ARGS(&factory))) || !factory) {
        dev->Release();
        return fail("GPU: create_dxgi_factory");
    }

    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ID3D12CommandQueue* queue = nullptr;
    if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) || !queue) {
        factory->Release();
        dev->Release();
        return fail("GPU: CreateCommandQueue");
    }

    HWND hwnd = create_test_window(128, 128);
    if (!hwnd) {
        queue->Release();
        factory->Release();
        dev->Release();
        return fail("GPU: CreateWindowExW");
    }

    IDXGIOutput* restrict_output = nullptr;
    if (with_output) {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(factory->EnumAdapters(0, &adapter)) && adapter) {
            adapter->EnumOutputs(0, &restrict_output);
            adapter->Release();
        }
    }

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = width;
    scd.Height = height;
    scd.Format = format;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Flags = flags;

    IDXGISwapChain1* sc1 = nullptr;
    const HRESULT hr = factory->CreateSwapChainForHwnd(queue, hwnd, &scd, nullptr, restrict_output, &sc1);
    if (restrict_output) restrict_output->Release();
    if (FAILED(hr) || !sc1) {
        DestroyWindow(hwnd);
        queue->Release();
        factory->Release();
        dev->Release();
        return fail("GPU: CreateSwapChainForHwnd");
    }
    IDXGISwapChain3* sc = nullptr;
    if (FAILED(sc1->QueryInterface(IID_PPV_ARGS(&sc))) || !sc) {
        sc1->Release();
        DestroyWindow(hwnd);
        queue->Release();
        factory->Release();
        dev->Release();
        return fail("GPU: QI IDXGISwapChain3");
    }
    sc1->Release();

    // Acquire backbuffer, optionally clear it on the GPU, and present so Vulkan swapchain resources are active.
    const UINT idx = sc->GetCurrentBackBufferIndex();
    ID3D12Resource* bb = nullptr;
    if (FAILED(sc->GetBuffer(idx, IID_PPV_ARGS(&bb))) || !bb) {
        sc->Release();
        DestroyWindow(hwnd);
        queue->Release();
        factory->Release();
        dev->Release();
        return fail("GPU: GetBuffer");
    }

    if (render_clear) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        ID3D12DescriptorHeap* rtv_heap = nullptr;
        ID3D12CommandAllocator* alloc = nullptr;
        ID3D12GraphicsCommandList* cl = nullptr;
        if (FAILED(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap))) ||
            FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl)))) {
            if (cl) cl->Release();
            if (alloc) alloc->Release();
            if (rtv_heap) rtv_heap->Release();
            bb->Release();
            sc->Release();
            DestroyWindow(hwnd);
            queue->Release();
            factory->Release();
            dev->Release();
            return fail("GPU: create RTV/allocator/command list");
        }
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dev->CreateRenderTargetView(bb, nullptr, rtv);
        const float rgba[4] = {0.2f, 0.4f, 0.8f, 1.0f};
        auto to_rt = make_transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cl->ResourceBarrier(1, &to_rt);
        cl->ClearRenderTargetView(rtv, rgba, 0, nullptr);
        auto to_present = make_transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        cl->ResourceBarrier(1, &to_present);
        cl->Close();
        ID3D12CommandList* lists[1] = {cl};
        queue->ExecuteCommandLists(1, lists);
        cl->Release();
        alloc->Release();
        rtv_heap->Release();
    }
    bb->Release();

    if (FAILED(sc->Present(0, 0))) {
        sc->Release();
        DestroyWindow(hwnd);
        queue->Release();
        factory->Release();
        dev->Release();
        return fail("GPU: Present");
    }

    // Release factory, device, and command queue so `sc` holds the LAST reference to the device/queue.
    const ULONG factory_refs = factory->Release();
    const ULONG dev_refs = dev->Release();
    const ULONG queue_refs = queue->Release();
    if (factory_refs == 0 || dev_refs == 0 || queue_refs == 0) {
        sc->Release();
        DestroyWindow(hwnd);
        return fail("GPU: swapchain did not hold live references to factory/device/queue");
    }

    // Final Release runs ~SwapChain() while holding the last reference to the standalone device/VulkanContext/VkDevice.
    if (sc->Release() != 0) {
        DestroyWindow(hwnd);
        return fail("GPU: swapchain still referenced after final Release");
    }
    DestroyWindow(hwnd);
    return 0;
}

int run_all() {
    if (!has_interactive_window_station()) {
        std::printf("dxgi_swapchain_lifetime_test [gpu]: skipped (no interactive window station)\n");
        return 0;
    }

    // Probe standalone Vulkan availability and print the chosen adapter description.
    ID3D12Device* probe = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&probe))) || !probe) {
        const std::string filter = leonida::vulkan_device_filter();
        if (!filter.empty()) {
            std::fprintf(stderr, "FAIL: LEONIDA_VK_DEVICE=%s: standalone Vulkan device creation failed\n",
                         filter.c_str());
            return 1;
        }
        std::printf("dxgi_swapchain_lifetime_test [gpu]: skipped (no standalone Vulkan 1.3 GPU)\n");
        return 0;
    }
    std::string gpu_name = "(unknown)";
    IDXGIFactory4* probe_factory = nullptr;
    if (SUCCEEDED(rt::d3d12::create_dxgi_factory(probe, IID_PPV_ARGS(&probe_factory))) && probe_factory) {
        IDXGIAdapter1* adapter = nullptr;
        if (SUCCEEDED(probe_factory->EnumAdapters1(0, &adapter)) && adapter) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                char utf8[256]{};
                if (WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr) > 0)
                    gpu_name = utf8;
            }
            adapter->Release();
        }
        probe_factory->Release();
    }

    // Check whether this GPU's Vulkan driver can actually create a swapchain on a window on the current display
    // (a render-only secondary GPU, such as an AMD iGPU while X11/XWayland runs on an NVIDIA dGPU, can fail
    // vkGetPhysicalDeviceSurfaceSupportKHR or vkCreateSwapchainKHR even though Vulkan compute/render works).
    bool can_present = false;
    VkResult probe_vk_result = VK_ERROR_INITIALIZATION_FAILED;
    const rt::d3d12::VulkanDeviceInfo* probe_info = rt::d3d12::vulkan_device_info(probe);
    if (probe_info && probe_info->presentation && probe_info->get_instance_proc) {
        const auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            probe_info->get_instance_proc(probe_info->instance, "vkCreateWin32SurfaceKHR"));
        const auto destroy_surface = reinterpret_cast<PFN_vkDestroySurfaceKHR>(
            probe_info->get_instance_proc(probe_info->instance, "vkDestroySurfaceKHR"));
        const auto surface_support = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            probe_info->get_instance_proc(probe_info->instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));
        const auto surface_caps = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            probe_info->get_instance_proc(probe_info->instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
        const auto surface_formats = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            probe_info->get_instance_proc(probe_info->instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
        const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            probe_info->get_instance_proc(probe_info->instance, "vkGetDeviceProcAddr"));
        const auto create_sc =
            gdpa ? reinterpret_cast<PFN_vkCreateSwapchainKHR>(gdpa(probe_info->device, "vkCreateSwapchainKHR"))
                 : nullptr;
        const auto destroy_sc =
            gdpa ? reinterpret_cast<PFN_vkDestroySwapchainKHR>(gdpa(probe_info->device, "vkDestroySwapchainKHR"))
                 : nullptr;
        HWND probe_hwnd = create_test_window(64, 64);
        if (probe_hwnd && create_surface && destroy_surface && surface_support && surface_caps && surface_formats &&
            create_sc && destroy_sc) {
            VkWin32SurfaceCreateInfoKHR sci{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
            sci.hinstance = GetModuleHandleW(nullptr);
            sci.hwnd = probe_hwnd;
            VkSurfaceKHR surf = VK_NULL_HANDLE;
            probe_vk_result = create_surface(probe_info->instance, &sci, nullptr, &surf);
            if (probe_vk_result == VK_SUCCESS && surf) {
                VkBool32 supported = VK_FALSE;
                VkSurfaceCapabilitiesKHR caps{};
                std::uint32_t fmt_count = 0;
                if (surface_support(probe_info->physical, probe_info->queue_family, surf, &supported) == VK_SUCCESS &&
                    supported == VK_TRUE && surface_caps(probe_info->physical, surf, &caps) == VK_SUCCESS &&
                    surface_formats(probe_info->physical, surf, &fmt_count, nullptr) == VK_SUCCESS && fmt_count > 0) {
                    VkSurfaceFormatKHR fmt{};
                    std::uint32_t one = 1;
                    if (surface_formats(probe_info->physical, surf, &one, &fmt) >= VK_SUCCESS && one > 0) {
                        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
                        ci.surface = surf;
                        ci.minImageCount = caps.minImageCount > 0 ? caps.minImageCount : 2;
                        ci.imageFormat = fmt.format;
                        ci.imageColorSpace = fmt.colorSpace;
                        ci.imageExtent = (caps.currentExtent.width != UINT32_MAX && caps.currentExtent.width > 0 &&
                                          caps.currentExtent.height > 0)
                                             ? caps.currentExtent
                                             : VkExtent2D{64, 64};
                        ci.imageArrayLayers = 1;
                        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                        ci.preTransform = caps.currentTransform;
                        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
                        ci.clipped = VK_TRUE;
                        VkSwapchainKHR sc_probe = VK_NULL_HANDLE;
                        probe_vk_result = create_sc(probe_info->device, &ci, nullptr, &sc_probe);
                        if (probe_vk_result == VK_SUCCESS && sc_probe) {
                            can_present = true;
                            destroy_sc(probe_info->device, sc_probe, nullptr);
                        }
                    }
                }
                destroy_surface(probe_info->instance, surf, nullptr);
            }
        }
        if (probe_hwnd) DestroyWindow(probe_hwnd);
    }
    probe->Release();

    if (!can_present) {
        // Still exercise real-GPU partial-init teardown on this standalone device (CreateSwapChainForHwnd creates
        // fences, command pool, and surface, then fails swapchain creation and destroys SwapChain cleanly).
        ID3D12Device* dev = nullptr;
        IDXGIFactory4* factory = nullptr;
        ID3D12CommandQueue* queue = nullptr;
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        HWND hwnd = create_test_window(64, 64);
        if (hwnd && SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&dev))) && dev &&
            SUCCEEDED(rt::d3d12::create_dxgi_factory(dev, IID_PPV_ARGS(&factory))) && factory &&
            SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) && queue) {
            DXGI_SWAP_CHAIN_DESC1 scd{};
            scd.Width = 64;
            scd.Height = 64;
            scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            scd.SampleDesc.Count = 1;
            scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            scd.BufferCount = 2;
            scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            IDXGISwapChain1* sc1 = nullptr;
            const HRESULT hr = factory->CreateSwapChainForHwnd(queue, hwnd, &scd, nullptr, nullptr, &sc1);
            if (sc1) sc1->Release();
            if (SUCCEEDED(hr)) {
                queue->Release();
                factory->Release();
                dev->Release();
                DestroyWindow(hwnd);
                return fail("GPU: unexpected CreateSwapChainForHwnd success on non-presentable surface");
            }
        }
        if (queue) queue->Release();
        if (factory) factory->Release();
        if (dev) dev->Release();
        if (hwnd) DestroyWindow(hwnd);
        std::printf("dxgi_swapchain_lifetime_test [gpu]: %s cannot create a Vulkan swapchain on this display "
                    "(VkResult %d; partial-init teardown OK, present cases skipped)\n",
                    gpu_name.c_str(), int(probe_vk_result));
        return 0;
    }

    if (const int r = run_standalone_case("plain 128x128", 128, 128, DXGI_FORMAT_B8G8R8A8_UNORM, 0, false, false))
        return r;
    if (const int r = run_standalone_case("offscreen+waitable+output 96x64", 96, 64, DXGI_FORMAT_R10G10B10A2_UNORM,
                                          DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT, true, false))
        return r;
    if (const int r = run_standalone_case("cleared+presented 128x128", 128, 128, DXGI_FORMAT_B8G8R8A8_UNORM, 0, false,
                                          true))
        return r;

    std::printf("dxgi_swapchain_lifetime_test [gpu]: PASS on %s (3 standalone last-ref cases)\n", gpu_name.c_str());
    return 0;
}

}  // namespace gpu

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const int synth = synthetic::run_all();
    const int real = gpu::run_all();
    return (synth || real) ? 1 : 0;
}

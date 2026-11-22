// A Vulkan-backed ID3D12Device (PC ABI) for the Xbox-ABI device proxy (d3d12_xs_device.cpp) to forward to.
// Implemented: queues, allocators, command lists, fences, descriptor heaps, render-target and depth-stencil views,
// committed buffers and textures (2D, multisampled 2D, 3D), a few feature queries. Everything else is a generated stub that reports
// itself once. See docs/d3d12_vulkan.md section 4.1/4.2 for what each answer means and what is still open.
#include "d3d12_com.h"

#include "../src/descriptor_handles.h"  // slot arithmetic and dirty ranges (P0/P1)
#include "../src/descriptor_views.h"    // D3D12 view descriptions -> GFX10 T#/V#
#include "../src/bindless_registry.h"   // key -> bindless index
#include "../src/retirement.h"          // fence-value deferred release
#include "../src/vulkan_bindless.h"     // set 0: images, samplers, descriptor table
#include "../src/pipeline_stream.h"     // D3D12 pipeline-state stream (PR C)
#include "../src/d3d12_root_signature.h"  // serialized root signature (PR A)
#include "../src/root_signature_layout.h" // root signature -> Vulkan layout (PR B)
#include "../src/descriptor_sizes.h"    // per-type slot sizes
#include "d3d12_stubs.h"  // generated
#include "d3d12_com_object.h"
#include "d3d12_xs_device.h"

#include <vulkan/vulkan_win32.h>  // VK_KHR_win32_surface: the instance extension a presentable device needs

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace rt::d3d12 {
namespace {

// Slot sizes are per heap type (descriptor_sizes.h): a sampler heap no longer shares the CBV/SRV/UAV stride.
// The console's own increments are open question 4 in docs/d3d12_vulkan.md; the application reads them from the
// device, and GetDescriptorHandleIncrementSize answers per type. The RTV/DSV payload is our own TargetView.
static_assert(sizeof(leonida::TargetView) <= rt::d3d12::kDescriptorSlotBytesRtv);
static_assert(sizeof(leonida::TargetView) <= rt::d3d12::kDescriptorSlotBytesDsv);

void log_once(const char* what, int value) {
    static std::mutex mutex;
    static std::set<std::pair<std::string, int>> seen;
    std::lock_guard lock(mutex);
    if (seen.insert({what, value}).second) std::fprintf(stderr, "[rt] d3d12: %s %d\n", what, value);
}

struct Format {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects = 0;
    bool typeless = false;  // created with MUTABLE_FORMAT so typed views can reinterpret it
};

// DXGI -> Vulkan for the formats render targets, depth buffers and common textures use. Typeless depth families
// become depth formats when the resource allows depth-stencil.
Format vulkan_format(DXGI_FORMAT format, bool depth) {
    constexpr VkImageAspectFlags color = VK_IMAGE_ASPECT_COLOR_BIT, d = VK_IMAGE_ASPECT_DEPTH_BIT,
                                 ds = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return {VK_FORMAT_R32G32B32A32_SFLOAT, color, true};
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return {VK_FORMAT_R32G32B32A32_SFLOAT, color};
    case DXGI_FORMAT_R32G32B32A32_UINT: return {VK_FORMAT_R32G32B32A32_UINT, color};
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return {VK_FORMAT_R16G16B16A16_SFLOAT, color, true};
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return {VK_FORMAT_R16G16B16A16_SFLOAT, color};
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {VK_FORMAT_R16G16B16A16_UNORM, color};
    case DXGI_FORMAT_R32G32_FLOAT: return {VK_FORMAT_R32G32_SFLOAT, color};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color, true};
    case DXGI_FORMAT_R10G10B10A2_UNORM: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32, color};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32, color};
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return {VK_FORMAT_R8G8B8A8_UNORM, color, true};
    case DXGI_FORMAT_R8G8B8A8_UNORM: return {VK_FORMAT_R8G8B8A8_UNORM, color};
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return {VK_FORMAT_R8G8B8A8_SRGB, color};
    case DXGI_FORMAT_R8G8B8A8_UINT: return {VK_FORMAT_R8G8B8A8_UINT, color};
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return {VK_FORMAT_B8G8R8A8_UNORM, color, true};
    case DXGI_FORMAT_B8G8R8A8_UNORM: return {VK_FORMAT_B8G8R8A8_UNORM, color};
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {VK_FORMAT_B8G8R8A8_SRGB, color};
    case DXGI_FORMAT_R16G16_FLOAT: return {VK_FORMAT_R16G16_SFLOAT, color};
    case DXGI_FORMAT_R8G8_UNORM: return {VK_FORMAT_R8G8_UNORM, color};
    case DXGI_FORMAT_R16_FLOAT: return {VK_FORMAT_R16_SFLOAT, color};
    case DXGI_FORMAT_R8_UNORM: return {VK_FORMAT_R8_UNORM, color};
    case DXGI_FORMAT_R32_FLOAT: return {VK_FORMAT_R32_SFLOAT, color};
    case DXGI_FORMAT_R32_UINT: return {VK_FORMAT_R32_UINT, color};
    case DXGI_FORMAT_BC1_TYPELESS: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC1_UNORM: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC1_UNORM_SRGB: return {VK_FORMAT_BC1_RGBA_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC2_TYPELESS: return {VK_FORMAT_BC2_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC2_UNORM: return {VK_FORMAT_BC2_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC2_UNORM_SRGB: return {VK_FORMAT_BC2_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC3_TYPELESS: return {VK_FORMAT_BC3_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC3_UNORM: return {VK_FORMAT_BC3_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC3_UNORM_SRGB: return {VK_FORMAT_BC3_SRGB_BLOCK, color};
    case DXGI_FORMAT_BC4_TYPELESS: return {VK_FORMAT_BC4_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC4_UNORM: return {VK_FORMAT_BC4_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC4_SNORM: return {VK_FORMAT_BC4_SNORM_BLOCK, color};
    case DXGI_FORMAT_BC5_TYPELESS: return {VK_FORMAT_BC5_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC5_UNORM: return {VK_FORMAT_BC5_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC5_SNORM: return {VK_FORMAT_BC5_SNORM_BLOCK, color};
    case DXGI_FORMAT_BC6H_TYPELESS: return {VK_FORMAT_BC6H_UFLOAT_BLOCK, color, true};
    case DXGI_FORMAT_BC6H_UF16: return {VK_FORMAT_BC6H_UFLOAT_BLOCK, color};
    case DXGI_FORMAT_BC6H_SF16: return {VK_FORMAT_BC6H_SFLOAT_BLOCK, color};
    case DXGI_FORMAT_BC7_TYPELESS: return {VK_FORMAT_BC7_UNORM_BLOCK, color, true};
    case DXGI_FORMAT_BC7_UNORM: return {VK_FORMAT_BC7_UNORM_BLOCK, color};
    case DXGI_FORMAT_BC7_UNORM_SRGB: return {VK_FORMAT_BC7_SRGB_BLOCK, color};
    case DXGI_FORMAT_D32_FLOAT: return {VK_FORMAT_D32_SFLOAT, d};
    case DXGI_FORMAT_D16_UNORM: return {VK_FORMAT_D16_UNORM, d};
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return {VK_FORMAT_D32_SFLOAT_S8_UINT, ds};
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return {VK_FORMAT_D24_UNORM_S8_UINT, ds};
    case DXGI_FORMAT_R32_TYPELESS:
        return depth ? Format{VK_FORMAT_D32_SFLOAT, d, true} : Format{VK_FORMAT_R32_SFLOAT, color, true};
    case DXGI_FORMAT_R16_TYPELESS:
        return depth ? Format{VK_FORMAT_D16_UNORM, d, true} : Format{VK_FORMAT_R16_UNORM, color, true};
    case DXGI_FORMAT_R32G8X24_TYPELESS:
        return depth ? Format{VK_FORMAT_D32_SFLOAT_S8_UINT, ds, true} : Format{};
    case DXGI_FORMAT_R24G8_TYPELESS:
        return depth ? Format{VK_FORMAT_D24_UNORM_S8_UINT, ds, true} : Format{};
    default: return {};
    }
}

class DescriptorHeap;  // defined below; DeviceState only keeps the registry

// What a device's children need from it, kept apart from the COM class so children can hold a plain pointer.
struct DeviceState {
    VulkanDeviceInfo info;
    std::unique_ptr<leonida::VulkanContext> context;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkGetPhysicalDeviceFormatProperties format_properties = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties image_format_properties = nullptr;

    bool depth_usable(VkFormat format) const {
        VkFormatProperties properties{};
        format_properties(info.physical, format, &properties);
        return properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    // Whether the GPU has this format and usage in this sample count: not every format has every count.
    bool samples_usable(const VkImageCreateInfo& image) const {
        VkImageFormatProperties properties{};
        return image_format_properties(info.physical, image.format, image.imageType, image.tiling, image.usage,
                                       image.flags, &properties) == VK_SUCCESS &&
               (properties.sampleCounts & image.samples);
    }

    // The first memory type with `required`, preferring one that also has `preferred`.
    VkDeviceMemory allocate(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred) const {
        for (const VkMemoryPropertyFlags flags : {required | preferred, required}) {
            for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
                if (!(requirements.memoryTypeBits & (1u << i))) continue;
                if ((memory.memoryTypes[i].propertyFlags & flags) != flags) continue;
                VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocate.allocationSize = requirements.size;
                allocate.memoryTypeIndex = i;
                VkDeviceMemory out = VK_NULL_HANDLE;
                if (context->vk.vkAllocateMemory(info.device, &allocate, nullptr, &out) == VK_SUCCESS) return out;
            }
        }
        return VK_NULL_HANDLE;
    }

    // Every live descriptor heap, so a CPU handle maps back to its stride and dirty range. Heaps register in their
    // constructor and unregister in their destructor; a lookup holds the lock only for the scan.
    std::mutex descriptor_heaps_mutex;
    std::vector<DescriptorHeap*> descriptor_heaps;

    // The console GPU VA space: one linear arena for now, so a resource can hand a valid address to a descriptor.
    // Reserving a real range and growing it at a frame boundary is docs/d3d12_vulkan.md section 3.3's plan.
    // Set 0 for every pipeline: images, samplers and the descriptor hash table. Created once the device exists.
    std::map<std::uint64_t, void*> va_resources;  // console VA -> Resource*, filled by Resource's constructor
    std::unique_ptr<leonida::BindlessSet> bindless;
    leonida::BindlessRegistry bindless_registry;
    leonida::RetirementQueue retirement;
    std::map<leonida::DescriptorKey, VkSampler> samplers;  // live S# -> VkSampler, retired on replacement
    std::map<leonida::DescriptorKey, VkImageView> image_views;  // live T# -> VkImageView, retired on replacement
    std::uint64_t retire_timeline = 0;

    std::uint64_t next_console_va = 0x100000000ull;
    std::uint64_t allocate_va(std::uint64_t bytes) {
        const std::uint64_t aligned = (std::max<std::uint64_t>(bytes, 0x10000) + 0xFFFF) & ~std::uint64_t(0xFFFF);
        const std::uint64_t va = next_console_va;
        next_console_va += aligned;
        return va;
    }
};

class DescriptorHeap final
    : public Child<DescriptorHeap, ID3D12DescriptorHeapStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12DescriptorHeap> {
public:
    // {4a7d2c90-5e1b-4f63-8c0a-91d3b6e2f417}
    static constexpr GUID kTag{0x4a7d2c90, 0x5e1b, 0x4f63, {0x8c, 0x0a, 0x91, 0xd3, 0xb6, 0xe2, 0xf4, 0x17}};

    DescriptorHeap(DeviceState& device, const D3D12_DESCRIPTOR_HEAP_DESC& desc)
        : device_(device), desc_(desc), stride_(rt::d3d12::descriptor_slot_bytes(desc.Type)),
          storage_(new std::byte[slot_bytes(desc)]()) {
        std::lock_guard lock(device_.descriptor_heaps_mutex);
        device_.descriptor_heaps.push_back(this);
    }

    ~DescriptorHeap() override {
        std::lock_guard lock(device_.descriptor_heaps_mutex);
        auto& heaps = device_.descriptor_heaps;
        heaps.erase(std::remove(heaps.begin(), heaps.end(), this), heaps.end());
    }

    D3D12_DESCRIPTOR_HEAP_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }
    D3D12_CPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetCPUDescriptorHandleForHeapStart() override {
        return {reinterpret_cast<SIZE_T>(storage_.get())};
    }
    // Shader-visible heaps get a GPU handle; the console-visible arena/VGPR mapping is P0's next step, so the host
    // pointer stays a placeholder here (docs/d3d12_vulkan.md:337-341).
    D3D12_GPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetGPUDescriptorHandleForHeapStart() override {
        if (!(desc_.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)) return {0};
        return {reinterpret_cast<UINT64>(storage_.get())};
    }

    UINT stride() const { return stride_; }
    UINT slot_count() const { return desc_.NumDescriptors; }
    D3D12_DESCRIPTOR_HEAP_TYPE type() const { return desc_.Type; }
    std::uint8_t* base() const { return reinterpret_cast<std::uint8_t*>(storage_.get()); }

    // True when `ptr` is slot-aligned inside this heap; `slot` receives the slot number.
    bool slot_of(const void* ptr, std::uint32_t* slot) const {
        return rt::d3d12::slot_index(reinterpret_cast<std::uintptr_t>(ptr),
                                     reinterpret_cast<std::uintptr_t>(storage_.get()), desc_.NumDescriptors, stride_,
                                     slot);
    }
    bool contains(const void* ptr) const { return slot_of(ptr, nullptr); }

    // Marks [first, first + count) as written since the last bindless scan; copies and view creation call it.
    void mark_dirty(std::uint32_t first, std::uint32_t count) {
        if (!count || !stride_) return;
        const std::uint32_t last = std::min(desc_.NumDescriptors, first + count);
        if (last <= first) return;
        std::lock_guard lock(dirty_mutex_);
        dirty_ = rt::d3d12::merge(dirty_, rt::d3d12::slot_range(first, last - first));
    }
    rt::d3d12::DirtyRange dirty() const {
        std::lock_guard lock(dirty_mutex_);
        return dirty_;
    }
    void clear_dirty() {
        std::lock_guard lock(dirty_mutex_);
        dirty_ = {};
    }

    // Copies `count` slots of this heap into `source` (P1: CopyDescriptors goes through here so the destination
    // stride is the heap's, never the CBV/SRV/UAV default). Bounds are checked against the heap.
    bool copy_slots(std::uint32_t first, const std::uint8_t* source, std::uint32_t count) {
        std::size_t bytes = 0;
        if (!rt::d3d12::slot_run_bytes(count, stride_, &bytes) || !bytes) return false;
        const std::uint64_t offset = std::uint64_t(first) * stride_;
        if (offset + bytes > std::uint64_t(desc_.NumDescriptors) * stride_) return false;
        std::memmove(base() + offset, source, bytes);
        mark_dirty(first, count);
        return true;
    }

private:
    static std::size_t slot_bytes(const D3D12_DESCRIPTOR_HEAP_DESC& desc) {
        // CreateDescriptorHeap rejects a 0 stride before constructing, so this is never 0 in practice; the max keeps
        // the allocation non-empty for a defensive caller.
        return std::max<std::size_t>(1, std::size_t(desc.NumDescriptors) * rt::d3d12::descriptor_slot_bytes(desc.Type));
    }

    DeviceState& device_;
    D3D12_DESCRIPTOR_HEAP_DESC desc_;
    UINT stride_ = 0;
    std::unique_ptr<std::byte[]> storage_;
    mutable std::mutex dirty_mutex_;
    rt::d3d12::DirtyRange dirty_;
};

// The heap that owns a CPU descriptor handle, or nullptr. The pointer stays valid for one call: D3D12 does not
// destroy a heap while a copy or a view creation uses its handles.
inline DescriptorHeap* heap_of(DeviceState& state, const void* ptr) {
    std::lock_guard lock(state.descriptor_heaps_mutex);
    for (DescriptorHeap* heap : state.descriptor_heaps) {
        if (heap->contains(ptr)) return heap;
    }
    return nullptr;
}

class Resource final
    : public Child<Resource, ID3D12ResourceStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable, ID3D12Resource> {
public:
    // {9b3e5f12-7c48-4d0e-a6b1-2f8d4c7e9a35}
    static constexpr GUID kTag{0x9b3e5f12, 0x7c48, 0x4d0e, {0xa6, 0xb1, 0x2f, 0x8d, 0x4c, 0x7e, 0x9a, 0x35}};

    Resource(DeviceState& device, const D3D12_RESOURCE_DESC& desc, const D3D12_HEAP_PROPERTIES& heap,
             D3D12_HEAP_FLAGS heap_flags)
        : device_(device), desc_(desc), heap_(heap), heap_flags_(heap_flags),
          console_va_(device.allocate_va(std::max<std::uint64_t>(desc.Width * std::max<UINT>(desc.Height, 1),
                                                                0x10000))) {
        device_.va_resources[console_va_] = this;
    }

    ~Resource() override {
        device_.va_resources.erase(console_va_);
        auto& vk = device_.context->vk;
        const VkDevice dev = device_.info.device;
        device_.context->remove_resource(id());
        if (image_) vk.vkDestroyImage(dev, image_, nullptr);
        if (buffer_) vk.vkDestroyBuffer(dev, buffer_, nullptr);
        if (memory_) vk.vkFreeMemory(dev, memory_, nullptr);
    }

    leonida::ResourceId id() const { return resource_id(this); }
    VkImage image() const { return image_; }
    VkBuffer buffer() const { return buffer_; }
    std::uint64_t console_va() const { return console_va_; }
    UINT64 STDMETHODCALLTYPE GetGPUVirtualAddress() override { return console_va_; }

    // What a descriptor view needs from the resource (PR A): the image format and range the view may name.
    VkFormat format() const { return format_; }
    VkImageAspectFlags aspects() const { return aspects_; }
    std::uint32_t mip_levels() const { return mip_levels_; }
    std::uint32_t array_layers() const { return array_layers_; }
    std::uint32_t depth() const { return depth_; }
    std::uint32_t samples() const { return samples_; }
    std::uint64_t size_bytes() const { return desc_.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? desc_.Width : 0; }

    HRESULT create_buffer() {
        format_ = VK_FORMAT_UNDEFINED;
        aspects_ = 0;
        auto& vk = device_.context->vk;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = desc_.Width;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (vk.vkCreateBuffer(device_.info.device, &info, nullptr, &buffer_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetBufferMemoryRequirements(device_.info.device, buffer_, &requirements);
        const bool host = cpu_visible();
        memory_ = host ? device_.allocate(requirements,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                          heap_.Type == D3D12_HEAP_TYPE_READBACK ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0)
                       : device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!memory_ || vk.vkBindBufferMemory(device_.info.device, buffer_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;
        if (host) {
            if (vk.vkMapMemory(device_.info.device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_) != VK_SUCCESS)
                return E_OUTOFMEMORY;
            std::memset(mapped_, 0, std::size_t(desc_.Width));  // committed resources start zeroed
        } else {
            log_once("default-heap buffers are not zero-filled yet; dimension", int(desc_.Dimension));
        }
        leonida::ResourceBinding binding{};  // a buffer: no image
        binding.buffer = buffer_;
        binding.size = desc_.Width;
        device_.context->add_resource(id(), binding);
        return S_OK;
    }

    HRESULT create_texture(const Format& format) {
        auto& vk = device_.context->vk;
        const bool volume = desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const UINT depth = volume ? desc_.DepthOrArraySize : 1;
        if (desc_.MipLevels == 0) {
            UINT64 size = std::max<UINT64>({desc_.Width, desc_.Height, depth});
            UINT16 mips = 1;
            while (size > 1) {
                size >>= 1;
                ++mips;
            }
            desc_.MipLevels = mips;
        }
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = format.typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
        info.imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = format.format;
        info.extent = {std::uint32_t(desc_.Width), desc_.Height, depth};
        info.mipLevels = desc_.MipLevels;
        info.arrayLayers = volume ? 1 : desc_.DepthOrArraySize;
        const std::uint32_t samples = std::max<UINT>(1, desc_.SampleDesc.Count);
        info.samples = VkSampleCountFlagBits(samples);
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (!(desc_.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (samples > 1 && !device_.samples_usable(info)) {
            log_once("sample count not supported by the GPU for this format:", int(samples));
            return E_INVALIDARG;
        }
        if (vk.vkCreateImage(device_.info.device, &info, nullptr, &image_) != VK_SUCCESS) return E_OUTOFMEMORY;
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device_.info.device, image_, &requirements);
        memory_ = device_.allocate(requirements, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!memory_ || vk.vkBindImageMemory(device_.info.device, image_, memory_, 0) != VK_SUCCESS)
            return E_OUTOFMEMORY;

        leonida::ResourceBinding binding{};
        binding.image = image_;
        binding.format = format.format;
        binding.aspects = format.aspects;
        binding.width = std::uint32_t(desc_.Width);
        binding.height = desc_.Height;
        binding.mip_levels = desc_.MipLevels;
        binding.array_layers = info.arrayLayers;
        binding.depth = depth;
        binding.volume = volume;
        format_ = binding.format;
        aspects_ = binding.aspects;
        mip_levels_ = binding.mip_levels;
        array_layers_ = binding.array_layers;
        depth_ = depth;
        samples_ = samples;
        binding.samples = samples;
        binding.simultaneous_access = (desc_.Flags & D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS) != 0;
        binding.not_zeroed = (heap_flags_ & D3D12_HEAP_FLAG_CREATE_NOT_ZEROED) != 0;
        // Never used yet: the first command that touches a part zero-fills it and moves it to the layout of the
        // state that command expects (vulkan_replay.h, initialize), so no submission is needed here.
        device_.context->add_resource(id(), binding, VK_IMAGE_LAYOUT_UNDEFINED);
        return S_OK;
    }

    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }

    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* properties, D3D12_HEAP_FLAGS* flags) override {
        if (properties) *properties = heap_;
        if (flags) *flags = heap_flags_;
        return S_OK;
    }

    // Buffers on upload and readback heaps are mapped persistently; the ranges are hints, as on D3D12.
    HRESULT STDMETHODCALLTYPE Map(UINT subresource, const D3D12_RANGE*, void** data) override {
        if (!mapped_ || subresource != 0) return E_INVALIDARG;
        if (data) *data = mapped_;
        return S_OK;
    }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}

private:
    bool cpu_visible() const {
        if (heap_.Type == D3D12_HEAP_TYPE_UPLOAD || heap_.Type == D3D12_HEAP_TYPE_READBACK) return true;
        return heap_.Type == D3D12_HEAP_TYPE_CUSTOM && heap_.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE &&
               heap_.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    }

    DeviceState& device_;
    D3D12_RESOURCE_DESC desc_;
    D3D12_HEAP_PROPERTIES heap_;
    D3D12_HEAP_FLAGS heap_flags_;
    std::uint64_t console_va_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspects_ = 0;
    std::uint32_t mip_levels_ = 1;
    std::uint32_t array_layers_ = 1;
    std::uint32_t depth_ = 1;
    std::uint32_t samples_ = 1;
    VkImage image_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
};

// The buffer a console VA points into, or nullptr. Only buffers have a size a descriptor can name; a texture VA is
// resolved through its ID3D12Resource instead.
inline Resource* resource_for_va(DeviceState& state, std::uint64_t va) {
    auto it = state.va_resources.upper_bound(va);
    if (it == state.va_resources.begin()) return nullptr;
    --it;
    auto* resource = static_cast<Resource*>(it->second);
    const std::uint64_t size = resource->size_bytes();
    return size && va - it->first < size ? resource : nullptr;
}

// ReplayFunctions' console VA -> VkBuffer lookup for vertex and index buffers; `user` is the owning DeviceState.
inline VkBuffer lookup_buffer(std::uint64_t va, std::uint64_t* base_va, void* user) {
    Resource* resource = resource_for_va(*static_cast<DeviceState*>(user), va);
    if (!resource) return VK_NULL_HANDLE;
    if (base_va) *base_va = resource->console_va();
    return resource->buffer();
}
namespace sampler_detail {

inline leonida::gfx10::Filter point_or_bilinear(bool linear, bool anisotropic) {
    if (anisotropic) return leonida::gfx10::Filter::aniso_bilinear;
    return linear ? leonida::gfx10::Filter::bilinear : leonida::gfx10::Filter::point;
}

inline leonida::gfx10::AddressMode address(D3D12_TEXTURE_ADDRESS_MODE mode) {
    switch (mode) {
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR: return leonida::gfx10::AddressMode::mirror;
        case D3D12_TEXTURE_ADDRESS_MODE_CLAMP: return leonida::gfx10::AddressMode::clamp_last_texel;
        case D3D12_TEXTURE_ADDRESS_MODE_BORDER: return leonida::gfx10::AddressMode::clamp_border;
        case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return leonida::gfx10::AddressMode::mirror_once_border;
        default: return leonida::gfx10::AddressMode::wrap;
    }
}

inline leonida::gfx10::CompareFunction compare(D3D12_COMPARISON_FUNC func) {
    const unsigned value = static_cast<unsigned>(func);
    if (value >= 1 && value <= 8) return static_cast<leonida::gfx10::CompareFunction>(value - 1);
    return leonida::gfx10::CompareFunction::never;
}

inline leonida::gfx10::Sampler from_d3d12(const D3D12_SAMPLER_DESC& desc) {
    leonida::gfx10::Sampler sampler{};
    const bool anisotropic = desc.Filter == D3D12_FILTER_ANISOTROPIC ||
                             desc.Filter == D3D12_FILTER_COMPARISON_ANISOTROPIC;
    const bool point = desc.Filter == D3D12_FILTER_MIN_MAG_MIP_POINT ||
                       desc.Filter == D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
    sampler.mag_filter = sampler.min_filter = anisotropic ? leonida::gfx10::Filter::aniso_bilinear
                                             : point      ? leonida::gfx10::Filter::point
                                                          : leonida::gfx10::Filter::bilinear;
    sampler.mip_filter = point ? leonida::gfx10::MipFilter::none : leonida::gfx10::MipFilter::linear;
    sampler.address_u = address(desc.AddressU);
    sampler.address_v = address(desc.AddressV);
    sampler.address_w = address(desc.AddressW);
    sampler.max_anisotropy = std::clamp<UINT>(desc.MaxAnisotropy, 1, 16);
    sampler.compare = compare(desc.ComparisonFunc);
    sampler.min_lod = desc.MinLOD;
    sampler.max_lod = desc.MaxLOD;
    sampler.lod_bias = desc.MipLODBias;
    switch (desc.BorderColor[0] == 0.0f && desc.BorderColor[1] == 0.0f && desc.BorderColor[2] == 0.0f
                ? 0u
                : (desc.BorderColor[0] == 1.0f ? 2u : 1u)) {
        case 2: sampler.border_color = leonida::gfx10::BorderColor::opaque_white; break;
        case 1: sampler.border_color = leonida::gfx10::BorderColor::opaque_black; break;
        default: sampler.border_color = leonida::gfx10::BorderColor::transparent_black; break;
    }
    sampler.filter_mode = (desc.Filter & D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT) == D3D12_FILTER_MINIMUM_MIN_MAG_MIP_POINT
                              ? leonida::gfx10::FilterMode::minimum
                          : (desc.Filter & D3D12_FILTER_MAXIMUM_MIN_MAG_MIP_POINT) ==
                                    D3D12_FILTER_MAXIMUM_MIN_MAG_MIP_POINT
                              ? leonida::gfx10::FilterMode::maximum
                              : leonida::gfx10::FilterMode::blend;
    return sampler;
}

}  // namespace sampler_detail
// --- PR C: D3D12 pipeline state -> VkPipeline ---------------------------------------------------------------

inline VkFormat dxgi_format(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
        case DXGI_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        case DXGI_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case DXGI_FORMAT_D32_FLOAT: return VK_FORMAT_D32_SFLOAT;
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
        case DXGI_FORMAT_UNKNOWN: return VK_FORMAT_UNDEFINED;
        default: return VK_FORMAT_UNDEFINED;  // the caller logs and falls back
    }
}

inline VkFormat element_format(DXGI_FORMAT format) {
    const VkFormat converted = dxgi_format(format);
    if (converted != VK_FORMAT_UNDEFINED) return converted;
    log_once("input element format not mapped, using R32G32B32A32_SFLOAT; DXGI format", int(format));
    return VK_FORMAT_R32G32B32A32_SFLOAT;
}

inline UINT element_size(DXGI_FORMAT format) {
    switch (format) {
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
        case DXGI_FORMAT_R32G32B32_FLOAT: return 12;
        case DXGI_FORMAT_R32G32_FLOAT: return 8;
        case DXGI_FORMAT_R32_FLOAT: return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
        case DXGI_FORMAT_R16G16_FLOAT: return 4;
        case DXGI_FORMAT_R8G8B8A8_UNORM: return 4;
        default: return 4;
    }
}

inline VkBlendFactor blend_factor(D3D12_BLEND blend) {
    switch (blend) {
        case D3D12_BLEND_ZERO: return VK_BLEND_FACTOR_ZERO;
        case D3D12_BLEND_ONE: return VK_BLEND_FACTOR_ONE;
        case D3D12_BLEND_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR;
        case D3D12_BLEND_INV_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case D3D12_BLEND_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
        case D3D12_BLEND_INV_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case D3D12_BLEND_DEST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
        case D3D12_BLEND_INV_DEST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case D3D12_BLEND_DEST_COLOR: return VK_BLEND_FACTOR_DST_COLOR;
        case D3D12_BLEND_INV_DEST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        default: return VK_BLEND_FACTOR_ONE;
    }
}

inline VkBlendOp blend_op(D3D12_BLEND_OP op) {
    switch (op) {
        case D3D12_BLEND_OP_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
        case D3D12_BLEND_OP_REV_SUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case D3D12_BLEND_OP_MIN: return VK_BLEND_OP_MIN;
        case D3D12_BLEND_OP_MAX: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}

inline VkCompareOp compare_op(D3D12_COMPARISON_FUNC func) {
    const unsigned value = static_cast<unsigned>(func);
    return value >= 1 && value <= 8 ? static_cast<VkCompareOp>(value - 1) : VK_COMPARE_OP_ALWAYS;
}

class PipelineState final
    : public Child<PipelineState, ID3D12PipelineStateStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12PipelineState> {
public:
    // {2a6f5b1c-3f8e-4b2a-9c1d-7e4b8a6f2d10}
    static constexpr GUID kTag{0x2a6f5b1c, 0x3f8e, 0x4b2a, {0x9c, 0x1d, 0x7e, 0x4b, 0x8a, 0x6f, 0x2d, 0x10}};

    PipelineState(DeviceState& device, VkPipeline pipeline, VkPipelineLayout layout)
        : device_(device), pipeline_(pipeline), layout_(layout) {
        // A recorded draw names the state by this object's address; the device finds the VkPipeline through
        // VulkanContext's registry, which is why the object registers itself here.
        if (pipeline_) device_.context->add_pipeline(reinterpret_cast<std::uintptr_t>(this), pipeline_, layout_);
    }

    ~PipelineState() override {
        if (pipeline_) device_.context->remove_pipeline(reinterpret_cast<std::uintptr_t>(this));
        auto& vk = device_.context->vk;
        if (pipeline_) vk.vkDestroyPipeline(device_.info.device, pipeline_, nullptr);
        if (layout_) vk.vkDestroyPipelineLayout(device_.info.device, layout_, nullptr);
    }

    VkPipeline pipeline() const { return pipeline_; }

private:
    DeviceState& device_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
};

// --- PR B: D3D12 root signature object and its Vulkan layout -----------------------------------------------

class RootSignature final
    : public Child<RootSignature, ID3D12RootSignatureStubs, ID3D12Object, ID3D12DeviceChild, ID3D12Pageable,
                   ID3D12RootSignature> {
public:
    // {6f1c2a44-8d3e-4c7a-b1f2-2a9d5e6c7b31}
    static constexpr GUID kTag{0x6f1c2a44, 0x8d3e, 0x4c7a, {0xb1, 0xf2, 0x2a, 0x9d, 0x5e, 0x6c, 0x7b, 0x31}};

    RootSignature(DeviceState& device, VkPipelineLayout layout, std::vector<VkDescriptorSetLayout> sets)
        : device_(device), layout_(layout), sets_(std::move(sets)) {}

    ~RootSignature() override {
        auto& vk = device_.context->vk;
        if (layout_) vk.vkDestroyPipelineLayout(device_.info.device, layout_, nullptr);
        for (VkDescriptorSetLayout set : sets_)
            if (set) vk.vkDestroyDescriptorSetLayout(device_.info.device, set, nullptr);
    }

    VkPipelineLayout layout() const { return layout_; }

private:
    DeviceState& device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> sets_;
};

// Deserializer object: holds the parsed description; the Vulkan layout needs a device, so that step stays in
// CreateRootSignature. Kept separate so a tool can inspect a blob without a device.
class RootSignatureDeserializer final : public ID3D12RootSignatureDeserializer {
public:
    RootSignatureDeserializer(rt::d3d12::RootSignatureDesc desc) : desc_(std::move(desc)) {}

    ULONG STDMETHODCALLTYPE AddRef() override { return ++references_; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG remaining = --references_;
        if (!remaining) delete this;
        return remaining;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12RootSignatureDeserializer)) {
            *out = static_cast<ID3D12RootSignatureDeserializer*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    const D3D12_ROOT_SIGNATURE_DESC* STDMETHODCALLTYPE GetRootSignatureDesc() override {
        return nullptr;  // the parsed description is exposed through desc(); the D3D12 struct is not rebuilt
    }

    const rt::d3d12::RootSignatureDesc& desc() const { return desc_; }

private:
    ~RootSignatureDeserializer() = default;
    std::atomic<ULONG> references_{1};
    rt::d3d12::RootSignatureDesc desc_;
};
inline HRESULT create_root_layout(DeviceState& state, const rt::d3d12::RootSignatureDesc& desc,
                                  VkPipelineLayout* layout_out, std::vector<VkDescriptorSetLayout>* sets_out) {
    rt::d3d12::RootLayoutDesc mapped{};
    std::string error;
    if (!rt::d3d12::build_root_layout(desc, mapped, &error)) {
        std::fprintf(stderr, "[rt] d3d12: root signature rejected: %s\n", error.c_str());
        return E_INVALIDARG;
    }
    auto& vk = state.context->vk;
    const VkDevice device = state.info.device;
    sets_out->assign(mapped.set_count, VK_NULL_HANDLE);
    for (std::uint32_t set = 0; set < mapped.set_count; ++set) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (const auto& binding : mapped.bindings) {
            if (binding.set != set) continue;
            VkDescriptorSetLayoutBinding entry{};
            entry.binding = binding.binding;
            entry.descriptorType = binding.type;
            entry.descriptorCount = binding.count;
            entry.stageFlags = binding.stage;
            bindings.push_back(entry);
        }
        VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        info.bindingCount = std::uint32_t(bindings.size());
        info.pBindings = bindings.data();
        if (vk.vkCreateDescriptorSetLayout(device, &info, nullptr, &(*sets_out)[set]) != VK_SUCCESS) {
            for (VkDescriptorSetLayout created : *sets_out)
                if (created) vk.vkDestroyDescriptorSetLayout(device, created, nullptr);
            sets_out->clear();
            return E_FAIL;
        }
    }
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = std::uint32_t(sets_out->size());
    layout_info.pSetLayouts = sets_out->data();
    layout_info.pushConstantRangeCount = std::uint32_t(mapped.push_ranges.size());
    layout_info.pPushConstantRanges = mapped.push_ranges.data();
    if (vk.vkCreatePipelineLayout(device, &layout_info, nullptr, layout_out) != VK_SUCCESS) {
        for (VkDescriptorSetLayout created : *sets_out)
            if (created) vk.vkDestroyDescriptorSetLayout(device, created, nullptr);
        sets_out->clear();
        return E_FAIL;
    }
    return S_OK;
}
// Builds a VkPipeline from a parsed D3D12 stream. Returns E_INVALIDARG with a log line on a stream this layer cannot
// express yet; the caller (CreatePipelineState) reports it instead of crashing.
inline HRESULT build_pipeline(DeviceState& state, const D3D12_PIPELINE_STATE_STREAM_DESC& desc,
                              VkPipeline* pipeline_out, VkPipelineLayout* layout_out) {
    rt::d3d12::PipelineStream stream{};
    std::string error;
    if (!rt::d3d12::parse_pipeline_stream(desc.pPipelineStateSubobjectStream, desc.SizeInBytes, stream, &error)) {
        std::fprintf(stderr, "[rt] d3d12: pipeline stream rejected: %s\n", error.c_str());
        return E_INVALIDARG;
    }
    if (!stream.vs || !stream.ps || !stream.vs->pShaderBytecode || !stream.ps->pShaderBytecode) return E_INVALIDARG;
    if (!state.bindless) {
        std::fprintf(stderr, "[rt] d3d12: no bindless set for a pipeline\n");
        return E_FAIL;
    }

    auto& vk = state.context->vk;
    const VkDevice device = state.info.device;
    VkShaderModule vs = VK_NULL_HANDLE, ps = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module.codeSize = stream.vs->BytecodeLength;
    module.pCode = static_cast<const std::uint32_t*>(stream.vs->pShaderBytecode);
    if (vk.vkCreateShaderModule(device, &module, nullptr, &vs) != VK_SUCCESS) return E_FAIL;
    module.codeSize = stream.ps->BytecodeLength;
    module.pCode = static_cast<const std::uint32_t*>(stream.ps->pShaderBytecode);
    if (vk.vkCreateShaderModule(device, &module, nullptr, &ps) != VK_SUCCESS) {
        vk.vkDestroyShaderModule(device, vs, nullptr);
        return E_FAIL;
    }

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = 144;  // the translator's contract: s0..s31, heap base, code base
    const VkDescriptorSetLayout set_layout = state.bindless->layout();
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vk.vkCreatePipelineLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
        vk.vkDestroyShaderModule(device, vs, nullptr);
        vk.vkDestroyShaderModule(device, ps, nullptr);
        return E_FAIL;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = ps;

    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    if (stream.input_layout) {
        for (UINT i = 0; i < stream.input_layout->NumElements; ++i) {
            const D3D12_INPUT_ELEMENT_DESC& e = stream.input_layout->pInputElementDescs[i];
            VkVertexInputAttributeDescription a{};
            a.location = e.SemanticIndex;
            a.binding = e.InputSlot;
            a.format = element_format(e.Format);
            a.offset = e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT ? 0 : e.AlignedByteOffset;
            attributes.push_back(a);
        }
        std::uint32_t slots = 0;
        for (const auto& a : attributes) slots = std::max(slots, a.binding + 1);
        bindings.resize(slots);
        for (std::uint32_t slot = 0; slot < slots; ++slot) {
            std::uint32_t stride = 0;
            for (std::size_t i = 0; i < attributes.size(); ++i)
                if (attributes[i].binding == slot)
                    stride = std::max(stride, attributes[i].offset +
                                                  element_size(stream.input_layout->pInputElementDescs[i].Format));
            bindings[slot] = {slot, stride, VK_VERTEX_INPUT_RATE_VERTEX};
        }
    }
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = std::uint32_t(bindings.size());
    vertex_input.pVertexBindingDescriptions = bindings.data();
    vertex_input.vertexAttributeDescriptionCount = std::uint32_t(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    switch (stream.topology) {
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT: assembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break;
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE: assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE: assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
        default: assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break;
    }
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.lineWidth = 1.0f;
    raster.cullMode = VK_CULL_MODE_NONE;
    if (stream.rasterizer) {
        raster.depthClampEnable = VK_FALSE;  // the depthClamp feature is not enabled: depth clipping stays on
        raster.polygonMode = stream.rasterizer->FillMode == D3D12_FILL_MODE_WIREFRAME ? VK_POLYGON_MODE_LINE
                                                                                     : VK_POLYGON_MODE_FILL;
        raster.cullMode = stream.rasterizer->CullMode == D3D12_CULL_MODE_FRONT ? VK_CULL_MODE_FRONT_BIT
                          : stream.rasterizer->CullMode == D3D12_CULL_MODE_BACK ? VK_CULL_MODE_BACK_BIT
                                                                                : VK_CULL_MODE_NONE;
        raster.frontFace = stream.rasterizer->FrontCounterClockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                    : VK_FRONT_FACE_CLOCKWISE;
        raster.depthBiasEnable = stream.rasterizer->DepthBias != 0 ? VK_TRUE : VK_FALSE;
        raster.depthBiasConstantFactor = float(stream.rasterizer->DepthBias);
        raster.depthBiasSlopeFactor = stream.rasterizer->SlopeScaledDepthBias;
        raster.depthBiasClamp = stream.rasterizer->DepthBiasClamp;
    }
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkPipelineColorBlendAttachmentState> attachments(std::max<std::uint32_t>(stream.rtv_count, 1));
    for (std::uint32_t i = 0; i < attachments.size(); ++i) {
        auto& a = attachments[i];
        if (stream.blend && i < 8) {
            const D3D12_RENDER_TARGET_BLEND_DESC& b = stream.blend->RenderTarget[i];
            a.blendEnable = b.BlendEnable;
            a.srcColorBlendFactor = blend_factor(b.SrcBlend);
            a.dstColorBlendFactor = blend_factor(b.DestBlend);
            a.colorBlendOp = blend_op(b.BlendOp);
            a.srcAlphaBlendFactor = blend_factor(b.SrcBlendAlpha);
            a.dstAlphaBlendFactor = blend_factor(b.DestBlendAlpha);
            a.alphaBlendOp = blend_op(b.BlendOpAlpha);
            a.colorWriteMask = (b.RenderTargetWriteMask & D3D12_COLOR_WRITE_ENABLE_RED ? VK_COLOR_COMPONENT_R_BIT : 0) |
                               (b.RenderTargetWriteMask & D3D12_COLOR_WRITE_ENABLE_GREEN ? VK_COLOR_COMPONENT_G_BIT : 0) |
                               (b.RenderTargetWriteMask & D3D12_COLOR_WRITE_ENABLE_BLUE ? VK_COLOR_COMPONENT_B_BIT : 0) |
                               (b.RenderTargetWriteMask & D3D12_COLOR_WRITE_ENABLE_ALPHA ? VK_COLOR_COMPONENT_A_BIT : 0);
        } else {
            a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                               VK_COLOR_COMPONENT_A_BIT;
        }
    }
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    if (!state.info.independent_blend && attachments.size() > 1) {
        // Without independentBlend Vulkan requires every attachment state to be identical; D3D12 does not, so the
        // first target's state is used for all of them and the collapse is reported once.
        log_once("independentBlend is not enabled: per-target blend state collapsed", int(attachments.size()));
        std::fill(attachments.begin() + 1, attachments.end(), attachments.front());
    }
    blend.attachmentCount = std::uint32_t(attachments.size());
    blend.pAttachments = attachments.data();
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    bool has_depth = false;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    if (stream.depth_stencil) {
        depth.depthTestEnable = stream.depth_stencil->DepthEnable;
        depth.depthWriteEnable = stream.depth_stencil->DepthWriteMask != D3D12_DEPTH_WRITE_MASK_ZERO;
        depth.depthCompareOp = compare_op(stream.depth_stencil->DepthFunc);
        depth.stencilTestEnable = stream.depth_stencil->StencilEnable;
        has_depth = depth.depthTestEnable || depth.stencilTestEnable;
        depth_format = dxgi_format(stream.dsv_format);
        if (depth_format == VK_FORMAT_UNDEFINED) has_depth = false;
    }
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;

    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    // Every render-target format the stream declares, in order: a deferred-rendering pipeline writes several
    // attachments in one pass, and dynamic rendering needs the whole list up front.
    std::vector<VkFormat> color_formats;
    for (std::uint32_t i = 0; i < stream.rtv_count && i < leonida::max_render_targets; ++i) {
        VkFormat format = dxgi_format(stream.rtv_formats[i]);
        if (format == VK_FORMAT_UNDEFINED) {
            log_once("pipeline color format not mapped, using R8G8B8A8_UNORM; DXGI format", int(stream.rtv_formats[i]));
            format = VK_FORMAT_R8G8B8A8_UNORM;
        }
        color_formats.push_back(format);
    }
    if (color_formats.empty()) color_formats.push_back(VK_FORMAT_R8G8B8A8_UNORM);  // a pipeline with no render target
    rendering.colorAttachmentCount = std::uint32_t(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    if (has_depth) {
        rendering.depthAttachmentFormat = depth_format;
        rendering.stencilAttachmentFormat =
            depth_format == VK_FORMAT_D24_UNORM_S8_UINT ? depth_format : VK_FORMAT_UNDEFINED;
    }

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &rendering;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex_input;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = has_depth ? &depth : nullptr;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const VkResult result = vk.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
    vk.vkDestroyShaderModule(device, vs, nullptr);
    vk.vkDestroyShaderModule(device, ps, nullptr);
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[rt] d3d12: vkCreateGraphicsPipelines returned %d\n", int(result));
        vk.vkDestroyPipelineLayout(device, layout, nullptr);
        return E_FAIL;
    }
    *pipeline_out = pipeline;
    *layout_out = layout;
    return S_OK;
}
// The Vulkan objects create_vulkan_device_standalone made, destroyed with the device.
struct OwnedRuntime {
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkDestroyInstance destroy_instance = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
};

class Device final
    : public Object<Device, ID3D12Device11Stubs, ID3D12Object, ID3D12Device, ID3D12Device1, ID3D12Device2,
                    ID3D12Device3, ID3D12Device4, ID3D12Device5, ID3D12Device6, ID3D12Device7, ID3D12Device8,
                    ID3D12Device9, ID3D12Device10, ID3D12Device11> {
public:
    // {c58e1d27-3a96-4b4f-9e72-6d0a8b5f3c61}
    static constexpr GUID kTag{0xc58e1d27, 0x3a96, 0x4b4f, {0x9e, 0x72, 0x6d, 0x0a, 0x8b, 0x5f, 0x3c, 0x61}};

    explicit Device(const VulkanDeviceInfo& info) {
        state_.info = info;
        const auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            info.get_instance_proc(info.instance, "vkGetDeviceProcAddr"));
        state_.format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceFormatProperties"));
        state_.image_format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceImageFormatProperties"));
        const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            info.get_instance_proc(info.instance, "vkGetPhysicalDeviceMemoryProperties"));
        if (!get_device || !state_.format_properties || !state_.image_format_properties || !memory_properties) return;
        memory_properties(info.physical, &state_.memory);

        if (state_.info.max_push_constants_size == 0 && info.physical && info.get_instance_proc) {
            const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
                info.get_instance_proc(info.instance, "vkGetPhysicalDeviceProperties"));
            if (get_props) {
                VkPhysicalDeviceProperties props{};
                get_props(info.physical, &props);
                state_.info.max_push_constants_size = props.limits.maxPushConstantsSize;
                state_.info.push_constants_need_fallback = (state_.info.max_push_constants_size < 144);
            }
        }

        state_.context = std::make_unique<leonida::VulkanContext>(get_device, info.device,
                                                                  info.separate_depth_stencil_layouts);
    }

    ~Device() override {
        state_.bindless.reset();  // views/samplers must go before the Vulkan device they belong to
        if (state_.context) {
            for (const auto& entry : state_.image_views)
                state_.context->vk.vkDestroyImageView(state_.info.device, entry.second, nullptr);
            state_.image_views.clear();
            for (const auto& entry : state_.samplers)
                state_.context->vk.vkDestroySampler(state_.info.device, entry.second, nullptr);
            state_.samplers.clear();
        }
        if (state_.context) {
            {
                std::lock_guard lock(state_.context->submit_mutex());
                state_.context->vk.vkDeviceWaitIdle(state_.info.device);
            }
            // Nothing can be submitted any more. The VkPipelines belong to the PipelineState objects (which may be
            // destroyed after this device), so the registry is only emptied, never destroyed from here.
            state_.context->clear_pipelines();
        }
        state_.context.reset();
        if (owned_.device) owned_.destroy_device(owned_.device, nullptr);
        if (owned_.instance) owned_.destroy_instance(owned_.instance, nullptr);
    }

    bool ok() const { return state_.context && state_.context->loaded(); }
    const VulkanDeviceInfo& info() const { return state_.info; }
    void own(const OwnedRuntime& owned) { owned_ = owned; }
    void set_outer(IUnknown* outer) { outer_ = outer; }

    UINT STDMETHODCALLTYPE GetNodeCount() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return S_OK; }

    HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* desc, REFIID riid,
                                                 void** out) override {
        if (!desc) return E_INVALIDARG;
        // Copy and compute queues share the one graphics VkQueue for now (section 8.2, question 12).
        if (desc->Type != D3D12_COMMAND_LIST_TYPE_DIRECT) log_once("command queue type on the direct VkQueue:", desc->Type);
        return create_command_queue(*state_.context, state_.info.queue, state_.info.queue_family, *desc, riid, out,
                                    identity());
    }

    HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** out) override {
        return create_command_allocator(type, riid, out, identity());
    }

    HRESULT STDMETHODCALLTYPE CreateCommandList(UINT, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* allocator,
                                                ID3D12PipelineState* state, REFIID riid, void** out) override {
        if (state) stub("ID3D12Device::CreateCommandList(initial pipeline state)");
        return create_command_list(type, allocator, riid, out, identity());
    }

    // Device4: a list created closed, without an allocator.
    HRESULT STDMETHODCALLTYPE CreateCommandList1(UINT, D3D12_COMMAND_LIST_TYPE type, D3D12_COMMAND_LIST_FLAGS,
                                                 REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        ID3D12CommandAllocator* allocator = nullptr;
        HRESULT hr = create_command_allocator(type, IID_PPV_ARGS(&allocator), identity());
        if (FAILED(hr)) return hr;
        ID3D12GraphicsCommandList* list = nullptr;
        hr = create_command_list(type, allocator, IID_PPV_ARGS(&list), identity());
        allocator->Release();
        if (FAILED(hr)) return hr;
        list->Close();
        hr = list->QueryInterface(riid, out);
        list->Release();
        return hr;
    }

    HRESULT STDMETHODCALLTYPE CreateFence(UINT64 initial, D3D12_FENCE_FLAGS flags, REFIID riid, void** out) override {
        return create_fence(*state_.context, initial, flags, riid, out, identity());
    }

    HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* desc, REFIID riid,
                                                   void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc || !desc->NumDescriptors) return E_INVALIDARG;
        if (!rt::d3d12::descriptor_slot_bytes(desc->Type)) return E_INVALIDARG;  // no slot size, no heap
        auto* heap = new DescriptorHeap(state_, *desc);
        heap->attach_device(identity());
        return hand_out(heap, riid, out);
    }

    UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE type) override {
        return rt::d3d12::descriptor_slot_bytes(type);
    }

    void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource* resource, const D3D12_RENDER_TARGET_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        leonida::TargetView view{};
        if (const Resource* r = Resource::from<Resource>(resource)) {
            const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
            view = {r->id(), 0, 0, rd.DepthOrArraySize};
            if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D) {
                view = {r->id(), desc->Texture2D.MipSlice, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DMS) {
                view = {r->id(), 0, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
                view = {r->id(), desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc) {
                log_once("render-target view dimension not supported:", desc->ViewDimension);
                view = {};
            }
        } else if (resource) {
            stub("ID3D12Device::CreateRenderTargetView(resource from another device)");
        }
        std::memcpy(reinterpret_cast<void*>(destination.ptr), &view, sizeof(view));
    }

    void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource* resource, const D3D12_DEPTH_STENCIL_VIEW_DESC* desc,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!destination.ptr) return;
        leonida::TargetView view{};
        if (const Resource* r = Resource::from<Resource>(resource)) {
            const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
            view = {r->id(), 0, 0, rd.DepthOrArraySize};
            if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2D) {
                view = {r->id(), desc->Texture2D.MipSlice, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2DMS) {
                view = {r->id(), 0, 0, 1};
            } else if (desc && desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2DARRAY) {
                view = {r->id(), desc->Texture2DArray.MipSlice, desc->Texture2DArray.FirstArraySlice,
                        desc->Texture2DArray.ArraySize};
            } else if (desc) {
                log_once("depth-stencil view dimension not supported:", desc->ViewDimension);
                view = {};
            }
        } else if (resource) {
            stub("ID3D12Device::CreateDepthStencilView(resource from another device)");
        }
        std::memcpy(reinterpret_cast<void*>(destination.ptr), &view, sizeof(view));
    }

    // ---- CBV/SRV/UAV views (P2 slice): encode the GFX10 descriptor and write the heap slot ----

    // Writes an 8-dword T#/V# into a CBV/SRV/UAV heap slot and marks it dirty for the bindless scan. The view's
    // registration in bindless set 0 (VkImageView/buffer index) belongs to the #157 work; this writes what a shader
    // reads from a descriptor table.
    bool write_view(D3D12_CPU_DESCRIPTOR_HANDLE destination, const void* words, std::size_t bytes) {
        if (!destination.ptr) return false;
        DescriptorHeap* heap = heap_of(state_, reinterpret_cast<const void*>(destination.ptr));
        std::uint32_t slot = 0;
        if (!heap || heap->type() != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ||
            !heap->slot_of(reinterpret_cast<const void*>(destination.ptr), &slot)) {
            log_once("descriptor write outside a CBV/SRV/UAV heap", 0);
            return false;
        }
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words, std::min(bytes, std::size_t(heap->stride())));
        heap->mark_dirty(slot, 1);
        return true;
    }

    void STDMETHODCALLTYPE CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!desc) return;
        leonida::gfx10::BufferWords words{};
        if (!rt::d3d12::views::encode_cbv(desc->BufferLocation, desc->SizeInBytes, words))
            log_once("constant buffer view rejected, size", int(desc->SizeInBytes));
        else
            write_view(destination, words.data(), sizeof(words));
    }

    void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource* resource,
                                                    const D3D12_SHADER_RESOURCE_VIEW_DESC* desc,
                                                    D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        const Resource* r = Resource::from<Resource>(resource);
        if (!r || !desc) return;
        const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
        const VkFormat view_format =
            vulkan_format(desc->Format == DXGI_FORMAT_UNKNOWN ? rd.Format : desc->Format, false).format;
        if (desc->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
            leonida::gfx10::BufferWords words{};
            const rt::d3d12::views::BufferResource buffer{r->console_va(), rd.Width};
            if (rt::d3d12::views::encode_buffer_srv(*desc, buffer, view_format, words))
                write_view(destination, words.data(), sizeof(words));
            else
                log_once("buffer SRV not encoded, format", int(desc->Format));
            return;
        }
        leonida::gfx10::ImageWords words{};
        rt::d3d12::views::TextureResource texture{};
        texture.address = r->console_va();
        texture.width = static_cast<std::uint32_t>(rd.Width);
        texture.height = rd.Height;
        texture.depth = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? rd.DepthOrArraySize : 1;
        texture.mip_levels = rd.MipLevels;
        texture.array_layers = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : rd.DepthOrArraySize;
        texture.samples = std::max(rd.SampleDesc.Count, 1u);
        if (!rt::d3d12::views::encode_texture_srv(*desc, texture, view_format, words)) {
            log_once("texture SRV not encoded, dimension", int(desc->ViewDimension));
            return;
        }
        if (!write_view(destination, words.data(), sizeof(words))) return;

        // PR B: the shader reaches this texture through the bindless set, not through the console VA. Create the
        // image view the descriptor array needs, register it and insert the T# -> index mapping, then flush.
        if (!state_.bindless) return;
        VkImageSubresourceRange range{};
        range.aspectMask = r->aspects() ? r->aspects() : VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel = 0;
        range.levelCount = VK_REMAINING_MIP_LEVELS;
        range.baseArrayLayer = 0;
        range.layerCount = VK_REMAINING_ARRAY_LAYERS;
        VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        leonida::ImageDimension dimension = leonida::ImageDimension::dim_2d_array;
        switch (desc->ViewDimension) {
            case D3D12_SRV_DIMENSION_TEXTURE1D:
                view_type = VK_IMAGE_VIEW_TYPE_1D;
                dimension = leonida::ImageDimension::dim_1d;
                range.baseMipLevel = desc->Texture1D.MostDetailedMip;
                range.levelCount = desc->Texture1D.MipLevels ? desc->Texture1D.MipLevels : VK_REMAINING_MIP_LEVELS;
                range.layerCount = 1;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2D:
                view_type = VK_IMAGE_VIEW_TYPE_2D;
                dimension = leonida::ImageDimension::dim_2d;
                range.baseMipLevel = desc->Texture2D.MostDetailedMip;
                range.levelCount = desc->Texture2D.MipLevels ? desc->Texture2D.MipLevels : VK_REMAINING_MIP_LEVELS;
                range.layerCount = 1;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE3D:
                view_type = VK_IMAGE_VIEW_TYPE_3D;
                dimension = leonida::ImageDimension::dim_3d;
                range.baseMipLevel = desc->Texture3D.MostDetailedMip;
                range.levelCount = desc->Texture3D.MipLevels ? desc->Texture3D.MipLevels : VK_REMAINING_MIP_LEVELS;
                range.layerCount = 1;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
            case D3D12_SRV_DIMENSION_TEXTURECUBE:
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
                view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                dimension = leonida::ImageDimension::dim_2d_array;
                range.baseMipLevel = desc->Texture2DArray.MostDetailedMip;
                range.levelCount = desc->Texture2DArray.MipLevels ? desc->Texture2DArray.MipLevels
                                                                  : VK_REMAINING_MIP_LEVELS;
                range.baseArrayLayer = desc->Texture2DArray.FirstArraySlice;
                range.layerCount = desc->Texture2DArray.ArraySize;
                break;
            default:
                range.levelCount = 1;
                range.layerCount = 1;
                break;
        }
        leonida::ResourceBinding viewer{};
        viewer.image = r->image();
        viewer.format = r->format();
        viewer.aspects = r->aspects();
        viewer.width = static_cast<std::uint32_t>(rd.Width);
        viewer.height = rd.Height;
        viewer.mip_levels = r->mip_levels();
        viewer.array_layers = r->array_layers();
        viewer.depth = r->depth();
        VkImageViewCreateInfo srv_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        srv_view_info.image = viewer.image;
        srv_view_info.viewType = view_type;
        srv_view_info.format = viewer.format;
        srv_view_info.subresourceRange = range;
        VkImageView image_view = VK_NULL_HANDLE;
        state_.context->vk.vkCreateImageView(state_.info.device, &srv_view_info, nullptr, &image_view);
        if (!image_view) {
            log_once("image view creation failed for an SRV", int(desc->ViewDimension));
            return;
        }
        const std::uint32_t index = state_.bindless->add_image(dimension, image_view, VK_IMAGE_LAYOUT_GENERAL);
        if (!index) {
            log_once("bindless image array is full, dimension", int(dimension));
            return;
        }        const leonida::DescriptorKey view_key = leonida::make_tsharp_key(words.data());
        state_.bindless->insert_key(view_key, index);
        if (const auto it = state_.image_views.find(view_key); it != state_.image_views.end()) {
            const VkImageView old = it->second;
            state_.retirement.retire(++state_.retire_timeline,
                                     [vk = state_.context->vk, device = state_.info.device, old] {
                                         vk.vkDestroyImageView(device, old, nullptr);
                                     });
            it->second = image_view;
        } else {
            state_.image_views.emplace(view_key, image_view);
        }
        if (state_.bindless->flush_needs_idle()) state_.context->vk.vkDeviceWaitIdle(state_.info.device);
        state_.bindless->flush();
        if (state_.retire_timeline) {
            state_.context->vk.vkDeviceWaitIdle(state_.info.device);
            state_.retirement.collect(state_.retire_timeline);
        }
    }

    HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT node_mask, const void* blob, SIZE_T size, REFIID riid,
                                                  void** out) override {
        (void)node_mask;
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!blob || !size) return E_INVALIDARG;
        rt::d3d12::RootSignatureDesc desc{};
        std::string error;
        if (!rt::d3d12::parse_root_signature(blob, size, desc, &error)) {
            std::fprintf(stderr, "[rt] d3d12: root signature parse failed: %s\n", error.c_str());
            return E_INVALIDARG;
        }
        VkPipelineLayout layout = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> sets;
        const HRESULT hr = create_root_layout(state_, desc, &layout, &sets);
        if (FAILED(hr)) return hr;
        auto* object = new RootSignature(state_, layout, std::move(sets));
        object->attach_device(identity());
        return hand_out(object, riid, out);
    }
    HRESULT STDMETHODCALLTYPE CreatePipelineState(const D3D12_PIPELINE_STATE_STREAM_DESC* desc, REFIID riid,
                                                  void** out) override {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (!desc || !desc->pPipelineStateSubobjectStream || !desc->SizeInBytes) return E_INVALIDARG;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        const HRESULT hr = build_pipeline(state_, *desc, &pipeline, &layout);
        if (FAILED(hr)) return hr;
        auto* object = new PipelineState(state_, pipeline, layout);
        object->attach_device(identity());
        return hand_out(object, riid, out);
    }
    void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (!desc || !destination.ptr) return;
        DescriptorHeap* heap = heap_of(state_, reinterpret_cast<const void*>(destination.ptr));
        std::uint32_t slot = 0;
        if (!heap || heap->type() != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ||
            !heap->slot_of(reinterpret_cast<const void*>(destination.ptr), &slot)) {
            log_once("sampler write outside a sampler heap", 0);
            return;
        }
        const leonida::gfx10::Sampler sampler = sampler_detail::from_d3d12(*desc);
        const leonida::gfx10::SamplerWords words = leonida::gfx10::encode(sampler);
        std::memcpy(reinterpret_cast<void*>(destination.ptr), words.data(),
                    std::min<std::size_t>(sizeof(words), heap->stride()));
        heap->mark_dirty(slot, 1);
        if (!state_.bindless) return;  // no set 0: the descriptor is written, the shader just cannot look it up
        const VkSamplerCreateInfo info = leonida::gfx10::vulkan_sampler(sampler);
        VkSampler vk_sampler = VK_NULL_HANDLE;
        if (state_.context->vk.vkCreateSampler(state_.info.device, &info, nullptr, &vk_sampler) != VK_SUCCESS) {
            log_once("vkCreateSampler failed", 0);
            return;
        }
        const std::uint32_t index = state_.bindless->add_sampler(vk_sampler);
        if (!index) {
            log_once("bindless sampler array is full", 0);
            return;
        }
        const leonida::DescriptorKey key = leonida::make_ssharp_key(words.data());
        state_.bindless->insert_key(key, index);
        // Replacing an S# retires the old VkSampler at the next timeline value; the collect() below runs only after
        // the queue is idle, so no command buffer can still be sampling through it.
        if (const auto it = state_.samplers.find(key); it != state_.samplers.end()) {
            const VkSampler old = it->second;
            state_.retirement.retire(++state_.retire_timeline,
                                     [vk = state_.context->vk, device = state_.info.device, old] {
                                         vk.vkDestroySampler(device, old, nullptr);
                                     });
            it->second = vk_sampler;
        } else {
            state_.samplers.emplace(key, vk_sampler);
        }
        if (state_.bindless->flush_needs_idle()) state_.context->vk.vkDeviceWaitIdle(state_.info.device);
        state_.bindless->flush();
        if (state_.retire_timeline) {
            state_.context->vk.vkDeviceWaitIdle(state_.info.device);
            state_.retirement.collect(state_.retire_timeline);
        }
    }
    void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource* resource,
                                                     ID3D12Resource* counter_resource,
                                                     const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                                                     D3D12_CPU_DESCRIPTOR_HANDLE destination) override {
        if (counter_resource) log_once("UAV counter resources are not bound yet", 0);
        const Resource* r = Resource::from<Resource>(resource);
        if (!r || !desc) return;
        const D3D12_RESOURCE_DESC rd = const_cast<Resource*>(r)->GetDesc();
        const VkFormat view_format =
            vulkan_format(desc->Format == DXGI_FORMAT_UNKNOWN ? rd.Format : desc->Format, false).format;
        if (desc->ViewDimension == D3D12_UAV_DIMENSION_BUFFER) {
            leonida::gfx10::BufferWords words{};
            const rt::d3d12::views::BufferResource buffer{r->console_va(), rd.Width};
            if (rt::d3d12::views::encode_buffer_uav(*desc, buffer, view_format, words))
                write_view(destination, words.data(), sizeof(words));
            else
                log_once("buffer UAV not encoded, format", int(desc->Format));
            return;
        }
        leonida::gfx10::ImageWords words{};
        rt::d3d12::views::TextureResource texture{};
        texture.address = r->console_va();
        texture.width = static_cast<std::uint32_t>(rd.Width);
        texture.height = rd.Height;
        texture.depth = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? rd.DepthOrArraySize : 1;
        texture.mip_levels = rd.MipLevels;
        texture.array_layers = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : rd.DepthOrArraySize;
        texture.samples = std::max(rd.SampleDesc.Count, 1u);
        if (rt::d3d12::views::encode_texture_uav(*desc, texture, view_format, words))
            write_view(destination, words.data(), sizeof(words));
        else
            log_once("texture UAV not encoded, dimension", int(desc->ViewDimension));
    }
    // A descriptor is its slot's bytes (a TargetView in RTV and DSV heaps, otherwise what the shader loads), so a
    // copy is a byte copy. D3D12 walks the destination and source ranges together, one descriptor at a time; a
    // null size array means every range holds one descriptor. Whichever side runs out first ends the copy.
    void STDMETHODCALLTYPE CopyDescriptors(UINT destination_ranges, const D3D12_CPU_DESCRIPTOR_HANDLE* destinations,
                                           const UINT* destination_sizes, UINT source_ranges,
                                           const D3D12_CPU_DESCRIPTOR_HANDLE* sources, const UINT* source_sizes,
                                           D3D12_DESCRIPTOR_HEAP_TYPE type) override {
        if (!destinations || !sources) return;
        const UINT type_stride = rt::d3d12::descriptor_slot_bytes(type);
        if (!type_stride) return;  // an unknown heap type has no slot size
        UINT d = 0, s = 0, d_at = 0, s_at = 0;  // the range on each side, and the position in it
        while (d < destination_ranges && s < source_ranges) {
            const UINT d_size = destination_sizes ? destination_sizes[d] : 1;
            const UINT s_size = source_sizes ? source_sizes[s] : 1;
            if (d_at >= d_size || !destinations[d].ptr) {
                ++d;
                d_at = 0;
                continue;
            }
            if (s_at >= s_size || !sources[s].ptr) {
                ++s;
                s_at = 0;
                continue;
            }
            DescriptorHeap* destination_heap = heap_of(state_, reinterpret_cast<const void*>(destinations[d].ptr));
            DescriptorHeap* source_heap = heap_of(state_, reinterpret_cast<const void*>(sources[s].ptr));
            std::uint32_t d_slot = 0, s_slot = 0;
            if (!destination_heap || !source_heap ||
                !destination_heap->slot_of(reinterpret_cast<const void*>(destinations[d].ptr), &d_slot) ||
                !source_heap->slot_of(reinterpret_cast<const void*>(sources[s].ptr), &s_slot)) {
                ++d;
                d_at = 0;  // a handle outside every heap: nothing to copy, advance both sides by one range
                ++s;
                s_at = 0;
                continue;
            }
            // The handles name the start of the range; d_at/s_at is how far into it this pass already copied. Both
            // sides must be offset, or a multi-range copy writes its second pass over the first one.
            const UINT copied = rt::d3d12::copy_slots_between(
                destination_heap->base(), destination_heap->slot_count(), d_slot + d_at, source_heap->base(),
                source_heap->slot_count(), s_slot + s_at, destination_heap->stride(),
                std::min(d_size - d_at, s_size - s_at));
            if (!copied) {
                ++d;
                d_at = 0;
                ++s;
                s_at = 0;
                continue;
            }
            destination_heap->mark_dirty(d_slot + d_at, copied);
            d_at += copied;
            s_at += copied;
            if (d_at >= d_size) {
                ++d;
                d_at = 0;
            }
            if (s_at >= s_size) {
                ++s;
                s_at = 0;
            }
        }
    }
    void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT count, D3D12_CPU_DESCRIPTOR_HANDLE destination,
                                                 D3D12_CPU_DESCRIPTOR_HANDLE source,
                                                 D3D12_DESCRIPTOR_HEAP_TYPE type) override {
        if (!destination.ptr || !source.ptr || !count) return;
        const UINT type_stride = rt::d3d12::descriptor_slot_bytes(type);
        if (!type_stride) return;
        DescriptorHeap* destination_heap = heap_of(state_, reinterpret_cast<const void*>(destination.ptr));
        std::uint32_t d_slot = 0;
        if (destination_heap &&
            destination_heap->slot_of(reinterpret_cast<const void*>(destination.ptr), &d_slot)) {
            const UINT stride = destination_heap->stride();
            DescriptorHeap* source_heap = heap_of(state_, reinterpret_cast<const void*>(source.ptr));
            std::uint32_t s_slot = 0;
            if (source_heap && source_heap->slot_of(reinterpret_cast<const void*>(source.ptr), &s_slot)) {
                const UINT copied = rt::d3d12::copy_slots_between(
                    destination_heap->base(), destination_heap->slot_count(), d_slot, source_heap->base(),
                    source_heap->slot_count(), s_slot, stride, count);
                destination_heap->mark_dirty(d_slot, copied);
                return;
            }
            // The source is not inside a known heap: copy raw bytes with the destination heap's stride.
            const UINT n = std::min(count, destination_heap->slot_count() - d_slot);
            std::size_t bytes = 0;
            if (!rt::d3d12::slot_run_bytes(n, stride, &bytes)) return;
            std::memmove(destination_heap->base() + std::size_t(d_slot) * stride,
                         reinterpret_cast<const void*>(source.ptr), bytes);
            destination_heap->mark_dirty(d_slot, n);
            return;
        }
        // A handle outside every heap: fall back to the type stride (the slot range is unknown, so no dirty mark).
        std::size_t bytes = 0;
        if (!rt::d3d12::slot_run_bytes(count, type_stride, &bytes)) return;
        std::memmove(reinterpret_cast<void*>(destination.ptr), reinterpret_cast<const void*>(source.ptr), bytes);
    }
    HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES* heap, D3D12_HEAP_FLAGS heap_flags,
                                                      const D3D12_RESOURCE_DESC* desc, D3D12_RESOURCE_STATES,
                                                      const D3D12_CLEAR_VALUE*, REFIID riid, void** out) override {
        if (out) *out = nullptr;
        if (!heap || !desc) return E_INVALIDARG;
        auto* resource = new Resource(state_, *desc, *heap, heap_flags);
        resource->attach_device(identity());
        HRESULT hr = E_NOTIMPL;
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            hr = resource->create_buffer();
        } else if (desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
            log_once("committed resource dimension not supported:", desc->Dimension);
        } else if (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
                   (desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))) {
            log_once("volume attachment views not supported, flags:", int(desc->Flags));
        } else if (heap->Type == D3D12_HEAP_TYPE_UPLOAD || heap->Type == D3D12_HEAP_TYPE_READBACK) {
            hr = E_INVALIDARG;  // D3D12 places textures on default heaps unless they are row-major
        } else if (const UINT samples = desc->SampleDesc.Count;
                   samples > 1 && ((samples & (samples - 1)) || desc->MipLevels != 1 ||
                                   desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                                   !(desc->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)))) {
            hr = E_INVALIDARG;  // D3D12: a multisampled texture is a single-mip 2D render target or depth-stencil
        } else {
            Format format = vulkan_format(desc->Format, desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
            if (format.format == VK_FORMAT_D24_UNORM_S8_UINT && !state_.depth_usable(format.format)) {
                log_once("D24_UNORM_S8_UINT unavailable, using D32_SFLOAT_S8_UINT; format", desc->Format);
                format.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
            }
            if (format.format == VK_FORMAT_UNDEFINED) {
                log_once("texture format not supported yet:", desc->Format);
                hr = E_INVALIDARG;
            } else {
                hr = resource->create_texture(format);
            }
        }
        if (FAILED(hr)) {
            resource->Release();
            return hr;
        }
        if (!out) {  // a capability check: D3D12 allows a null output
            resource->Release();
            return S_FALSE;
        }
        return hand_out(resource, riid, out);
    }

    // Footprints for copies through a buffer: rows padded to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (512 for planar
    // depth-stencil), subresources placed at D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, block-compressed sizes rounded up to whole blocks, depth-stencil
    // formats as two planes. d3d12_vulkan_device_test compares every output with the PC runtime's.
    void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC* desc, UINT first, UINT count, UINT64 base,
                                                 D3D12_PLACED_SUBRESOURCE_FOOTPRINT* layouts, UINT* rows,
                                                 UINT64* row_sizes, UINT64* total) override {
        const auto fail = [&] {
            for (UINT i = 0; i < count; ++i) {
                if (layouts) std::memset(&layouts[i], 0xff, sizeof(layouts[i]));
                if (rows) rows[i] = ~0u;
                if (row_sizes) row_sizes[i] = ~0ull;
            }
            if (total) *total = ~0ull;
        };
        if (!desc) return fail();
        const auto align = [](UINT64 value, UINT64 to) { return (value + to - 1) / to * to; };
        if (desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            if (first != 0 || count != 1) return fail();
            if (layouts) layouts[0] = {base, {DXGI_FORMAT_UNKNOWN, UINT(desc->Width), 1, 1,
                                              UINT(align(desc->Width, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT))}};
            if (rows) rows[0] = 1;
            if (row_sizes) row_sizes[0] = desc->Width;
            if (total) *total = desc->Width;
            return;
        }
        const bool volume = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if (!volume && desc->Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
            log_once("GetCopyableFootprints dimension not supported:", desc->Dimension);
            return fail();
        }
        const Format format = vulkan_format(desc->Format, true);  // typeless depth families are planar either way
        UINT mips = desc->MipLevels;
        const UINT depth = volume ? desc->DepthOrArraySize : 1;
        if (!mips)
            for (UINT64 size = std::max<UINT64>({desc->Width, desc->Height, depth}); size; size >>= 1) ++mips;
        const UINT planes = (format.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? 2 : 1;
        const UINT per_plane = mips * (volume ? 1 : desc->DepthOrArraySize);
        if (!format.format || UINT64(first) + count > UINT64(per_plane) * planes) return fail();

        UINT64 offset = base, end = base;
        for (UINT i = 0; i < count; ++i) {
            const UINT index = first + i, plane = index / per_plane, mip = index % mips;
            const VkImageAspectFlags aspect =
                planes == 2 ? (plane ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT) : format.aspects;
            const leonida::BlockFormat block = leonida::block_format(format.format, aspect);
            if (!block.bytes) return fail();
            const UINT width = std::max<UINT>(1, UINT(desc->Width >> mip)), height = std::max<UINT>(1, desc->Height >> mip);
            const UINT blocks_wide = (width + block.block - 1) / block.block;
            const UINT blocks_high = (height + block.block - 1) / block.block;
            const UINT64 row_size = UINT64(blocks_wide) * block.bytes;
            // Measured against the PC runtime (d3d12_vulkan_device_test): planar depth-stencil rows are padded to
            // 512 bytes, every other format's to D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256).
            const UINT pitch = UINT(align(row_size, planes == 2 ? 2 * D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
                                                                : D3D12_TEXTURE_DATA_PITCH_ALIGNMENT));
            offset = align(offset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            DXGI_FORMAT footprint_format = desc->Format;
            if (planes == 2) footprint_format = plane ? DXGI_FORMAT_R8_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
            const UINT slices = std::max(1u, depth >> mip);
            if (layouts)
                layouts[i] = {offset, {footprint_format, blocks_wide * block.block, blocks_high * block.block, slices, pitch}};
            if (rows) rows[i] = blocks_high;
            if (row_sizes) row_sizes[i] = row_size;
            const UINT64 total_rows = UINT64(blocks_high) * slices;
            end = offset + (total_rows - 1) * pitch + row_size;
            offset += total_rows * pitch;
        }
        if (total) *total = end - base;
    }

    HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE feature, void* data, UINT size) override {
        log_once("CheckFeatureSupport feature", feature);
        if (!data) return E_INVALIDARG;
        switch (feature) {
        case D3D12_FEATURE_FEATURE_LEVELS: {
            if (size != sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS)) return E_INVALIDARG;
            auto* levels = static_cast<D3D12_FEATURE_DATA_FEATURE_LEVELS*>(data);
            D3D_FEATURE_LEVEL best = D3D_FEATURE_LEVEL(0);
            for (UINT i = 0; i < levels->NumFeatureLevels; ++i)
                if (levels->pFeatureLevelsRequested[i] <= D3D_FEATURE_LEVEL_12_1)
                    best = std::max(best, levels->pFeatureLevelsRequested[i]);
            if (!best) return DXGI_ERROR_UNSUPPORTED;
            levels->MaxSupportedFeatureLevel = best;
            return S_OK;
        }
        case D3D12_FEATURE_ARCHITECTURE:
        case D3D12_FEATURE_ARCHITECTURE1: {
            const bool v1 = feature == D3D12_FEATURE_ARCHITECTURE1;
            if (size != (v1 ? sizeof(D3D12_FEATURE_DATA_ARCHITECTURE1) : sizeof(D3D12_FEATURE_DATA_ARCHITECTURE)))
                return E_INVALIDARG;
            auto* arch = static_cast<D3D12_FEATURE_DATA_ARCHITECTURE1*>(data);
            const UINT node = arch->NodeIndex;
            if (node != 0) return E_INVALIDARG;
            // A discrete desktop GPU: not tile-based, no UMA. The console is UMA (question 11); the layer keeps
            // upload heaps host-visible, so the game's upload path still works.
            arch->TileBasedRenderer = FALSE;
            arch->UMA = FALSE;
            arch->CacheCoherentUMA = FALSE;
            if (v1) arch->IsolatedMMU = TRUE;
            return S_OK;
        }
        case D3D12_FEATURE_D3D12_OPTIONS: {
            if (size != sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS)) return E_INVALIDARG;
            auto* opts = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS*>(data);
            *opts = {};
            opts->DoublePrecisionFloatShaderOps = TRUE;
            opts->OutputMergerLogicOp = TRUE;
            opts->MinPrecisionSupport = D3D12_SHADER_MIN_PRECISION_SUPPORT_NONE;
            opts->TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED;
            opts->ResourceBindingTier = D3D12_RESOURCE_BINDING_TIER_3;  // Bindless descriptor indexing supported
            opts->PSSpecifiedStencilRefSupported = TRUE;
            opts->TypedUAVLoadAdditionalFormats = TRUE;
            opts->ROVsSupported = TRUE;
            opts->ConservativeRasterizationTier = D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
            opts->MaxGPUVirtualAddressBitsPerResource = 40;
            opts->StandardSwizzle64KBSupported = FALSE;
            opts->CrossNodeSharingTier = D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED;
            opts->CrossAdapterRowMajorTextureSupported = FALSE;
            opts->VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation = TRUE;
            opts->ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER_1;
            return S_OK;
        }
        case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT)) return E_INVALIDARG;
            auto* va = static_cast<D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT*>(data);
            // The console and PC BDA support 48-bit GPU virtual addresses (section 4.1).
            va->MaxGPUVirtualAddressBitsPerResource = 40;
            va->MaxGPUVirtualAddressBitsPerProcess = 48;
            return S_OK;
        }
        case D3D12_FEATURE_FORMAT_SUPPORT: {
            if (size != sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT)) return E_INVALIDARG;
            auto* fs = static_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(data);
            fs->Support1 = D3D12_FORMAT_SUPPORT1_NONE;
            fs->Support2 = D3D12_FORMAT_SUPPORT2_NONE;
            const Format fmt = vulkan_format(fs->Format, false);
            const Format depth_fmt = vulkan_format(fs->Format, true);
            const VkFormat vk_fmt = fmt.format != VK_FORMAT_UNDEFINED ? fmt.format : depth_fmt.format;
            if (vk_fmt == VK_FORMAT_UNDEFINED || !state_.format_properties) {
                return DXGI_ERROR_UNSUPPORTED;
            }
            VkFormatProperties props{};
            state_.format_properties(state_.info.physical, vk_fmt, &props);
            const VkFormatFeatureFlags optimal = props.optimalTilingFeatures;
            const VkFormatFeatureFlags buffer = props.bufferFeatures;

            D3D12_FORMAT_SUPPORT1 s1 = D3D12_FORMAT_SUPPORT1_NONE;
            D3D12_FORMAT_SUPPORT2 s2 = D3D12_FORMAT_SUPPORT2_NONE;

            if (buffer != 0) s1 |= D3D12_FORMAT_SUPPORT1_BUFFER;
            if (buffer & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) s1 |= D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER;
            if (fs->Format == DXGI_FORMAT_R16_UINT || fs->Format == DXGI_FORMAT_R32_UINT)
                s1 |= D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER;

            if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                      D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                      D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
            }
            if (buffer & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_SHADER_LOAD;
            }
            if (optimal & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
            }
            if (optimal & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_RENDER_TARGET;
            }
            if (optimal & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_BLENDABLE;
            }
            if (optimal & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                s1 |= D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL;
            }
            if ((optimal & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
                (buffer & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)) {
                s1 |= D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
                s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
                if ((optimal & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ||
                    (buffer & VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT)) {
                    s2 |= D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD;
                }
            }
            fs->Support1 = s1;
            fs->Support2 = s2;
            return S_OK;
        }
        default:
            // Not answered yet. The log above lists every query; answers are added as the trace asks for them.
            return E_INVALIDARG;
        }
    }

private:
    IUnknown* identity() { return outer_ ? outer_ : static_cast<ID3D12Device*>(this); }

public:
    // Set 0 and its layout for the pipeline layouts the device hands out. Null when the set could not be made.
    VkDescriptorSet bindless_set() const { return state_.bindless ? state_.bindless->set() : VK_NULL_HANDLE; }
    VkDescriptorSetLayout bindless_layout() const {
        return state_.bindless ? state_.bindless->layout() : VK_NULL_HANDLE;
    }
    leonida::BindlessSet* bindless() const { return state_.bindless.get(); }
    std::uintptr_t resource_id_for_va(std::uint64_t va) {
        Resource* resource = resource_for_va(state_, va);
        return resource ? resource->id() : 0;
    }
    leonida::RetirementQueue& retirement() { return state_.retirement; }
    DeviceState& mutable_state() { return state_; }

    void create_bindless(PFN_vkGetDeviceProcAddr get_device_proc) {
        if (state_.bindless || !get_device_proc) return;
        state_.bindless = std::make_unique<leonida::BindlessSet>(get_device_proc, state_.info.device, state_.memory);
        if (!state_.context || !state_.bindless->ok()) return;
        // Draws need set 0 and the console VA -> VkBuffer lookup. The fallback images of the set have undefined
        // contents, so the first submission that uses a pipeline also initialises them.
        state_.context->set_draw_resources(
            state_.bindless->set(), &lookup_buffer, &state_,
            [bindless = state_.bindless.get()](VkCommandBuffer command_buffer) {
                if (!bindless->fallbacks_ready()) bindless->record_fallback_init(command_buffer);
            });
    }

    DeviceState state_;
    OwnedRuntime owned_;
    IUnknown* outer_ = nullptr;  // the Xbox-ABI proxy, which owns a reference to this device
};

inline HRESULT create_root_signature_deserializer_impl(const void* data, SIZE_T size, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!data || !size) return E_INVALIDARG;
    rt::d3d12::RootSignatureDesc desc{};
    std::string error;
    if (!rt::d3d12::parse_root_signature(data, size, desc, &error)) {
        std::fprintf(stderr, "[rt] d3d12: root signature deserializer: %s\n", error.c_str());
        return E_INVALIDARG;
    }
    if (riid != __uuidof(IUnknown) && riid != __uuidof(ID3D12RootSignatureDeserializer)) return E_NOINTERFACE;
    *out = static_cast<ID3D12RootSignatureDeserializer*>(new RootSignatureDeserializer(std::move(desc)));
    return S_OK;
}
}  // namespace

HRESULT create_root_signature_deserializer(const void* data, SIZE_T size, REFIID riid, void** out) {
    return create_root_signature_deserializer_impl(data, size, riid, out);
}
HRESULT create_vulkan_device(const VulkanDeviceInfo& info, REFIID riid, void** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    auto* device = new Device(info);
    if (!device->ok()) {
        device->Release();
        return E_FAIL;
    }
    PFN_vkGetInstanceProcAddr get_instance = info.get_instance_proc;
    if (!get_instance && info.instance) {
        static const HMODULE loader = LoadLibraryA("vulkan-1.dll");
        if (loader) get_instance = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader, "vkGetInstanceProcAddr"));
    }
    const auto get_device_proc = get_instance
                                     ? reinterpret_cast<PFN_vkGetDeviceProcAddr>(
                                           get_instance(info.instance, "vkGetDeviceProcAddr"))
                                     : nullptr;
    device->create_bindless(get_device_proc);
    if (!device->bindless_set()) std::fprintf(stderr, "[rt] bindless: set 0 missing (instance=%p, get_device=%p)\n",
                                               static_cast<void*>(info.instance), reinterpret_cast<void*>(get_device_proc));
    return hand_out(device, riid, out);
}

void set_device_outer(ID3D12Device* device, IUnknown* outer) {
    if (Device* d = Device::from<Device>(device)) d->set_outer(outer);
}

VkImage vulkan_image(ID3D12Resource* resource) {
    const Resource* r = Resource::from<Resource>(resource);
    return r ? r->image() : VK_NULL_HANDLE;
}

VkPipeline vulkan_pipeline(ID3D12PipelineState* state) {
    const PipelineState* p = PipelineState::from<PipelineState>(state);
    return p ? p->pipeline() : VK_NULL_HANDLE;
}

VkPipelineLayout vulkan_root_signature_layout(ID3D12RootSignature* root_signature) {
    const RootSignature* r = RootSignature::from<RootSignature>(root_signature);
    return r ? r->layout() : VK_NULL_HANDLE;
}

VkDevice vulkan_device(ID3D12Device* device) {
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    Device* d = Device::from<Device>(real);
    return d ? d->info().device : VK_NULL_HANDLE;
}

VulkanDeviceInfo vulkan_device_info(ID3D12Device* device) {
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    Device* d = Device::from<Device>(real);
    return d ? d->info() : VulkanDeviceInfo{};
}

VkImageLayout vulkan_image_layout(ID3D12Device* device, ID3D12Resource* resource) {
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    Device* d = Device::from<Device>(real);
    const Resource* r = Resource::from<Resource>(resource);
    if (!d || !r || !d->state_.context) return VK_IMAGE_LAYOUT_UNDEFINED;
    return d->state_.context->layout(r->id(), 0);
}

VkBuffer vulkan_buffer_for_va(ID3D12Device* device, std::uint64_t va, std::uint64_t* base_va) {
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    Device* d = Device::from<Device>(real);
    if (!d) return VK_NULL_HANDLE;
    Resource* resource = resource_for_va(d->mutable_state(), va);
    if (!resource) return VK_NULL_HANDLE;
    if (base_va) *base_va = resource->console_va();
    return resource->buffer();
}

std::uint32_t device_max_push_constants_size(ID3D12Device* device) {
    if (!device) return 0;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    if (auto* d = Device::from<Device>(real)) return d->info().max_push_constants_size;
    return 0;
}

bool device_push_constants_need_fallback(ID3D12Device* device) {
    if (!device) return false;
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    if (!real) real = device;
    if (auto* d = Device::from<Device>(real)) return d->info().push_constants_need_fallback;
    return false;
}

// Creates its own Vulkan 1.3 instance and device. LEONIDA_VK_VALIDATION=1 adds VK_LAYER_KHRONOS_validation.
// With `presentable` the instance gets the surface extensions and the device VK_KHR_swapchain, so a presenter can
// create a swapchain on exactly these objects (shared-device WSI, docs/d3d12_vulkan.md section 8).
HRESULT create_vulkan_device_standalone(REFIID riid, void** out, bool presentable) {
    if (!out) return E_POINTER;
    *out = nullptr;
    static const HMODULE library = LoadLibraryA("vulkan-1.dll");
    const auto get = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                   reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")))
                             : nullptr;
    if (!get) {
        std::fprintf(stderr, "[rt] d3d12: vulkan-1.dll not found\n");
        return E_FAIL;
    }
    const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gta6rt";
    app.apiVersion = VK_API_VERSION_1_3;
    const char* surface_extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    if (presentable) {
        instance_info.enabledExtensionCount = 2;
        instance_info.ppEnabledExtensionNames = surface_extensions;
    }
    const char* layer = "VK_LAYER_KHRONOS_validation";
    char validation[8] = {};
    size_t length = 0;
    if (getenv_s(&length, validation, "LEONIDA_VK_VALIDATION") == 0 && length > 1 && validation[0] != '0') {
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = &layer;
    }
    OwnedRuntime owned{};
    if (create_instance(&instance_info, nullptr, &owned.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "[rt] d3d12: vkCreateInstance (Vulkan 1.3) failed\n");
        return E_FAIL;
    }
    owned.destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(get(owned.instance, "vkDestroyInstance"));
    const auto fail = [&](const char* why) {
        std::fprintf(stderr, "[rt] d3d12: %s\n", why);
        owned.destroy_instance(owned.instance, nullptr);
        return E_FAIL;
    };

#define LEONIDA_INSTANCE(name) const auto name = reinterpret_cast<PFN_##name>(get(owned.instance, #name))
    LEONIDA_INSTANCE(vkEnumeratePhysicalDevices);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceProperties);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    LEONIDA_INSTANCE(vkGetPhysicalDeviceFeatures2);
    LEONIDA_INSTANCE(vkCreateDevice);
    LEONIDA_INSTANCE(vkGetDeviceProcAddr);
#undef LEONIDA_INSTANCE
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(owned.instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    vkEnumeratePhysicalDevices(owned.instance, &count, physicals.data());
    VulkanDeviceInfo info{};
    info.get_instance_proc = get;
    info.instance = owned.instance;
    info.separate_depth_stencil_layouts = true;
    VkPhysicalDeviceProperties chosen{};
    bool chosen_bc = false;
    bool chosen_independent_blend = false;
    for (VkPhysicalDevice candidate : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.apiVersion < VK_API_VERSION_1_3) continue;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
        vkGetPhysicalDeviceFeatures2(candidate, &features);
        const bool bc = features.features.textureCompressionBC;
        if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore ||
            !f12.separateDepthStencilLayouts)
            continue;
        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count, families.data());
        for (std::uint32_t i = 0; i < family_count; ++i) {
            if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (!info.physical || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                info.physical = candidate;
                info.queue_family = i;
                chosen = props;
                chosen_bc = bc;
                chosen_independent_blend = features.features.independentBlend == VK_TRUE;
            }
            break;
        }
    }
    if (!info.physical)
        return fail("no Vulkan 1.3 GPU with dynamicRendering, synchronization2, timelineSemaphore and "
                    "separateDepthStencilLayouts");

    VkPhysicalDeviceVulkan13Features enable13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    enable13.dynamicRendering = VK_TRUE;
    enable13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enable13};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.separateDepthStencilLayouts = VK_TRUE;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = info.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enable{};
    enable.textureCompressionBC = chosen_bc ? VK_TRUE : VK_FALSE;  // the game's textures are BC1-7
    // D3D12 blends each render target separately; Vulkan needs this feature for that (MRT/GBuffer pipelines).
    enable.independentBlend = chosen_independent_blend ? VK_TRUE : VK_FALSE;
    if (!enable.independentBlend)
        std::fprintf(stderr, "[rt] d3d12: the GPU has no independentBlend: per-target blends collapse to target 0\n");
    if (!chosen_bc) std::fprintf(stderr, "[rt] d3d12: the GPU has no BC texture support\n");
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable12};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.pEnabledFeatures = &enable;
    const char* device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (presentable) {
        device_info.enabledExtensionCount = 1;
        device_info.ppEnabledExtensionNames = device_extensions;
    }
    if (vkCreateDevice(info.physical, &device_info, nullptr, &owned.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
    info.device = owned.device;
    owned.destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(owned.device, "vkDestroyDevice"));
    reinterpret_cast<PFN_vkGetDeviceQueue>(vkGetDeviceProcAddr(owned.device, "vkGetDeviceQueue"))(
        owned.device, info.queue_family, 0, &info.queue);
    info.max_push_constants_size = chosen.limits.maxPushConstantsSize;
    info.push_constants_need_fallback = (info.max_push_constants_size < 144);
    info.independent_blend = enable.independentBlend == VK_TRUE;
    std::fprintf(stderr, "[rt] d3d12: Vulkan device: %s (maxPushConstantsSize: %u bytes)\n", chosen.deviceName,
                 chosen.limits.maxPushConstantsSize);
    if (info.push_constants_need_fallback) {
        std::fprintf(stderr,
                     "[rt] d3d12: WARNING: GPU maxPushConstantsSize (%u bytes) is below the 144 bytes required by "
                     "the shader contract; uniform-buffer fallback required for the 16-byte tail\n",
                     chosen.limits.maxPushConstantsSize);
    }

    auto* device = new Device(info);
    device->own(owned);  // destroyed with the device from here on, even on failure
    if (!device->ok()) {
        device->Release();
        return E_FAIL;
    }
    const auto get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get(info.instance, "vkGetDeviceProcAddr"));
    device->create_bindless(get_device_proc);
    return hand_out(device, riid, out);
}

}  // namespace rt::d3d12

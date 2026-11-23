// Hardening and real-GPU regression tests for IndirectDispatchCounter
// (runtime/src/vulkan_indirect_dispatch_count.h):
//   1. Deterministic MockVk checks:
//      a) null get_ / null device_ safety (no crash in prepare, emit, or destructors).
//      b) missing device entry points and partial-init cleanup without destroying garbage handles.
//      c) Batch::prepare() rejects stride < 12, unaligned argument_offset / count_offset (% 4 != 0),
//         address-range overflows, and cleans up failed allocations without freeing garbage handles.
//      d) memoryTypeCount > VK_MAX_MEMORY_TYPES is safely clamped without out-of-bounds reads or UB shifts.
//      e) emit() guards null cb / unbuilt pipeline / invalid jobs and chunks dispatches with overflow-safe math.
//   2. Real Vulkan GPU end-to-end BDA count-gating execution (clamping count < max_count, count == 0,
//      count > max_count, non-zero aligned offsets, stride > 12) with zero validation errors.
// Skips GPU section with exit code 77 when no suitable Vulkan 1.3 BDA device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_indirect_dispatch_count.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace mock {
struct State {
    std::string missing_proc;
    bool fail_shader = false;
    bool fail_layout = false;
    bool fail_pipeline = false;
    bool fail_create_buffer = false;
    bool fail_alloc = false;
    bool fail_bind = false;
    bool write_garbage_on_fail = false;

    std::uint64_t next_handle = 0x1000;
    std::uint32_t req_memory_type_bits = 0x1u;
    VkDeviceSize req_size = 256;
    VkDeviceAddress args_va = 0x100000;
    VkDeviceAddress count_va = 0x200000;
    VkDeviceAddress output_va = 0x300000;

    std::unordered_set<std::uint64_t> live_shaders;
    std::unordered_set<std::uint64_t> live_layouts;
    std::unordered_set<std::uint64_t> live_pipelines;
    std::unordered_set<std::uint64_t> live_buffers;
    std::unordered_set<std::uint64_t> live_memories;
    bool destroyed_unknown = false;
    std::uint32_t create_calls = 0;

    struct DispatchRec {
        std::uint32_t groups_x = 0;
        std::uint32_t stride_words = 0;
        std::uint32_t first_record = 0;
        std::uint32_t records = 0;
        VkDeviceAddress args = 0;
        VkDeviceAddress output = 0;
    };
    DispatchRec pending_push{};
    std::vector<DispatchRec> dispatches;
    std::uint32_t barriers = 0;
    std::uint32_t binds = 0;
};

State g;

template<typename H>
H make_h(std::uint64_t v) {
    return reinterpret_cast<H>(static_cast<std::uintptr_t>(v));
}
template<typename H>
std::uint64_t raw_h(H h) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h));
}

constexpr std::uint64_t kGarbageShader = 0xDEAD0001ull;
constexpr std::uint64_t kGarbageLayout = 0xDEAD0002ull;
constexpr std::uint64_t kGarbagePipeline = 0xDEAD0003ull;
constexpr std::uint64_t kGarbageBuffer = 0xDEAD0004ull;
constexpr std::uint64_t kGarbageMemory = 0xDEAD0005ull;

VKAPI_ATTR VkResult VKAPI_CALL create_shader(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
                                             VkShaderModule* out) {
    ++g.create_calls;
    if (g.fail_shader) {
        if (g.write_garbage_on_fail) *out = make_h<VkShaderModule>(kGarbageShader);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const std::uint64_t id = ++g.next_handle;
    g.live_shaders.insert(id);
    *out = make_h<VkShaderModule>(id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_shader(VkDevice, VkShaderModule h, const VkAllocationCallbacks*) {
    if (!g.live_shaders.erase(raw_h(h))) g.destroyed_unknown = true;
}
VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*,
                                             VkPipelineLayout* out) {
    ++g.create_calls;
    if (g.fail_layout) {
        if (g.write_garbage_on_fail) *out = make_h<VkPipelineLayout>(kGarbageLayout);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const std::uint64_t id = ++g.next_handle;
    g.live_layouts.insert(id);
    *out = make_h<VkPipelineLayout>(id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice, VkPipelineLayout h, const VkAllocationCallbacks*) {
    if (!g.live_layouts.erase(raw_h(h))) g.destroyed_unknown = true;
}
VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                VkPipeline* out) {
    ++g.create_calls;
    if (g.fail_pipeline) {
        if (g.write_garbage_on_fail) *out = make_h<VkPipeline>(kGarbagePipeline);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const std::uint64_t id = ++g.next_handle;
    g.live_pipelines.insert(id);
    *out = make_h<VkPipeline>(id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline h, const VkAllocationCallbacks*) {
    if (!g.live_pipelines.erase(raw_h(h))) g.destroyed_unknown = true;
}
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    ++g.create_calls;
    if (g.fail_create_buffer) {
        if (g.write_garbage_on_fail) *out = make_h<VkBuffer>(kGarbageBuffer);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const std::uint64_t id = ++g.next_handle;
    g.live_buffers.insert(id);
    g.req_size = info ? info->size : 256;
    *out = make_h<VkBuffer>(id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer h, const VkAllocationCallbacks*) {
    if (!g.live_buffers.erase(raw_h(h))) g.destroyed_unknown = true;
}
VKAPI_ATTR void VKAPI_CALL get_buffer_reqs(VkDevice, VkBuffer, VkMemoryRequirements* reqs) {
    reqs->size = g.req_size;
    reqs->alignment = 16;
    reqs->memoryTypeBits = g.req_memory_type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_memory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                            VkDeviceMemory* out) {
    ++g.create_calls;
    if (g.fail_alloc) {
        if (g.write_garbage_on_fail) *out = make_h<VkDeviceMemory>(kGarbageMemory);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    const std::uint64_t id = ++g.next_handle;
    g.live_memories.insert(id);
    *out = make_h<VkDeviceMemory>(id);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory h, const VkAllocationCallbacks*) {
    if (!g.live_memories.erase(raw_h(h))) g.destroyed_unknown = true;
}
VKAPI_ATTR VkResult VKAPI_CALL bind_buffer(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return g.fail_bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL get_bda(VkDevice, const VkBufferDeviceAddressInfo* info) {
    const std::uint64_t id = raw_h(info->buffer);
    if (id == 0xAA01ull) return g.args_va;
    if (id == 0xAA02ull) return g.count_va;
    return g.output_va;
}
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                       std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                       const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    ++g.barriers;
}
VKAPI_ATTR void VKAPI_CALL cmd_bind(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) { ++g.binds; }
VKAPI_ATTR void VKAPI_CALL cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t,
                                    std::uint32_t size, const void* values) {
    struct Push {
        VkDeviceAddress arguments, count, output;
        std::uint32_t stride_words, first_record, records;
    };
    if (size == sizeof(Push) && values) {
        Push p{};
        std::memcpy(&p, values, sizeof(p));
        g.pending_push.args = p.arguments;
        g.pending_push.output = p.output;
        g.pending_push.stride_words = p.stride_words;
        g.pending_push.first_record = p.first_record;
        g.pending_push.records = p.records;
    }
}
VKAPI_ATTR void VKAPI_CALL cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t, std::uint32_t) {
    g.pending_push.groups_x = x;
    g.dispatches.push_back(g.pending_push);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice, const char* name) {
    const std::string_view n(name);
    if (!g.missing_proc.empty() && n == g.missing_proc) return nullptr;
    if (n == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(create_buffer);
    if (n == "vkDestroyBuffer") return reinterpret_cast<PFN_vkVoidFunction>(destroy_buffer);
    if (n == "vkGetBufferMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(get_buffer_reqs);
    if (n == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(alloc_memory);
    if (n == "vkFreeMemory") return reinterpret_cast<PFN_vkVoidFunction>(free_memory);
    if (n == "vkBindBufferMemory") return reinterpret_cast<PFN_vkVoidFunction>(bind_buffer);
    if (n == "vkGetBufferDeviceAddress") return reinterpret_cast<PFN_vkVoidFunction>(get_bda);
    if (n == "vkCreateShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(create_shader);
    if (n == "vkDestroyShaderModule") return reinterpret_cast<PFN_vkVoidFunction>(destroy_shader);
    if (n == "vkCreatePipelineLayout") return reinterpret_cast<PFN_vkVoidFunction>(create_layout);
    if (n == "vkDestroyPipelineLayout") return reinterpret_cast<PFN_vkVoidFunction>(destroy_layout);
    if (n == "vkCreateComputePipelines") return reinterpret_cast<PFN_vkVoidFunction>(create_pipelines);
    if (n == "vkDestroyPipeline") return reinterpret_cast<PFN_vkVoidFunction>(destroy_pipeline);
    if (n == "vkCmdPipelineBarrier") return reinterpret_cast<PFN_vkVoidFunction>(cmd_barrier);
    if (n == "vkCmdBindPipeline") return reinterpret_cast<PFN_vkVoidFunction>(cmd_bind);
    if (n == "vkCmdPushConstants") return reinterpret_cast<PFN_vkVoidFunction>(cmd_push);
    if (n == "vkCmdDispatch") return reinterpret_cast<PFN_vkVoidFunction>(cmd_dispatch);
    return nullptr;
}

bool clean() {
    return !g.destroyed_unknown && g.live_shaders.empty() && g.live_layouts.empty() && g.live_pipelines.empty() &&
           g.live_buffers.empty() && g.live_memories.empty();
}
}  // namespace mock

int run_mock_checks() {
    using namespace leonida;
    const VkDevice fake_dev = mock::make_h<VkDevice>(0x7777);
    const VkCommandBuffer fake_cb = mock::make_h<VkCommandBuffer>(0x8888);
    const VkBuffer args_buf = mock::make_h<VkBuffer>(0xAA01);
    const VkBuffer count_buf = mock::make_h<VkBuffer>(0xAA02);

    VkPhysicalDeviceMemoryProperties mem{};
    mem.memoryTypeCount = 1;
    mem.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkPhysicalDeviceLimits lim{};
    lim.maxComputeWorkGroupCount[0] = 65535;

    // 1a. Stride < 12 and unaligned offsets must be rejected before creating any Vulkan object.
    {
        mock::g = {};
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 4, 2, out));
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 8, 2, out));
        EXPECT(!b->prepare(args_buf, 2, count_buf, 0, 12, 2, out));
        EXPECT(!b->prepare(args_buf, 0, count_buf, 6, 12, 2, out));
        EXPECT(mock::g.create_calls == 0);
        EXPECT(b->allocations.empty());
    }
    EXPECT(mock::clean());

    // 1b. Null get_ / null device_ must safely fail without crashing in prepare(), emit(), or destructors.
    {
        mock::g = {};
        IndirectDispatchCounter c1(nullptr, fake_dev, mem, lim);
        auto b1 = c1.batch();
        CountedDispatch out{};
        EXPECT(!b1->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        c1.emit(fake_cb, out);

        IndirectDispatchCounter c2(mock::get_device, VK_NULL_HANDLE, mem, lim);
        auto b2 = c2.batch();
        EXPECT(!b2->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        c2.emit(fake_cb, out);
        EXPECT(mock::g.create_calls == 0);
    }
    EXPECT(mock::clean());

    // 1c. Missing entry point and partial-init cleanup without destroying garbage output handles.
    {
        mock::g = {};
        mock::g.missing_proc = "vkCmdDispatch";
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        EXPECT(mock::g.create_calls == 0);
    }
    EXPECT(mock::clean());

    {
        mock::g = {};
        mock::g.fail_pipeline = true;
        mock::g.write_garbage_on_fail = true;
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
    }
    EXPECT(mock::clean());

    {
        mock::g = {};
        mock::g.fail_layout = true;
        mock::g.write_garbage_on_fail = true;
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
    }
    EXPECT(mock::clean());

    // 1d. memoryTypeCount > VK_MAX_MEMORY_TYPES clamp + failed vkAllocateMemory garbage handle + bind/overflow cleanup.
    {
        mock::g = {};
        VkPhysicalDeviceMemoryProperties bad_mem{};
        bad_mem.memoryTypeCount = VK_MAX_MEMORY_TYPES + 16;
        mock::g.req_memory_type_bits = 0;  // no valid type in 0..31; must not read past VK_MAX_MEMORY_TYPES
        IndirectDispatchCounter c(mock::get_device, fake_dev, bad_mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        EXPECT(b->allocations.empty());
    }
    EXPECT(mock::clean());

    {
        mock::g = {};
        mock::g.fail_alloc = true;
        mock::g.write_garbage_on_fail = true;
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        EXPECT(b->allocations.empty());
    }
    EXPECT(mock::clean());

    {
        mock::g = {};
        mock::g.fail_bind = true;
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 12, 1, out));
        EXPECT(b->allocations.empty());
    }
    EXPECT(mock::clean());

    // Address-range overflow in arguments, count, or output must fail and clean up scratch allocation immediately.
    {
        mock::g = {};
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, lim);
        auto b = c.batch();
        CountedDispatch out{};

        mock::g.args_va = UINT64_MAX - 16;
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 16, 3, out));
        EXPECT(b->allocations.empty());

        mock::g.args_va = 0x100000;
        mock::g.count_va = UINT64_MAX - 2;
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 16, 2, out));
        EXPECT(b->allocations.empty());

        mock::g.count_va = 0x200000;
        mock::g.output_va = UINT64_MAX - 16;
        EXPECT(!b->prepare(args_buf, 0, count_buf, 0, 16, 2, out));
        EXPECT(b->allocations.empty());
    }
    EXPECT(mock::clean());

    // 1e. emit() guards & multi-chunk dispatch slicing.
    {
        mock::g = {};
        VkPhysicalDeviceLimits small_lim = lim;
        small_lim.maxComputeWorkGroupCount[0] = 2;  // 128 records per dispatch
        IndirectDispatchCounter c(mock::get_device, fake_dev, mem, small_lim);

        // Before pipeline is built, emit() must be a safe no-op.
        CountedDispatch dummy{};
        dummy.arguments = 0x100000;
        dummy.count = 0x200000;
        dummy.output = 0x300000;
        dummy.stride = 16;
        dummy.max_count = 4;
        c.emit(fake_cb, dummy);
        EXPECT(mock::g.barriers == 0 && mock::g.binds == 0 && mock::g.dispatches.empty());

        auto b = c.batch();
        CountedDispatch job{};
        EXPECT(b->prepare(args_buf, 16, count_buf, 4, 16, 300, job));
        EXPECT(b->allocations.size() == 1);

        // Invalid job/cb guards after pipeline exists:
        c.emit(VK_NULL_HANDLE, job);
        CountedDispatch bad_stride = job;
        bad_stride.stride = 8;
        c.emit(fake_cb, bad_stride);
        CountedDispatch zero_cnt = job;
        zero_cnt.max_count = 0;
        c.emit(fake_cb, zero_cnt);
        CountedDispatch zero_va = job;
        zero_va.output = 0;
        c.emit(fake_cb, zero_va);
        EXPECT(mock::g.barriers == 0 && mock::g.binds == 0 && mock::g.dispatches.empty());

        // Valid chunked emit: 300 records -> 128 (2 groups), 128 (2 groups), 44 (1 group).
        c.emit(fake_cb, job);
        EXPECT(mock::g.barriers == 2 && mock::g.binds == 1 && mock::g.dispatches.size() == 3);
        EXPECT(mock::g.dispatches[0].first_record == 0 && mock::g.dispatches[0].records == 128 &&
               mock::g.dispatches[0].groups_x == 2 && mock::g.dispatches[0].stride_words == 4);
        EXPECT(mock::g.dispatches[1].first_record == 128 && mock::g.dispatches[1].records == 128 &&
               mock::g.dispatches[1].groups_x == 2);
        EXPECT(mock::g.dispatches[2].first_record == 256 && mock::g.dispatches[2].records == 44 &&
               mock::g.dispatches[2].groups_x == 1);
    }
    EXPECT(mock::clean());

    std::printf("mock hardening checks (1a-1e): ok\n");
    return 0;
}

}  // namespace

int main() {
    using namespace leonida;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (run_mock_checks() != 0) return 1;

    vktest::Device d;
    d.request_buffer_device_address = true;
    if (!d.create(false) || !d.buffer_device_address) {
        std::printf("SKIP: no suitable Vulkan 1.3 BDA device\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties mem{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &mem);
    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);

    void* args_ptr = nullptr;
    void* count_ptr = nullptr;
    void* copy_count_ptr = nullptr;
    void* readback_ptr = nullptr;
    const VkBuffer args_buf = d.buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &args_ptr);
    const VkBuffer count_buf = d.buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &count_ptr);
    const VkBuffer copy_count_buf = d.buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &copy_count_ptr);
    const VkBuffer readback_buf = d.buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &readback_ptr);
    EXPECT(args_buf && count_buf && copy_count_buf && readback_buf);
    EXPECT(args_ptr && count_ptr && copy_count_ptr && readback_ptr);

    // 5 dispatch records with stride = 16 (3 uint32 args + 1 uint32 padding), starting at argument_offset = 16.
    constexpr std::uint32_t kMaxCount = 5;
    constexpr std::uint32_t kStride = 16;
    constexpr VkDeviceSize kArgOffset = 16;
    constexpr VkDeviceSize kCountOffset = 4;
    std::memset(args_ptr, 0, 256);
    auto* args_u32 = static_cast<std::uint32_t*>(args_ptr);
    for (std::uint32_t r = 0; r < kMaxCount; ++r) {
        const std::size_t base = (kArgOffset / 4) + r * (kStride / 4);
        args_u32[base + 0] = r * 3u + 1u;
        args_u32[base + 1] = r * 3u + 2u;
        args_u32[base + 2] = r * 3u + 3u;
        args_u32[base + 3] = 0xDEADBEEFu;
    }

    const std::uint32_t copy_all = kMaxCount;
    std::memcpy(copy_count_ptr, &copy_all, sizeof(copy_all));
    const VkDeviceAddress copy_count_va = d.device_address(copy_count_buf);
    const VkDeviceAddress readback_va = d.device_address(readback_buf);
    EXPECT(copy_count_va && readback_va);

    IndirectDispatchCounter counter(d.device_proc, d.device, mem, props.limits);
    for (const std::uint32_t gpu_count : std::array<std::uint32_t, 3>{3u, 0u, 99u}) {
        std::memset(count_ptr, 0, 64);
        std::memcpy(static_cast<std::uint8_t*>(count_ptr) + kCountOffset, &gpu_count, sizeof(gpu_count));
        std::memset(readback_ptr, 0xCC, 256);

        auto batch = counter.batch();
        CountedDispatch job{};
        EXPECT(batch->prepare(args_buf, kArgOffset, count_buf, kCountOffset, kStride, kMaxCount, job));

        // Copy job.output (device-local BDA storage buffer) back into host-visible readback_buf using
        // IndirectDispatchCounter itself so we never need TRANSFER_SRC_BIT on the private indirect scratch buffer.
        CountedDispatch copy_back{};
        copy_back.arguments = job.output;
        copy_back.count = copy_count_va;
        copy_back.output = readback_va;
        copy_back.stride = 12;
        copy_back.max_count = kMaxCount;

        const bool ok = d.run([&](VkCommandBuffer cb) {
            counter.emit(cb, job);
            counter.emit(cb, copy_back);
            VkMemoryBarrier host{};
            host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                                   VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr,
                                                                   0, nullptr);
            return true;
        });
        EXPECT(ok);

        const auto* got = static_cast<const std::uint32_t*>(readback_ptr);
        const std::uint32_t active = std::min(gpu_count, kMaxCount);
        for (std::uint32_t r = 0; r < kMaxCount; ++r) {
            const std::uint32_t e0 = r < active ? r * 3u + 1u : 0u;
            const std::uint32_t e1 = r < active ? r * 3u + 2u : 0u;
            const std::uint32_t e2 = r < active ? r * 3u + 3u : 0u;
            EXPECT(got[r * 3u + 0u] == e0);
            EXPECT(got[r * 3u + 1u] == e1);
            EXPECT(got[r * 3u + 2u] == e2);
        }
    }
    std::printf("real GPU BDA count-gating (count=3, 0, 99; offset+stride): ok\n");

    const unsigned val_errors = d.validation_errors();
    std::printf("Vulkan validation errors: %u\n", val_errors);
    if (val_errors != 0) return 1;
    std::printf("vk_indirect_dispatch_count_test: ok\n");
    return 0;
}

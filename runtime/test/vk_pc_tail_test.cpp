// CPU-only unit tests for vk_pc_tail.h (no Vulkan instance, no GPU): SPIR-V DescriptorSet scan bounds,
// LEONIDA_PC_TAIL_UBO environment parsing, alignment/dynamic-offset overflow checks, partial-failure cleanup,
// multi-slot tail writes, and safe destruction.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "vk_pc_tail.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

std::uint32_t word0(std::uint32_t op, std::uint32_t count) { return (count << 16) | op; }

template <typename Handle>
Handle make_handle(std::uint64_t id) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
    } else {
        return static_cast<Handle>(id);
    }
}

template <typename Handle>
std::uint64_t handle_id(Handle h) {
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h));
    } else {
        return static_cast<std::uint64_t>(h);
    }
}

void set_pc_tail_env(const char* value) {
#ifdef _WIN32
    _putenv_s("LEONIDA_PC_TAIL_UBO", value ? value : "");
#else
    if (!value || value[0] == '\0') {
        unsetenv("LEONIDA_PC_TAIL_UBO");
    } else {
        setenv("LEONIDA_PC_TAIL_UBO", value, 1);
    }
#endif
}

struct FakeVulkan {
    static constexpr VkDeviceSize max_alloc = 64u << 20;
    std::uint64_t next_id = 1;
    std::unordered_set<std::uint64_t> layouts;
    std::unordered_set<std::uint64_t> pools;
    std::unordered_set<std::uint64_t> buffers;
    std::unordered_set<std::uint64_t> memories;
    std::unordered_map<std::uint64_t, VkDeviceSize> buffer_sizes;
    std::unordered_map<std::uint64_t, std::uint64_t> bound_buffer_mem;
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> memory_storage;

    int create_layout_calls = 0;
    int update_calls = 0;
    VkDeviceSize last_buffer_size = 0;
    VkDescriptorBufferInfo last_buffer_info{};
    VkDescriptorType last_descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::uint32_t last_alloc_mem_type = UINT32_MAX;
    bool freed_mem_while_buffer_bound = false;

    const char* omit_proc = nullptr;
    int fail_layout_call = 0;
    bool fail_pool = false;
    bool fail_alloc_set = false;
    bool fail_buffer = false;
    bool fail_alloc_mem = false;
    bool fail_bind = false;
    bool fail_map = false;
    bool map_null_ptr = false;
    std::uint32_t memory_type_bits = 0x1u;

    std::size_t live_count() const {
        return layouts.size() + pools.size() + buffers.size() + memories.size();
    }
};

FakeVulkan* g_vk = nullptr;
const VkDevice k_dev = make_handle<VkDevice>(0x1000);

VKAPI_ATTR VkResult VKAPI_CALL fake_create_layout(
    VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    ++g_vk->create_layout_calls;
    if (g_vk->fail_layout_call == g_vk->create_layout_calls) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const std::uint64_t id = g_vk->next_id++;
    g_vk->layouts.insert(id);
    *out = make_handle<VkDescriptorSetLayout>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_layout(
    VkDevice, VkDescriptorSetLayout layout, const VkAllocationCallbacks*) {
    g_vk->layouts.erase(handle_id(layout));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_pool(
    VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* out) {
    if (g_vk->fail_pool) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const std::uint64_t id = g_vk->next_id++;
    g_vk->pools.insert(id);
    *out = make_handle<VkDescriptorPool>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_pool(
    VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
    g_vk->pools.erase(handle_id(pool));
}

VKAPI_ATTR VkResult VKAPI_CALL fake_alloc_sets(
    VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet* out) {
    if (g_vk->fail_alloc_set) return VK_ERROR_OUT_OF_POOL_MEMORY;
    *out = make_handle<VkDescriptorSet>(g_vk->next_id++);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_update_sets(
    VkDevice, std::uint32_t write_count, const VkWriteDescriptorSet* writes, std::uint32_t, const VkCopyDescriptorSet*) {
    if (write_count > 0 && writes && writes[0].pBufferInfo) {
        ++g_vk->update_calls;
        g_vk->last_buffer_info = *writes[0].pBufferInfo;
        g_vk->last_descriptor_type = writes[0].descriptorType;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL fake_create_buffer(
    VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    g_vk->last_buffer_size = info->size;
    if (g_vk->fail_buffer) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const std::uint64_t id = g_vk->next_id++;
    g_vk->buffers.insert(id);
    g_vk->buffer_sizes[id] = info->size;
    *out = make_handle<VkBuffer>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_buffer(
    VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    const std::uint64_t id = handle_id(buffer);
    g_vk->buffers.erase(id);
    g_vk->buffer_sizes.erase(id);
    g_vk->bound_buffer_mem.erase(id);
}

VKAPI_ATTR void VKAPI_CALL fake_get_buffer_mem_reqs(
    VkDevice, VkBuffer buffer, VkMemoryRequirements* reqs) {
    const auto it = g_vk->buffer_sizes.find(handle_id(buffer));
    reqs->size = it != g_vk->buffer_sizes.end() ? it->second : 0;
    reqs->alignment = 256;
    reqs->memoryTypeBits = g_vk->memory_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_alloc_memory(
    VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    g_vk->last_alloc_mem_type = info->memoryTypeIndex;
    if (g_vk->fail_alloc_mem || info->allocationSize > FakeVulkan::max_alloc) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    const std::uint64_t id = g_vk->next_id++;
    g_vk->memories.insert(id);
    g_vk->memory_storage.emplace(id, std::vector<std::uint8_t>(static_cast<std::size_t>(info->allocationSize), 0xCDu));
    *out = make_handle<VkDeviceMemory>(id);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_free_memory(
    VkDevice, VkDeviceMemory mem, const VkAllocationCallbacks*) {
    const std::uint64_t id = handle_id(mem);
    for (const auto& [buf_id, mem_id] : g_vk->bound_buffer_mem) {
        if (mem_id == id && g_vk->buffers.count(buf_id) != 0) g_vk->freed_mem_while_buffer_bound = true;
    }
    g_vk->memories.erase(id);
    g_vk->memory_storage.erase(id);
}

VKAPI_ATTR VkResult VKAPI_CALL fake_bind_buffer_memory(
    VkDevice, VkBuffer buffer, VkDeviceMemory mem, VkDeviceSize) {
    if (g_vk->fail_bind) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    g_vk->bound_buffer_mem[handle_id(buffer)] = handle_id(mem);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_map_memory(
    VkDevice, VkDeviceMemory mem, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** pp_data) {
    if (g_vk->fail_map) return VK_ERROR_MEMORY_MAP_FAILED;
    if (g_vk->map_null_ptr) {
        *pp_data = nullptr;
        return VK_SUCCESS;
    }
    auto it = g_vk->memory_storage.find(handle_id(mem));
    if (it == g_vk->memory_storage.end()) return VK_ERROR_MEMORY_MAP_FAILED;
    *pp_data = it->second.data();
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_gdpa(VkDevice, const char* name) {
    if (g_vk->omit_proc && std::strcmp(name, g_vk->omit_proc) == 0) return nullptr;
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_create_layout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_layout);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_create_pool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_pool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_alloc_sets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_update_sets);
    if (std::strcmp(name, "vkCreateBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_create_buffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_destroy_buffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_get_buffer_mem_reqs);
    if (std::strcmp(name, "vkAllocateMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_alloc_memory);
    if (std::strcmp(name, "vkFreeMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_free_memory);
    if (std::strcmp(name, "vkBindBufferMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_bind_buffer_memory);
    if (std::strcmp(name, "vkMapMemory") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(fake_map_memory);
    return nullptr;
}

VkPhysicalDeviceMemoryProperties host_memory_props() {
    VkPhysicalDeviceMemoryProperties mp{};
    mp.memoryTypeCount = 1;
    mp.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return mp;
}

bool is_reset(const leonida::vk_test::PcTail& tail) {
    return tail.set_layout == VK_NULL_HANDLE && tail.empty_layout == VK_NULL_HANDLE &&
           tail.pool == VK_NULL_HANDLE && tail.set == VK_NULL_HANDLE &&
           tail.buffer == VK_NULL_HANDLE && tail.mem == VK_NULL_HANDLE &&
           tail.stride == 0 && tail.slots == 0 && tail.map == nullptr;
}

void test_spirv_declares_set() {
    using leonida::vk_test::spirv_declares_set;

    // OpDecorate (71) %10 DescriptorSet (34) 1 and Binding (33) 1.
    const std::vector<std::uint32_t> mod_set1 = {
        0x07230203, 0x00010500, 0, 32, 0,
        word0(71, 4), 10, 33, 1,
        word0(71, 4), 10, 34, 1,
    };
    expect(spirv_declares_set(mod_set1.data(), mod_set1.size() * 4, 1), "DescriptorSet 1 is detected");
    expect(!spirv_declares_set(mod_set1.data(), mod_set1.size() * 4, 0), "DescriptorSet 0 is absent in set-1 module");

    const std::vector<std::uint32_t> mod_set0 = {
        0x07230203, 0x00010500, 0, 32, 0,
        word0(71, 4), 10, 34, 0,
    };
    expect(spirv_declares_set(mod_set0.data(), mod_set0.size() * 4, 0), "DescriptorSet 0 is detected");
    expect(!spirv_declares_set(mod_set0.data(), mod_set0.size() * 4, 1), "DescriptorSet 1 is absent in set-0 module");

    expect(!spirv_declares_set(nullptr, 24, 1), "null data returns false");
    expect(!spirv_declares_set(mod_set1.data(), 0, 1), "0 bytes returns false");
    expect(!spirv_declares_set(mod_set1.data(), 16, 1), "short header (< 20 bytes) returns false");
    expect(!spirv_declares_set(mod_set1.data(), 22, 1), "unaligned byte size returns false");

    std::vector<std::uint32_t> bad_magic = mod_set1;
    bad_magic[0] = 0x07230202;
    expect(!spirv_declares_set(bad_magic.data(), bad_magic.size() * 4, 1), "non-SPIR-V magic returns false");

    // Truncated instruction whose word count exceeds remaining words must stop without reading out of bounds.
    const std::vector<std::uint32_t> truncated_decorate = {
        0x07230203, 0x00010500, 0, 32, 0,
        word0(71, 4), 10, 34,
    };
    expect(!spirv_declares_set(truncated_decorate.data(), truncated_decorate.size() * 4, 1),
           "truncated OpDecorate (count > remaining words) returns false");

    const std::vector<std::uint32_t> huge_count = {
        0x07230203, 0x00010500, 0, 32, 0,
        word0(17, 0xffff), 1,
        word0(71, 4), 10, 34, 1,
    };
    expect(!spirv_declares_set(huge_count.data(), huge_count.size() * 4, 1),
           "instruction with count past module end stops scan cleanly");

    const std::vector<std::uint32_t> zero_count = {
        0x07230203, 0x00010500, 0, 32, 0,
        word0(71, 0), 10, 34, 1,
    };
    expect(!spirv_declares_set(zero_count.data(), zero_count.size() * 4, 1),
           "instruction with zero word count stops without looping");
}

void test_pc_tail_requested() {
    using leonida::vk_test::pc_tail_requested;

    const char* saved_ptr = std::getenv("LEONIDA_PC_TAIL_UBO");
    const bool had_saved = saved_ptr != nullptr;
    const std::string saved = had_saved ? saved_ptr : "";

    set_pc_tail_env("");
    expect(!pc_tail_requested(), "unset LEONIDA_PC_TAIL_UBO is false");

    for (const char* truthy : {"1", "TRUE ", " on ", "Yes", "\tTrUe\r\n"}) {
        set_pc_tail_env(truthy);
        expect(pc_tail_requested(), truthy);
    }
    for (const char* falsy : {"0", "off", "false", "no", "   ", "2"}) {
        set_pc_tail_env(falsy);
        expect(!pc_tail_requested(), falsy);
    }

    set_pc_tail_env(had_saved ? saved.c_str() : "");
}

void test_pctail_null_and_missing_procs() {
    using leonida::vk_test::PcTail;
    const auto mp = host_memory_props();

    FakeVulkan vk;
    g_vk = &vk;

    PcTail tail;
    expect(!tail.create(nullptr, k_dev, mp, 256, VK_SHADER_STAGE_COMPUTE_BIT, 1), "null gdpa returns false");
    expect(!tail.create(fake_gdpa, VK_NULL_HANDLE, mp, 256, VK_SHADER_STAGE_COMPUTE_BIT, 1), "null dev returns false");
    tail.destroy(nullptr, k_dev);
    tail.destroy(fake_gdpa, VK_NULL_HANDLE);
    expect(is_reset(tail) && vk.live_count() == 0, "null destroy arguments safely reset PcTail");

    for (const char* proc : {
             "vkCreateDescriptorSetLayout",
             "vkCreateDescriptorPool",
             "vkAllocateDescriptorSets",
             "vkUpdateDescriptorSets",
             "vkCreateBuffer",
             "vkGetBufferMemoryRequirements",
             "vkAllocateMemory",
             "vkBindBufferMemory",
             "vkMapMemory",
         }) {
        FakeVulkan missing_vk;
        missing_vk.omit_proc = proc;
        g_vk = &missing_vk;
        PcTail t;
        expect(!t.create(fake_gdpa, k_dev, mp, 256, VK_SHADER_STAGE_COMPUTE_BIT, 1), proc);
        expect(is_reset(t) && missing_vk.live_count() == 0, "missing proc leaves zero handles");
    }
}

void test_pctail_partial_failure_cleanup() {
    using leonida::vk_test::PcTail;
    const auto mp = host_memory_props();

    const auto check_failure = [&](const char* label, auto configure_vk, const VkPhysicalDeviceMemoryProperties& mem) {
        FakeVulkan vk;
        configure_vk(vk);
        g_vk = &vk;
        PcTail tail;
        const bool ok = tail.create(fake_gdpa, k_dev, mem, 256, VK_SHADER_STAGE_COMPUTE_BIT, 2);
        expect(!ok, label);
        expect(is_reset(tail), label);
        expect(vk.live_count() == 0, label);
        expect(!vk.freed_mem_while_buffer_bound, label);
    };

    check_failure("1st vkCreateDescriptorSetLayout failure", [](FakeVulkan& vk) { vk.fail_layout_call = 1; }, mp);
    check_failure("2nd vkCreateDescriptorSetLayout failure cleans 1st layout",
                  [](FakeVulkan& vk) { vk.fail_layout_call = 2; }, mp);
    check_failure("vkCreateDescriptorPool failure cleans layouts", [](FakeVulkan& vk) { vk.fail_pool = true; }, mp);
    check_failure("vkAllocateDescriptorSets failure cleans pool and layouts",
                  [](FakeVulkan& vk) { vk.fail_alloc_set = true; }, mp);
    check_failure("vkCreateBuffer failure cleans pool and layouts", [](FakeVulkan& vk) { vk.fail_buffer = true; }, mp);

    VkPhysicalDeviceMemoryProperties device_only_mp{};
    device_only_mp.memoryTypeCount = 2;
    device_only_mp.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    device_only_mp.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;  // missing HOST_COHERENT
    check_failure("no host-visible+coherent memory type cleans buffer, pool, and layouts",
                  [](FakeVulkan& vk) { vk.memory_type_bits = 0x3u; }, device_only_mp);

    // memoryTypeCount > 32 (VK_MAX_MEMORY_TYPES) must clamp at 32 without shifting 32-bit memoryTypeBits out of range.
    VkPhysicalDeviceMemoryProperties over32_none{};
    over32_none.memoryTypeCount = 40;
    check_failure("memoryTypeCount == 40 without matching type avoids UB and cleans up",
                  [](FakeVulkan& vk) { vk.memory_type_bits = 0xffff'ffffu; }, over32_none);

    {
        VkPhysicalDeviceMemoryProperties over32_bit31{};
        over32_bit31.memoryTypeCount = 40;
        over32_bit31.memoryTypes[31].propertyFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        FakeVulkan vk;
        vk.memory_type_bits = std::uint32_t(1) << 31;
        g_vk = &vk;
        PcTail tail;
        expect(tail.create(fake_gdpa, k_dev, over32_bit31, 256, VK_SHADER_STAGE_COMPUTE_BIT, 1),
               "memoryTypeCount == 40 selects bit 31 cleanly");
        expect(vk.last_alloc_mem_type == 31, "selected memory type 31");
        tail.destroy(fake_gdpa, k_dev);
        expect(vk.live_count() == 0, "memoryTypeCount == 40 cleanup leaves zero handles");
    }

    check_failure("vkAllocateMemory failure cleans buffer, pool, and layouts",
                  [](FakeVulkan& vk) { vk.fail_alloc_mem = true; }, mp);
    check_failure("vkBindBufferMemory failure destroys buffer before freeing memory",
                  [](FakeVulkan& vk) { vk.fail_bind = true; }, mp);
    check_failure("vkMapMemory error cleans memory, buffer, pool, and layouts",
                  [](FakeVulkan& vk) { vk.fail_map = true; }, mp);
    check_failure("vkMapMemory null pointer cleans memory, buffer, pool, and layouts",
                  [](FakeVulkan& vk) { vk.map_null_ptr = true; }, mp);
}

void test_pctail_overflow_bounds() {
    using leonida::vk_test::PcTail;
    const auto mp = host_memory_props();

    // Near-UINT64_MAX alignment with 2 slots overflows stride * slots and must fail before creating Vulkan objects.
    {
        FakeVulkan vk;
        g_vk = &vk;
        PcTail tail;
        expect(!tail.create(fake_gdpa, k_dev, mp, UINT64_MAX - 5, VK_SHADER_STAGE_COMPUTE_BIT, 2),
               "near-UINT64_MAX alignment with 2 slots rejected");
        expect(!tail.create(fake_gdpa, k_dev, mp, UINT64_MAX, VK_SHADER_STAGE_COMPUTE_BIT, 2),
               "UINT64_MAX alignment with 2 slots rejected");
        expect(vk.create_layout_calls == 0 && is_reset(tail), "overflow rejected before Vulkan calls");
    }

    // Near-UINT64_MAX alignment with 1 slot must not wrap (slot_bytes + align - 1) to 0; huge allocation fails cleanly.
    {
        FakeVulkan vk;
        g_vk = &vk;
        PcTail tail;
        expect(!tail.create(fake_gdpa, k_dev, mp, UINT64_MAX - 5, VK_SHADER_STAGE_COMPUTE_BIT, 1),
               "near-UINT64_MAX alignment with 1 slot fails allocation cleanly");
        expect(vk.last_buffer_size == UINT64_MAX - 5, "stride did not wrap around to 0");
        expect(vk.live_count() == 0 && is_reset(tail), "huge single-slot failure leaves zero handles");
    }

    // Dynamic offset must fit in uint32_t for every slot: align = 256, slot_count = (1 << 24) + 2 -> last offset =
    // 256 * ((1 << 24) + 1) = 0x100000100 > UINT32_MAX.
    {
        FakeVulkan vk;
        g_vk = &vk;
        PcTail tail;
        expect(!tail.create(fake_gdpa, k_dev, mp, 256, VK_SHADER_STAGE_COMPUTE_BIT, (1u << 24) + 2u),
               "dynamic offset exceeding UINT32_MAX rejected");
        expect(vk.create_layout_calls == 0 && is_reset(tail), "dynamic-offset overflow rejected before Vulkan calls");
    }
}

void test_pctail_slots_write_and_recreate() {
    using leonida::vk_test::PcTail;
    const auto mp = host_memory_props();

    FakeVulkan vk;
    g_vk = &vk;
    PcTail tail;
    expect(tail.create(fake_gdpa, k_dev, mp, 64, VK_SHADER_STAGE_COMPUTE_BIT, 3), "3-slot PcTail created");
    expect(tail.stride == 64 && tail.slots == 3, "stride aligned to 64 bytes for 3 slots");
    expect(tail.offset(0) == 0 && tail.offset(1) == 64 && tail.offset(2) == 128, "dynamic offsets match slot * stride");
    expect(vk.update_calls == 1 && vk.last_descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
               vk.last_buffer_info.range == PcTail::slot_bytes,
           "descriptor write binds 24-byte dynamic uniform range");

    // Memory was initialized to 0xCD in fake_alloc_memory and zeroed by PcTail::create().
    for (std::size_t i = 0; i < 192; ++i) {
        if (tail.map[i] != 0) {
            expect(false, "PcTail::create zero-initializes mapped buffer");
            break;
        }
    }

    std::uint32_t pc[40]{};
    for (std::uint32_t i = 0; i < 40; ++i) pc[i] = 0x1000u + i;

    // Slot 0: write 4 words (pc[32..35]); words 4..5 of slot 0 remain zero.
    tail.write(0, pc, 4);
    // Slot 1: write all 6 words (pc[32..37]).
    for (std::uint32_t i = 32; i < 40; ++i) pc[i] = 0x2000u + i;
    tail.write(1, pc, 6);
    // Slot 2: words > 6 is clamped to 6 so words past pc[37] never spill into padding or next slot.
    for (std::uint32_t i = 32; i < 40; ++i) pc[i] = 0x3000u + i;
    tail.write(2, pc, 8);

    // Ignored writes: out-of-range slot, null pc, zero words.
    for (std::uint32_t i = 32; i < 40; ++i) pc[i] = 0x9999u;
    tail.write(3, pc, 6);
    tail.write(0, nullptr, 6);
    tail.write(0, pc, 0);

    std::uint32_t slot0[6]{}, slot1[6]{}, slot2[8]{};
    std::memcpy(slot0, tail.map + 0, sizeof(slot0));
    std::memcpy(slot1, tail.map + 64, sizeof(slot1));
    std::memcpy(slot2, tail.map + 128, sizeof(slot2));

    expect(slot0[0] == 0x1020u && slot0[3] == 0x1023u && slot0[4] == 0 && slot0[5] == 0,
           "4-dword write copies pc[32..35] and leaves words 4..5 zero");
    expect(slot1[0] == 0x2020u && slot1[5] == 0x2025u, "6-dword write copies pc[32..37]");
    expect(slot2[0] == 0x3020u && slot2[5] == 0x3025u && slot2[6] == 0 && slot2[7] == 0,
           "words > 6 clamps at 6 dwords without writing into slot padding");

    // Re-creating on an already-live PcTail destroys old handles before creating new ones.
    expect(tail.create(fake_gdpa, k_dev, mp, 0, VK_SHADER_STAGE_VERTEX_BIT, 0),
           "re-create with align=0, slot_count=0 succeeds");
    expect(tail.stride == 24 && tail.slots == 1, "align=0 and slot_count=0 normalize to stride=24, slots=1");
    expect(vk.layouts.size() == 2 && vk.pools.size() == 1 && vk.buffers.size() == 1 && vk.memories.size() == 1,
           "re-create destroys previous Vulkan objects without leaking");

    tail.destroy(fake_gdpa, k_dev);
    expect(is_reset(tail) && vk.live_count() == 0 && !vk.freed_mem_while_buffer_bound,
           "destroy releases all handles in valid order");
    tail.destroy(fake_gdpa, k_dev);
    expect(is_reset(tail) && vk.live_count() == 0, "second destroy is a no-op");
}

}  // namespace

int main() {
    test_spirv_declares_set();
    test_pc_tail_requested();
    test_pctail_null_and_missing_procs();
    test_pctail_partial_failure_cleanup();
    test_pctail_overflow_bounds();
    test_pctail_slots_write_and_recreate();

    if (failures) return 1;
    std::puts("vk_pc_tail: all checks passed");
    return 0;
}

// --pc-tail-ubo for the test programs: shaders translated with `rdna2spirv.py --pc-tail-ubo` keep s0..s31 (128 bytes)
// in push constants and read the tail (heap base, code base; the --mesh-compute output address) from a uniform block
// at set 1, binding 0 (docs/design.md, "Push-constant tail as a uniform block"). LEONIDA_PC_TAIL_UBO=1 says the
// shaders were translated that way (build.py and run_gpu_tests.py pass the flag then). A test program then declares
// 128 push-constant bytes and, when the module declares set 1, binds a PcTail: a host-visible buffer of one slot per
// dispatch or draw, bound as a dynamic uniform buffer so each one can select its own tail.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace leonida::vk_test {

inline constexpr std::uint32_t pc_tail_set = 1;
inline constexpr std::uint32_t pc_tail_binding = 0;

// LEONIDA_PC_TAIL_UBO=1: the shaders were translated with --pc-tail-ubo. The values run_gpu_tests.py accepts:
// 1/true/yes/on, any case, with the trailing space cmd.exe's `set VAR=1 &` leaves.
inline bool pc_tail_requested() {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* v = std::getenv("LEONIDA_PC_TAIL_UBO");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    std::string s = v ? v : "";
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// OpDecorate (71) <id> DescriptorSet (34) <set>: a --pc-tail-ubo module declares set 1 only when it reads the tail.
inline bool spirv_declares_set(const void* data, std::size_t bytes, std::uint32_t set) {
    if (!data || bytes < 20 || bytes % 4) return false;
    std::vector<std::uint32_t> words(bytes / 4);
    std::memcpy(words.data(), data, bytes);
    if (words[0] != 0x07230203) return false;
    for (std::size_t i = 5; i < words.size() && (words[i] >> 16);) {
        const std::uint32_t op = words[i] & 0xffff, count = words[i] >> 16;
        if (count > words.size() - i) break;
        if (op == 71 && count == 4 && words[i + 2] == 34 && words[i + 3] == set) return true;
        i += count;
    }
    return false;
}

// The tail block, `slots` slots of up to 6 dwords each (the 24-byte --mesh-compute tail), each at a multiple of the
// device's minUniformBufferOffsetAlignment. create() destroys what it already made and returns false if any call or
// size check fails. set_layout goes at index pc_tail_set of the pipeline layout; a program with no set 0 of its own
// puts empty_layout at index 0. A program that destroys its objects before vkDestroyDevice (so the validation layer
// can count leaks) calls destroy().
struct PcTail {
    static constexpr VkDeviceSize slot_bytes = 6 * sizeof(std::uint32_t);
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout empty_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize stride = 0;
    std::uint32_t slots = 0;
    std::uint8_t* map = nullptr;

    bool create(PFN_vkGetDeviceProcAddr gdpa, VkDevice dev, const VkPhysicalDeviceMemoryProperties& memory,
                VkDeviceSize min_offset_alignment, VkShaderStageFlags stages, std::uint32_t slot_count) {
        if (!gdpa || !dev) return false;
        if (set_layout || empty_layout || pool || set || buffer || mem || stride || slots || map) destroy(gdpa, dev);
#define PC_TAIL_LOAD(f) auto f = reinterpret_cast<PFN_##f>(gdpa(dev, #f)); if (!f) return false
        PC_TAIL_LOAD(vkCreateDescriptorSetLayout); PC_TAIL_LOAD(vkCreateDescriptorPool);
        PC_TAIL_LOAD(vkAllocateDescriptorSets); PC_TAIL_LOAD(vkUpdateDescriptorSets); PC_TAIL_LOAD(vkCreateBuffer);
        PC_TAIL_LOAD(vkGetBufferMemoryRequirements); PC_TAIL_LOAD(vkAllocateMemory); PC_TAIL_LOAD(vkBindBufferMemory);
        PC_TAIL_LOAD(vkMapMemory);
#undef PC_TAIL_LOAD
        const VkDeviceSize align = min_offset_alignment ? min_offset_alignment : 1;
        const VkDeviceSize rem = slot_bytes % align;
        const VkDeviceSize pad = rem ? align - rem : 0;
        if (pad > UINT64_MAX - slot_bytes) return false;
        const VkDeviceSize slot_stride = slot_bytes + pad;
        const std::uint32_t count = slot_count ? slot_count : 1;
        if (slot_stride > UINT64_MAX / count) return false;
        if (slot_stride * count > std::numeric_limits<std::size_t>::max()) return false;
        if ((count - 1) > 0 && slot_stride > UINT32_MAX / (count - 1)) return false;
        stride = slot_stride;
        slots = count;

        const auto fail = [&]() {
            destroy(gdpa, dev);
            return false;
        };

        VkDescriptorSetLayoutBinding b{pc_tail_binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, stages, nullptr};
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1; li.pBindings = &b;
        VkDescriptorSetLayout out_set_layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(dev, &li, nullptr, &out_set_layout) != VK_SUCCESS || !out_set_layout)
            return fail();
        set_layout = out_set_layout;
        VkDescriptorSetLayoutCreateInfo ei{};
        ei.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        VkDescriptorSetLayout out_empty_layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(dev, &ei, nullptr, &out_empty_layout) != VK_SUCCESS || !out_empty_layout)
            return fail();
        empty_layout = out_empty_layout;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &ps;
        VkDescriptorPool out_pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(dev, &pi, nullptr, &out_pool) != VK_SUCCESS || !out_pool) return fail();
        pool = out_pool;
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &set_layout;
        VkDescriptorSet out_set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev, &ai, &out_set) != VK_SUCCESS || !out_set) return fail();
        set = out_set;

        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = stride * slots; bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VkBuffer out_buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(dev, &bi, nullptr, &out_buffer) != VK_SUCCESS || !out_buffer) return fail();
        buffer = out_buffer;
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, buffer, &mr);
        constexpr VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const std::uint32_t type_limit = std::min<std::uint32_t>(memory.memoryTypeCount, VK_MAX_MEMORY_TYPES);
        std::uint32_t type = 0;
        while (type < type_limit &&
               !((mr.memoryTypeBits & (std::uint32_t(1) << type)) != 0 &&
                 (memory.memoryTypes[type].propertyFlags & host) == host))
            ++type;
        if (type == type_limit) return fail();
        VkMemoryAllocateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mi.allocationSize = mr.size; mi.memoryTypeIndex = type;
        VkDeviceMemory out_mem = VK_NULL_HANDLE;
        if (vkAllocateMemory(dev, &mi, nullptr, &out_mem) != VK_SUCCESS || !out_mem) return fail();
        mem = out_mem;
        if (vkBindBufferMemory(dev, buffer, mem, 0) != VK_SUCCESS) return fail();
        void* p = nullptr;
        if (vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) return fail();
        map = static_cast<std::uint8_t*>(p);
        std::memset(map, 0, std::size_t(stride * slots));

        VkDescriptorBufferInfo di{buffer, 0, slot_bytes};
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = set; w.dstBinding = pc_tail_binding; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w.pBufferInfo = &di;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
        return true;
    }

    // The tail of slot `slot` from a push-constant array laid out as without the flag: words 32.. of `pc`, `words`
    // of them (4, or 6 with --mesh-compute), clamped to the 6-dword slot.
    void write(std::uint32_t slot, const std::uint32_t* pc, std::uint32_t words) {
        if (!map || !pc || slot >= slots || !words) return;
        const std::uint32_t count = std::min<std::uint32_t>(words, 6u);
        std::memcpy(map + stride * slot, pc + 32, count * sizeof(std::uint32_t));
    }

    std::uint32_t offset(std::uint32_t slot) const { return std::uint32_t(stride * slot); }

    // Destroys what create() made (the set goes with its pool; freeing the memory unmaps it). Null handles are
    // skipped, so it is safe after a failed or skipped create().
    void destroy(PFN_vkGetDeviceProcAddr gdpa, VkDevice dev) {
        if (!gdpa || !dev) {
            *this = PcTail{};
            return;
        }
        const auto destroy_pool = reinterpret_cast<PFN_vkDestroyDescriptorPool>(gdpa(dev, "vkDestroyDescriptorPool"));
        const auto destroy_layout =
            reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(gdpa(dev, "vkDestroyDescriptorSetLayout"));
        const auto destroy_buffer = reinterpret_cast<PFN_vkDestroyBuffer>(gdpa(dev, "vkDestroyBuffer"));
        const auto free_memory = reinterpret_cast<PFN_vkFreeMemory>(gdpa(dev, "vkFreeMemory"));
        if (pool && destroy_pool) destroy_pool(dev, pool, nullptr);
        if (buffer && destroy_buffer) destroy_buffer(dev, buffer, nullptr);
        if (mem && free_memory) free_memory(dev, mem, nullptr);
        for (const auto layout : {set_layout, empty_layout})
            if (layout && destroy_layout) destroy_layout(dev, layout, nullptr);
        *this = PcTail{};
    }
};

} // namespace leonida::vk_test

// Cross-queue synchronization regressions for BindlessSet / VulkanContext / VulkanQueue (shdann audit item 7):
//   1. First bindless upload (fallback image initialization + vkCmdCopyBuffer) on queue A must gate queue B's
//      bindless dispatch via a shared timeline semaphore so queue B cannot read stale/empty table bytes.
//   2. Once queue A's upload completes, subsequent queue B dispatches emit zero redundant bindless waits.
//   3. Incremental updates (vkCmdUpdateBuffer) on queue A also gate queue B until completed.
//   4. A failed vkQueueSubmit rolls back both fallbacks_ready() and table_dirty_ so the retry uploads cleanly.
//   5. Destroying queue A after its upload does not break queue B's bindless sync or leak/invalidate semaphores.
// Skips with exit code 77 when no Vulkan 1.3 device with bindless, bufferDeviceAddress, and 144-byte push
// constants is available. Set LEONIDA_VK_VALIDATION=1 to check with VK_LAYER_KHRONOS_validation. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/vulkan_bindless.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#define EXPECT(condition)                                                               \
    do {                                                                                \
        if (!(condition)) {                                                             \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);                \
            return 1;                                                                   \
        }                                                                               \
    } while (0)

namespace {

// Validated SPIR-V 1.2 compute shader (compiled from GLSL 450):
// Reads set=0 binding=2 `table[]`, scans for push_constant `key[8]`, writes {table[0], found_index} to
// buffer_reference `pc.out_addr` (uvec2 at push-constant dword offset 8).
constexpr std::uint32_t kBindlessProbeSpv[] = {
    0x07230203u, 0x00010500u, 0x0008000bu, 0x00000069u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x000014e3u, 0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu,
    0x000014e4u, 0x00000001u, 0x0007000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x0000000cu,
    0x00000040u, 0x00060010u, 0x00000004u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, 0x00030003u,
    0x00000002u, 0x000001c2u, 0x00070004u, 0x455f4c47u, 0x625f5458u, 0x65666675u, 0x65725f72u, 0x65726566u,
    0x0065636eu, 0x00090004u, 0x455f4c47u, 0x625f5458u, 0x65666675u, 0x65725f72u, 0x65726566u, 0x5f65636eu,
    0x63657675u, 0x00000032u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00040005u, 0x00000008u,
    0x6b73616du, 0x00000000u, 0x00040005u, 0x0000000au, 0x6c626154u, 0x00000065u, 0x00050006u, 0x0000000au,
    0x00000000u, 0x6c626174u, 0x00000065u, 0x00030005u, 0x0000000cu, 0x00000000u, 0x00040005u, 0x00000012u,
    0x6e756f66u, 0x00000064u, 0x00040005u, 0x00000014u, 0x6e756f63u, 0x00000074u, 0x00040005u, 0x0000001au,
    0x746f6c73u, 0x00000000u, 0x00040005u, 0x00000024u, 0x65736162u, 0x00000000u, 0x00040005u, 0x0000002au,
    0x6374616du, 0x00000068u, 0x00030005u, 0x0000002cu, 0x0000006au, 0x00030005u, 0x0000003eu, 0x00004350u,
    0x00040006u, 0x0000003eu, 0x00000000u, 0x0079656bu, 0x00060006u, 0x0000003eu, 0x00000001u, 0x5f74756fu,
    0x72646461u, 0x00000000u, 0x00040006u, 0x0000003eu, 0x00000002u, 0x00646170u, 0x00030005u, 0x00000040u,
    0x00006370u, 0x00040005u, 0x00000059u, 0x4274754fu, 0x00006675u, 0x00050006u, 0x00000059u, 0x00000000u,
    0x64726f77u, 0x00000073u, 0x00040005u, 0x0000005bu, 0x5f74756fu, 0x00667562u, 0x00040047u, 0x00000009u,
    0x00000006u, 0x00000004u, 0x00030047u, 0x0000000au, 0x00000002u, 0x00040048u, 0x0000000au, 0x00000000u,
    0x00000018u, 0x00050048u, 0x0000000au, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u, 0x0000000cu,
    0x00000018u, 0x00040047u, 0x0000000cu, 0x00000021u, 0x00000002u, 0x00040047u, 0x0000000cu, 0x00000022u,
    0x00000000u, 0x00040047u, 0x0000003au, 0x00000006u, 0x00000004u, 0x00040047u, 0x0000003du, 0x00000006u,
    0x00000004u, 0x00030047u, 0x0000003eu, 0x00000002u, 0x00050048u, 0x0000003eu, 0x00000000u, 0x00000023u,
    0x00000000u, 0x00050048u, 0x0000003eu, 0x00000001u, 0x00000023u, 0x00000020u, 0x00050048u, 0x0000003eu,
    0x00000002u, 0x00000023u, 0x00000028u, 0x00040047u, 0x00000058u, 0x00000006u, 0x00000004u, 0x00030047u,
    0x00000059u, 0x00000002u, 0x00050048u, 0x00000059u, 0x00000000u, 0x00000023u, 0x00000000u, 0x00030047u,
    0x0000005bu, 0x000014ecu, 0x00040047u, 0x00000068u, 0x0000000bu, 0x00000019u, 0x00020013u, 0x00000002u,
    0x00030021u, 0x00000003u, 0x00000002u, 0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x00040020u,
    0x00000007u, 0x00000007u, 0x00000006u, 0x0003001du, 0x00000009u, 0x00000006u, 0x0003001eu, 0x0000000au,
    0x00000009u, 0x00040020u, 0x0000000bu, 0x0000000cu, 0x0000000au, 0x0004003bu, 0x0000000bu, 0x0000000cu,
    0x0000000cu, 0x00040015u, 0x0000000du, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000du, 0x0000000eu,
    0x00000000u, 0x00040020u, 0x0000000fu, 0x0000000cu, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000013u,
    0x00000000u, 0x0004002bu, 0x00000006u, 0x00000016u, 0x00000001u, 0x0004002bu, 0x00000006u, 0x00000018u,
    0x00001000u, 0x00020014u, 0x00000022u, 0x0004002bu, 0x00000006u, 0x00000026u, 0x0000000au, 0x00040020u,
    0x00000029u, 0x00000007u, 0x00000022u, 0x00030029u, 0x00000022u, 0x0000002bu, 0x0004002bu, 0x00000006u,
    0x00000033u, 0x00000008u, 0x0004001cu, 0x0000003au, 0x00000006u, 0x00000033u, 0x00040017u, 0x0000003bu,
    0x00000006u, 0x00000002u, 0x0004002bu, 0x00000006u, 0x0000003cu, 0x0000001au, 0x0004001cu, 0x0000003du,
    0x00000006u, 0x0000003cu, 0x0005001eu, 0x0000003eu, 0x0000003au, 0x0000003bu, 0x0000003du, 0x00040020u,
    0x0000003fu, 0x00000009u, 0x0000003eu, 0x0004003bu, 0x0000003fu, 0x00000040u, 0x00000009u, 0x00040020u,
    0x00000042u, 0x00000009u, 0x00000006u, 0x0003002au, 0x00000022u, 0x00000048u, 0x0004002bu, 0x0000000du,
    0x0000004bu, 0x00000001u, 0x00030027u, 0x00000057u, 0x000014e5u, 0x0003001du, 0x00000058u, 0x00000006u,
    0x0003001eu, 0x00000059u, 0x00000058u, 0x00040020u, 0x00000057u, 0x000014e5u, 0x00000059u, 0x00040020u,
    0x0000005au, 0x00000007u, 0x00000057u, 0x00040020u, 0x0000005cu, 0x00000009u, 0x0000003bu, 0x00040020u,
    0x00000062u, 0x000014e5u, 0x00000006u, 0x00040017u, 0x00000067u, 0x00000006u, 0x00000003u, 0x0006002cu,
    0x00000067u, 0x00000068u, 0x00000016u, 0x00000016u, 0x00000016u, 0x00050036u, 0x00000002u, 0x00000004u,
    0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0004003bu, 0x00000007u, 0x00000008u, 0x00000007u,
    0x0004003bu, 0x00000007u, 0x00000012u, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000014u, 0x00000007u,
    0x0004003bu, 0x00000007u, 0x0000001au, 0x00000007u, 0x0004003bu, 0x00000007u, 0x00000024u, 0x00000007u,
    0x0004003bu, 0x00000029u, 0x0000002au, 0x00000007u, 0x0004003bu, 0x00000007u, 0x0000002cu, 0x00000007u,
    0x0004003bu, 0x0000005au, 0x0000005bu, 0x00000007u, 0x00060041u, 0x0000000fu, 0x00000010u, 0x0000000cu,
    0x0000000eu, 0x0000000eu, 0x0004003du, 0x00000006u, 0x00000011u, 0x00000010u, 0x0003003eu, 0x00000008u,
    0x00000011u, 0x0003003eu, 0x00000012u, 0x00000013u, 0x0004003du, 0x00000006u, 0x00000015u, 0x00000008u,
    0x00050080u, 0x00000006u, 0x00000017u, 0x00000015u, 0x00000016u, 0x0007000cu, 0x00000006u, 0x00000019u,
    0x00000001u, 0x00000026u, 0x00000017u, 0x00000018u, 0x0003003eu, 0x00000014u, 0x00000019u, 0x0003003eu,
    0x0000001au, 0x00000013u, 0x000200f9u, 0x0000001bu, 0x000200f8u, 0x0000001bu, 0x000400f6u, 0x0000001du,
    0x0000001eu, 0x00000000u, 0x000200f9u, 0x0000001fu, 0x000200f8u, 0x0000001fu, 0x0004003du, 0x00000006u,
    0x00000020u, 0x0000001au, 0x0004003du, 0x00000006u, 0x00000021u, 0x00000014u, 0x000500b0u, 0x00000022u,
    0x00000023u, 0x00000020u, 0x00000021u, 0x000400fau, 0x00000023u, 0x0000001cu, 0x0000001du, 0x000200f8u,
    0x0000001cu, 0x0004003du, 0x00000006u, 0x00000025u, 0x0000001au, 0x00050084u, 0x00000006u, 0x00000027u,
    0x00000025u, 0x00000026u, 0x00050080u, 0x00000006u, 0x00000028u, 0x00000016u, 0x00000027u, 0x0003003eu,
    0x00000024u, 0x00000028u, 0x0003003eu, 0x0000002au, 0x0000002bu, 0x0003003eu, 0x0000002cu, 0x00000013u,
    0x000200f9u, 0x0000002du, 0x000200f8u, 0x0000002du, 0x000400f6u, 0x0000002fu, 0x00000030u, 0x00000000u,
    0x000200f9u, 0x00000031u, 0x000200f8u, 0x00000031u, 0x0004003du, 0x00000006u, 0x00000032u, 0x0000002cu,
    0x000500b0u, 0x00000022u, 0x00000034u, 0x00000032u, 0x00000033u, 0x000400fau, 0x00000034u, 0x0000002eu,
    0x0000002fu, 0x000200f8u, 0x0000002eu, 0x0004003du, 0x00000006u, 0x00000035u, 0x00000024u, 0x0004003du,
    0x00000006u, 0x00000036u, 0x0000002cu, 0x00050080u, 0x00000006u, 0x00000037u, 0x00000035u, 0x00000036u,
    0x00060041u, 0x0000000fu, 0x00000038u, 0x0000000cu, 0x0000000eu, 0x00000037u, 0x0004003du, 0x00000006u,
    0x00000039u, 0x00000038u, 0x0004003du, 0x00000006u, 0x00000041u, 0x0000002cu, 0x00060041u, 0x00000042u,
    0x00000043u, 0x00000040u, 0x0000000eu, 0x00000041u, 0x0004003du, 0x00000006u, 0x00000044u, 0x00000043u,
    0x000500abu, 0x00000022u, 0x00000045u, 0x00000039u, 0x00000044u, 0x000300f7u, 0x00000047u, 0x00000000u,
    0x000400fau, 0x00000045u, 0x00000046u, 0x00000047u, 0x000200f8u, 0x00000046u, 0x0003003eu, 0x0000002au,
    0x00000048u, 0x000200f9u, 0x0000002fu, 0x000200f8u, 0x00000047u, 0x000200f9u, 0x00000030u, 0x000200f8u,
    0x00000030u, 0x0004003du, 0x00000006u, 0x0000004au, 0x0000002cu, 0x00050080u, 0x00000006u, 0x0000004cu,
    0x0000004au, 0x0000004bu, 0x0003003eu, 0x0000002cu, 0x0000004cu, 0x000200f9u, 0x0000002du, 0x000200f8u,
    0x0000002fu, 0x0004003du, 0x00000022u, 0x0000004du, 0x0000002au, 0x000300f7u, 0x0000004fu, 0x00000000u,
    0x000400fau, 0x0000004du, 0x0000004eu, 0x0000004fu, 0x000200f8u, 0x0000004eu, 0x0004003du, 0x00000006u,
    0x00000050u, 0x00000024u, 0x00050080u, 0x00000006u, 0x00000051u, 0x00000050u, 0x00000033u, 0x00060041u,
    0x0000000fu, 0x00000052u, 0x0000000cu, 0x0000000eu, 0x00000051u, 0x0004003du, 0x00000006u, 0x00000053u,
    0x00000052u, 0x0003003eu, 0x00000012u, 0x00000053u, 0x000200f9u, 0x0000001du, 0x000200f8u, 0x0000004fu,
    0x000200f9u, 0x0000001eu, 0x000200f8u, 0x0000001eu, 0x0004003du, 0x00000006u, 0x00000055u, 0x0000001au,
    0x00050080u, 0x00000006u, 0x00000056u, 0x00000055u, 0x0000004bu, 0x0003003eu, 0x0000001au, 0x00000056u,
    0x000200f9u, 0x0000001bu, 0x000200f8u, 0x0000001du, 0x00050041u, 0x0000005cu, 0x0000005du, 0x00000040u,
    0x0000004bu, 0x0004003du, 0x0000003bu, 0x0000005eu, 0x0000005du, 0x0004007cu, 0x00000057u, 0x0000005fu,
    0x0000005eu, 0x0003003eu, 0x0000005bu, 0x0000005fu, 0x0004003du, 0x00000057u, 0x00000060u, 0x0000005bu,
    0x0004003du, 0x00000006u, 0x00000061u, 0x00000008u, 0x00060041u, 0x00000062u, 0x00000063u, 0x00000060u,
    0x0000000eu, 0x0000000eu, 0x0005003eu, 0x00000063u, 0x00000061u, 0x00000002u, 0x00000004u, 0x0004003du,
    0x00000057u, 0x00000064u, 0x0000005bu, 0x0004003du, 0x00000006u, 0x00000065u, 0x00000012u, 0x00060041u,
    0x00000062u, 0x00000066u, 0x00000064u, 0x0000000eu, 0x0000004bu, 0x0005003eu, 0x00000066u, 0x00000065u,
    0x00000002u, 0x00000004u, 0x000100fdu, 0x00010038u,
};

namespace driver {

PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;

std::atomic<VkResult> fail_next_submit{VK_SUCCESS};
std::atomic<std::uint32_t> last_wait_count{0};
std::atomic<std::uint32_t> last_signal_count{0};
std::atomic<std::uintptr_t> last_queue{0};

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    const VkResult armed = fail_next_submit.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;
    if (count > 0 && submits) {
        last_queue.store(reinterpret_cast<std::uintptr_t>(queue));
        last_wait_count.store(submits[0].waitSemaphoreCount);
        last_signal_count.store(submits[0].signalSemaphoreCount);
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkQueueSubmit") == 0) return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr proc, VkDevice device) {
    real_get_device = proc;
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(proc(device, "vkQueueSubmit"));
}

}  // namespace driver

leonida::DescriptorKey make_key(std::uint32_t tag) {
    return {tag, 0x9e3779b9u * tag, tag ^ 0x55555555u, tag * 17u, 7u, tag * tag, 0x80000000u, ~tag};
}

leonida::Dispatch make_probe_dispatch(std::uintptr_t pipe_id, const leonida::DescriptorKey& key, std::uint64_t out_addr) {
    leonida::Dispatch d{};
    d.pipeline = pipe_id;
    for (std::size_t i = 0; i < 8; ++i) d.user_data[i] = key[i];
    d.user_data[8] = static_cast<std::uint32_t>(out_addr & 0xffffffffu);
    d.user_data[9] = static_cast<std::uint32_t>(out_addr >> 32);
    d.x = 1;
    d.y = 1;
    d.z = 1;
    return d;
}

struct GateCleanup {
    PFN_vkSignalSemaphore signal = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore gate = VK_NULL_HANDLE;
    std::uint64_t target = 100;
    ~GateCleanup() {
        if (signal && device && gate) {
            VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            info.semaphore = gate;
            info.value = target;
            signal(device, &info);
        }
    }
};

struct SemaphoreOwner {
    PFN_vkDestroySemaphore destroy = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore sem = VK_NULL_HANDLE;
    ~SemaphoreOwner() {
        if (destroy && device && sem) destroy(device, sem, nullptr);
    }
};

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    d.all_queue_types = true;
    d.queue_count_wanted = 2;
    d.request_buffer_device_address = true;
    if (!d.create()) {
        std::printf("SKIP: no usable Vulkan 1.3 device\n");
        return 77;
    }

    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);
    if (!d.bindless || !d.buffer_device_address || props.limits.maxPushConstantsSize < sizeof(ShaderUserData)) {
        std::printf("SKIP: device lacks bindless, bufferDeviceAddress, or 144-byte push constants\n");
        return 77;
    }

    VkPhysicalDeviceMemoryProperties mem_props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &mem_props);

    driver::install(d.device_proc, d.device);
    const auto create_semaphore = d.fn<PFN_vkCreateSemaphore>("vkCreateSemaphore");
    const auto destroy_semaphore = d.fn<PFN_vkDestroySemaphore>("vkDestroySemaphore");
    const auto signal_semaphore = d.fn<PFN_vkSignalSemaphore>("vkSignalSemaphore");
    const auto wait_semaphores = d.fn<PFN_vkWaitSemaphores>("vkWaitSemaphores");
    const auto create_shader = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto create_pipe_layout = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto destroy_pipe_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto create_compute = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto destroy_pipe = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    EXPECT(create_semaphore && destroy_semaphore && signal_semaphore && wait_semaphores &&
           create_shader && destroy_shader && create_pipe_layout && destroy_pipe_layout &&
           create_compute && destroy_pipe);

    auto make_timeline = [&]() -> VkSemaphore {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = 0;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        VkSemaphore sem = VK_NULL_HANDLE;
        if (create_semaphore(d.device, &info, nullptr, &sem) != VK_SUCCESS) return VK_NULL_HANDLE;
        return sem;
    };
    auto signal_val = [&](VkSemaphore sem, std::uint64_t val) -> bool {
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = sem;
        info.value = val;
        return signal_semaphore(d.device, &info) == VK_SUCCESS;
    };
    auto wait_val = [&](VkSemaphore sem, std::uint64_t val, std::uint64_t timeout_ns) -> VkResult {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &sem;
        info.pValues = &val;
        return wait_semaphores(d.device, &info, timeout_ns);
    };

    VkQueue queue_a_handle = d.queue;
    std::uint32_t queue_a_family = d.family;
    VkQueue queue_b_handle = d.queue;
    std::uint32_t queue_b_family = d.family;
    VulkanQueue::QueueKind queue_b_kind = VulkanQueue::QueueKind::direct;
    if (d.compute_queue && d.compute_queue != d.queue) {
        queue_b_handle = d.compute_queue;
        queue_b_family = d.compute_family;
        queue_b_kind = VulkanQueue::QueueKind::compute;
    } else if (d.queues.size() >= 2 && d.queues[1] != d.queue) {
        queue_b_handle = d.queues[1];
    }
    const bool distinct_vk_queues = (queue_a_handle != queue_b_handle);
    std::printf("queues: A family=%u, B family=%u (%s)\n", queue_a_family, queue_b_family,
                distinct_vk_queues ? "distinct VkQueues" : "shared VkQueue");

    BindlessFeatures bfeat{};
    bfeat.partially_bound = d.partially_bound;
    bfeat.update_after_bind = d.bindless;
    bfeat.queue_families = {queue_a_family, queue_b_family};
    BindlessSet bindless(driver::get_device, d.device, mem_props, bfeat);
    EXPECT(bindless.ok());

    VkShaderModuleCreateInfo sm_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_info.codeSize = sizeof(kBindlessProbeSpv);
    sm_info.pCode = kBindlessProbeSpv;
    VkShaderModule shader = VK_NULL_HANDLE;
    EXPECT(create_shader(d.device, &sm_info, nullptr, &shader) == VK_SUCCESS);

    const VkDescriptorSetLayout set_layout = bindless.layout();
    const VkPushConstantRange pc_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
    VkPipelineLayoutCreateInfo pl_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &set_layout;
    pl_info.pushConstantRangeCount = 1;
    pl_info.pPushConstantRanges = &pc_range;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    EXPECT(create_pipe_layout(d.device, &pl_info, nullptr, &pipe_layout) == VK_SUCCESS);

    VkComputePipelineCreateInfo cp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_info.stage.module = shader;
    cp_info.stage.pName = "main";
    cp_info.layout = pipe_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    EXPECT(create_compute(d.device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &pipeline) == VK_SUCCESS);

    void* out_mapped = nullptr;
    const VkBuffer out_buf = d.buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_mapped);
    EXPECT(out_buf && out_mapped);
    const std::uint64_t out_addr = d.device_address(out_buf);
    EXPECT(out_addr != 0);
    auto* out_words = static_cast<volatile std::uint32_t*>(out_mapped);

    SemaphoreOwner gate_owner{destroy_semaphore, d.device, make_timeline()};
    SemaphoreOwner done_b_owner{destroy_semaphore, d.device, make_timeline()};
    EXPECT(gate_owner.sem && done_b_owner.sem);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        context.set_bindless_prepare([&](VkCommandBuffer cb, bool* recorded) {
            if (!bindless.fallbacks_ready()) bindless.record_fallback_init(cb);
            const bool ok = bindless.flush(cb);
            if (recorded) *recorded = bindless.has_recorded_upload();
            return ok;
        });
        context.set_bindless_commit([&](bool submitted) { bindless.table_upload_result(submitted); });

        constexpr std::uintptr_t kPipeId = 0x1001;
        ComputePipeline cp{};
        cp.pipeline = pipeline;
        cp.layout = pipe_layout;
        cp.bindless_set = bindless.set();
        context.add_pipeline(kPipeId, cp);

        auto queue_a = std::make_unique<VulkanQueue>(context, queue_a_handle, queue_a_family, VulkanQueue::QueueKind::direct);
        VulkanQueue queue_b(context, queue_b_handle, queue_b_family, queue_b_kind);
        EXPECT(queue_a->ok() && queue_b.ok());

        // Declared after queues so ~GateCleanup() fires before ~VulkanQueue()::vkQueueWaitIdle() on early error.
        GateCleanup gate_cleanup{signal_semaphore, d.device, gate_owner.sem, 100};

        CommandList empty_list;
        EXPECT(empty_list.close());
        const CommandList* empty_ptr = &empty_list;

        // 1. First upload (fallback init + vkCmdCopyBuffer) gated on queue A must block queue B.
        const DescriptorKey k1 = make_key(1);
        EXPECT(bindless.insert_key(k1, 42));
        queue_a->wait(gate_owner.sem, 1);
        EXPECT(queue_a->execute({&empty_ptr, 1}) == ReplayResult::success);

        out_words[0] = 0xdeadbeefu;
        out_words[1] = 0xdeadbeefu;
        CommandList probe1;
        EXPECT(probe1.dispatch(make_probe_dispatch(kPipeId, k1, out_addr)));
        EXPECT(probe1.close());
        const CommandList* probe1_ptr = &probe1;

        driver::last_wait_count.store(0);
        EXPECT(queue_b.execute({&probe1_ptr, 1}) == ReplayResult::success);
        const std::uint32_t waits_first = driver::last_wait_count.load();
        EXPECT(queue_b.signal(done_b_owner.sem, 1));

        bool blocked_first = true;
        if (distinct_vk_queues) {
            const VkResult r = wait_val(done_b_owner.sem, 1, 50'000'000ull);
            blocked_first = (r == VK_TIMEOUT);
        }
        EXPECT(signal_val(gate_owner.sem, 1));
        EXPECT(wait_val(done_b_owner.sem, 1, 5'000'000'000ull) == VK_SUCCESS);
        EXPECT(queue_a->idle() && queue_b.idle());

        if (waits_first == 0 || !blocked_first || out_words[0] != 63u || out_words[1] != 42u) {
            std::fprintf(stderr, "subtest 1 failed: waits=%u blocked=%d mask=%u found=%u\n",
                         waits_first, int(blocked_first), out_words[0], out_words[1]);
            return 1;
        }
        std::printf("bindless queue sync 1: first upload gated queue B (mask=%u found=%u)\n",
                    out_words[0], out_words[1]);

        // 2. After queue A's upload completed, subsequent dispatches on queue B must emit zero redundant waits.
        out_words[0] = 0u;
        out_words[1] = 0u;
        driver::last_wait_count.store(99);
        EXPECT(queue_b.execute({&probe1_ptr, 1}) == ReplayResult::success);
        EXPECT(queue_b.idle());
        EXPECT(driver::last_wait_count.load() == 0);
        EXPECT(out_words[0] == 63u && out_words[1] == 42u);
        std::printf("bindless queue sync 2: no redundant wait after upload completed\n");

        // 3. Incremental slot update (vkCmdUpdateBuffer) on queue A must also gate queue B.
        const DescriptorKey k2 = make_key(2);
        EXPECT(bindless.insert_key(k2, 77));
        queue_a->wait(gate_owner.sem, 2);
        EXPECT(queue_a->execute({&empty_ptr, 1}) == ReplayResult::success);

        out_words[0] = 0xdeadbeefu;
        out_words[1] = 0xdeadbeefu;
        CommandList probe2;
        EXPECT(probe2.dispatch(make_probe_dispatch(kPipeId, k2, out_addr)));
        EXPECT(probe2.close());
        const CommandList* probe2_ptr = &probe2;

        driver::last_wait_count.store(0);
        EXPECT(queue_b.execute({&probe2_ptr, 1}) == ReplayResult::success);
        const std::uint32_t waits_inc = driver::last_wait_count.load();
        EXPECT(queue_b.signal(done_b_owner.sem, 2));

        bool blocked_inc = true;
        if (distinct_vk_queues) {
            const VkResult r = wait_val(done_b_owner.sem, 2, 50'000'000ull);
            blocked_inc = (r == VK_TIMEOUT);
        }
        EXPECT(signal_val(gate_owner.sem, 2));
        EXPECT(wait_val(done_b_owner.sem, 2, 5'000'000'000ull) == VK_SUCCESS);
        EXPECT(queue_a->idle() && queue_b.idle());
        EXPECT(waits_inc >= 1 && blocked_inc && out_words[0] == 63u && out_words[1] == 77u);
        std::printf("bindless queue sync 3: incremental vkCmdUpdateBuffer gated queue B (found=%u)\n",
                    out_words[1]);

        // 4. Destroying queue A leaves queue B able to synchronize and upload safely.
        queue_a.reset();
        const DescriptorKey k3 = make_key(3);
        EXPECT(bindless.insert_key(k3, 99));
        out_words[0] = 0u;
        out_words[1] = 0u;
        CommandList probe3;
        EXPECT(probe3.dispatch(make_probe_dispatch(kPipeId, k3, out_addr)));
        EXPECT(probe3.close());
        const CommandList* probe3_ptr = &probe3;
        EXPECT(queue_b.execute({&probe3_ptr, 1}) == ReplayResult::success);
        EXPECT(queue_b.idle());
        EXPECT(out_words[0] == 63u && out_words[1] == 99u);
        std::printf("bindless queue sync 4: queue B works after queue A destroyed (found=%u)\n", out_words[1]);
    }

    // 5. Failed vkQueueSubmit must roll back both fallbacks_ready() and table_dirty_ so the retry succeeds.
    {
        BindlessSet fresh_bindless(driver::get_device, d.device, mem_props, bfeat);
        EXPECT(fresh_bindless.ok());

        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        context.set_bindless_prepare([&](VkCommandBuffer cb, bool* recorded) {
            if (!fresh_bindless.fallbacks_ready()) fresh_bindless.record_fallback_init(cb);
            const bool ok = fresh_bindless.flush(cb);
            if (recorded) *recorded = fresh_bindless.has_recorded_upload();
            return ok;
        });
        context.set_bindless_commit([&](bool submitted) { fresh_bindless.table_upload_result(submitted); });

        constexpr std::uintptr_t kPipeId = 0x1002;
        ComputePipeline cp{};
        cp.pipeline = pipeline;
        cp.layout = pipe_layout;
        cp.bindless_set = fresh_bindless.set();
        context.add_pipeline(kPipeId, cp);

        VulkanQueue queue_a(context, queue_a_handle, queue_a_family, VulkanQueue::QueueKind::direct);
        VulkanQueue queue_b(context, queue_b_handle, queue_b_family, queue_b_kind);
        EXPECT(queue_a.ok() && queue_b.ok());

        const DescriptorKey k4 = make_key(4);
        EXPECT(fresh_bindless.insert_key(k4, 123));

        CommandList empty_list;
        EXPECT(empty_list.close());
        const CommandList* empty_ptr = &empty_list;

        driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(queue_a.execute({&empty_ptr, 1}) == ReplayResult::unsupported);
        EXPECT(!fresh_bindless.fallbacks_ready());

        // Retry on queue_a must now initialize fallbacks, upload k4 -> 123, and gate queue_b.
        EXPECT(queue_a.execute({&empty_ptr, 1}) == ReplayResult::success);
        EXPECT(fresh_bindless.fallbacks_ready());

        out_words[0] = 0u;
        out_words[1] = 0u;
        CommandList probe4;
        EXPECT(probe4.dispatch(make_probe_dispatch(kPipeId, k4, out_addr)));
        EXPECT(probe4.close());
        const CommandList* probe4_ptr = &probe4;
        EXPECT(queue_b.execute({&probe4_ptr, 1}) == ReplayResult::success);
        EXPECT(queue_a.idle() && queue_b.idle());
        EXPECT(out_words[0] == 63u && out_words[1] == 123u);
        std::printf("bindless queue sync 5: failed submit rolled back fallbacks_ready and retried cleanly\n");
    }

    destroy_pipe(d.device, pipeline, nullptr);
    destroy_pipe_layout(d.device, pipe_layout, nullptr);
    destroy_shader(d.device, shader, nullptr);

    EXPECT(d.validation_errors() == 0);
    std::printf("vk_bindless_queue_sync_test: ok\n");
    return 0;
}

// Cross-queue synchronization and lifetime regressions for storage-image initialization (runtime/src/vulkan_backend.h):
//   1. Destroying the initializing queue before another queue submits must not leave storage_init waiting on a
//      destroyed VkSemaphore (VUID-VkSubmitInfo-pWaitSemaphores-parameter); the first consumer (including a fresh
//      shader queue submitting after GPU completion) retains a queue timeline wait to establish Vulkan's
//      signal-to-batch memory dependency, and subsequent submissions on that queue omit redundant waits.
//   2. Destroying the initializing queue while another queue's submission waiting on storage initialization is still
//      in flight on the GPU must not destroy a waited-on VkSemaphore (VUID-vkDestroySemaphore-semaphore-05149).
//   3. While storage initialization on Queue A is still in flight, unrelated empty and CopyBuffer submissions on Queue B
//      emit zero waits and finish immediately; a submission touching an initialized storage image waits on
//      storage_init_timeline_, and a subsequent shader submission on Queue B emits zero redundant waits.
//   4. Chained storage initializations across different queues stay strictly ordered via the context-owned timeline
//      semaphore.
//   5. Failed submission rolls back storage_pending_ without leaking waits_ entries or advancing storage_init_seen_, and
//      remove_resource() clears both pending and initialized storage-init entries.
//   6. Timeline semaphore creation failure marks VulkanContext unloaded, VulkanQueue not ok, and execute unsupported.
// Skips (77) without a Vulkan 1.3 device. No shaders or game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
using Kind = VulkanQueue::QueueKind;

constexpr std::uint32_t kTex = 64;
constexpr std::uint32_t kReadbackBytes = kTex * kTex * 4;
constexpr std::uintptr_t kPipeId = 1;
constexpr std::uint32_t kPrefillWord = 0xdeadbeefu;

// Validated SPIR-V 1.2 no-op compute shader (LocalSize 1,1,1):
//   OpCapability Shader; OpMemoryModel Logical GLSL450; OpEntryPoint GLCompute %main "main"; OpReturn
constexpr std::array<std::uint32_t, 35> kNoopComputeSpirv = {
    0x07230203u, 0x00010200u, 0x00070000u, 0x00000005u, 0x00000000u, 0x00020011u, 0x00000001u,
    0x0003000eu, 0x00000000u, 0x00000001u, 0x0005000fu, 0x00000005u, 0x00000001u, 0x6e69616du,
    0x00000000u, 0x00060010u, 0x00000001u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u,
    0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00050036u, 0x00000002u,
    0x00000001u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000004u, 0x000100fdu, 0x00010038u,
};

template<typename Fn>
struct ScopeExit {
    Fn fn;
    bool active = true;
    explicit ScopeExit(Fn f) : fn(std::move(f)) {}
    ~ScopeExit() {
        if (active) fn();
    }
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
};

template<typename Fn>
ScopeExit<Fn> make_scope_exit(Fn fn) {
    return ScopeExit<Fn>(std::move(fn));
}

namespace driver {

struct PendingWaitRef {
    VkSemaphore wait_sem = VK_NULL_HANDLE;
    VkSemaphore signal_sem = VK_NULL_HANDLE;
    std::uint64_t signal_value = 0;
};

PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateSemaphore real_create_semaphore = nullptr;
PFN_vkDestroySemaphore real_destroy_semaphore = nullptr;
PFN_vkGetSemaphoreCounterValue real_get_counter = nullptr;
PFN_vkQueueSubmit real_queue_submit = nullptr;

std::mutex mu;
std::unordered_set<VkSemaphore> destroyed_semaphores;
std::vector<PendingWaitRef> pending_wait_refs;
std::vector<std::pair<VkSemaphore, std::uint64_t>> last_waits;
std::vector<std::pair<VkSemaphore, std::uint64_t>> last_signals;
std::atomic<bool> waited_on_destroyed{false};
std::atomic<bool> destroyed_while_in_flight{false};
std::atomic<VkResult> fail_next_submit{VK_SUCCESS};
std::atomic<VkResult> fail_next_create_semaphore{VK_SUCCESS};

const VkTimelineSemaphoreSubmitInfo* find_timeline_info(const void* pNext) {
    for (const auto* next = static_cast<const VkBaseInStructure*>(pNext); next; next = next->pNext) {
        if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
            return reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
    }
    return nullptr;
}

VKAPI_ATTR VkResult VKAPI_CALL create_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info,
                                                  const VkAllocationCallbacks* alloc, VkSemaphore* out) {
    const VkResult armed = fail_next_create_semaphore.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;
    const VkResult res = real_create_semaphore(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out) {
        std::lock_guard lock(mu);
        destroyed_semaphores.erase(*out);
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* alloc) {
    bool in_flight = false;
    if (sem) {
        std::lock_guard lock(mu);
        for (auto it = pending_wait_refs.begin(); it != pending_wait_refs.end();) {
            if (destroyed_semaphores.count(it->signal_sem)) {
                it = pending_wait_refs.erase(it);
                continue;
            }
            std::uint64_t done = 0;
            if (real_get_counter &&
                real_get_counter(device, it->signal_sem, &done) == VK_SUCCESS &&
                done >= it->signal_value) {
                it = pending_wait_refs.erase(it);
                continue;
            }
            if (it->wait_sem == sem) in_flight = true;
            ++it;
        }
        if (in_flight) destroyed_while_in_flight.store(true, std::memory_order_relaxed);
        destroyed_semaphores.insert(sem);
    }
    if (!in_flight) real_destroy_semaphore(device, sem, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    std::vector<PendingWaitRef> added;
    bool hit_destroyed = false;
    {
        std::lock_guard lock(mu);
        last_waits.clear();
        last_signals.clear();
        if (count > 0 && submits) {
            const VkSubmitInfo& s0 = submits[0];
            const auto* tl0 = find_timeline_info(s0.pNext);
            for (std::uint32_t i = 0; i < s0.waitSemaphoreCount; ++i) {
                const VkSemaphore w = s0.pWaitSemaphores ? s0.pWaitSemaphores[i] : VK_NULL_HANDLE;
                const std::uint64_t v =
                    (tl0 && tl0->pWaitSemaphoreValues && i < tl0->waitSemaphoreValueCount) ? tl0->pWaitSemaphoreValues[i]
                                                                                            : 0;
                last_waits.emplace_back(w, v);
            }
            for (std::uint32_t i = 0; i < s0.signalSemaphoreCount; ++i) {
                const VkSemaphore sig = s0.pSignalSemaphores ? s0.pSignalSemaphores[i] : VK_NULL_HANDLE;
                const std::uint64_t v = (tl0 && tl0->pSignalSemaphoreValues && i < tl0->signalSemaphoreValueCount)
                                            ? tl0->pSignalSemaphoreValues[i]
                                            : 0;
                last_signals.emplace_back(sig, v);
            }
        }
        for (std::uint32_t s = 0; s < count && submits; ++s) {
            const VkSubmitInfo& info = submits[s];
            for (std::uint32_t i = 0; i < info.waitSemaphoreCount; ++i) {
                const VkSemaphore w = info.pWaitSemaphores ? info.pWaitSemaphores[i] : VK_NULL_HANDLE;
                if (w && destroyed_semaphores.count(w)) hit_destroyed = true;
            }
        }
        if (hit_destroyed) {
            waited_on_destroyed.store(true, std::memory_order_relaxed);
        } else if (fail_next_submit.load(std::memory_order_relaxed) == VK_SUCCESS) {
            for (std::uint32_t s = 0; s < count && submits; ++s) {
                const VkSubmitInfo& info = submits[s];
                const auto* tl = find_timeline_info(info.pNext);
                if (info.signalSemaphoreCount > 0 && info.pSignalSemaphores && info.pSignalSemaphores[0] && tl &&
                    tl->signalSemaphoreValueCount > 0 && tl->pSignalSemaphoreValues && tl->pSignalSemaphoreValues[0] > 0) {
                    const VkSemaphore track_sem = info.pSignalSemaphores[0];
                    const std::uint64_t track_val = tl->pSignalSemaphoreValues[0];
                    for (std::uint32_t i = 0; i < info.waitSemaphoreCount; ++i) {
                        const VkSemaphore w = info.pWaitSemaphores ? info.pWaitSemaphores[i] : VK_NULL_HANDLE;
                        if (w) added.push_back({w, track_sem, track_val});
                    }
                }
            }
            pending_wait_refs.insert(pending_wait_refs.end(), added.begin(), added.end());
        }
    }
    if (hit_destroyed) return VK_ERROR_OUT_OF_HOST_MEMORY;
    const VkResult armed = fail_next_submit.exchange(VK_SUCCESS);
    if (armed != VK_SUCCESS) return armed;

    const VkResult res = real_queue_submit(queue, count, submits, fence);
    if (res != VK_SUCCESS && !added.empty()) {
        std::lock_guard lock(mu);
        for (const auto& r : added) {
            for (auto it = pending_wait_refs.begin(); it != pending_wait_refs.end(); ++it) {
                if (it->wait_sem == r.wait_sem && it->signal_sem == r.signal_sem &&
                    it->signal_value == r.signal_value) {
                    pending_wait_refs.erase(it);
                    break;
                }
            }
        }
    }
    return res;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateSemaphore") return reinterpret_cast<PFN_vkVoidFunction>(create_semaphore);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr proc, VkDevice device) {
    real_get_device = proc;
    real_create_semaphore = reinterpret_cast<PFN_vkCreateSemaphore>(proc(device, "vkCreateSemaphore"));
    real_destroy_semaphore = reinterpret_cast<PFN_vkDestroySemaphore>(proc(device, "vkDestroySemaphore"));
    real_get_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(proc(device, "vkGetSemaphoreCounterValue"));
    real_queue_submit = reinterpret_cast<PFN_vkQueueSubmit>(proc(device, "vkQueueSubmit"));
}

std::vector<std::pair<VkSemaphore, std::uint64_t>> waits() {
    std::lock_guard lock(mu);
    return last_waits;
}

std::vector<std::pair<VkSemaphore, std::uint64_t>> signals() {
    std::lock_guard lock(mu);
    return last_signals;
}

bool waited_for(VkSemaphore sem, std::uint64_t* out_val = nullptr) {
    std::lock_guard lock(mu);
    for (const auto& [w, v] : last_waits) {
        if (w == sem) {
            if (out_val) *out_val = v;
            return true;
        }
    }
    return false;
}

bool waited_other_than(VkSemaphore extra_wait, VkSemaphore* out_sem = nullptr, std::uint64_t* out_val = nullptr) {
    std::lock_guard lock(mu);
    for (const auto& [w, v] : last_waits) {
        if (w != extra_wait) {
            if (out_sem) *out_sem = w;
            if (out_val) *out_val = v;
            return true;
        }
    }
    return false;
}

}  // namespace driver

bool prefill_image(vktest::Device& d, VkImage img, VkImageLayout final_layout = VK_IMAGE_LAYOUT_UNDEFINED) {
    const auto cmd_barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    const auto cmd_clear = d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    if (!cmd_barrier || !cmd_clear) return false;
    return d.run([&](VkCommandBuffer cb) {
        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = img;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        cmd_barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                    &to_dst);

        VkClearColorValue val{};
        val.uint32[0] = kPrefillWord;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        cmd_clear(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &val, 1, &range);

        if (final_layout != VK_IMAGE_LAYOUT_UNDEFINED && final_layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            VkImageMemoryBarrier to_final{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            to_final.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_final.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            to_final.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            to_final.newLayout = final_layout;
            to_final.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_final.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            to_final.image = img;
            to_final.subresourceRange = range;
            cmd_barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                        nullptr, 1, &to_final);
        }
        return true;
    });
}

ResourceBinding make_storage_binding(VkImage img) {
    ResourceBinding r{};
    r.image = img;
    r.format = VK_FORMAT_R32_UINT;
    r.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    r.width = kTex;
    r.height = kTex;
    r.mip_levels = 1;
    r.array_layers = 1;
    return r;
}

ResourceBinding make_buffer_binding(VkBuffer buf, VkDeviceSize size, bool host_readable = false) {
    ResourceBinding r{};
    r.buffer = buf;
    r.size = size;
    r.host_readable = host_readable;
    return r;
}

CommandList make_dispatch_list() {
    CommandList list;
    Dispatch dp{};
    dp.pipeline = kPipeId;
    dp.x = 1;
    dp.y = 1;
    dp.z = 1;
    list.dispatch(dp);
    list.close();
    return list;
}

CommandList make_copy_tex_list(ResourceId dst_buf, ResourceId src_tex) {
    CommandList list;
    CopyTexture ct{};
    ct.dst = {dst_buf, true, 0, 0, kTex, kTex, 1, kTex * 4};
    ct.src = {src_tex, false, 0, 0, 0, 0, 0, 0};
    list.copy_texture(ct);
    list.close();
    return list;
}

CommandList make_copy_buf_list(ResourceId dst_buf, ResourceId src_buf, std::uint64_t bytes = 256) {
    CommandList list;
    list.copy_buffer({dst_buf, 0, src_buf, 0, bytes});
    list.close();
    return list;
}

CommandList make_empty_list() {
    CommandList list;
    list.close();
    return list;
}

bool check_zero_texels(const void* ptr) {
    const auto* words = static_cast<const std::uint32_t*>(ptr);
    for (std::uint32_t i = 0; i < kTex * kTex; ++i) {
        if (words[i] != 0u) {
            std::fprintf(stderr, "readback[%u] = 0x%08x, expected 0x00000000\n", i, words[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device d;
    d.queue_count_wanted = 2;
    d.all_queue_types = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no Vulkan 1.3 device\n");
        return 77;
    }
    if (status != vktest::DeviceStatus::ready) return 1;

    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);
    if (props.limits.maxPushConstantsSize < sizeof(ShaderUserData)) {
        std::printf("SKIP: maxPushConstantsSize (%u) < %zu\n", props.limits.maxPushConstantsSize,
                    sizeof(ShaderUserData));
        return 77;
    }

    driver::install(d.device_proc, d.device);
    EXPECT(driver::real_create_semaphore && driver::real_destroy_semaphore && driver::real_get_counter &&
           driver::real_queue_submit);

    const auto signal_sem = d.fn<PFN_vkSignalSemaphore>("vkSignalSemaphore");
    const auto create_pipe_layout = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto destroy_pipe_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto create_shader = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto create_compute = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto destroy_pipe = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    EXPECT(signal_sem && create_pipe_layout && destroy_pipe_layout && create_shader && destroy_shader &&
           create_compute && destroy_pipe);

    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline compute_pipe = VK_NULL_HANDLE;
    std::vector<VkSemaphore> test_semaphores;
    const auto cleanup_vk = make_scope_exit([&] {
        for (VkSemaphore s : test_semaphores) driver::destroy_semaphore(d.device, s, nullptr);
        if (compute_pipe) destroy_pipe(d.device, compute_pipe, nullptr);
        if (pipe_layout) destroy_pipe_layout(d.device, pipe_layout, nullptr);
    });

    {
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo linfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        linfo.pushConstantRangeCount = 1;
        linfo.pPushConstantRanges = &push;
        EXPECT(create_pipe_layout(d.device, &linfo, nullptr, &pipe_layout) == VK_SUCCESS);

        VkShaderModuleCreateInfo minfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        minfo.codeSize = kNoopComputeSpirv.size() * sizeof(std::uint32_t);
        minfo.pCode = kNoopComputeSpirv.data();
        VkShaderModule mod = VK_NULL_HANDLE;
        EXPECT(create_shader(d.device, &minfo, nullptr, &mod) == VK_SUCCESS);

        VkComputePipelineCreateInfo pinfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pinfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                       mod, "main", nullptr};
        pinfo.layout = pipe_layout;
        const VkResult pres = create_compute(d.device, VK_NULL_HANDLE, 1, &pinfo, nullptr, &compute_pipe);
        destroy_shader(d.device, mod, nullptr);
        EXPECT(pres == VK_SUCCESS);
    }

    auto make_timeline = [&]() -> VkSemaphore {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        VkSemaphore s = VK_NULL_HANDLE;
        if (driver::create_semaphore(d.device, &info, nullptr, &s) == VK_SUCCESS) test_semaphores.push_back(s);
        return s;
    };
    auto host_signal = [&](VkSemaphore s, std::uint64_t v) {
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = s;
        info.value = v;
        return signal_sem(d.device, &info) == VK_SUCCESS;
    };

    VkQueue qB = VK_NULL_HANDLE;
    std::uint32_t qB_family = d.family;
    Kind qB_kind = Kind::direct;
    bool distinct_vk_queues = true;
    if (d.compute_queue && d.compute_queue != d.queue) {
        qB = d.compute_queue;
        qB_family = d.compute_family;
        qB_kind = Kind::compute;
    } else if (d.queues.size() > 1 && d.queues[1] != d.queue) {
        qB = d.queues[1];
        qB_family = d.family;
        qB_kind = Kind::direct;
    } else {
        qB = d.queue;
        qB_family = d.family;
        qB_kind = Kind::direct;
        distinct_vk_queues = false;
    }

    VulkanContext ctx(driver::get_device, d.device, true);
    EXPECT(ctx.loaded());
    ctx.add_pipeline(kPipeId, {compute_pipe, pipe_layout, 0, VK_NULL_HANDLE});

    void* dummy_src_map = nullptr;
    void* dummy_dst_map = nullptr;
    VkBuffer dummy_src_buf = d.buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &dummy_src_map);
    VkBuffer dummy_dst_buf = d.buffer(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dummy_dst_map);
    EXPECT(dummy_src_buf && dummy_dst_buf);
    constexpr ResourceId kDummySrc = 1000, kDummyDst = 1001;
    ctx.add_resource(kDummySrc, make_buffer_binding(dummy_src_buf, 256));
    ctx.add_resource(kDummyDst, make_buffer_binding(dummy_dst_buf, 256, true));

    const CommandList dispatch_list = make_dispatch_list();
    const CommandList* dispatch_ptr = &dispatch_list;

    // 1. Destroying the initializing queue before another queue submits must not leave storage-init waiting on a
    //    destroyed VkSemaphore; first consumers (including a fresh shader queue after GPU completion) still wait once
    //    for memory visibility, while subsequent submissions on the same consumer queue skip redundant waits.
    {
        void* rb1_map = nullptr;
        VkBuffer rb1_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb1_map);
        VkImage img1 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(rb1_buf && img1 && prefill_image(d, img1));
        constexpr ResourceId kRb1 = 10, kImg1 = 11;
        ctx.add_resource(kRb1, make_buffer_binding(rb1_buf, kReadbackBytes, true));
        ctx.add_resource(kImg1, make_storage_binding(img1), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kImg1);

        VulkanQueue queueB(ctx, qB, qB_family, qB_kind);
        EXPECT(queueB.ok());
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t val1 = 0;
        {
            VulkanQueue queueA(ctx, d.queue, d.family, Kind::direct);
            EXPECT(queueA.ok());
            EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
            const auto a_signals = driver::signals();
            EXPECT(a_signals.size() == 2);
            timeline = a_signals[1].first;
            val1 = a_signals[1].second;
            EXPECT(timeline != VK_NULL_HANDLE && val1 > 0);
        }

        const CommandList copy1 = make_copy_tex_list(kRb1, kImg1);
        const CommandList* copy1_ptr = &copy1;
        EXPECT(queueB.execute({&copy1_ptr, 1}) == ReplayResult::success);
        EXPECT(!driver::waited_on_destroyed.load());
        std::uint64_t b1_val = 0;
        EXPECT(driver::waited_for(timeline, &b1_val) && b1_val == val1);
        EXPECT(queueB.idle());
        EXPECT(check_zero_texels(rb1_map));

        EXPECT(queueB.execute({&copy1_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueB.idle());

        VulkanQueue queueC(ctx, qB, qB_family, qB_kind);
        EXPECT(queueC.ok());
        EXPECT(queueC.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        std::uint64_t c1_val = 0;
        EXPECT(driver::waited_for(timeline, &c1_val) && c1_val == val1);
        EXPECT(queueC.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueC.idle());
    }
    std::printf("1/6 destroyed initializing queue and first-consumer memory-dep wait: ok\n");

    // 2. Destroying the initializing queue while Queue B has an in-flight gated submission waiting on storage init
    //    must keep the awaited semaphore alive (no VUID-vkDestroySemaphore-semaphore-05149).
    if (distinct_vk_queues) {
        VkSemaphore gateA = make_timeline();
        VkSemaphore gateB = make_timeline();
        EXPECT(gateA && gateB);

        void* rb2_map = nullptr;
        VkBuffer rb2_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb2_map);
        VkImage img2 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(rb2_buf && img2 && prefill_image(d, img2));
        constexpr ResourceId kRb2 = 20, kImg2 = 21;
        ctx.add_resource(kRb2, make_buffer_binding(rb2_buf, kReadbackBytes, true));
        ctx.add_resource(kImg2, make_storage_binding(img2), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kImg2);

        VulkanQueue queueB(ctx, qB, qB_family, qB_kind);
        EXPECT(queueB.ok());
        {
            auto queueA = std::make_unique<VulkanQueue>(ctx, d.queue, d.family, Kind::direct);
            auto unblock = make_scope_exit([&] {
                host_signal(gateA, 10);
                host_signal(gateB, 10);
            });
            EXPECT(queueA->ok());

            queueA->wait(gateA, 1);
            EXPECT(queueA->execute({&dispatch_ptr, 1}) == ReplayResult::success);

            queueB.wait(gateB, 1);
            const CommandList copy2 = make_copy_tex_list(kRb2, kImg2);
            const CommandList* copy2_ptr = &copy2;
            EXPECT(queueB.execute({&copy2_ptr, 1}) == ReplayResult::success);
            EXPECT(driver::waited_other_than(gateB));

            EXPECT(host_signal(gateA, 1));
            queueA.reset();
            EXPECT(!driver::destroyed_while_in_flight.load());

            EXPECT(host_signal(gateB, 1));
            unblock.active = false;
        }
        EXPECT(queueB.idle());
        EXPECT(check_zero_texels(rb2_map));
        std::printf("2/6 destroyed initializing queue during in-flight waiter: ok\n");
    } else {
        std::printf("2/6 skipped: device exposes a single queue\n");
    }

    // 3. Unrelated empty and CopyBuffer work on Queue B must not block while Queue A's init is in flight, whereas
    //    touching the initialized storage image waits once on storage_init_timeline_ and covers subsequent shaders.
    if (distinct_vk_queues) {
        VkSemaphore gateA3 = make_timeline();
        EXPECT(gateA3);

        void* rb3_map = nullptr;
        VkBuffer rb3_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb3_map);
        VkImage img3 = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(rb3_buf && img3 && prefill_image(d, img3));
        constexpr ResourceId kRb3 = 30, kImg3 = 31;
        ctx.add_resource(kRb3, make_buffer_binding(rb3_buf, kReadbackBytes, true));
        ctx.add_resource(kImg3, make_storage_binding(img3), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kImg3);

        VulkanQueue queueA(ctx, d.queue, d.family, Kind::direct);
        VulkanQueue queueB(ctx, qB, qB_family, qB_kind);
        auto unblock = make_scope_exit([&] { host_signal(gateA3, 10); });
        EXPECT(queueA.ok() && queueB.ok());

        queueA.wait(gateA3, 1);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        const auto a_signals = driver::signals();
        EXPECT(a_signals.size() == 2);
        const VkSemaphore timeline = a_signals[1].first;
        const std::uint64_t val3 = a_signals[1].second;
        EXPECT(timeline != VK_NULL_HANDLE && val3 > 1);
        std::uint64_t gate_val = 0, chain_val = 0;
        EXPECT(driver::waited_for(gateA3, &gate_val) && gate_val == 1);
        EXPECT(driver::waited_for(timeline, &chain_val) && chain_val == val3 - 1);

        const CommandList empty_list = make_empty_list();
        const CommandList* empty_ptr = &empty_list;
        EXPECT(queueB.execute({&empty_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueB.idle());

        const CommandList dummy_copy = make_copy_buf_list(kDummyDst, kDummySrc);
        const CommandList* dummy_ptr = &dummy_copy;
        EXPECT(queueB.execute({&dummy_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueB.idle());

        const CommandList copy3 = make_copy_tex_list(kRb3, kImg3);
        const CommandList* copy3_ptr = &copy3;
        EXPECT(queueB.execute({&copy3_ptr, 1}) == ReplayResult::success);
        std::uint64_t b_copy_val = 0;
        EXPECT(driver::waited_for(timeline, &b_copy_val) && b_copy_val == val3);

        EXPECT(queueB.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());

        EXPECT(host_signal(gateA3, 1));
        unblock.active = false;
        EXPECT(queueA.idle() && queueB.idle());
        EXPECT(check_zero_texels(rb3_map));
        std::printf("3/6 unrelated non-shader Queue B work unblocked while storage-init waits once: ok\n");
    } else {
        std::printf("3/6 skipped: device exposes a single queue\n");
    }

    // 4. Chained monotonic storage-init signaling across two queues.
    if (distinct_vk_queues) {
        VkSemaphore gateA4 = make_timeline();
        EXPECT(gateA4);

        void* rb4a_map = nullptr;
        void* rb4b_map = nullptr;
        VkBuffer rb4a_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb4a_map);
        VkBuffer rb4b_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb4b_map);
        VkImage img4a = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        VkImage img4b = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(rb4a_buf && rb4b_buf && img4a && img4b && prefill_image(d, img4a) && prefill_image(d, img4b));
        constexpr ResourceId kRb4a = 40, kImg4a = 41, kRb4b = 42, kImg4b = 43;
        ctx.add_resource(kRb4a, make_buffer_binding(rb4a_buf, kReadbackBytes, true));
        ctx.add_resource(kImg4a, make_storage_binding(img4a), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.add_resource(kRb4b, make_buffer_binding(rb4b_buf, kReadbackBytes, true));
        ctx.add_resource(kImg4b, make_storage_binding(img4b), VK_IMAGE_LAYOUT_UNDEFINED);

        VulkanQueue queueA(ctx, d.queue, d.family, Kind::direct);
        VulkanQueue queueB(ctx, qB, qB_family, qB_kind);
        auto unblock = make_scope_exit([&] { host_signal(gateA4, 10); });
        EXPECT(queueA.ok() && queueB.ok());

        ctx.note_storage_image(kImg4a);
        queueA.wait(gateA4, 1);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        const auto sigA = driver::signals();
        EXPECT(sigA.size() == 2);
        const VkSemaphore timeline = sigA[1].first;
        const std::uint64_t valA = sigA[1].second;

        ctx.note_storage_image(kImg4b);
        EXPECT(queueB.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        std::uint64_t b_waited = 0;
        EXPECT(driver::waited_for(timeline, &b_waited) && b_waited == valA);
        const auto sigB = driver::signals();
        EXPECT(sigB.size() == 2 && sigB[1].first == timeline && sigB[1].second == valA + 1);

        EXPECT(host_signal(gateA4, 1));
        unblock.active = false;
        const CommandList copy4a = make_copy_tex_list(kRb4a, kImg4a);
        const CommandList copy4b = make_copy_tex_list(kRb4b, kImg4b);
        const CommandList* copies4[] = {&copy4a, &copy4b};
        EXPECT(queueB.execute(copies4) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueA.idle() && queueB.idle());
        EXPECT(check_zero_texels(rb4a_map) && check_zero_texels(rb4b_map));
        std::printf("4/6 chained cross-queue storage-init signaling: ok\n");
    } else {
        std::printf("4/6 skipped: device exposes a single queue\n");
    }

    // 5. Failed submit rollback (storage_pending_ restored, no leaked wait or stale storage_init_seen_) and
    //    remove_resource() cleanup of both pending and initialized entries.
    {
        // 5a: Failed submit on initializer preserves storage_pending_ for retry.
        void* rb5a_map = nullptr;
        VkBuffer rb5a_buf = d.buffer(kReadbackBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb5a_map);
        VkImage img5a = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(rb5a_buf && img5a && prefill_image(d, img5a));
        constexpr ResourceId kRb5a = 50, kImg5a = 51;
        ctx.add_resource(kRb5a, make_buffer_binding(rb5a_buf, kReadbackBytes, true));
        ctx.add_resource(kImg5a, make_storage_binding(img5a), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kImg5a);

        VulkanQueue queueA(ctx, d.queue, d.family, Kind::direct);
        EXPECT(queueA.ok());
        driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::unsupported);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::signals().size() == 2);

        const CommandList copy5a = make_copy_tex_list(kRb5a, kImg5a);
        const CommandList* copy5a_ptr = &copy5a;
        EXPECT(queueA.execute({&copy5a_ptr, 1}) == ReplayResult::success);
        EXPECT(queueA.idle());
        EXPECT(check_zero_texels(rb5a_map));

        // 5b: Failed submit while waiting on storage init must not leak waits_ or advance storage_init_seen_.
        if (distinct_vk_queues) {
            VkSemaphore gate5b = make_timeline();
            EXPECT(gate5b);
            VkImage img5b = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            EXPECT(img5b && prefill_image(d, img5b));
            constexpr ResourceId kImg5b = 52;
            ctx.add_resource(kImg5b, make_storage_binding(img5b), VK_IMAGE_LAYOUT_UNDEFINED);
            ctx.note_storage_image(kImg5b);

            VulkanQueue queueB(ctx, qB, qB_family, qB_kind);
            auto queueA2 = std::make_unique<VulkanQueue>(ctx, d.queue, d.family, Kind::direct);
            auto unblock = make_scope_exit([&] { host_signal(gate5b, 10); });
            EXPECT(queueB.ok() && queueA2->ok());

            queueA2->wait(gate5b, 1);
            EXPECT(queueA2->execute({&dispatch_ptr, 1}) == ReplayResult::success);
            const auto sig = driver::signals();
            EXPECT(sig.size() == 2);

            driver::fail_next_submit.store(VK_ERROR_OUT_OF_HOST_MEMORY);
            EXPECT(queueB.execute({&dispatch_ptr, 1}) == ReplayResult::unsupported);
            std::uint64_t failed_wait_val = 0;
            EXPECT(driver::waited_for(sig[1].first, &failed_wait_val) && failed_wait_val == sig[1].second);

            EXPECT(host_signal(gate5b, 1));
            unblock.active = false;
            queueA2.reset();

            const CommandList dummy_copy = make_copy_buf_list(kDummyDst, kDummySrc);
            const CommandList* dummy_ptr = &dummy_copy;
            EXPECT(queueB.execute({&dummy_ptr, 1}) == ReplayResult::success);
            EXPECT(!driver::waited_on_destroyed.load());
            EXPECT(driver::waits().empty());

            EXPECT(queueB.execute({&dispatch_ptr, 1}) == ReplayResult::success);
            std::uint64_t retry_wait_val = 0;
            EXPECT(driver::waited_for(sig[1].first, &retry_wait_val) && retry_wait_val == sig[1].second);
            EXPECT(queueB.execute({&dispatch_ptr, 1}) == ReplayResult::success);
            EXPECT(driver::waits().empty());
            EXPECT(queueB.idle());
        }

        // 5c: remove_resource() clears both storage_pending_ and storage_initialized_at_.
        VkImage img5c = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        VkImage img5c_reuse = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(img5c && img5c_reuse && prefill_image(d, img5c_reuse, VK_IMAGE_LAYOUT_GENERAL));
        constexpr ResourceId kReuse = 53;
        ctx.add_resource(kReuse, make_storage_binding(img5c), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kReuse);
        ctx.remove_resource(kReuse);
        ctx.add_resource(kReuse, make_storage_binding(img5c_reuse), VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::signals().size() == 1);
        EXPECT(queueA.idle());

        VkImage img5d = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        VkImage img5d_reuse = d.image(VK_FORMAT_R32_UINT, kTex, kTex, 1,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(img5d && img5d_reuse && prefill_image(d, img5d) &&
               prefill_image(d, img5d_reuse, VK_IMAGE_LAYOUT_GENERAL));
        constexpr ResourceId kReuseInit = 54;
        ctx.add_resource(kReuseInit, make_storage_binding(img5d), VK_IMAGE_LAYOUT_UNDEFINED);
        ctx.note_storage_image(kReuseInit);
        EXPECT(queueA.execute({&dispatch_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::signals().size() == 2);
        EXPECT(queueA.idle());
        ctx.remove_resource(kReuseInit);
        ctx.add_resource(kReuseInit, make_storage_binding(img5d_reuse), VK_IMAGE_LAYOUT_GENERAL);

        VulkanQueue queueB_5c(ctx, qB, qB_family, qB_kind);
        EXPECT(queueB_5c.ok());
        const CommandList copy5d = make_copy_tex_list(kRb5a, kReuseInit);
        const CommandList* copy5d_ptr = &copy5d;
        EXPECT(queueB_5c.execute({&copy5d_ptr, 1}) == ReplayResult::success);
        EXPECT(driver::waits().empty());
        EXPECT(queueB_5c.idle());
    }
    std::printf("5/6 failed-submit rollback and remove_resource cleanup: ok\n");

    // 6. Failed storage_init_timeline_ creation propagates through loaded(), ok(), and execute().
    {
        driver::fail_next_create_semaphore.store(VK_ERROR_OUT_OF_HOST_MEMORY);
        VulkanContext bad_ctx(driver::get_device, d.device, true);
        EXPECT(!bad_ctx.loaded());
        VulkanQueue bad_queue(bad_ctx, d.queue, d.family, Kind::direct);
        EXPECT(!bad_queue.ok());
        EXPECT(bad_queue.execute({&dispatch_ptr, 1}) == ReplayResult::unsupported);
    }
    std::printf("6/6 timeline semaphore creation failure propagation: ok\n");

    if (d.validation) {
        std::printf("validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    std::printf("vk_storage_init_sync_test: ok\n");
    return 0;
}

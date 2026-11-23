// Headless Vulkan compute test for translated compute shaders.
// Usage: vk_compute_test cs.spv groups expect [--require-wave64]
//   expect = "mirror64": out[g*64 + t] == g*64 + (63 - t)   (runtime/test/shaders/cs_lds.asm)
//            "workgroup_yz": pass 2x3x4 for groups; check the Y/Z histogram (cs_workgroup_yz.asm)
//            "exec64", "wavemask", "bvh": see cs_exec64.asm, cs_wavemask.asm, cs_bvh.asm; anything else prints out[0..7]
//   VKC_BENCH=N (env): dispatch N more times after the checked one and print the median / min / max wall time of one
//   dispatch (submit to queue idle) as "bench ms: median=.. min=.. max=.. n=..". The buffer is not cleared in between.
//   VKC_OUT_WORDS=N (env): make the output buffer at least N words (default groups * 64), so a single group can
//   write more than one word per thread. VKC_DUMP=<file> (env): write the whole buffer to <file> after the dispatch.
//   VKC_HEAP_INIT=<file> (env): copy the file's bytes to the start of the buffer (Xbox VA 0) after the default fill
//   and before the dispatch, so a program can read prepared inputs (a root table, descriptors, data) from the heap.
//   The file must fit in the buffer (raise VKC_OUT_WORDS); bytes past it keep the fill.
//   --require-wave64: require full 64-wide subgroups (local size X must be a multiple of 64), or exit 77.
//   VKC_SUBGROUP_SIZE=N (env): run at N lanes. Where the device can't require a compute subgroup size but N is its
//   default subgroupSize (MoltenVK: 32, and requiredSubgroupSizeStages 0), the size is still fixed for a SPIR-V 1.5
//   module (no varying subgroup size); cs_wave_probe.comp then checks, with the test's local size and flags, that every
//   subgroup is N lanes wide (by its lane numbers and a ballot), or the test fails (exit 1) without running; N-lane
//   subgroups not in invocation order print "not available", as a size that can't be pinned does.
//   VKC_SGPRS=w4,w5,... (env): set s4, s5, ... (decimal or 0x hex, up to s31), the user SGPRs after the output V#,
//   so a test can take input words (cs_user_sgprs.asm). They are zero otherwise. Same syntax as vk_harness's VKH_SGPRS.
// Contract (docs/design.md): push constants = s0..s31 (128 bytes), heap base lo/hi, code base lo/hi (144 bytes).
// LEONIDA_PC_TAIL_UBO=1 (modules from rdna2spirv.py --pc-tail-ubo): s0..s31 only, the rest at set 1 (vk_pc_tail.h).
// The output buffer is the "heap": Xbox VA 0 maps to its device address, and s[0:3] hold a V# with base 0.
// LEONIDA_VK_VALIDATION=1 (env): run under VK_LAYER_KHRONOS_validation; a validation error fails the run (exit 1),
// so scripts/run_gpu_tests.py reports it. The layer's own variables still apply, for example
// VK_KHRONOS_VALIDATION_GPUAV_ENABLE=1 for GPU-assisted validation of what the translated shader does. In batch mode
// the process environment's LEONIDA_VK_VALIDATION applies to the whole batch; a validation error is charged to the
// job during which the layer reported it (its messages are in that job's stderr, and its exit code is 1 unless the
// job failed otherwise).
//
// Batch mode: vk_compute_test --batch FILE [--job-timeout SECONDS] opens the device once and runs every job of FILE
// ("-" reads standard input) in order, so a fuzzer pays the device open (seconds on NVIDIA, more when many processes
// open it at once) once per batch instead of once per program. One job per line, fields separated by tabs:
//   cs.spv <TAB> groups <TAB> expect [<TAB> --require-wave64] [<TAB> NAME=VALUE]...
// with NAME one of VKC_OUT_WORDS, VKC_DUMP, VKC_HEAP_INIT, VKC_HEAP_FILE, VKC_SUBGROUP_SIZE, VKC_SGPRS, VKC_BENCH and
// LEONIDA_PC_TAIL_UBO: what the environment gives a single program. In batch mode they come from the job line only
// (the process environment's values are not used), so one job's settings never reach the next. Empty lines and lines
// starting with # are not jobs. Each job gets a fresh shader module, pipeline, buffers, command pool and fence, and
// the same checks and messages as a single run. On stdout, per job:
//   @vkc-job-begin I                      (flushed before the job starts: a crash is charged to job I)
//   @vkc-job-end I EXIT STATUS OUT ERR    then OUT bytes of the job's stdout and ERR bytes of its stderr
// EXIT is the exit code a single run of the job would have had; STATUS is done, timeout (a dispatch ran past
// --job-timeout, default 60 s, 0 = no limit; EXIT 124) or lost (VK_ERROR_DEVICE_LOST). After timeout or lost the
// batch stops at once with exit 3 and the caller runs the remaining jobs in a new process (tools/vk_compute_batch.py
// does). A complete batch ends with "@vkc-batch-end N" and exit 0, whatever the jobs' own exit codes. The device is
// opened with every feature a job may need that it has (computeFullSubgroups, shaderFloat64, VK_KHR_shader_fma).
// vk_compute_test --batch-version prints "vk_compute_test batch 1" (an older build prints its usage and exits 2).
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "vk_shader_fma.h"
#include "vk_pc_tail.h"
#include "vk_validation.h"
#include "cs_wave_probe_spv.h"
#include "spirv_capability.h"
#include "spirv_capability.h"

#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static PFN_vkGetInstanceProcAddr gipa;

#if defined(__GNUC__) || defined(__clang__)
#define VKC_FORMAT(f, a) __attribute__((format(printf, f, a)))
#else
#define VKC_FORMAT(f, a)
#endif

// Where one program's messages go: straight to stdout / stderr for a single run (as they always did), into a string
// for a batch job, which then prints them framed after the job.
struct Sink {
    std::FILE* direct = nullptr;
    std::string text;
    VKC_FORMAT(2, 3) void print(const char* fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        if (direct) {
            std::vfprintf(direct, fmt, ap);
        } else {
            va_list again;
            va_copy(again, ap);
            const int n = std::vsnprintf(nullptr, 0, fmt, ap);
            if (n > 0) {
                const std::size_t at = text.size();
                text.resize(at + std::size_t(n) + 1);
                std::vsnprintf(&text[at], std::size_t(n) + 1, fmt, again);
                text.resize(at + std::size_t(n));
            }
            va_end(again);
        }
        va_end(ap);
    }
};

// One program's output and how its run ended: lost = the device was lost, timed_out = a dispatch ran past the limit.
struct Io {
    Sink out, err;
    bool lost = false, timed_out = false;
};

#define CHECK(x)                                                                         \
    do {                                                                                 \
        VkResult r_ = (x);                                                               \
        if (r_ != VK_SUCCESS) {                                                          \
            io.err.print("FAILED %s -> %d (line %d)\n", #x, r_, __LINE__);               \
            if (r_ == VK_ERROR_DEVICE_LOST) io.lost = true;                              \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// The same word list parser as vk_harness: decimal or 0x hex, no sign or space, and "1,2," is rejected.
static bool parse_u32(const char*& s, uint32_t& v) {
    const bool hex = s[0] == '0' && (s[1] == 'x' || s[1] == 'X');
    const char* digits = hex ? s + 2 : s;
    if (!(hex ? std::isxdigit(uint8_t(*digits)) : std::isdigit(uint8_t(*digits)))) return false;
    char* end;
    unsigned long long x = std::strtoull(digits, &end, hex ? 16 : 10);
    if (x > 0xFFFFFFFFull) return false;
    v = uint32_t(x); s = end;
    return true;
}

// The module's SPIR-V version and the OpExecutionMode LocalSize of its GLCompute entry point "main", the one the
// pipeline is made from. All zero when there is none (no such entry point, or LocalSizeId), and then the size is
// unknown: nothing that needs it (the subgroup probe, full subgroups) is done.
static uint32_t spirv_local_size(const std::vector<char>& code, uint32_t local[3]) {
    local[0] = local[1] = local[2] = 0;
    const std::size_t n = code.size() / 4;
    std::vector<uint32_t> w(n);
    if (n) std::memcpy(w.data(), code.data(), n * 4);
    if (n < 5 || w[0] != 0x07230203u) return 0;
    constexpr uint32_t kOpEntryPoint = 15, kOpExecutionMode = 16, kGLCompute = 5, kLocalSize = 17;
    uint32_t main_id = 0;
    bool have_main = false;
    for (std::size_t i = 5; i < n;) {
        const uint32_t len = w[i] >> 16, op = w[i] & 0xFFFFu;
        if (!len || i + len > n) break;
        // OpEntryPoint model id "name" ...: the name is a nul-terminated string packed little-endian into words.
        if (op == kOpEntryPoint && len >= 4 && w[i + 1] == kGLCompute) {
            const char* name = reinterpret_cast<const char*>(&w[i + 3]);
            if (std::strncmp(name, "main", 4 * (len - 3)) == 0 && std::memchr(name, 0, 4 * (len - 3))) {
                main_id = w[i + 2];
                have_main = true;
            }
        }
        if (op == kOpExecutionMode && len == 6 && have_main && w[i + 1] == main_id && w[i + 2] == kLocalSize) {
            local[0] = w[i + 3]; local[1] = w[i + 4]; local[2] = w[i + 5];
        }
        i += len;
    }
    return w[1];
}

static bool next_item(const char*& s) {
    if (!*s) return true;
    return *s++ == ',' && *s;
}

// VKC_SGPRS: s4, s5, ... in order, at most s31 (s[0:3] stay the output V#).
static bool parse_user_sgprs(const char* s, uint32_t* pc) {
    for (uint32_t i = 4; *s; i++) {
        if (i == 32 || !parse_u32(s, pc[i]) || !next_item(s)) return false;
    }
    return true;
}

// The settings a single run reads from the environment; a batch job gives them on its line.
static const char* const kJobSettings[] = {"VKC_OUT_WORDS", "VKC_DUMP", "VKC_HEAP_INIT", "VKC_HEAP_FILE",
                                           "VKC_SUBGROUP_SIZE",
                                           "VKC_SGPRS", "VKC_BENCH", "LEONIDA_PC_TAIL_UBO"};

// One program to run: the command line of a single run, or one line of a batch.
struct Job {
    std::string spv, groups, expect;
    bool require_wave64 = false;
    bool from_env = true;                                        // a single run: settings from the environment
    std::vector<std::pair<std::string, std::string>> settings;   // a batch job: from its line (the last one wins)
    std::string error;                                           // why a batch line is not a job
    std::size_t line = 0;

    const char* get(const char* name) const {
        if (from_env) return std::getenv(name);
        for (auto it = settings.rbegin(); it != settings.rend(); ++it)
            if (it->first == name) return it->second.c_str();
        return nullptr;
    }
};

// A job read and checked before it touches the device.
struct Program {
    uint32_t pc[36] = {};
    bool workgroup_yz = false;
    uint32_t groups = 0, groups_y = 1, groups_z = 1, yz_bins = 0, origins = 0, out_words = 0;
    const char* dump_path = nullptr;
    const char* heap_init_path = nullptr;
    const char* bench = nullptr;
    std::vector<char> heap_init, code;
    std::string heap_file_path;          // VKC_HEAP_FILE: a file copied into the heap at heap_file_at
    uint64_t heap_file_at = 0;
    std::vector<char> heap_file;
    uint32_t local[3] = {};
    uint32_t spirv_version = 0;
    uint32_t req_sg = 0;
    bool pc_tail = false;
};

// Reads the job's settings and files. -1 to go on, else the exit code (2: a bad setting or file).
static int prepare(const Job& job, Io& io, Program& p) {
    const char* sgprs_env = job.get("VKC_SGPRS");
    if (sgprs_env && !parse_user_sgprs(sgprs_env, p.pc)) {
        io.err.print("VKC_SGPRS: expected up to 28 comma-separated words for s4..s31, got \"%s\"\n", sgprs_env);
        return 2;
    }
    p.workgroup_yz = job.expect == "workgroup_yz";
    if (p.workgroup_yz && job.groups != "2x3x4") {
        io.err.print("workgroup_yz expects a 2x3x4 dispatch\n");
        return 2;
    }
    p.groups = 2;
    if (!p.workgroup_yz) {
        const char* count = job.groups.c_str();
        if (!parse_u32(count, p.groups) || *count || !p.groups) {
            io.err.print("groups: expected a positive 32-bit workgroup count, got \"%s\"\n", job.groups.c_str());
            return 2;
        }
    }
    p.groups_y = p.workgroup_yz ? 3u : 1u;
    p.groups_z = p.workgroup_yz ? 4u : 1u;
    p.yz_bins = (p.groups_y + 1) * (p.groups_z + 1);
    p.origins = p.workgroup_yz ? p.groups * p.groups_y * p.groups_z : 0;
    const char* out_words_env = job.get("VKC_OUT_WORDS");
    p.dump_path = job.get("VKC_DUMP");
    p.heap_init_path = job.get("VKC_HEAP_INIT");
    if (p.heap_init_path) {
        std::ifstream f(p.heap_init_path, std::ios::binary);
        if (!f) {
            io.err.print("VKC_HEAP_INIT: could not read %s\n", p.heap_init_path);
            return 2;
        }
        p.heap_init.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    // VKC_HEAP_FILE=<path>[@<offset>]: copy a file into the heap so a shader can read a buffer (a BVH, its
    // descriptor blocks and the records) without the test hard-coding it. The offset is a decimal or 0x hex byte
    // offset (no octal, sign or spaces); addresses the shader sees are Xbox VAs, i.e. heap offsets, as for every
    // other heap buffer here. Two overlaps to keep in mind: "bvh" mode writes its synthetic scene after this copy,
    // so a file at 0x100..0x264 is overwritten there, and the output V# starts at heap offset 0, so a file at @0
    // shares bytes with the dispatch's output.
    if (const char* heap_file_env = job.get("VKC_HEAP_FILE")) {
        const std::string spec(heap_file_env);
        const std::size_t at_sign = spec.rfind('@');
        p.heap_file_path = spec.substr(0, at_sign);
        if (at_sign != std::string::npos) {
            const char* digits = spec.c_str() + at_sign + 1;
            const bool hex = digits[0] == '0' && (digits[1] == 'x' || digits[1] == 'X');
            const char* body = hex ? digits + 2 : digits;
            const auto first = static_cast<unsigned char>(*body);
            if (!(hex ? std::isxdigit(first) : std::isdigit(first))) {
                io.err.print("VKC_HEAP_FILE: expected <path>[@<offset>], got \"%s\"\n", heap_file_env);
                return 2;
            }
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(body, &end, hex ? 16 : 10);
            if (end == body || *end || parsed > (1ull << 32) || p.heap_file_path.empty()) {
                io.err.print("VKC_HEAP_FILE: expected <path>[@<offset>], got \"%s\"\n", heap_file_env);
                return 2;
            }
            p.heap_file_at = parsed;
        }
        std::ifstream hf(p.heap_file_path, std::ios::binary);
        p.heap_file.assign(std::istreambuf_iterator<char>(hf), {});
        if (p.heap_file.empty()) {
            io.err.print("VKC_HEAP_FILE: cannot read %s\n", p.heap_file_path.c_str());
            return 2;
        }
    }
    if (out_words_env) {
        char* end;
        const unsigned long n = std::strtoul(out_words_env, &end, 10);
        if (*out_words_env < '0' || *out_words_env > '9' || *end || n == 0 || n > (1ul << 24)) {
            io.err.print("VKC_OUT_WORDS: expected 1..16777216, got \"%s\"\n", out_words_env);
            return 2;
        }
        p.out_words = uint32_t(n);
    }
    std::ifstream f(job.spv, std::ios::binary);
    p.code.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    p.spirv_version = spirv_local_size(p.code, p.local);
    const char* sg_env = job.get("VKC_SUBGROUP_SIZE");
    p.req_sg = job.require_wave64 ? 64u : (sg_env ? uint32_t(std::atoi(sg_env)) : 0u);
    p.pc_tail = leonida::vk_test::pc_tail_value(job.get("LEONIDA_PC_TAIL_UBO"));
    p.bench = job.get("VKC_BENCH");
    return -1;
}

#define VKC_DEVICE_FUNCTIONS(X)                                                                                      \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkGetBufferDeviceAddress) X(vkCreateShaderModule) X(vkCreatePipelineLayout)                     \
    X(vkCreateComputePipelines) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)            \
    X(vkCmdBindPipeline) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkEndCommandBuffer) X(vkQueueSubmit)                \
    X(vkCmdPipelineBarrier) X(vkCmdDispatchBase) X(vkCmdBindDescriptorSets) X(vkCreateFence) X(vkResetFences)       \
    X(vkWaitForFences) X(vkDestroyFence) X(vkDestroyBuffer) X(vkFreeMemory) X(vkDestroyShaderModule)                 \
    X(vkDestroyPipelineLayout) X(vkDestroyPipeline) X(vkDestroyCommandPool) X(vkDeviceWaitIdle) X(vkDestroyDevice)

// The instance, the chosen physical device with what it reports, and (once created) the device and its queue.
// Not copied: the property and feature structures point at each other.
struct Device {
    VkInstance inst = VK_NULL_HANDLE;
    leonida::vk_test::OptInValidation validation;   // LEONIDA_VK_VALIDATION (its create info outlives the instance call)
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceSubgroupSizeControlProperties sgProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceVulkan13Features f13q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceMemoryProperties mp{};
    uint32_t qi = 0;
    leonida::vk_test::ShaderFma fma;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;
#define VKC_MEMBER(name) PFN_##name name = nullptr;
    VKC_DEVICE_FUNCTIONS(VKC_MEMBER)
#undef VKC_MEMBER
    Device() = default;
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

// Loads Vulkan, creates the instance and picks the physical device; prints the "device:" line. -1 or an exit code.
static int open_instance(Device& d, Io& io) {
    gipa = load_vulkan();
    if (!gipa) return 1;
    VkInstance inst = VK_NULL_HANDLE;
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    LOADI(vkCreateInstance);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (!d.validation.prepare(gipa, ici)) return 1;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    d.inst = inst;
    CHECK(d.validation.attach(gipa, inst));
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties);
    LOADI(vkGetPhysicalDeviceProperties2); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetPhysicalDeviceMemoryProperties);
#undef LOADI
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(inst, &n, pds.data());
    std::vector<VkPhysicalDeviceProperties> candidates(pds.size());
    for (std::size_t i = 0; i < pds.size(); ++i) vkGetPhysicalDeviceProperties(pds[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);
    d.pd = chosen < pds.size() ? pds[chosen] : VK_NULL_HANDLE;
    if (!d.pd) {
        leonida::vk_test::report_no_device();
        return 1;
    }
    vkGetPhysicalDeviceProperties(d.pd, &d.props);
    d.subgroup.pNext = &d.sgProps;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &d.subgroup;
    vkGetPhysicalDeviceProperties2(d.pd, &props2);
    io.err.print("device: %s (subgroup %u..%u)%s\n", d.props.deviceName, d.sgProps.minSubgroupSize,
                 d.sgProps.maxSubgroupSize, d.validation.enabled() ? " (validation on)" : "");
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(d.pd, &qn, qf.data());
    while (d.qi < qn && !(qf[d.qi].queueFlags & VK_QUEUE_COMPUTE_BIT)) d.qi++;
    d.f12q.pNext = &d.f13q;
    d.f2q.pNext = &d.f12q;
    vkGetPhysicalDeviceFeatures2(d.pd, &d.f2q);
    vkGetPhysicalDeviceMemoryProperties(d.pd, &d.mp);
    // OpFmaKHR (rdna2spirv.py --fma khr): VK_KHR_shader_fma, when the device has it.
    d.fma.query(gipa, inst, d.pd);
    return -1;
}

// How one job runs on this device: the subgroup size it can request or observe, and what the device must enable.
struct Plan {
    bool can_request_size = false, fixed = false, observed = false, observed_full = false, needs_float64 = false,
         needs_int64 = false, int64_atomics = false;
    uint32_t local_total = 0;
};

static int plan_job(const Device& d, const Job& job, const Program& p, Io& io, Plan& pl) {
    const uint32_t max_groups_x = d.props.limits.maxComputeWorkGroupCount[0];
    if (p.groups > max_groups_x) {
        io.err.print("groups: %u exceeds the X dispatch limit of %u on %s\n", p.groups, max_groups_x,
                     d.props.deviceName);
        return 2;
    }
    const uint32_t req_sg = p.req_sg;
    pl.can_request_size = req_sg > 0 && d.f13q.subgroupSizeControl &&
        req_sg >= d.sgProps.minSubgroupSize && req_sg <= d.sgProps.maxSubgroupSize &&
        (d.sgProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    if (job.require_wave64 && (!d.f13q.computeFullSubgroups || (!pl.can_request_size && d.subgroup.subgroupSize != 64))) {
        io.err.print("full 64-wide compute subgroups unavailable\n");
        return 77;
    }
    pl.needs_float64 = leonida::vk_test::spirv_declares_capability(p.code.data(), p.code.size(), 10);  // Float64
    if (pl.needs_float64 && !d.f2q.features.shaderFloat64) {
        io.err.print("unsupported: %s uses Float64 and the device has no shaderFloat64\n", job.spv.c_str());
        return 77;
    }
    // shaderFloat64 and shaderInt64 only for a module that declares them: SHADER-423's address pairs need no
    // 64-bit integers (TEST-468).
    pl.needs_int64 = leonida::vk_test::needs_shader_int64(p.code.data(), p.code.size());
    if (pl.needs_int64 && !d.f2q.features.shaderInt64) {
        io.err.print("unsupported: %s declares Int64 and the device has no shaderInt64\n", job.spv.c_str());
        return 77;
    }
    // 64-bit atomics: the translator's are buffer atomics (rdna2spirv.py --int64-atomics, *_atomic_swap_x2 on
    // PhysicalStorageBuffer memory), which need shaderBufferInt64Atomics. Without the check a device that lacks it
    // (MoltenVK) failed the pipeline instead of reporting 77. A shared-memory one gets shaderSharedInt64Atomics too
    // where the device has it.
    pl.int64_atomics = leonida::vk_test::declares_capability(p.code.data(), p.code.size(),
                                                            leonida::vk_test::spirv_capability_int64_atomics);
    if (pl.int64_atomics && !d.f12q.shaderBufferInt64Atomics) {
        io.err.print("unsupported: %s declares Int64Atomics and the device has no shaderBufferInt64Atomics\n",
                     job.spv.c_str());
        return 77;
    }
    // A size the device can't require but runs anyway: one min..max range (fixed), or its default subgroupSize in a
    // module below SPIR-V 1.6 (Vulkan "Varying Subgroup Size": SubgroupSize then always equals subgroupSize), which
    // cs_wave_probe checks before the test runs (observed). Full subgroups where local size X is a multiple of
    // maxSubgroupSize (VUID-VkPipelineShaderStageCreateInfo-flags-02759).
    pl.fixed = req_sg && d.sgProps.minSubgroupSize == d.sgProps.maxSubgroupSize && d.subgroup.subgroupSize == req_sg;
    pl.local_total = p.local[0] * p.local[1] * p.local[2];
    // The probe itself needs subgroup operations in compute: the basic built-ins and a ballot.
    const bool probe_runs = (d.subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (d.subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
        (d.subgroup.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT);
    pl.observed = req_sg && !job.require_wave64 && !pl.can_request_size && !pl.fixed && probe_runs &&
        d.subgroup.subgroupSize == req_sg && p.spirv_version && p.spirv_version < 0x00010600u &&
        pl.local_total && pl.local_total <= 1024;
    pl.observed_full = pl.observed && d.f13q.computeFullSubgroups && p.local[0] % d.sgProps.maxSubgroupSize == 0;
    return -1;
}

// Creates the device and its queue. chain13: pass the Vulkan 1.3 features at all; full_subgroups and float64: what
// to enable. A single run enables what its program needs; a batch everything the device has.
static int create_device(Device& d, Io& io, bool chain13, bool full_subgroups, bool float64, bool int64,
                         bool int64_atomics) {
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = d.qi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.shaderBufferInt64Atomics = int64_atomics ? VK_TRUE : VK_FALSE;
    f12.shaderSharedInt64Atomics = int64_atomics && d.f12q.shaderSharedInt64Atomics ? VK_TRUE : VK_FALSE;
    if (chain13) {
        f13.subgroupSizeControl = d.f13q.subgroupSizeControl;
        f13.computeFullSubgroups = full_subgroups ? VK_TRUE : VK_FALSE;
        f12.pNext = &f13;
    }
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.features.shaderInt64 = int64 ? VK_TRUE : VK_FALSE;
    f2.features.shaderFloat64 = float64 ? VK_TRUE : VK_FALSE;
    f2.pNext = &f12;
    d.fma.link_after(&f2);
    const char* fma_extension = d.fma.extension();
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pNext = &f2;
    if (d.fma.supported()) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &fma_extension; }
    const auto vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(gipa(d.inst, "vkCreateDevice"));
    d.vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(d.inst, "vkGetDeviceProcAddr"));
    CHECK(vkCreateDevice(d.pd, &dci, nullptr, &d.dev));
#define VKC_LOAD(name) d.name = reinterpret_cast<PFN_##name>(d.vkGetDeviceProcAddr(d.dev, #name));
    VKC_DEVICE_FUNCTIONS(VKC_LOAD)
#undef VKC_LOAD
    d.vkGetDeviceQueue(d.dev, d.qi, 0, &d.q);
    return -1;
}

// Destroys what one job made, last first, when the job ends; nothing after a timeout, when the GPU may still be
// running it (the batch then ends the process).
struct Cleanup {
    const Io& io;
    std::vector<std::function<void()>> steps;
    explicit Cleanup(const Io& owner) : io(owner) {}
    void add(std::function<void()> step) { steps.push_back(std::move(step)); }
    ~Cleanup() {
        if (io.timed_out) return;
        for (auto it = steps.rbegin(); it != steps.rend(); ++it) (*it)();
    }
};

// Submits one command buffer and waits for its fence, at most timeout_ns (UINT64_MAX: no limit).
#define SUBMIT_AND_WAIT(cmdbuf)                                                                                   \
    do {                                                                                                          \
        CHECK(vkResetFences(dev, 1, &fence));                                                                     \
        VkSubmitInfo si_{VK_STRUCTURE_TYPE_SUBMIT_INFO};                                                          \
        si_.commandBufferCount = 1; si_.pCommandBuffers = &(cmdbuf);                                              \
        CHECK(vkQueueSubmit(q, 1, &si_, fence));                                                                  \
        const VkResult w_ = vkWaitForFences(dev, 1, &fence, VK_TRUE, timeout_ns);                                 \
        if (w_ == VK_TIMEOUT) {                                                                                   \
            io.timed_out = true;                                                                                  \
            io.err.print("timed out: the dispatch did not finish in %g s (--job-timeout)\n", timeout_ns / 1e9);   \
            return 124;                                                                                           \
        }                                                                                                         \
        if (w_ != VK_SUCCESS) {                                                                                   \
            io.err.print("FAILED vkWaitForFences -> %d (line %d)\n", w_, __LINE__);                              \
            if (w_ == VK_ERROR_DEVICE_LOST) io.lost = true;                                                       \
            return 1;                                                                                             \
        }                                                                                                         \
    } while (0)

// Runs one prepared job on an open device: the buffers, the pipeline, the dispatch and the checks of a single run,
// with the same output. -1 is never returned; the result is the job's exit code.
static int run_job(Device& d, const Job& job, Program& p, const Plan& plan, Io& io, uint64_t timeout_ns) {
    Cleanup cleanup(io);
    const char* expect = job.expect.c_str();
    const bool workgroup_yz = p.workgroup_yz, require_wave64 = job.require_wave64;
    const uint32_t groups = p.groups, groups_y = p.groups_y, groups_z = p.groups_z, yz_bins = p.yz_bins;
    const uint32_t origins = p.origins, out_words = p.out_words, req_sg = p.req_sg, local_total = plan.local_total;
    const char* const dump_path = p.dump_path;
    const char* const heap_init_path = p.heap_init_path;
    const std::vector<char>& heap_init = p.heap_init;
    const std::vector<char>& code = p.code;
    const uint32_t* const local = p.local;
    uint32_t* const pc = p.pc;
    const bool can_request_size = plan.can_request_size, fixed = plan.fixed, observed = plan.observed;
    const bool observed_full = plan.observed_full;
    const VkPhysicalDeviceMemoryProperties& mp = d.mp;
    const VkDevice dev = d.dev;
    const VkQueue q = d.q;
    const uint32_t qi = d.qi;
#define VKC_LOCAL(name) const auto name = d.name; (void)name;
    VKC_DEVICE_FUNCTIONS(VKC_LOCAL)
#undef VKC_LOCAL

    (void)qi; (void)require_wave64;
    // Output buffer = heap. Host-visible so the result can be read back directly.
    VkDeviceSize words = workgroup_yz ? yz_bins * (1 + origins) : VkDeviceSize(groups) * 64;
    if (out_words > words) words = out_words;
    // The heap file may need more than the dispatch's own output words; round its end up to a word.
    if (!p.heap_file.empty()) {
        const VkDeviceSize need = VkDeviceSize(p.heap_file_at + p.heap_file.size() + 3) / 4;
        if (need > words) words = need;
    }
    // "bvh" writes its nodes at heap offsets 0x100 to 0x264 below, past the 256 bytes one group's 64 words give: the
    // shader then read them through the device address outside the buffer (VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819).
    const bool bvh = !std::strcmp(expect, "bvh");
    if (bvh && words < 0x280 / sizeof(uint32_t)) words = 0x280 / sizeof(uint32_t);
    const VkDeviceSize bytes = words * sizeof(uint32_t);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer buf; CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));
    cleanup.add([&d, buf] { d.vkDestroyBuffer(d.dev, buf, nullptr); });
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    uint32_t mt = 0;
    for (; mt < mp.memoryTypeCount; mt++)
        if ((mr.memoryTypeBits >> mt & 1) && (mp.memoryTypes[mt].propertyFlags &
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    VkMemoryAllocateFlagsInfo maf{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    maf.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt; mai.pNext = &maf;
    VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem));
    cleanup.add([&d, mem] { d.vkFreeMemory(d.dev, mem, nullptr); });
    CHECK(vkBindBufferMemory(dev, buf, mem, 0));
    void* map; CHECK(vkMapMemory(dev, mem, 0, bytes, 0, &map));
    std::memset(map, workgroup_yz || !std::strcmp(expect, "exec64") ? 0 : 0xEE, size_t(bytes));
    if (heap_init_path) {
        if (heap_init.size() > size_t(bytes)) {
            io.err.print("VKC_HEAP_INIT: %s is %zu bytes, the buffer %llu (raise VKC_OUT_WORDS)\n",
                         heap_init_path, heap_init.size(), static_cast<unsigned long long>(bytes));
            return 2;
        }
        if (!heap_init.empty()) std::memcpy(map, heap_init.data(), heap_init.size());
    }
    if (!p.heap_file.empty()) {
        std::memcpy(static_cast<char*>(map) + p.heap_file_at, p.heap_file.data(), p.heap_file.size());
        io.out.print("heap file: %s at 0x%llx (%zu bytes)\n", p.heap_file_path.c_str(),
                     static_cast<unsigned long long>(p.heap_file_at), p.heap_file.size());
    }
    if (!std::strcmp(expect, "bvh")) {
        // AMD-format BVH in the heap (runtime/test/shaders/cs_bvh.asm): box32 root at 0x100 whose children are a
        // triangle at z=10 (node 0x40, addr 0x200), a triangle at z=5 (node 0x48, addr 0x240), a box the ray
        // misses, and a box with a NaN min.x that RDNA2 still enters at t = 0 (SHADER-650). Node id = address >> 3 | type.
        auto f32 = [&](uint32_t at, float v) { std::memcpy(static_cast<char*>(map) + at, &v, 4); };
        auto u32 = [&](uint32_t at, uint32_t v) { std::memcpy(static_cast<char*>(map) + at, &v, 4); };
        u32(0x100, 0x40); u32(0x104, 0x48); u32(0x108, 0x60); u32(0x10C, 0x68);
        const float boxes[4][6] = {{0, 0, 10, 1, 1, 10}, {0, 0, 5, 1, 1, 5}, {5, 0, 0, 6, 1, 1}, {NAN, 0, 0, 1, 1, 1}};
        for (int i = 0; i < 4; i++) for (int c = 0; c < 6; c++) f32(0x110 + 24 * i + 4 * c, boxes[i][c]);
        const float tri[2][9] = {{0, 0, 10, 1, 0, 10, 0, 1, 10}, {0, 0, 5, 1, 0, 5, 0, 1, 5}};
        for (int t = 0; t < 2; t++) for (int c = 0; c < 9; c++) f32(0x200 + 0x40 * t + 4 * c, tri[t][c]);
        // Triangle-ID word at node +60: byte 0 (TRI0) = 0x09 returns vertex 1's and vertex 2's weights, what a
        // builder writes (AMD GPURT WriteTriangleIdField); the translator reads it since the remap fix.
        for (int t = 0; t < 2; t++) u32(0x200 + 0x40 * t + 60, 0x09);
    }
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = buf;
    const uint64_t heap = vkGetBufferDeviceAddress(dev, &bai);

    {
        std::string why;
        if (!d.fma.accepts(code.data(), code.size(), job.spv.c_str(), &why)) {
            io.err.print("%s", why.c_str());
            return 77;
        }
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size(); smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    cleanup.add([&d, sm] { d.vkDestroyShaderModule(d.dev, sm, nullptr); });
    // 144 push-constant bytes: s0..s31, heap base, code base. With LEONIDA_PC_TAIL_UBO=1 (rdna2spirv.py
    // --pc-tail-ubo) only s0..s31; the tail, if the module reads it, is a uniform block at set 1 (vk_pc_tail.h), one
    // slot for the dispatch and one per workgroup_yz origin.
    const bool pc_tail = p.pc_tail;
    const uint32_t pc_bytes = pc_tail ? 128u : 144u;
    leonida::vk_test::PcTail tail;
    const bool tail_block = pc_tail &&
        leonida::vk_test::spirv_declares_set(code.data(), code.size(), leonida::vk_test::pc_tail_set);
    if (tail_block) {
        const bool made = tail.create(d.vkGetDeviceProcAddr, dev, mp, d.props.limits.minUniformBufferOffsetAlignment,
                                      VK_SHADER_STAGE_COMPUTE_BIT, 1 + origins);
        cleanup.add([&d, tail] { leonida::vk_test::PcTail(tail).destroy(d.vkGetDeviceProcAddr, d.dev); });
        if (!made) {
            io.err.print("FAILED creating the --pc-tail-ubo block\n");
            return 1;
        }
    }
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    const VkDescriptorSetLayout tail_layouts[] = {tail.empty_layout, tail.set_layout};
    if (tail_block) { pli.setLayoutCount = 2; pli.pSetLayouts = tail_layouts; }
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    cleanup.add([&d, pl] { d.vkDestroyPipelineLayout(d.dev, pl, nullptr); });
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    if (workgroup_yz) cpi.flags |= VK_PIPELINE_CREATE_DISPATCH_BASE_BIT;
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSg{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    reqSg.requiredSubgroupSize = req_sg;
    if (can_request_size)
        cpi.stage.pNext = &reqSg;
    if (require_wave64 || observed_full)
        cpi.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    cpi.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
    cleanup.add([&d, pipe] { d.vkDestroyPipeline(d.dev, pipe, nullptr); });
    if (require_wave64) io.out.print("compute subgroup size: 64\n");
    else if (req_sg) {
        // VKC_SUBGROUP_SIZE: say whether the pipeline really runs at that size, so a runner can skip a test the device
        // can't pin instead of judging words from whatever size the driver picked (#774).
        if (can_request_size || fixed) io.out.print("compute subgroup size: %u\n", req_sg);
        else if (!observed) io.out.print("compute subgroup size: %u not available\n", req_sg);
    }

    pc[0] = 0; pc[1] = 0; pc[2] = 0xFFFFFFFFu; pc[3] = 0x31016FACu;     // V#: base 0, stride 0, num_records max, OOB_SELECT 3 (raw)
    pc[32] = uint32_t(heap); pc[33] = uint32_t(heap >> 32);    // heap base
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qi;
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &pool));
    cleanup.add([&d, pool] { d.vkDestroyCommandPool(d.dev, pool, nullptr); });
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
    cleanup.add([&d, fence] { d.vkDestroyFence(d.dev, fence, nullptr); });
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    if (observed) {
        // cs_wave_probe with the test's local size and flags, one group, four words per invocation in its own buffer.
        VkBufferCreateInfo pbi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        pbi.size = VkDeviceSize(local_total) * 16;
        pbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer pbuf; CHECK(vkCreateBuffer(dev, &pbi, nullptr, &pbuf));
        cleanup.add([&d, pbuf] { d.vkDestroyBuffer(d.dev, pbuf, nullptr); });
        VkMemoryRequirements pmr; vkGetBufferMemoryRequirements(dev, pbuf, &pmr);
        VkMemoryAllocateInfo pmai = mai;
        pmai.allocationSize = pmr.size;
        VkDeviceMemory pmem; CHECK(vkAllocateMemory(dev, &pmai, nullptr, &pmem));
        cleanup.add([&d, pmem] { d.vkFreeMemory(d.dev, pmem, nullptr); });
        CHECK(vkBindBufferMemory(dev, pbuf, pmem, 0));
        void* pmap; CHECK(vkMapMemory(dev, pmem, 0, pbi.size, 0, &pmap));
        std::memset(pmap, 0xEE, size_t(pbi.size));
        VkBufferDeviceAddressInfo pbai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        pbai.buffer = pbuf;
        const uint64_t paddr = vkGetBufferDeviceAddress(dev, &pbai);
        VkShaderModuleCreateInfo psmi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        psmi.codeSize = sizeof(leonida::vk_test::cs_wave_probe_spirv);
        psmi.pCode = leonida::vk_test::cs_wave_probe_spirv;
        VkShaderModule psm; CHECK(vkCreateShaderModule(dev, &psmi, nullptr, &psm));
        cleanup.add([&d, psm] { d.vkDestroyShaderModule(d.dev, psm, nullptr); });
        const VkSpecializationMapEntry entries[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
        const VkSpecializationInfo spec{3, entries, sizeof(local), local};
        VkComputePipelineCreateInfo pcpi = cpi;
        pcpi.flags = 0;
        pcpi.stage.module = psm;
        pcpi.stage.pSpecializationInfo = &spec;
        VkPipeline ppipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pcpi, nullptr, &ppipe));
        cleanup.add([&d, ppipe] { d.vkDestroyPipeline(d.dev, ppipe, nullptr); });
        const uint32_t ppc[2] = {uint32_t(paddr), uint32_t(paddr >> 32)};   // cs_wave_probe's buffer address
        VkCommandBuffer pcmd; CHECK(vkAllocateCommandBuffers(dev, &cai, &pcmd));
        VkCommandBufferBeginInfo pcbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        CHECK(vkBeginCommandBuffer(pcmd, &pcbi));
        vkCmdBindPipeline(pcmd, VK_PIPELINE_BIND_POINT_COMPUTE, ppipe);
        vkCmdPushConstants(pcmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), ppc);
        vkCmdDispatch(pcmd, 1, 1, 1);
        VkMemoryBarrier pread{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        pread.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pread.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(pcmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &pread, 0, nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(pcmd));
        SUBMIT_AND_WAIT(pcmd);
        // Width first, from what each invocation reports: every subgroup has N lanes (one may have the rest of a
        // group that isn't a multiple of N), each lane once, and a ballot of the subgroup counts them all. A wrong
        // width fails the test. Then the order: the translator takes a wave32 lane as LocalInvocationIndex % N, so
        // invocation l must be lane l % N of subgroup l / N. Vulkan doesn't promise that mapping; where it differs,
        // the size is reported as not available, as on a device that can't give N at all (the runners skip).
        const uint32_t* pw = static_cast<const uint32_t*>(pmap);
        const uint32_t subgroups = (local_total + req_sg - 1) / req_sg, tail = local_total % req_sg;
        std::vector<uint32_t> count(subgroups, 0), seen(std::size_t(subgroups) * req_sg, 0);
        std::vector<bool> bad(local_total, false);
        for (uint32_t l = 0; l < local_total; l++) {
            const uint32_t* w = pw + 4 * l;
            if (w[0] == req_sg && w[2] < subgroups && w[1] < req_sg && !seen[std::size_t(w[2]) * req_sg + w[1]]++)
                count[w[2]]++;
            else
                bad[l] = true;
        }
        uint32_t wrong = 0, first = 0, misplaced = 0, first_misplaced = 0, partial = 0;
        for (uint32_t c : count)
            if (c != req_sg) partial++;
        for (uint32_t l = 0; l < local_total; l++) {
            const uint32_t* w = pw + 4 * l;
            if (bad[l] || w[3] != count[w[2]]) {
                if (!wrong++) first = l;
            }
            if ((w[1] != l % req_sg || w[2] != l / req_sg) && !misplaced++) first_misplaced = l;
        }
        if (!wrong && partial > (tail ? 1u : 0u)) wrong = partial;   // e.g. two half-filled subgroups
        if (wrong) {
            const uint32_t* w = pw + 4 * first;
            io.out.print("compute subgroup size: %u observed wrong in %u of %u invocations; invocation %u has "
                        "SubgroupSize %u, lane %u, subgroup %u, a ballot of %u lanes\n", req_sg, wrong, local_total,
                        first, w[0], w[1], w[2], w[3]);
            return 1;
        }
        if (misplaced) {
            const uint32_t* w = pw + 4 * first_misplaced;
            io.err.print("%u-lane subgroups, but not in invocation order: invocation %u is lane %u of "
                         "subgroup %u\n", req_sg, first_misplaced, w[1], w[2]);
            // As where the size can't be pinned at all: say so, run the test, and let the runner skip it.
            io.out.print("compute subgroup size: %u not available\n", req_sg);
        } else {
            io.out.print("compute subgroup size: %u (observed: all %u invocations of a %ux%ux%u group%s)\n", req_sg,
                        local_total, local[0], local[1], local[2], observed_full ? ", full subgroups" : "");
        }
    }
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes, pc);
    auto bind_tail = [&](uint32_t slot) {
        if (!tail_block) return;
        tail.write(slot, pc, 4);
        const uint32_t offset = tail.offset(slot);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, leonida::vk_test::pc_tail_set, 1, &tail.set,
                                1, &offset);
    };
    bind_tail(0);
    vkCmdDispatch(cmd, groups, groups_y, groups_z);
    // A permutation within one axis can leave the full-grid histogram unchanged. Dispatch one group at
    // each known origin into its own histogram, selected by the host independently of the shader's SGPRs.
    for (uint32_t g = 0; g < origins; ++g) {
        const uint64_t base = heap + (g + 1) * yz_bins * sizeof(uint32_t);
        const uint32_t address[] = {uint32_t(base), uint32_t(base >> 32)};
        if (pc_tail) {
            pc[32] = address[0]; pc[33] = address[1];     // the heap base is the tail's dwords 0..1
            bind_tail(g + 1);
        } else {
            vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 32 * sizeof(uint32_t), sizeof(address), address);
        }
        vkCmdDispatchBase(cmd, g % groups, (g / groups) % groups_y, g / (groups * groups_y), 1, 1, 1);
    }
    VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    readback.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                         1, &readback, 0, nullptr, 0, nullptr);
    CHECK(vkEndCommandBuffer(cmd));
    SUBMIT_AND_WAIT(cmd);
    if (const char* b = p.bench) {
        std::vector<double> ms;
        for (int i = 0, n = std::atoi(b); i < n; i++) {
            auto t0 = std::chrono::steady_clock::now();
            SUBMIT_AND_WAIT(cmd);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        if (!ms.empty()) {
            std::sort(ms.begin(), ms.end());
            io.out.print("bench ms: median=%.3f min=%.3f max=%.3f n=%zu\n", ms[ms.size() / 2], ms.front(), ms.back(), ms.size());
        }
        uint32_t h = 2166136261u;   // FNV-1a of the whole output buffer, to compare configurations
        for (VkDeviceSize i = 0; i < bytes; i++) h = (h ^ static_cast<const unsigned char*>(map)[i]) * 16777619u;
        io.out.print("bench out fnv1a: %08x\n", h);
    }

    if (dump_path) {
        std::ofstream dump(dump_path, std::ios::binary);
        dump.write(static_cast<const char*>(map), std::streamsize(bytes));
        dump.close();
        if (dump.fail()) {
            io.err.print("VKC_DUMP: could not write %s\n", dump_path);
            return 1;
        }
    }

    const uint32_t* out = static_cast<const uint32_t*>(map);
    uint32_t bad = 0, first_bad = ~0u;
    if (workgroup_yz) {
        // X is disabled in RSRC2. Both X groups must contribute to each valid (Y,Z) bin.
        // The last column and row catch out-of-range register values without indexing past the buffer.
        for (uint32_t z = 0; z <= groups_z; ++z) {
            for (uint32_t y = 0; y <= groups_y; ++y) {
                const uint32_t want = y < groups_y && z < groups_z ? groups : 0;
                const uint32_t got = out[z * (groups_y + 1) + y];
                if (got != want) {
                    io.err.print("(Y=%u, Z=%u): got %u, expected %u\n", y, z, got, want);
                    ++bad;
                }
            }
        }
        io.out.print("2x3x4 workgroup Y/Z: %u mismatches\n", bad);
        uint32_t wrong_origins = 0;
        for (uint32_t g = 0; g < origins; ++g) {
            const uint32_t y = (g / groups) % groups_y, z = g / (groups * groups_y);
            const uint32_t* histogram = out + (g + 1) * yz_bins;
            for (uint32_t bin = 0; bin < yz_bins; ++bin) {
                const uint32_t want = bin == z * (groups_y + 1) + y ? 1u : 0u;
                if (histogram[bin] != want) {
                    io.err.print("origin (%u,%u,%u), bin %u: got %u, expected %u\n",
                                 g % groups, y, z, bin, histogram[bin], want);
                    ++wrong_origins;
                    break;
                }
            }
        }
        io.out.print("single-group origins: %u/%u matched\n", origins - wrong_origins, origins);
        return bad || wrong_origins ? 1 : 0;
    }
    if (!std::strcmp(expect, "mirror64")) {
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                uint32_t want = g * 64 + (63 - t), got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(expect, "exec64")) {    // runtime/test/shaders/cs_exec64.asm
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                uint32_t want = (t >= 16 && t < 48) ? 7u : 0u, got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(expect, "wavemask")) {  // runtime/test/shaders/cs_wavemask.asm
        for (uint32_t g = 0; g < groups; g++)
            for (uint32_t t = 0; t < 64; t++) {
                // lanes 16-47: on again after the parked mask (0x80) and after the one through a VGPR (0x40),
                // plus how many set lanes are below this one
                uint32_t top = (t >= 16 && t < 48) ? 0xc0u + t - 16 : 0;
                uint32_t want = 65u | 16u << 8 | 8u << 16 | top << 24, got = out[g * 64 + t];
                if (got != want) { if (!bad) first_bad = g * 64 + t; bad++; }
            }
    }
    if (!std::strcmp(expect, "bvh")) {
        const uint32_t want[8] = {0x68, 0x48, 0x40, 0xFFFFFFFFu, 0xC1200000u, 0xBF800000u, 0xBE800000u, 0xBE800000u};
        for (int i = 0; i < 8; i++) if (out[i] != want[i]) { if (!bad) first_bad = i; bad++; }
        io.out.print("bvh out = %08x %08x %08x %08x | %08x %08x %08x %08x\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    }
    io.out.print("%u groups x 64: %u mismatches", groups, bad);
    if (bad) io.out.print(" (first at %u: got 0x%08x)", first_bad, out[first_bad]);
    io.out.print("\nout[0..7] = %u %u %u %u %u %u %u %u\n", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    return bad ? 1 : 0;
}

// The device for a whole batch: every feature a job may need that the device has.
static int open_batch_device(Device& d, Io& io) {
    const int rc = open_instance(d, io);
    if (rc >= 0) return rc;
    const bool full = d.f13q.computeFullSubgroups == VK_TRUE;
    return create_device(d, io, d.f13q.subgroupSizeControl || full, full, d.f2q.features.shaderFloat64 == VK_TRUE,
                         d.f2q.features.shaderInt64 == VK_TRUE, d.f12q.shaderBufferInt64Atomics == VK_TRUE);
}

// One line of a batch file: spv, groups, expect, then --require-wave64 and NAME=VALUE settings, tab-separated.
static Job parse_job_line(const std::string& text, std::size_t line) {
    Job job;
    job.from_env = false;
    job.line = line;
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (;;) {
        const std::size_t tab = text.find('\t', start);
        fields.push_back(text.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
        if (tab == std::string::npos) break;
        start = tab + 1;
    }
    if (fields.size() < 3) {
        job.error = "expected cs.spv, groups and expect separated by tabs";
        return job;
    }
    job.spv = fields[0]; job.groups = fields[1]; job.expect = fields[2];
    for (std::size_t i = 3; i < fields.size(); i++) {
        const std::string& f = fields[i];
        if (f == "--require-wave64") {
            job.require_wave64 = true;
            continue;
        }
        const std::size_t eq = f.find('=');
        const std::string name = f.substr(0, eq);
        const bool known = eq != std::string::npos &&
            std::any_of(std::begin(kJobSettings), std::end(kJobSettings), [&](const char* s) { return name == s; });
        if (!known) {
            job.error = "unknown field \"" + f + "\" (--require-wave64 or NAME=VALUE with NAME one of VKC_OUT_WORDS, "
                        "VKC_DUMP, VKC_HEAP_INIT, VKC_HEAP_FILE, VKC_SUBGROUP_SIZE, VKC_SGPRS, VKC_BENCH, "
                        "LEONIDA_PC_TAIL_UBO)";
            return job;
        }
        job.settings.emplace_back(name, f.substr(eq + 1));
    }
    return job;
}

static bool read_jobs(const char* path, std::vector<Job>& jobs) {
    std::ifstream file;
    std::istream* in = &std::cin;
    if (std::strcmp(path, "-")) {
        file.open(path, std::ios::binary);
        if (!file) return false;
        in = &file;
    }
    std::string text;
    for (std::size_t line = 1; std::getline(*in, text); line++) {
        if (!text.empty() && text.back() == '\r') text.pop_back();
        if (text.empty() || text[0] == '#') continue;
        jobs.push_back(parse_job_line(text, line));
    }
    return true;
}

static void write_bytes(const std::string& s) {
    if (!s.empty()) std::fwrite(s.data(), 1, s.size(), stdout);
}

// Validation errors the layer reported while one job ran fail that job, as exit_status does for a single run;
// `charged` is how many were counted so far.
static int charge_validation(int code, unsigned& charged, Io& io) {
    const unsigned total = leonida::vk_test::OptInValidation::errors();
    if (total == charged) return code;
    io.err.print("%u validation errors\n", total - charged);
    charged = total;
    return code != 0 && code != 77 ? code : 1;
}

static int batch_main(const char* path, double timeout_s) {
    std::vector<Job> jobs;
    if (!read_jobs(path, jobs)) {
        std::fprintf(stderr, "--batch: could not read %s\n", path);
        return 2;
    }
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);   // the frame lengths count bytes as written
#endif
    const uint64_t timeout_ns = timeout_s > 0 ? uint64_t(timeout_s * 1e9) : UINT64_MAX;
    Device d;
    Io device_io;
    const int rc = open_batch_device(d, device_io);
    std::fputs(device_io.err.text.c_str(), stderr);
    if (rc >= 0) return rc;   // no device: no job ran, and the caller sees this exit code and message
    // A single run prints the "device:" line first on stderr: so does every job that gets that far.
    const std::string device_line = device_io.err.text;
    std::printf("@vkc-batch 1 jobs %zu\n", jobs.size());
    std::fflush(stdout);
    unsigned charged = 0;   // validation errors already charged to a job (those of the device open go to the first)
    for (std::size_t i = 0; i < jobs.size(); i++) {
        std::printf("@vkc-job-begin %zu\n", i);
        std::fflush(stdout);
        const Job& job = jobs[i];
        Io io;
        leonida::vk_test::OptInValidation::capture(&io.err.text);   // the layer's messages join this job's stderr
        int code;
        if (!job.error.empty()) {
            io.err.print("batch line %zu: %s\n", job.line, job.error.c_str());
            code = 2;
        } else {
            Program p;
            code = prepare(job, io, p);
            if (code < 0) {
                io.err.text = device_line;
                Plan pl;
                code = plan_job(d, job, p, io, pl);
                if (code < 0) code = run_job(d, job, p, pl, io, timeout_ns);
            }
        }
        leonida::vk_test::OptInValidation::capture(nullptr);
        code = charge_validation(code, charged, io);
        const char* status = io.timed_out ? "timeout" : io.lost ? "lost" : "done";
        std::printf("@vkc-job-end %zu %d %s %zu %zu\n", i, code, status, io.out.text.size(), io.err.text.size());
        write_bytes(io.out.text);
        write_bytes(io.err.text);
        std::fflush(stdout);
        if (io.timed_out || io.lost) {
            // The GPU may still run the job: leave everything to the process exit, and the caller restarts.
            std::fflush(stderr);
            std::_Exit(3);
        }
    }
    std::printf("@vkc-batch-end %zu\n", jobs.size());
    std::fflush(stdout);
    d.vkDeviceWaitIdle(d.dev);
    d.vkDestroyDevice(d.dev, nullptr);
    d.validation.detach(gipa, d.inst);
    if (const auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(gipa(d.inst, "vkDestroyInstance")))
        destroy(d.inst, nullptr);
    return 0;
}

static int single_main(const Job& job) {
    Io io;
    io.out.direct = stdout;
    io.err.direct = stderr;
    Program p;
    int rc = prepare(job, io, p);
    if (rc >= 0) return rc;
    Device d;
    rc = open_instance(d, io);
    if (rc >= 0) return rc;
    Plan pl;
    rc = plan_job(d, job, p, io, pl);
    if (rc >= 0) return rc;
    rc = create_device(d, io, d.f13q.subgroupSizeControl || job.require_wave64,
                       job.require_wave64 || pl.observed_full, pl.needs_float64, pl.needs_int64, pl.int64_atomics);
    if (rc >= 0) return rc;
    return run_job(d, job, p, pl, io, UINT64_MAX);
}

int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--batch-version")) {
        std::printf("vk_compute_test batch 1\n");
        return 0;
    }
    if (argc >= 3 && !std::strcmp(argv[1], "--batch")) {
        double timeout_s = 60;
        if (argc == 5 && !std::strcmp(argv[3], "--job-timeout")) {
            char* end;
            timeout_s = std::strtod(argv[4], &end);
            if (end == argv[4] || *end || !(timeout_s >= 0) || timeout_s > 1e6) {
                std::fprintf(stderr, "--job-timeout: expected seconds (0 = no limit), got \"%s\"\n", argv[4]);
                return 2;
            }
        } else if (argc != 3) {
            std::fprintf(stderr, "usage: vk_compute_test --batch FILE|- [--job-timeout SECONDS]\n");
            return 2;
        }
        return batch_main(argv[2], timeout_s);
    }
    const bool require_wave64 = argc == 5 && !std::strcmp(argv[4], "--require-wave64");
    if (argc != 4 && !require_wave64) {
        std::fprintf(stderr, "usage: vk_compute_test cs.spv groups expect [--require-wave64]\n"
                             "       vk_compute_test --batch FILE|- [--job-timeout SECONDS]\n");
        return 2;
    }
    Job job;
    job.spv = argv[1]; job.groups = argv[2]; job.expect = argv[3];
    job.require_wave64 = require_wave64;
    // GPU-assisted validation reports what the shader did when the queue finishes, after the words were judged.
    return leonida::vk_test::OptInValidation::exit_status(single_main(job));
}

// What the runtime's VkPipelineCache keeps between sessions: the files it starts from and saves to, the check that a
// file belongs to this GPU and driver, and the pipeline creation feedback (VK_EXT_pipeline_creation_feedback, core in
// Vulkan 1.3) that says whether the cache paid off.
//
// The build's pipelines stage writes pipeline_cache.bin beside gta6rt.exe: a seed, never written here. A session saves
// what it learned to a second file in the same folder, or to LEONIDA_PIPELINE_CACHE, and the next one starts from
// that, so it has every pipeline the game has already made.
#pragma once

#include "cached_pso.h"

#include <vulkan/vulkan_core.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace leonida::pipeline_cache_store {

// A session cache file bigger than this is neither loaded nor saved (LEONIDA_PIPELINE_CACHE_MAX_MB overrides): saving
// reads the whole cache into memory next to the driver's copy and writes all of it, and the full-corpus seed is 3.8 GB
// (docs/design.md); a session that made a few pipelines should not write that at exit.
inline constexpr std::uint64_t kDefaultMaxSessionBytes = std::uint64_t{1} << 30;

// LEONIDA_PIPELINE_CACHE=off: no session file.
inline bool is_off(const std::wstring& value) {
    return value.size() == 3 && (value[0] | 0x20) == L'o' && (value[1] | 0x20) == L'f' && (value[2] | 0x20) == L'f';
}

inline constexpr const wchar_t* kSessionFileName = L"pipeline_cache.user.bin";

// A cache blob's VkPipelineCacheHeaderVersionOne names the GPU and the driver build it came from; a blob from another
// one would be ignored by the driver at best, so it is not even handed over.
inline bool header_matches(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32);
    if (!data || size < sizeof(VkPipelineCacheHeaderVersionOne)) return false;
    VkPipelineCacheHeaderVersionOne header{};
    // The header is little-endian, like the host. The driver interprets the opaque payload.
    std::memcpy(&header, data, sizeof(header));
    return header.headerSize == sizeof(header) && header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
           header.vendorID == identity.vendor && header.deviceID == identity.device &&
           std::memcmp(header.pipelineCacheUUID, identity.cache_uuid, VK_UUID_SIZE) == 0;
}

// True when both names reach the same file, or would once it exists.
inline bool same_file(const std::filesystem::path& a, const std::filesystem::path& b) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::equivalent(a, b, ec) && !ec) return true;
    const fs::path left = fs::weakly_canonical(a, ec);
    if (ec) return false;
    const fs::path right = fs::weakly_canonical(b, ec);
    return !ec && left == right;
}

// The session file starts a session only when it is at least as new as the seed: a rebuilt seed holds pipelines the
// older session file lacks, and the next save then replaces the session file with a superset.
inline bool session_file_is_current(const std::filesystem::path& session, const std::filesystem::path& seed) {
    std::error_code ec;
    const auto session_time = std::filesystem::last_write_time(session, ec);
    if (ec) return false;
    const auto seed_time = std::filesystem::last_write_time(seed, ec);
    return ec || session_time >= seed_time;  // no seed: nothing to be older than
}

// What the pipelines of one device took to create, from VK_EXT_pipeline_creation_feedback.
struct Stats {
    std::atomic<std::uint32_t> pipelines{0};   // created, with valid feedback
    std::atomic<std::uint32_t> cache_hits{0};  // of those, found in the application's VkPipelineCache
    std::atomic<std::uint64_t> nanoseconds{0}; // their total creation time
};

// Chained into one vkCreate*Pipelines call; a no-op when the device has no feedback (before Vulkan 1.3 and without the
// extension enabled, the struct must not be chained).
class CreationFeedback {
public:
    CreationFeedback(bool enabled, std::uint32_t stage_count)
        : enabled_(enabled), stages_(enabled ? stage_count : 0) {
        info_.pPipelineCreationFeedback = &pipeline_;
        info_.pipelineStageCreationFeedbackCount = std::uint32_t(stages_.size());
        info_.pPipelineStageCreationFeedbacks = stages_.data();
    }

    // The pNext for the create info: this struct, then `next`.
    const void* link(const void* next) {
        if (!enabled_) return next;
        info_.pNext = next;
        return &info_;
    }

    void record(Stats& stats) const {
        if (!enabled_ || !(pipeline_.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT)) return;
        stats.pipelines.fetch_add(1, std::memory_order_relaxed);
        stats.nanoseconds.fetch_add(pipeline_.duration, std::memory_order_relaxed);
        if (pipeline_.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT)
            stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
    }

private:
    bool enabled_;
    VkPipelineCreationFeedback pipeline_{};
    std::vector<VkPipelineCreationFeedback> stages_;
    VkPipelineCreationFeedbackCreateInfo info_{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO};
};

}  // namespace leonida::pipeline_cache_store

// Xbox GPU page mapping on Vulkan: physical page pools, a sparse arena with fixed GPU addresses, and the semantics of
// RegisterPagePoolX / CopyPageMappings(Batch)X / ClearPageMappingsX (docs/xbox_paging.md). Header-only and
// independent of the D3D12 layer, so the device proxy can wire it once the Xbox-only slots are measured.
//
// The model, as on the console:
//   - A physical page is 64 KiB. Its GPU half is a 64 KiB range of a device-local VkDeviceMemory chunk (on NVIDIA
//     only device-local memory can back a sparse page). PagePool hands pages out of 64 MiB chunks, so Windows'
//     video-memory manager pages at a useful granularity under pressure.
//   - PagedRegion is one sparse, device-addressable VkBuffer with a large virtual size. Console VA = offset into it,
//     and its device address never changes: only the backing of a page does. Unbound pages read 0 under strict
//     residency (residencyNonResidentStrict).
//   - PageMapper turns the Xbox calls into binds: every call becomes ONE vkQueueBindSparse (Submit), with runs that
//     are contiguous in the arena and in one chunk coalesced into a single VkSparseMemoryBind. Ordering against
//     command lists is the queue's (VulkanQueue::bind_sparse_ordered).
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace leonida::xbox {

constexpr VkDeviceSize kPageSize = 64 * 1024;
constexpr VkDeviceSize kChunkSize = 64ull << 20;
constexpr std::uint32_t kPagesPerChunk = std::uint32_t(kChunkSize / kPageSize);

// A physical page: (chunk << 10 | page) + 1. 0 is "no page" (an unbound arena page).
using PageId = std::uint32_t;
constexpr PageId kNoPage = 0;

inline PageId make_page(std::uint32_t chunk, std::uint32_t index) { return (chunk * kPagesPerChunk + index) + 1; }
inline std::uint32_t page_chunk(PageId id) { return (id - 1) / kPagesPerChunk; }
inline std::uint32_t page_index(PageId id) { return (id - 1) % kPagesPerChunk; }

// Picks the memory type for physical pages from the arena's memoryTypeBits: device-local first (the only kind NVIDIA
// allows), else any allowed type. UINT32_MAX if none.
inline std::uint32_t pick_page_memory_type(const VkPhysicalDeviceMemoryProperties& properties, std::uint32_t bits) {
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        if (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) return i;
        if (fallback == UINT32_MAX) fallback = i;
    }
    return fallback;
}

class PagePool {
public:
    // memory_priority: VK_EXT_memory_priority is enabled, so chunks carry set_priority_hint()'s value.
    PagePool(VkDevice device, PFN_vkGetDeviceProcAddr get, std::uint32_t memory_type, bool memory_priority = false)
        : device_(device), memory_type_(memory_type), memory_priority_(memory_priority) {
        allocate_ = reinterpret_cast<PFN_vkAllocateMemory>(get(device, "vkAllocateMemory"));
        free_ = reinterpret_cast<PFN_vkFreeMemory>(get(device, "vkFreeMemory"));
    }
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;
    ~PagePool() {
        for (auto& chunk : chunks_)
            if (chunk.memory) free_(device_, chunk.memory, nullptr);
    }

    bool ok() const { return allocate_ && free_ && memory_type_ != UINT32_MAX; }
    std::uint32_t memory_type() const { return memory_type_; }

    // Priority for chunks allocated from now on (0 lowest .. 1 highest): render targets high, streamed pages low.
    // Driven by the device's budget policy; ignored without VK_EXT_memory_priority.
    void set_priority_hint(float priority) {
        std::lock_guard lock(mutex_);
        priority_ = std::isnan(priority) ? 0.5f : std::clamp(priority, 0.0f, 1.0f);
    }

    // `count` pages, as one contiguous run inside one chunk where possible (so their binds coalesce: about 2 us a page
    // contiguous against 9 us scattered on a GTX 1660 SUPER), else gathered from the free runs in chunk order, with
    // new chunks as needed. A request of at least half a chunk that no free run holds gets a fresh chunk instead of
    // being scattered. All or nothing.
    bool allocate(std::uint32_t count, std::vector<PageId>& out) {
        if (!count) return true;
        if (!ok()) return false;
        std::lock_guard lock(mutex_);
        const std::size_t base = out.size();
        if (count <= kPagesPerChunk) {
            for (std::uint32_t c = 0; c < chunks_.size(); ++c) {
                if (!chunks_[c].memory || chunks_[c].free_count < count) continue;
                const std::uint32_t first = find_run(chunks_[c], count);
                if (first == UINT32_MAX) continue;
                take(c, first, count, out);
                return true;
            }
            if (count >= kPagesPerChunk / 2) {
                const std::uint32_t c = new_chunk();
                if (c != UINT32_MAX) {
                    take(c, 0, count, out);
                    return true;
                }
            }
        }
        std::uint32_t left = count;
        for (std::uint32_t c = 0; c < chunks_.size() && left; ++c) {
            if (!chunks_[c].memory) continue;
            for (std::uint32_t i = 0; i < kPagesPerChunk && left; ++i) {
                if (!available(chunks_[c], i)) continue;
                take(c, i, 1, out);
                --left;
            }
        }
        while (left) {
            const std::uint32_t c = new_chunk();
            if (c == UINT32_MAX) {
                release_locked(std::span<const PageId>(out).subspan(base));
                out.resize(base);
                return false;
            }
            const std::uint32_t n = std::min(left, kPagesPerChunk);
            take(c, 0, n, out);
            left -= n;
        }
        return true;
    }

    // Pages go back to the free list now, but a page an arena still maps is not handed out again until it is
    // unmapped, so a stale mapping never shows a new owner's data. Chunk memory is freed only by trim().
    void release(std::span<const PageId> pages) {
        std::lock_guard lock(mutex_);
        release_locked(pages);
    }

    // The arena maps (or unmaps) a page with a bind that completes at `ticket` on the bind queue's timeline. A pool's
    // binds all go through one queue: tickets of different timelines do not compare.
    void note_mapped(PageId id, int delta, std::uint64_t ticket) {
        std::lock_guard lock(mutex_);
        if (!known(id)) return;
        auto& chunk = chunks_[page_chunk(id)];
        chunk.mapped += delta;
        chunk.mapped_pages[page_index(id)] += delta;
        chunk.last_bind = std::max(chunk.last_bind, ticket);
    }

    // Frees chunks with no allocated and no mapped pages whose last bind has completed. Returns bytes freed.
    VkDeviceSize trim(std::uint64_t completed_ticket) {
        std::lock_guard lock(mutex_);
        VkDeviceSize freed = 0;
        for (auto& chunk : chunks_) {
            if (!chunk.memory || chunk.free_count != kPagesPerChunk || chunk.mapped ||
                chunk.last_bind > completed_ticket)
                continue;
            free_(device_, chunk.memory, nullptr);
            chunk = Chunk{};
            freed += kChunkSize;
            reserved_ -= kChunkSize;
        }
        return freed;
    }

    // Whether `id` is a page allocate() handed out and release() has not taken back.
    bool allocated(PageId id) const {
        std::lock_guard lock(mutex_);
        return known(id) && chunks_[page_chunk(id)].used[page_index(id)];
    }

    VkDeviceMemory memory(PageId id) const {
        std::lock_guard lock(mutex_);
        return known(id) ? chunks_[page_chunk(id)].memory : VK_NULL_HANDLE;
    }
    static VkDeviceSize offset(PageId id) { return VkDeviceSize(page_index(id)) * kPageSize; }
    VkDeviceSize bytes_reserved() const {
        std::lock_guard lock(mutex_);
        return reserved_;
    }
    std::uint64_t pages_in_use() const {
        std::lock_guard lock(mutex_);
        return in_use_;
    }
    std::uint32_t chunk_count() const {
        std::lock_guard lock(mutex_);
        std::uint32_t n = 0;
        for (const auto& chunk : chunks_) n += chunk.memory ? 1 : 0;
        return n;
    }

private:
    struct Chunk {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::vector<bool> used;
        std::vector<std::int32_t> mapped_pages;  // arena pages bound to each page
        std::uint32_t free_count = 0;
        std::int64_t mapped = 0;      // arena pages currently bound to pages of this chunk
        std::uint64_t last_bind = 0;  // ticket of the last bind that referenced this chunk
    };

    // Free and not mapped by any arena.
    static bool available(const Chunk& chunk, std::uint32_t i) { return !chunk.used[i] && !chunk.mapped_pages[i]; }

    static std::uint32_t find_run(const Chunk& chunk, std::uint32_t count) {
        std::uint32_t run = 0;
        for (std::uint32_t i = 0; i < kPagesPerChunk; ++i) {
            run = available(chunk, i) ? run + 1 : 0;
            if (run == count) return i + 1 - count;
        }
        return UINT32_MAX;
    }

    void take(std::uint32_t c, std::uint32_t first, std::uint32_t count, std::vector<PageId>& out) {
        auto& chunk = chunks_[c];
        for (std::uint32_t i = first; i < first + count; ++i) {
            chunk.used[i] = true;
            out.push_back(make_page(c, i));
        }
        chunk.free_count -= count;
        in_use_ += count;
    }

    // A page of a chunk that is allocated now (its pages may be free).
    bool known(PageId id) const {
        return id != kNoPage && page_chunk(id) < chunks_.size() && chunks_[page_chunk(id)].memory;
    }

    void release_locked(std::span<const PageId> pages) {
        for (PageId id : pages) {
            if (!known(id)) continue;
            auto& chunk = chunks_[page_chunk(id)];
            const std::uint32_t i = page_index(id);
            if (!chunk.used[i]) continue;
            chunk.used[i] = false;
            ++chunk.free_count;
            --in_use_;
        }
    }

    std::uint32_t new_chunk() {
        // The arena has SHADER_DEVICE_ADDRESS usage, so memory bound into it must be device-addressable.
        VkMemoryPriorityAllocateInfoEXT priority{VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT};
        priority.priority = priority_;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                        memory_priority_ ? &priority : nullptr};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        info.allocationSize = kChunkSize;
        info.memoryTypeIndex = memory_type_;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (allocate_(device_, &info, nullptr, &memory) != VK_SUCCESS) return UINT32_MAX;
        std::uint32_t c = 0;
        while (c < chunks_.size() && chunks_[c].memory) ++c;
        if (c == chunks_.size()) chunks_.emplace_back();
        chunks_[c].memory = memory;
        chunks_[c].used.assign(kPagesPerChunk, false);
        chunks_[c].mapped_pages.assign(kPagesPerChunk, 0);
        chunks_[c].free_count = kPagesPerChunk;
        chunks_[c].mapped = 0;
        chunks_[c].last_bind = 0;
        reserved_ += kChunkSize;
        return c;
    }

    VkDevice device_;
    PFN_vkAllocateMemory allocate_ = nullptr;
    PFN_vkFreeMemory free_ = nullptr;
    std::uint32_t memory_type_;
    bool memory_priority_;
    float priority_ = 0.5f;
    mutable std::mutex mutex_;  // allocate() can grow chunks_ while a mapper thread reads it
    std::vector<Chunk> chunks_;
    VkDeviceSize reserved_ = 0;
    std::uint64_t in_use_ = 0;
};

// The sparse arena: console VA = offset, device address fixed for its lifetime. Its pages come from one PagePool,
// which must outlive it, and one PageMapper changes it. Destroy it once the queue has finished with it.
class PagedRegion {
public:
    PagedRegion() = default;
    PagedRegion(const PagedRegion&) = delete;
    PagedRegion& operator=(const PagedRegion&) = delete;
    ~PagedRegion() {
        // Mappings still in place end with the buffer. The chunks keep their last bind's ticket, so trim() still
        // waits for it.
        if (pool_)
            for (PageId id : mapped_)
                if (id) pool_->note_mapped(id, -1, 0);
        if (buffer_) destroy_(device_, buffer_, nullptr);
    }

    // virtual_bytes is rounded up to whole pages. extra_usage adds to transfer, storage, uniform, vertex, index,
    // indirect and device-address usage.
    bool create(VkDevice device, PFN_vkGetDeviceProcAddr get, VkDeviceSize virtual_bytes,
                VkBufferUsageFlags extra_usage = 0) {
        if (buffer_ || !virtual_bytes || virtual_bytes > ~VkDeviceSize(0) - kPageSize) return false;
        device_ = device;
        destroy_ = reinterpret_cast<PFN_vkDestroyBuffer>(get(device, "vkDestroyBuffer"));
        const auto create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(get(device, "vkCreateBuffer"));
        const auto requirements_of =
            reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(get(device, "vkGetBufferMemoryRequirements"));
        auto address_of = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(get(device, "vkGetBufferDeviceAddress"));
        if (!address_of)
            address_of = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(get(device, "vkGetBufferDeviceAddressKHR"));
        if (!destroy_ || !create_buffer || !requirements_of || !address_of) return false;
        pages_ = (virtual_bytes + kPageSize - 1) / kPageSize;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = pages_ * kPageSize;
        info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                     VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | extra_usage;
        if (create_buffer(device, &info, nullptr, &buffer_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        requirements_of(device, buffer_, &requirements);
        memory_type_bits_ = requirements.memoryTypeBits;
        if (requirements.alignment > kPageSize || kPageSize % requirements.alignment) return false;  // page != block
        VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address.buffer = buffer_;
        address_ = address_of(device, &address);
        mapped_.assign(std::size_t(pages_), kNoPage);
        return address_ != 0;
    }

    VkBuffer buffer() const { return buffer_; }
    VkDeviceAddress address() const { return address_; }
    std::uint32_t memory_type_bits() const { return memory_type_bits_; }
    std::uint64_t page_count() const { return pages_; }
    VkDeviceSize size() const { return pages_ * kPageSize; }
    PageId mapped(std::uint64_t page) const { return page < pages_ ? mapped_[std::size_t(page)] : kNoPage; }

    // (arena page, physical page or kNoPage) changes, in any order; a later change to the same page wins.
    using Change = std::pair<std::uint64_t, PageId>;

    // Sorts and deduplicates `changes`, drops the ones that leave a page as it is, and returns the binds, with runs
    // contiguous in the arena and in one chunk (or unbound runs) merged into one VkSparseMemoryBind.
    std::vector<VkSparseMemoryBind> binds_for(std::vector<Change>& changes, const PagePool& pool) const {
        std::stable_sort(changes.begin(), changes.end(),
                         [](const Change& a, const Change& b) { return a.first < b.first; });
        std::vector<Change> unique;
        unique.reserve(changes.size());
        for (const auto& change : changes) {
            if (!unique.empty() && unique.back().first == change.first) unique.back() = change;
            else unique.push_back(change);
        }
        changes.clear();
        for (const auto& change : unique)
            if (change.first < pages_ && mapped_[std::size_t(change.first)] != change.second) changes.push_back(change);
        std::vector<VkSparseMemoryBind> binds;
        for (const auto& [page, id] : changes) {
            const VkDeviceMemory memory = id ? pool.memory(id) : VK_NULL_HANDLE;
            const VkDeviceSize offset = id ? pool.offset(id) : 0;
            const VkDeviceSize resource_offset = page * kPageSize;
            if (!binds.empty()) {
                auto& last = binds.back();
                if (last.resourceOffset + last.size == resource_offset && last.memory == memory &&
                    (!memory || last.memoryOffset + last.size == offset)) {
                    last.size += kPageSize;
                    continue;
                }
            }
            binds.push_back({resource_offset, kPageSize, memory, offset, 0});
        }
        return binds;
    }

    // Records `changes` (as left by binds_for) once their bind was submitted at `ticket`.
    void apply(const std::vector<Change>& changes, PagePool& pool, std::uint64_t ticket) {
        pool_ = &pool;
        for (const auto& [page, id] : changes) {
            PageId& slot = mapped_[std::size_t(page)];
            if (slot) pool.note_mapped(slot, -1, ticket);
            if (id) pool.note_mapped(id, +1, ticket);
            slot = id;
        }
    }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    PFN_vkDestroyBuffer destroy_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    std::uint32_t memory_type_bits_ = 0;
    std::uint64_t pages_ = 0;
    std::vector<PageId> mapped_;
    PagePool* pool_ = nullptr;
};

// D3D12XBOX_PAGE_MAPPING_RANGE_TYPE: only INCREMENTING_PAGE_INDICES is documented publicly (Xbox-GDK-Samples
// AdvancedESRAM README); other types are refused until a call site shows them.
enum class PageRangeType : std::uint32_t { incrementing = 0 };

struct PageMappingRange {
    PageRangeType type = PageRangeType::incrementing;
    std::uint32_t page_count = 0;
    std::uint32_t start_page_in_pool = 0;
};

struct PageMappingBatch {
    std::uint64_t destination_va = 0;  // 64 KiB aligned, inside the arena
    const PageMappingRange* ranges = nullptr;
    std::uint32_t range_count = 0;
};

// Submits the binds of one call on the right queue and returns its ticket (VulkanQueue::bind_sparse_ordered).
using SubmitBinds = std::function<std::uint64_t(VkBuffer, std::span<const VkSparseMemoryBind>)>;

class PageMapper {
public:
    PageMapper(PagedRegion& region, PagePool& pool) : region_(region), pool_(pool) {}

    // RegisterPagePoolX: a pool is an ordered list of physical pages; handles start at 1.
    std::uint32_t register_page_pool(std::vector<PageId> pages) {
        std::lock_guard lock(mutex_);
        pools_.emplace(next_handle_, std::move(pages));
        return next_handle_++;
    }

    bool unregister_page_pool(std::uint32_t handle) {
        std::lock_guard lock(mutex_);
        return pools_.erase(handle) == 1;
    }

    // CopyPageMappingsBatchX: every batch maps destination_va + i pages to pool pages start + i. Validated in full
    // before anything is bound; one bind for the whole call.
    bool copy_page_mappings_batch(std::uint32_t pool_handle, std::span<const PageMappingBatch> batches,
                                  const SubmitBinds& submit) {
        std::lock_guard lock(mutex_);
        const auto pool = pools_.find(pool_handle);
        if (pool == pools_.end()) return false;
        std::vector<PagedRegion::Change> changes;
        for (const auto& batch : batches) {
            if (batch.destination_va % kPageSize || (batch.range_count && !batch.ranges)) return false;
            std::uint64_t page = batch.destination_va / kPageSize;
            for (std::uint32_t r = 0; r < batch.range_count; ++r) {
                const auto& range = batch.ranges[r];
                if (range.type != PageRangeType::incrementing) return false;
                if (std::uint64_t(range.start_page_in_pool) + range.page_count > pool->second.size()) return false;
                if (page + range.page_count > region_.page_count()) return false;
                for (std::uint32_t i = 0; i < range.page_count; ++i) {
                    const PageId id = pool->second[range.start_page_in_pool + i];
                    if (!pool_.allocated(id)) return false;  // released, or never the pool's: nothing to bind
                    changes.emplace_back(page + i, id);
                }
                page += range.page_count;
            }
        }
        return commit(changes, submit);
    }

    // ClearPageMappingsX: unbind [va, va + pages); reads return 0 afterwards.
    bool clear_page_mappings(std::uint64_t va, std::uint64_t pages, const SubmitBinds& submit) {
        std::lock_guard lock(mutex_);
        const std::uint64_t first = va / kPageSize;
        if (va % kPageSize || first > region_.page_count() || pages > region_.page_count() - first) return false;
        std::vector<PagedRegion::Change> changes;
        changes.reserve(std::size_t(pages));
        for (std::uint64_t i = 0; i < pages; ++i) changes.emplace_back(first + i, kNoPage);
        return commit(changes, submit);
    }

    struct Stats {
        std::uint64_t calls = 0, binds = 0, pages = 0, submits = 0;
    };
    Stats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

private:
    bool commit(std::vector<PagedRegion::Change>& changes, const SubmitBinds& submit) {
        if (!submit) return false;
        ++stats_.calls;
        auto binds = region_.binds_for(changes, pool_);
        if (binds.empty()) return true;  // nothing changes: no bind, as on the console's no-op remap
        const std::uint64_t ticket = submit(region_.buffer(), binds);
        if (!ticket) return false;
        region_.apply(changes, pool_, ticket);
        ++stats_.submits;
        stats_.binds += binds.size();
        stats_.pages += changes.size();
        return true;
    }

    PagedRegion& region_;
    PagePool& pool_;
    mutable std::mutex mutex_;
    std::map<std::uint32_t, std::vector<PageId>> pools_;
    std::uint32_t next_handle_ = 1;
    Stats stats_;
};

}  // namespace leonida::xbox

// The CPU half of Xbox physical pages, and its coherence with the GPU half (xbox_paging.h's PagePool); see
// docs/xbox_paging.md.
//
// On the console a physical page is one piece of memory that the CPU and GPU both address. On NVIDIA a sparse page
// can only be device-local memory, so here a physical page has two halves:
//   - CPU half: a 64 KiB slot of one pagefile-backed section (SEC_RESERVE, committed per page on allocation). Views
//     of it are placed at the title's VAs over placeholder reservations (VirtualAlloc2 + MapViewOfFile3 with
//     MEM_REPLACE_PLACEHOLDER), so one physical page can be mapped at several VAs, as the console allows.
//   - GPU half: a PagePool page, allocated when the page is first uploaded or bound.
// Coherence:
//   - CPU -> GPU: clean pages are read-only in every view. The first write faults; a vectored exception handler
//     marks the page dirty, makes that view writable and resumes the write. take_dirty() lists dirty pages and
//     write-protects them again, and PhysicalGpu::record_upload copies them into their GPU halves, recorded ahead of
//     the title's commands on the same queue.
//   - GPU -> CPU: only on request (readback, query results): record_readback then finish_readback.
// The xmem.dll exports that drive this (XMemAllocatePhysicalPages, XMemMapPhysicalPages[Scatter]) are wired when a
// call site shows their arguments, as XMemAlloc's were measured; this is the mechanism they map onto.
#pragma once

#include <windows.h>

#include <vulkan/vulkan.h>

#include "xbox_paging.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <vector>

namespace leonida::xbox {

// A physical page number as the title sees it: the CPU half's slot index.
using PhysicalPage = std::uint32_t;

class PhysicalPageStore {
public:
    // capacity: the most physical pages that can exist at once (address space only until pages are allocated).
    explicit PhysicalPageStore(std::uint32_t capacity) : capacity_(capacity), dirty_(capacity), used_(capacity, false) {
        const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
        if (kernelbase) {
            virtual_alloc2_ = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelbase, "VirtualAlloc2"));
            map_view3_ = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelbase, "MapViewOfFile3"));
            unmap_view2_ = reinterpret_cast<UnmapViewOfFile2Fn>(GetProcAddress(kernelbase, "UnmapViewOfFile2"));
        }
        const std::uint64_t bytes = std::uint64_t(capacity) * kPageSize;
        section_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, DWORD(bytes >> 32),
                                      DWORD(bytes), nullptr);
        if (section_) store_ = static_cast<std::uint8_t*>(MapViewOfFile(section_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        if (ok()) register_store(this);
    }
    PhysicalPageStore(const PhysicalPageStore&) = delete;
    PhysicalPageStore& operator=(const PhysicalPageStore&) = delete;
    ~PhysicalPageStore() {
        if (ok()) unregister_store(this);
        for (auto& [base, region] : regions_) {
            for (std::uint32_t i = 0; i < region.pages.size(); ++i)
                if (region.pages[i] != kUnmapped)
                    unmap_view2_(GetCurrentProcess(), region.at(i), MEM_PRESERVE_PLACEHOLDER);
            // Split slots are placeholders of their own: merge them back into one, then release it.
            if (!region.split.empty())
                VirtualFree(reinterpret_cast<void*>(base), region.pages.size() * kPageSize,
                            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
            VirtualFree(reinterpret_cast<void*>(base), 0, MEM_RELEASE);
        }
        if (store_) UnmapViewOfFile(store_);
        if (section_) CloseHandle(section_);
    }

    bool ok() const { return store_ && virtual_alloc2_ && map_view3_ && unmap_view2_; }

    // Commits `count` physical pages, zeroed. All or nothing.
    bool allocate(std::uint32_t count, std::vector<PhysicalPage>& out) {
        if (!ok()) return false;
        std::unique_lock lock(mutex_);
        const std::size_t base = out.size();
        for (std::uint32_t p = 0; p < capacity_ && out.size() - base < count; ++p) {
            if (used_[p]) continue;
            if (!VirtualAlloc(store_ + std::uint64_t(p) * kPageSize, kPageSize, MEM_COMMIT, PAGE_READWRITE)) break;
            used_[p] = true;
            out.push_back(p);
        }
        if (out.size() - base == count) return true;
        for (std::size_t i = base; i < out.size(); ++i) used_[out[i]] = false;
        out.resize(base);
        return false;
    }

    // Placeholder reservation for `pages` 64 KiB pages; the title's mappings go here. Returns the base, or nullptr.
    void* reserve(std::uint32_t pages, void* at = nullptr) {
        if (!ok() || !pages) return nullptr;
        void* base = virtual_alloc2_(nullptr, at, std::uint64_t(pages) * kPageSize,
                                     MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
        if (!base) return nullptr;
        std::unique_lock lock(mutex_);
        Region region;
        region.base = reinterpret_cast<std::uintptr_t>(base);
        region.pages.assign(pages, kUnmapped);
        regions_.emplace(region.base, std::move(region));
        return base;
    }

    // XMemMapPhysicalPagesScatter's mechanism: page i of [va, va + n pages) shows physical page `pages[i]`
    // (kUnmapped unmaps). The mapping starts clean (read-only) unless the page is already dirty. `va` must be page
    // aligned, as for MapViewOfFile3. The call is checked in full first; only an OS failure part way leaves it partly
    // applied.
    static constexpr PhysicalPage kUnmapped = UINT32_MAX;
    bool map(void* va, std::span<const PhysicalPage> pages) {
        if (reinterpret_cast<std::uintptr_t>(va) % kPageSize) return false;
        std::unique_lock lock(mutex_);
        Region* region = region_of(reinterpret_cast<std::uintptr_t>(va));
        if (!region) return false;
        const std::size_t first = (reinterpret_cast<std::uintptr_t>(va) - region->base) / kPageSize;
        if (first + pages.size() > region->pages.size()) return false;
        for (PhysicalPage p : pages)
            if (p != kUnmapped && (p >= capacity_ || !used_[p])) return false;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            const std::size_t slot = first + i;
            void* at = region->at(slot);
            if (region->pages[slot] == pages[i]) continue;
            if (region->pages[slot] != kUnmapped) {
                unmap_view2_(GetCurrentProcess(), at, MEM_PRESERVE_PLACEHOLDER);
                forget_view(region->pages[slot], at);
                region->pages[slot] = kUnmapped;
            }
            if (pages[i] == kUnmapped) continue;
            // Split the slot out of the placeholder, then put the view in its place.
            if (!split(region, slot)) return false;
            // Mapped writable (so the view's maximum protection allows the fault handler's upgrade), then dropped to
            // read-only while the page is clean.
            if (!map_view3_(section_, GetCurrentProcess(), at, std::uint64_t(pages[i]) * kPageSize, kPageSize,
                            MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0))
                return false;
            if (!dirty_[pages[i]].load(std::memory_order_relaxed)) {
                DWORD old = 0;
                VirtualProtect(at, kPageSize, PAGE_READONLY, &old);
            }
            region->pages[slot] = pages[i];
            views_[pages[i]].push_back(at);
        }
        return true;
    }

    // Pages written by the CPU since they were last taken, write-protected again in every view, in page order.
    std::vector<PhysicalPage> take_dirty() {
        std::unique_lock lock(mutex_);
        std::vector<PhysicalPage> out;
        for (PhysicalPage p = 0; p < capacity_; ++p) {
            if (!dirty_[p].load(std::memory_order_relaxed) || !dirty_[p].exchange(0, std::memory_order_acq_rel))
                continue;
            out.push_back(p);
            protect_views(p, PAGE_READONLY);
        }
        return out;
    }

    // Marks an allocated page dirty without a write: a new GPU half, an upload that failed, or a test.
    void mark_dirty(PhysicalPage p) {
        if (allocated(p)) dirty_[p].store(1, std::memory_order_release);
    }

    // Whether the CPU wrote `p` since it was last taken.
    bool dirty(PhysicalPage p) const { return p < capacity_ && dirty_[p].load(std::memory_order_acquire); }

    // Whether `p` was allocated (its CPU half is committed).
    bool allocated(PhysicalPage p) const {
        std::shared_lock lock(mutex_);
        return p < capacity_ && used_[p];
    }

    // The CPU half through the store's own view (always writable; never tracked).
    std::uint8_t* data(PhysicalPage p) const { return p < capacity_ ? store_ + std::uint64_t(p) * kPageSize : nullptr; }

    std::uint64_t write_faults() const { return faults_.load(std::memory_order_relaxed); }

private:
    using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG,
                                            MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapViewOfFile2Fn = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

    struct Region {
        std::uintptr_t base = 0;
        std::vector<PhysicalPage> pages;
        std::vector<bool> split;  // the slot is its own placeholder
        void* at(std::size_t slot) const { return reinterpret_cast<void*>(base + slot * kPageSize); }
    };

    Region* region_of(std::uintptr_t address) {
        auto it = regions_.upper_bound(address);
        if (it == regions_.begin()) return nullptr;
        --it;
        return address < it->first + it->second.pages.size() * kPageSize ? &it->second : nullptr;
    }

    bool split(Region* region, std::size_t slot) {
        if (region->split.empty()) region->split.assign(region->pages.size(), false);
        if (region->split[slot]) return true;
        // Once its neighbours are split off, a slot is a placeholder of its own, and splitting it again fails.
        void* const at = region->at(slot);
        MEMORY_BASIC_INFORMATION info{};
        const bool alone = VirtualQuery(at, &info, sizeof info) && info.AllocationBase == at &&
                           info.RegionSize == kPageSize;
        if (!alone && !VirtualFree(at, kPageSize, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) return false;
        region->split[slot] = true;
        return true;
    }

    void forget_view(PhysicalPage p, void* at) {
        auto& list = views_[p];
        list.erase(std::remove(list.begin(), list.end(), at), list.end());
    }

    void protect_views(PhysicalPage p, DWORD protect) {
        const auto it = views_.find(p);
        if (it == views_.end()) return;
        for (void* view : it->second) {
            DWORD old = 0;
            VirtualProtect(view, kPageSize, protect, &old);
        }
    }

    // The vectored handler: a write to a read-only view of a tracked page marks it dirty and opens that view.
    bool on_write_fault(std::uintptr_t address) {
        std::shared_lock lock(mutex_);
        Region* region = region_of(address);
        if (!region) return false;
        const std::size_t slot = (address - region->base) / kPageSize;
        const PhysicalPage p = region->pages[slot];
        if (p == kUnmapped) return false;
        dirty_[p].store(1, std::memory_order_release);
        faults_.fetch_add(1, std::memory_order_relaxed);
        DWORD old = 0;
        return VirtualProtect(region->at(slot), kPageSize, PAGE_READWRITE, &old) != 0;
    }

    static LONG CALLBACK vectored(EXCEPTION_POINTERS* info) {
        const EXCEPTION_RECORD* record = info->ExceptionRecord;
        if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 ||
            record->ExceptionInformation[0] != 1)
            return EXCEPTION_CONTINUE_SEARCH;
        const std::uintptr_t address = record->ExceptionInformation[1];
        std::shared_lock lock(registry_mutex());
        for (PhysicalPageStore* store : registry())
            if (store->on_write_fault(address)) return EXCEPTION_CONTINUE_EXECUTION;
        return EXCEPTION_CONTINUE_SEARCH;
    }

    static std::shared_mutex& registry_mutex() {
        static std::shared_mutex m;
        return m;
    }
    static std::vector<PhysicalPageStore*>& registry() {
        static std::vector<PhysicalPageStore*> stores;
        return stores;
    }
    static void register_store(PhysicalPageStore* store) {
        std::unique_lock lock(registry_mutex());
        static PVOID handler = AddVectoredExceptionHandler(1, &vectored);
        (void)handler;
        registry().push_back(store);
    }
    static void unregister_store(PhysicalPageStore* store) {
        std::unique_lock lock(registry_mutex());
        auto& list = registry();
        list.erase(std::remove(list.begin(), list.end(), store), list.end());
    }

    std::uint32_t capacity_;
    HANDLE section_ = nullptr;
    std::uint8_t* store_ = nullptr;
    VirtualAlloc2Fn virtual_alloc2_ = nullptr;
    MapViewOfFile3Fn map_view3_ = nullptr;
    UnmapViewOfFile2Fn unmap_view2_ = nullptr;
    mutable std::shared_mutex mutex_;
    std::vector<std::atomic<std::uint8_t>> dirty_;
    std::vector<bool> used_;
    std::map<std::uintptr_t, Region> regions_;
    std::map<PhysicalPage, std::vector<void*>> views_;
    std::atomic<std::uint64_t> faults_{0};
};

// The GPU half of physical pages: a PagePool page per physical page, created on first use, plus the copies that keep
// the two halves coherent. Copies address a page through a plain buffer bound to its chunk's memory (memory may be
// bound to a sparse arena page and to a plain buffer at once). One thread records with it.
class PhysicalGpu {
public:
    PhysicalGpu(VkDevice device, PFN_vkGetDeviceProcAddr get, PagePool& pool, PhysicalPageStore& store,
                const VkPhysicalDeviceMemoryProperties& properties)
        : device_(device), pool_(pool), store_(store), properties_(properties) {
        create_buffer_ = reinterpret_cast<PFN_vkCreateBuffer>(get(device, "vkCreateBuffer"));
        destroy_buffer_ = reinterpret_cast<PFN_vkDestroyBuffer>(get(device, "vkDestroyBuffer"));
        bind_ = reinterpret_cast<PFN_vkBindBufferMemory>(get(device, "vkBindBufferMemory"));
        requirements_ =
            reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(get(device, "vkGetBufferMemoryRequirements"));
        allocate_ = reinterpret_cast<PFN_vkAllocateMemory>(get(device, "vkAllocateMemory"));
        free_ = reinterpret_cast<PFN_vkFreeMemory>(get(device, "vkFreeMemory"));
        map_ = reinterpret_cast<PFN_vkMapMemory>(get(device, "vkMapMemory"));
        copy_ = reinterpret_cast<PFN_vkCmdCopyBuffer>(get(device, "vkCmdCopyBuffer"));
        barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    }
    PhysicalGpu(const PhysicalGpu&) = delete;
    PhysicalGpu& operator=(const PhysicalGpu&) = delete;

    bool ok() const {
        return create_buffer_ && destroy_buffer_ && bind_ && requirements_ && allocate_ && free_ && map_ && copy_ &&
               barrier_;
    }
    ~PhysicalGpu() {
        for (auto& [memory, buffer] : access_) destroy_buffer_(device_, buffer, nullptr);
        std::vector<PageId> pages;
        for (auto& [physical, gpu] : gpu_) pages.push_back(gpu);
        pool_.release(pages);
        if (staging_) destroy_buffer_(device_, staging_, nullptr);
        if (staging_memory_) free_(device_, staging_memory_, nullptr);
    }

    // The GPU half of `p`, allocated on first use (kNoPage if the pool is out of memory). A new GPU half holds
    // whatever its pool page held, so `p` is marked dirty: the next upload gives it the CPU half's contents.
    PageId gpu_page(PhysicalPage p) {
        bool created = false;
        const PageId gpu = half(p, created);
        if (created) store_.mark_dirty(p);
        return gpu;
    }

    // CPU -> GPU: copies the CPU halves of `pages` (as take_dirty() returned them) into their GPU halves, recorded
    // into `cb` before the title's commands. Barriers order the copies after earlier GPU work on the queue and make
    // them visible to everything after. The staging copy happens now, on the CPU. On failure the pages are marked
    // dirty again, so their writes are uploaded by a later call.
    bool record_upload(VkCommandBuffer cb, std::span<const PhysicalPage> pages) {
        if (pages.empty()) return true;
        std::vector<Target> targets;
        if (!resolve(pages, targets, false) || !ensure_staging(pages.size() * kPageSize)) {
            for (PhysicalPage p : pages) store_.mark_dirty(p);
            return false;
        }
        readback_.clear();  // the staging data it was waiting for is overwritten
        std::map<VkBuffer, std::vector<VkBufferCopy>> copies;
        for (std::size_t i = 0; i < pages.size(); ++i) {
            std::memcpy(staging_map_ + i * kPageSize, store_.data(pages[i]), kPageSize);
            copies[targets[i].buffer].push_back({i * kPageSize, targets[i].offset, kPageSize});
        }
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier_(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                 nullptr);
        for (const auto& [buffer, regions] : copies)
            copy_(cb, staging_, buffer, std::uint32_t(regions.size()), regions.data());
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier_(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0,
                 nullptr);
        uploaded_ += pages.size();
        return true;
    }

    // GPU -> CPU, step 1: copy the GPU halves into staging (after the GPU work that wrote them). Refused for a page
    // the CPU wrote since its last upload (the readback would overwrite those writes) and for a page with no GPU
    // half yet (the GPU never wrote it).
    bool record_readback(VkCommandBuffer cb, std::span<const PhysicalPage> pages) {
        if (pages.empty()) return true;
        for (PhysicalPage p : pages)
            if (store_.dirty(p)) return false;
        std::vector<Target> sources;
        if (!resolve(pages, sources, true) || !ensure_staging(pages.size() * kPageSize)) return false;
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier_(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0,
                 nullptr);
        std::map<VkBuffer, std::vector<VkBufferCopy>> copies;
        for (std::size_t i = 0; i < pages.size(); ++i)
            copies[sources[i].buffer].push_back({sources[i].offset, i * kPageSize, kPageSize});
        for (const auto& [buffer, regions] : copies)
            copy_(cb, buffer, staging_, std::uint32_t(regions.size()), regions.data());
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier_(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
        readback_.assign(pages.begin(), pages.end());
        return true;
    }

    // GPU -> CPU, step 2, once that work is complete: the CPU halves take the GPU data. `pages` must be the last
    // record_readback's, with no upload recorded since; each readback is finished once. Refused, and dropped, if
    // the CPU wrote one of the pages meanwhile.
    bool finish_readback(std::span<const PhysicalPage> pages) {
        if (readback_.empty() || !std::equal(pages.begin(), pages.end(), readback_.begin(), readback_.end()))
            return false;
        for (PhysicalPage p : pages) {
            if (store_.dirty(p)) {
                readback_.clear();
                return false;
            }
        }
        for (std::size_t i = 0; i < pages.size(); ++i)
            std::memcpy(store_.data(pages[i]), staging_map_ + i * kPageSize, kPageSize);
        readback_.clear();
        return true;
    }

    std::uint64_t pages_uploaded() const { return uploaded_; }

private:
    struct Target {
        VkBuffer buffer;
        VkDeviceSize offset;
    };

    // The GPU half of `p`; `created` says whether it was allocated now.
    PageId half(PhysicalPage p, bool& created) {
        created = false;
        if (!store_.allocated(p)) return kNoPage;
        auto it = gpu_.find(p);
        if (it != gpu_.end()) return it->second;
        std::vector<PageId> one;
        if (!pool_.allocate(1, one)) return kNoPage;
        gpu_.emplace(p, one[0]);
        created = true;
        return one[0];
    }

    // Each page's GPU half as a buffer and offset, so a failure records nothing. `existing`: only pages that have
    // a GPU half already.
    bool resolve(std::span<const PhysicalPage> pages, std::vector<Target>& out, bool existing) {
        out.clear();
        if (!ok()) return false;
        for (PhysicalPage p : pages)
            if (!store_.allocated(p) || (existing && !gpu_.count(p))) return false;
        out.reserve(pages.size());
        for (PhysicalPage p : pages) {
            bool created = false;
            const PageId gpu = half(p, created);
            const VkBuffer buffer = gpu ? access(pool_.memory(gpu)) : VK_NULL_HANDLE;
            if (!buffer) return false;
            out.push_back({buffer, PagePool::offset(gpu)});
        }
        return true;
    }

    VkBuffer access(VkDeviceMemory memory) {
        auto it = access_.find(memory);
        if (it != access_.end()) return it->second;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = kChunkSize;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (create_buffer_(device_, &info, nullptr, &buffer) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkMemoryRequirements requirements{};
        requirements_(device_, buffer, &requirements);
        if (!(requirements.memoryTypeBits & (1u << pool_.memory_type())) || requirements.size > kChunkSize ||
            bind_(device_, buffer, memory, 0) != VK_SUCCESS) {
            destroy_buffer_(device_, buffer, nullptr);
            return VK_NULL_HANDLE;
        }
        access_.emplace(memory, buffer);
        return buffer;
    }

    // One host-visible, coherent staging buffer, grown as needed, used from offset 0 by every record call. So a
    // command buffer holds one record call, and it must complete (and finish_readback run) before the next call.
    bool ensure_staging(VkDeviceSize bytes) {
        if (bytes <= staging_size_) return true;
        if (staging_) destroy_buffer_(device_, staging_, nullptr);
        if (staging_memory_) free_(device_, staging_memory_, nullptr);
        staging_ = VK_NULL_HANDLE;
        staging_memory_ = VK_NULL_HANDLE;
        staging_size_ = 0;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (create_buffer_(device_, &info, nullptr, &staging_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        requirements_(device_, staging_, &requirements);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0; i < properties_.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i)) || (properties_.memoryTypes[i].propertyFlags & want) != want)
                continue;
            VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            alloc.allocationSize = requirements.size;
            alloc.memoryTypeIndex = i;
            if (allocate_(device_, &alloc, nullptr, &staging_memory_) == VK_SUCCESS) break;
        }
        void* mapped = nullptr;
        if (!staging_memory_ || bind_(device_, staging_, staging_memory_, 0) != VK_SUCCESS ||
            map_(device_, staging_memory_, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            return false;
        staging_map_ = static_cast<std::uint8_t*>(mapped);
        staging_size_ = bytes;
        return true;
    }

    VkDevice device_;
    PagePool& pool_;
    PhysicalPageStore& store_;
    VkPhysicalDeviceMemoryProperties properties_;
    PFN_vkCreateBuffer create_buffer_ = nullptr;
    PFN_vkDestroyBuffer destroy_buffer_ = nullptr;
    PFN_vkBindBufferMemory bind_ = nullptr;
    PFN_vkGetBufferMemoryRequirements requirements_ = nullptr;
    PFN_vkAllocateMemory allocate_ = nullptr;
    PFN_vkFreeMemory free_ = nullptr;
    PFN_vkMapMemory map_ = nullptr;
    PFN_vkCmdCopyBuffer copy_ = nullptr;
    PFN_vkCmdPipelineBarrier barrier_ = nullptr;
    std::map<VkDeviceMemory, VkBuffer> access_;
    std::map<PhysicalPage, PageId> gpu_;
    VkBuffer staging_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
    std::uint8_t* staging_map_ = nullptr;
    VkDeviceSize staging_size_ = 0;
    std::uint64_t uploaded_ = 0;
    std::vector<PhysicalPage> readback_;  // staged by the last record_readback, not finished yet
};

}  // namespace leonida::xbox

#pragma once

// Which registered buffer holds a console VA or range. Ranges are kept ordered by their first VA, so a lookup starts
// with a binary search instead of a scan of every registered resource. No Vulkan or Windows dependency; not
// thread-safe (VulkanContext guards it with resource_mutex_, like the bindings themselves).
//
// D3D12 lets placed buffers alias, so ranges may overlap. A lookup then returns the containing range that starts
// closest below the VA (the larger ID on a tie): the walk goes down from the nearest start and stops once no range can
// still hold the requested bytes, which `longest_` bounds.
// `longest_` is the longest range indexed now, not the longest ever: a large buffer that is freed stops lengthening
// later misses. A miss still walks every start within the longest live range below the VA.

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <utility>

namespace leonida {

template<typename Id>
class BufferVaIndex {
public:
    // Registers [va, va + size). A null VA or an empty range is not indexed (no lookup can land in it).
    void insert(std::uint64_t va, std::uint64_t size, Id id) {
        if (!va || !size) return;
        const auto [it, added] = ranges_.try_emplace({va, id}, size);
        if (!added) {  // the same (va, id) again: its new size replaces the old one
            drop_size(it->second);
            it->second = size;
        }
        sizes_.insert(size);
        longest_ = *sizes_.rbegin();
    }

    // Forgets what insert(va, size, id) registered; nothing when it was not indexed.
    void erase(std::uint64_t va, Id id) {
        const auto it = ranges_.find({va, id});
        if (it == ranges_.end()) return;
        drop_size(it->second);
        ranges_.erase(it);
        longest_ = sizes_.empty() ? 0 : *sizes_.rbegin();
    }

    struct Hit {
        Id id;
        std::uint64_t start;  // the range's first VA
        std::uint64_t size;
    };

    // The range that holds `va`; false when none does.
    bool find(std::uint64_t va, Hit& out) const { return find(va, 1, out); }

    // The range that holds every requested byte. A shorter alias or adjacent buffer cannot cover the request.
    bool find(std::uint64_t va, std::uint64_t bytes, Hit& out) const {
        if (!bytes || bytes > longest_ || bytes - 1 > UINT64_MAX - va) return false;
        const std::uint64_t max_offset = longest_ - bytes;
        auto it = ranges_.upper_bound({va, max_id()});
        while (it != ranges_.begin()) {
            --it;
            const std::uint64_t start = it->first.first;
            const std::uint64_t offset = va - start;  // start <= va here
            if (offset > max_offset) return false;   // no range that starts this low holds all the bytes
            if (offset < it->second && bytes <= it->second - offset) {
                out = {it->first.second, start, it->second};
                return true;
            }
        }
        return false;
    }

    std::size_t size() const { return ranges_.size(); }
    std::uint64_t longest() const { return longest_; }

private:
    static constexpr Id max_id() { return ~Id(0); }

    void drop_size(std::uint64_t size) { sizes_.erase(sizes_.find(size)); }  // one copy, not every range of that size

    std::map<std::pair<std::uint64_t, Id>, std::uint64_t> ranges_;  // (first VA, id) -> bytes
    std::multiset<std::uint64_t> sizes_;  // the size of every indexed range, so the longest is known after an erase
    std::uint64_t longest_ = 0;  // the longest indexed range, 0 when none is
};

} // namespace leonida

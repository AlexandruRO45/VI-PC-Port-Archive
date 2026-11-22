#include "era_title_va.h"

#include <iterator>
#include <new>

namespace era_title_va {
bool Table::add(std::uint64_t title_begin, std::uint64_t size, std::uint64_t gpu_begin)
{
    const std::uint64_t title_end = title_begin + size;
    if (size == 0 || title_end < title_begin || gpu_begin == 0 || gpu_begin + size < gpu_begin) {
        return false;
    }
    std::lock_guard<std::mutex> guard(lock_);
    auto next = ranges_.lower_bound(title_begin);
    if ((next != ranges_.end() && next->first < title_end) ||
        (next != ranges_.begin() && std::prev(next)->second.title_end > title_begin)) {
        return false;
    }
    try {
        ranges_.emplace_hint(next, title_begin, Range{title_begin, title_end, gpu_begin});
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

bool Table::remove(std::uint64_t title_begin)
{
    std::lock_guard<std::mutex> guard(lock_);
    return ranges_.erase(title_begin) == 1;
}

std::uint64_t Table::translate(std::uint64_t title_address, std::uint64_t bytes) const
{
    if (bytes == 0) {
        bytes = 1;
    }
    const std::uint64_t last = title_address + (bytes - 1);
    if (last < title_address) {
        return 0;
    }
    std::lock_guard<std::mutex> guard(lock_);
    auto next = ranges_.upper_bound(title_address);
    if (next == ranges_.begin()) {
        return 0;
    }
    const Range& range = std::prev(next)->second;
    if (last >= range.title_end) {
        return 0;
    }
    return range.gpu_begin + (title_address - range.title_begin);
}

bool shader_ranges(const std::vector<Range>& ranges, std::vector<ShaderRange>& packed)
{
    packed.clear();
    std::vector<ShaderRange> result;
    try {
        result.reserve(ranges.size());
    } catch (const std::bad_alloc&) {
        return false;
    }
    constexpr std::uint64_t view_limit = 1ull << 32;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const Range& range = ranges[i];
        if (range.title_end <= range.title_begin || range.title_end - range.title_begin >= view_limit ||
            (range.title_begin & 3) != 0 ||
            (i > 0 && ranges[i - 1].title_end > range.title_begin)) {
            return false;
        }
        result.push_back({static_cast<std::uint32_t>(range.title_begin),
                          static_cast<std::uint32_t>(range.title_begin >> 32),
                          static_cast<std::uint32_t>(range.title_end),
                          static_cast<std::uint32_t>(range.title_end >> 32), static_cast<std::uint32_t>(i), 0});
    }
    packed.swap(result);
    return true;
}

std::vector<Range> Table::ranges() const
{
    std::lock_guard<std::mutex> guard(lock_);
    std::vector<Range> result;
    result.reserve(ranges_.size());
    for (const auto& entry : ranges_) {
        result.push_back(entry.second);
    }
    return result;
}
}

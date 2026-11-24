#include "startup_stubs/placement_publish.h"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

namespace {
long fail_after = -1;
bool injected;
}

void* operator new(std::size_t size)
{
    if (fail_after == 0) {
        fail_after = -1;
        injected = true;
        throw std::bad_alloc();
    }
    if (fail_after > 0) --fail_after;
    if (void* ptr = std::malloc(size ? size : 1)) return ptr;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main()
{
    using Key = unsigned;
    struct Record { std::uint64_t bytes; };
    for (long n = 0; n < 32; ++n) {
        SRWLOCK lock = SRWLOCK_INIT;
        std::unordered_map<Key, Record> records{{1, {4}}};
        std::multimap<std::uint64_t, Key> by_start{{0x1000, 1}};
        std::uint64_t max_width = 4;
        const auto publish = [&](Key owner) {
            return x2w_startup::publish_placement(lock, [&]() -> auto& { return records; }, by_start,
                max_width, owner, 0x2000, Record{64});
        };

        fail_after = n;
        injected = false;
        const HRESULT hr = publish(2);
        fail_after = -1;
        if (!TryAcquireSRWLockExclusive(&lock)) return 1;
        ReleaseSRWLockExclusive(&lock);
        if (injected) {
            if (hr != E_OUTOFMEMORY || records.size() != 1 || by_start.size() != 1 || max_width != 4 ||
                records.at(1).bytes != 4 || by_start.begin()->second != 1) return 2;
            if (publish(2) != S_OK) return 3;
        } else if (hr != S_OK) return 4;
        if (records.size() != 2 || records.at(2).bytes != 64 || by_start.count(0x2000) != 1 || max_width != 64) return 5;
        if (publish(2) != HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS) || by_start.size() != 2) return 6;
        if (!injected) {
            std::printf("placement registration: %ld allocation failures rolled back and retried\n", n);
            return n >= 2 ? 0 : 7;
        }
    }
    return 8;
}

// Lanterns, stage 25: the call-class timers and the sampling profiler (lanterns_scale_prof.h).
#include "lanterns_scale_prof.h"

#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <utility>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace lanterns::scale {

Timers& timers() {
    static Timers t;
    return t;
}

// Dynamic TLS (TlsAlloc), not thread_local: the guest's image is mapped by gta6rt's loader, not Windows'.
ClassCounters& Timers::local() {
    static const DWORD slot = TlsAlloc();
    auto* mine = static_cast<ClassCounters*>(TlsGetValue(slot));
    if (!mine) {
        mine = new ClassCounters();   // kept until exit: the totals read it after the thread ends
        TlsSetValue(slot, mine);
        AcquireSRWLockExclusive(&lock_);
        threads_.push_back(mine);
        ReleaseSRWLockExclusive(&lock_);
    }
    return *mine;
}

void Timers::reset() {
    AcquireSRWLockExclusive(&lock_);
    for (ClassCounters* c : threads_) *c = ClassCounters{};
    ReleaseSRWLockExclusive(&lock_);
}

ClassCounters Timers::total() const {
    ClassCounters sum;
    AcquireSRWLockShared(&lock_);
    for (const ClassCounters* c : threads_)
        for (unsigned k = 0; k < kClassCount; ++k) sum.count[k] += c->count[k], sum.ticks[k] += c->ticks[k];
    ReleaseSRWLockShared(&lock_);
    return sum;
}

void Timers::start_clock() {
    QueryPerformanceCounter(&qpc0_);
    tsc0_ = __rdtsc();
}

double Timers::ns_per_tick() const {
    LARGE_INTEGER now, freq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&freq);
    const std::uint64_t ticks = __rdtsc() - tsc0_;
    const double ns = double(now.QuadPart - qpc0_.QuadPart) * 1e9 / double(freq.QuadPart);
    return ticks ? ns / double(ticks) : 0.0;
}

bool Sampler::start(unsigned period_us, std::size_t max_samples) {
    period_us_ = std::max(100u, period_us);
    samples_.resize(max_samples);
    thread_ = std::thread([this] { run(); });
    return true;
}

void Sampler::stop() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

void Sampler::run() {
    const DWORD self = GetCurrentThreadId(), process = GetCurrentProcessId();
    struct Target { DWORD id; HANDLE handle; std::uint32_t index; };
    std::vector<Target> targets;
    auto refresh = [&] {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if (snap == INVALID_HANDLE_VALUE) return;
        THREADENTRY32 te{};
        te.dwSize = sizeof(te);
        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
            if (te.th32OwnerProcessID != process || te.th32ThreadID == self) continue;
            if (std::any_of(targets.begin(), targets.end(), [&](const Target& t) { return t.id == te.th32ThreadID; }))
                continue;
            HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
                                  te.th32ThreadID);
            if (!h) continue;
            targets.push_back({te.th32ThreadID, h, std::uint32_t(thread_ids_.size())});
            thread_ids_.push_back(te.th32ThreadID);
        }
        CloseHandle(snap);
    };
    auto last_refresh = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    CONTEXT ctx;
    while (!stop_) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_refresh > std::chrono::milliseconds(500)) {
            refresh();
            last_refresh = now;
        }
        for (const Target& t : targets) {
            if (SuspendThread(t.handle) == DWORD(-1)) continue;
            ctx.ContextFlags = CONTEXT_CONTROL;
            const BOOL got = GetThreadContext(t.handle, &ctx);
            ResumeThread(t.handle);
            if (!got) continue;
            const std::size_t at = count_.load(std::memory_order_relaxed);
            if (at >= samples_.size()) { stop_ = true; break; }
            samples_[at] = {ctx.Rip, t.index};
            count_.store(at + 1, std::memory_order_relaxed);
        }
        if (period_us_ >= 1000) Sleep(period_us_ / 1000);
        else SwitchToThread();
    }
    for (const Target& t : targets) CloseHandle(t.handle);
}

bool Sampler::write(const std::filesystem::path& file) const {
    FILE* f = _wfopen(file.c_str(), L"w");
    if (!f) return false;
    struct Module { std::uint64_t base, size; std::string name; };
    std::vector<Module> modules;
    const auto guest_base = reinterpret_cast<std::uint64_t>(&__ImageBase);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(reinterpret_cast<const std::uint8_t*>(&__ImageBase) +
                                                                 __ImageBase.e_lfanew);
    modules.push_back({guest_base, nt->OptionalHeader.SizeOfImage, "lanterns.exe(guest)"});
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint64_t> counts;   // (rip, thread) -> samples
    const std::size_t n = count_.load();
    for (std::size_t i = 0; i < n; ++i) ++counts[{samples_[i].rip, samples_[i].thread}];
    std::map<std::uint64_t, int> module_of_base;
    for (const auto& [key, c] : counts) {
        const std::uint64_t rip = key.first;
        int m = -1;
        for (std::size_t k = 0; k < modules.size(); ++k)
            if (rip >= modules[k].base && rip < modules[k].base + modules[k].size) m = int(k);
        if (m < 0) {
            // The image a RIP is in: its allocation base (VirtualQuery, not the module APIs the shims answer for
            // the guest), named by the file mapped there.
            MEMORY_BASIC_INFORMATION mbi{};
            const auto* h = VirtualQuery(reinterpret_cast<const void*>(rip), &mbi, sizeof(mbi)) && mbi.Type == MEM_IMAGE
                                ? static_cast<const std::uint8_t*>(mbi.AllocationBase) : nullptr;
            if (h && reinterpret_cast<const IMAGE_DOS_HEADER*>(h)->e_magic == IMAGE_DOS_SIGNATURE) {
                char name[MAX_PATH] = "?";
                K32GetMappedFileNameA(GetCurrentProcess(), const_cast<std::uint8_t*>(h), name, MAX_PATH);
                const auto* mnt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
                    reinterpret_cast<const std::uint8_t*>(h) + reinterpret_cast<const IMAGE_DOS_HEADER*>(h)->e_lfanew);
                modules.push_back({reinterpret_cast<std::uint64_t>(h), mnt->OptionalHeader.SizeOfImage, name});
                m = int(modules.size() - 1);
            }
        }
        if (m >= 0)
            std::fprintf(f, "S %d %llx %u %llu\n", m, (unsigned long long)(rip - modules[m].base), key.second,
                         (unsigned long long)c);
        else
            std::fprintf(f, "S -1 %llx %u %llu\n", (unsigned long long)rip, key.second, (unsigned long long)c);
    }
    for (std::size_t k = 0; k < modules.size(); ++k)
        std::fprintf(f, "M %zu %llx %llx %s\n", k, (unsigned long long)modules[k].base,
                     (unsigned long long)modules[k].size, modules[k].name.c_str());
    for (std::size_t t = 0; t < thread_ids_.size(); ++t) std::fprintf(f, "T %zu %lu\n", t, thread_ids_[t]);
    std::fclose(f);
    return true;
}

}  // namespace lanterns::scale
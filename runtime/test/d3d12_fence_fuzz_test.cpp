// Differential fence/queue fuzzer: one seeded random program of queue Signal/Wait, CPU Signal, SetEventOnCompletion
// and ExecuteCommandLists runs on the layer's device and on the host D3D12 device, and what an application can observe
// is compared at settle points: every fence's completed value and which events are set.
//
// A program is deterministic once settled: each fence has one signaller (the CPU or one queue, in program order), a
// queue waits only for a value its signaller has already been given or for a CPU fence (CPU signals never block), so
// no wait cycle can form, and the program ends by signalling every CPU fence to the highest value anything waits for.
// At a settle point the queues run until they block on a CPU fence value not signalled yet; both devices must reach
// the same state.
//
// Default (what ctest runs): monotonic values, no fence released while in use, steps below 2^31. Options for cases
// the layer is known to differ on: --rewinds (a signaller goes back, D3D-472), --release (a fence released while queue
// waits or events use it, D3D-496), --big (steps above 2^31-1, the timeline limit; #4609). --seeds N --start S
// --ops N choose the programs; --verbose prints each program. A mismatch prints the seed and the program up to the
// first settle point that differs. LEONIDA_VK_VALIDATION=1 runs the test-made device under the validation layer, and
// any validation error fails the seed. Needs a Vulkan GPU and a host D3D12 adapter (exit 77 otherwise). No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

struct Options {
    unsigned seeds = 40, start = 1, ops = 48;
    bool rewinds = false, release = false, big = false, verbose = false;
    std::string fixed;  // --case NAME: run one fixed program instead of the seeds
};

constexpr int queue_count = 3;  // DIRECT, COMPUTE, COPY
constexpr int fence_slots = 4;
constexpr D3D12_COMMAND_LIST_TYPE queue_type[queue_count] = {
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
constexpr const char* queue_name[queue_count] = {"DIRECT", "COMPUTE", "COPY"};
constexpr int cpu = -1;

struct Op {
    enum Kind { queue_signal, queue_wait, cpu_signal, event, execute, release, settle } kind;
    int queue = 0, fence = 0;
    UINT64 value = 0;
};

// Fixed programs for findings, run with --case NAME.
std::vector<Op> fixed_program(const std::string& name) {
    if (name == "pending-wait-release")  // a queue Wait, the fence released, then the queue's next submission
        return {{Op::cpu_signal, 0, 0, 1}, {Op::queue_wait, 1, 0, 1}, {Op::release, 0, 0, 0},
                {Op::execute, 1, 0, 0}, {Op::settle}};
    return {};
}

std::string describe(const Op& op) {
    char text[128];
    const auto v = static_cast<unsigned long long>(op.value);
    switch (op.kind) {
    case Op::queue_signal: std::snprintf(text, sizeof text, "%s Signal(f%d, %llu)", queue_name[op.queue], op.fence, v); break;
    case Op::queue_wait: std::snprintf(text, sizeof text, "%s Wait(f%d, %llu)", queue_name[op.queue], op.fence, v); break;
    case Op::cpu_signal: std::snprintf(text, sizeof text, "CPU f%d->Signal(%llu)", op.fence, v); break;
    case Op::event: std::snprintf(text, sizeof text, "f%d->SetEventOnCompletion(%llu, e)", op.fence, v); break;
    case Op::execute: std::snprintf(text, sizeof text, "%s ExecuteCommandLists(empty list)", queue_name[op.queue]); break;
    case Op::release: std::snprintf(text, sizeof text, "release f%d (a new fence at 0 takes the slot)", op.fence); break;
    case Op::settle: std::snprintf(text, sizeof text, "-- settle"); break;
    }
    return text;
}

// The program. Each fence slot has an owner (cpu or a queue index) for its current fence; a released slot gets a new
// fence with a new owner.
std::vector<Op> generate(std::uint64_t seed, const Options& o) {
    std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ull + 1);
    auto pick = [&](int n) { return int(rng() % unsigned(n)); };
    int owner[fence_slots];
    UINT64 issued[fence_slots] = {}, waited[fence_slots] = {};
    for (int& w : owner) w = pick(2) ? cpu : pick(queue_count);
    std::vector<Op> program;
    const auto step = [&]() -> UINT64 {
        if (o.big && pick(8) == 0) return 0x80000000ull + rng() % 0x100000000ull;
        return 1 + rng() % 3;
    };
    const auto next_value = [&](int f) {
        if (o.rewinds && issued[f] > 2 && pick(6) == 0) return issued[f] - 1 - rng() % (issued[f] - 1);
        const UINT64 s = step();
        return issued[f] > UINT64_MAX - s ? UINT64_MAX : issued[f] + s;
    };
    for (unsigned i = 0; i < o.ops; ++i) {
        if (i % 8 == 7) {
            program.push_back({Op::settle});
            continue;
        }
        const int f = pick(fence_slots);
        switch (pick(o.release ? 11 : 10)) {
        case 0: case 1: case 2: {  // a signal by the fence's owner
            const UINT64 v = next_value(f);
            program.push_back({owner[f] == cpu ? Op::cpu_signal : Op::queue_signal, owner[f] == cpu ? 0 : owner[f], f, v});
            issued[f] = v;
            break;
        }
        case 3: case 4: case 5: {  // a queue wait: any value of a CPU fence, an already given value of a queue fence
            const int q = pick(queue_count);
            UINT64 v;
            if (owner[f] == cpu) {
                v = issued[f] + rng() % 3;  // may be ahead of every CPU signal so far
                if (v == 0) v = 1;
            } else {
                if (issued[f] == 0) continue;
                v = 1 + rng() % issued[f];
            }
            program.push_back({Op::queue_wait, q, f, v});
            if (v > waited[f]) waited[f] = v;
            break;
        }
        case 6: case 7: {  // an event, possibly for a value nothing will reach
            UINT64 v = issued[f] + rng() % 3;
            if (v == 0) v = 1;
            program.push_back({Op::event, 0, f, v});
            break;
        }
        case 8: case 9:
            program.push_back({Op::execute, pick(queue_count), 0, 0});
            break;
        default:
            program.push_back({Op::release, 0, f, 0});
            owner[f] = pick(2) ? cpu : pick(queue_count);
            issued[f] = waited[f] = 0;
            break;
        }
    }
    // Release every queue still waiting on a CPU fence, then a final settle point.
    for (int f = 0; f < fence_slots; ++f)
        if (owner[f] == cpu && waited[f] > issued[f]) program.push_back({Op::cpu_signal, 0, f, waited[f]});
    program.push_back({Op::settle});
    return program;
}

struct Snapshot {
    UINT64 value[fence_slots] = {};
    std::vector<bool> events;
    bool operator==(const Snapshot& other) const {
        return std::memcmp(value, other.value, sizeof value) == 0 && events == other.events;
    }
};

struct Run {
    std::vector<Snapshot> settles;
    bool settled = true;  // false: a settle point never became stable
    std::string error;
};

// The watchdog: a hang ends the process with the seed instead of a ctest timeout with no message.
std::atomic<ULONGLONG> deadline{0};
std::atomic<unsigned> current_seed{0};
std::atomic<const char*> current_device{"start"};

Run execute(ID3D12Device* device, const std::vector<Op>& program) {
    Run run;
    ComPtr<ID3D12CommandQueue> queues[queue_count];
    ComPtr<ID3D12CommandAllocator> allocators[queue_count];
    ComPtr<ID3D12GraphicsCommandList> lists[queue_count];
    for (int q = 0; q < queue_count; ++q) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = queue_type[q];
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queues[q]))) ||
            FAILED(device->CreateCommandAllocator(queue_type[q], IID_PPV_ARGS(&allocators[q]))) ||
            FAILED(device->CreateCommandList(0, queue_type[q], allocators[q].Get(), nullptr, IID_PPV_ARGS(&lists[q]))) ||
            FAILED(lists[q]->Close())) {
            run.error = std::string("cannot create the ") + queue_name[q] + " queue and its list";
            return run;
        }
    }
    ComPtr<ID3D12Fence> fences[fence_slots];
    const auto fresh = [&](int f) { return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fences[f]))); };
    for (int f = 0; f < fence_slots; ++f)
        if (!fresh(f)) {
            run.error = "CreateFence failed";
            return run;
        }
    struct Registered { int fence; HANDLE handle; };
    std::vector<Registered> events;
    const auto snapshot = [&] {
        Snapshot s;
        for (int f = 0; f < fence_slots; ++f) s.value[f] = fences[f]->GetCompletedValue();
        for (const auto& e : events) s.events.push_back(WaitForSingleObject(e.handle, 0) == WAIT_OBJECT_0);
        return s;
    };
    for (const Op& op : program) {
        switch (op.kind) {
        case Op::queue_signal: queues[op.queue]->Signal(fences[op.fence].Get(), op.value); break;
        case Op::queue_wait: queues[op.queue]->Wait(fences[op.fence].Get(), op.value); break;
        case Op::cpu_signal: fences[op.fence]->Signal(op.value); break;
        case Op::event: {
            HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // manual reset: reading it does not consume it
            fences[op.fence]->SetEventOnCompletion(op.value, h);
            events.push_back({op.fence, h});
            break;
        }
        case Op::execute: {
            ID3D12CommandList* list = lists[op.queue].Get();
            queues[op.queue]->ExecuteCommandLists(1, &list);
            break;
        }
        case Op::release:
            fences[op.fence].Reset();
            if (!fresh(op.fence)) run.error = "CreateFence failed";
            break;
        case Op::settle: {
            // Stable for three reads 20 ms apart, within 3 s.
            Snapshot last = snapshot();
            int stable = 0;
            const ULONGLONG until = GetTickCount64() + 3000;
            while (stable < 3) {
                Sleep(20);
                Snapshot now = snapshot();
                stable = now == last ? stable + 1 : 0;
                last = std::move(now);
                if (GetTickCount64() > until) {
                    run.settled = false;
                    break;
                }
            }
            run.settles.push_back(std::move(last));
            break;
        }
        }
        if (!run.error.empty()) break;
    }
    // Drain: every queue signals a fence of its own, which must complete (the program released every wait).
    ComPtr<ID3D12Fence> drain;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&drain));
    for (int q = 0; q < queue_count; ++q) queues[q]->Signal(drain.Get(), UINT64(q + 1));
    HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    drain->SetEventOnCompletion(queue_count, h);
    if (WaitForSingleObject(h, 5000) != WAIT_OBJECT_0 && run.error.empty()) run.error = "the queues did not drain";
    CloseHandle(h);
    for (const auto& e : events) CloseHandle(e.handle);
    return run;
}

ComPtr<ID3D12Device> native_device(std::wstring& name) {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        name = desc.Description;
        return device;
    }
}

void print_snapshot(const char* who, const Snapshot& s) {
    std::fprintf(stderr, "  %-5s values:", who);
    for (UINT64 v : s.value) std::fprintf(stderr, " %llu", static_cast<unsigned long long>(v));
    std::fprintf(stderr, "  events:");
    for (bool e : s.events) std::fprintf(stderr, "%c", e ? '1' : '0');
    std::fprintf(stderr, "\n");
}

// Prints the program up to settle point `index` (0-based); returns false.
bool report(unsigned seed, const std::vector<Op>& program, size_t index, const Snapshot& host, const Snapshot& layer) {
    std::fprintf(stderr, "MISMATCH seed %u at settle point %zu. Program up to it:\n", seed, index);
    size_t settles = 0;
    for (const Op& op : program) {
        std::fprintf(stderr, "  %s\n", describe(op).c_str());
        if (op.kind == Op::settle && settles++ == index) break;
    }
    print_snapshot("host", host);
    print_snapshot("layer", layer);
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a crash must not lose the program printed before it
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto number = [&](unsigned& out) { if (i + 1 < argc) out = unsigned(std::strtoul(argv[++i], nullptr, 10)); };
        if (a == "--seeds") number(o.seeds);
        else if (a == "--start") number(o.start);
        else if (a == "--ops") number(o.ops);
        else if (a == "--rewinds") o.rewinds = true;
        else if (a == "--release") o.release = true;
        else if (a == "--big") o.big = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--case" && i + 1 < argc) o.fixed = argv[++i];
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::thread([] {
        for (;;) {
            Sleep(100);
            const ULONGLONG limit = deadline.load();
            if (limit && GetTickCount64() > limit) {
                std::fprintf(stderr, "HANG: seed %u on the %s\n", current_seed.load(), current_device.load());
                std::fflush(stderr);
                std::_Exit(1);
            }
        }
    }).detach();

    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    ComPtr<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&layer)))) {
        std::fprintf(stderr, "create_vulkan_device failed\n");
        return 1;
    }
    // Under Wine the "host" D3D12 is vkd3d, itself a translation layer: no reference to compare with.
    if (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version")) {
        std::puts("SKIP: Wine's D3D12 is not a host reference");
        return 77;
    }
    std::wstring name;
    ComPtr<ID3D12Device> host = native_device(name);
    if (!host) {
        std::puts("SKIP: no host D3D12 adapter to compare with");
        return 77;
    }
    if (!o.fixed.empty()) o.seeds = 1;
    std::printf("host D3D12: %ls; seeds %u..%u, %u operations each%s%s%s\n", name.c_str(), o.start,
                o.start + o.seeds - 1, o.ops, o.rewinds ? ", rewinds" : "", o.release ? ", releases" : "",
                o.big ? ", big steps" : "");

    unsigned failures = 0, settle_points = 0;
    for (unsigned seed = o.start; seed < o.start + o.seeds; ++seed) {
        const std::vector<Op> program = o.fixed.empty() ? generate(seed, o) : fixed_program(o.fixed);
        if (program.empty()) {
            std::fprintf(stderr, "unknown case %s\n", o.fixed.c_str());
            return 2;
        }
        if (o.verbose) {
            std::printf("seed %u:\n", seed);
            for (const Op& op : program) std::printf("  %s\n", describe(op).c_str());
        }
        current_seed = seed;
        deadline = GetTickCount64() + 60000;
        current_device = "host";
        const Run theirs = execute(host.Get(), program);
        const unsigned errors_before = d.validation_errors();
        current_device = "layer";
        const Run ours = execute(layer.Get(), program);
        deadline = 0;
        const unsigned validation = d.validation_errors() - errors_before;
        bool ok = true;
        if (!theirs.error.empty() || !theirs.settled) {
            std::fprintf(stderr, "seed %u: the host run is not usable (%s); seed skipped\n", seed,
                         theirs.error.empty() ? "a settle point was not stable" : theirs.error.c_str());
            continue;
        }
        if (!ours.error.empty()) {
            std::fprintf(stderr, "seed %u: layer: %s\n", seed, ours.error.c_str());
            ok = false;
        }
        if (!ours.settled) {
            std::fprintf(stderr, "seed %u: layer: a settle point was not stable within 3 s\n", seed);
            ok = false;
        }
        for (size_t i = 0; ok && i < theirs.settles.size() && i < ours.settles.size(); ++i)
            if (!(theirs.settles[i] == ours.settles[i])) ok = report(seed, program, i, theirs.settles[i], ours.settles[i]);
        if (ok && ours.settles.size() != theirs.settles.size()) {
            std::fprintf(stderr, "seed %u: %zu settle points on the layer, %zu on the host\n", seed, ours.settles.size(),
                         theirs.settles.size());
            ok = false;
        }
        if (validation) {
            std::fprintf(stderr, "seed %u: %u validation errors\n", seed, validation);
            ok = false;
        }
        settle_points += unsigned(theirs.settles.size());
        if (!ok) ++failures;
    }
    if (failures) {
        std::fprintf(stderr, "%u of %u seeds differ from the host\n", failures, o.seeds);
        return 1;
    }
    std::printf("fence fuzz: %u seeds, %u settle points, layer and host identical\n", o.seeds, settle_points);
    return 0;
}

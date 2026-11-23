// Tests for the boot diagnostics (shims/boot_diag.h, boot_watchdog.h, boot_faults.h).
//
// Everything is synthetic: the test builds a small block of x86-64 code (a "mapped image" the shim registry's module lookup
// knows as synth.dll, with unwind data registered like the loader does) and runs it:
//   waiter     calls WaitForSingleObject through an import slot on an event nobody sets
//   spinner    loops until a flag byte is set
//   nullcall   calls through a null pointer
//   nullread   reads [rcx+0x18] with rcx = 0
//   hostfault  calls ntdll!RtlInitUnicodeString(NULL, NULL), which faults inside the host
//   shimcall   calls through an import slot that holds a registered shim
//   ret7       returns 7 (the breakpoint target)
// Each report is read back through the diagnostics' own sink and checked against the code's exact offsets. The
// negative controls: a slot that holds another function names another import, a return address that follows no call names
// nothing, a stack word that is not a return address is not listed, a thread that makes progress is not dumped.
// No game files, no GPU. Run as "boot_diag_test"; "boot_diag_test --unhandled" is the child the last check starts.
#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <climits>

#include "../shims/boot_diag_install.h"
#include "../shims/shims.h"

#ifdef LEONIDA_TEST_PRESENT_PROGRESS
#include "../shims/xbox_present.h"
#include "../src/vulkan_present.h"

namespace {
HRESULT blank_present_result = E_FAIL;
}

namespace rt::d3d12 {
// Link the real XboxPresentation against a CPU-only presenter so the result is deterministic.
WindowPresenter::WindowPresenter(IUnknown* device, std::unique_ptr<leonida::VulkanPresenter> presenter)
    : device_(device), presenter_(std::move(presenter)) {}
WindowPresenter::~WindowPresenter() = default;
void WindowPresenter::set_abort(std::function<bool()>) {}
bool WindowPresenter::tear_supported() const { return false; }
bool WindowPresenter::last_present_tore() const { return false; }
bool WindowPresenter::hdr_active() const { return false; }
float WindowPresenter::display_sdr_white_nits() const { return 0; }
HRESULT WindowPresenter::present_blank(UINT) { return blank_present_result; }
HRESULT WindowPresenter::present(ID3D12Resource*, UINT, const D3D12_RECT*, bool) { return E_NOTIMPL; }
HRESULT WindowPresenter::present_planes(const leonida::PresentPlanes&, UINT, bool) { return E_NOTIMPL; }
}  // namespace rt::d3d12
#endif

namespace {

using namespace rt::boot_diag;

int g_checks = 0;
int g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what.c_str());
    if (!ok) ++g_failed;
}

// ---- what the diagnostics said ----------------------------------------------------------------------------------------

std::mutex g_out_mutex;
std::vector<std::string> g_out;

void collect(void*, const std::string& line) {
    const std::lock_guard lock(g_out_mutex);
    g_out.push_back(line);
}
void clear_out() {
    const std::lock_guard lock(g_out_mutex);
    g_out.clear();
}
std::string all_text() {
    const std::lock_guard lock(g_out_mutex);
    std::string s;
    for (const auto& l : g_out) s += l + "\n";
    return s;
}
size_t count_of(const std::string& hay, const std::string& needle) {
    size_t n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1)) ++n;
    return n;
}
// A failed check shows the text it looked at.
void check_in(bool ok, const std::string& what, const std::string& text) {
    check(ok, what);
    if (!ok) std::printf("---- text checked ----\n%s----\n", text.c_str());
}
bool has(const std::string& hay, const std::string& needle) { return hay.find(needle) != std::string::npos; }
std::string hex_rva(uint32_t rva) { return format("synth.dll+0x%x", rva); }

// ---- the synthetic image ----------------------------------------------------------------------------------------------

constexpr uint32_t kImageSize = 0x10000;
constexpr uint32_t kUnwindRva = 0x100;  // sub rsp,0x28 in a 4-byte prologue: UWOP_ALLOC_SMALL
constexpr uint32_t kCodeStart = 0x200;
constexpr uint32_t kDataStart = 0x8000;

uint8_t* g_image = nullptr;

bool lookup(void*, uint64_t address, std::string& module, uint32_t& rva) {
    const auto base = reinterpret_cast<uint64_t>(g_image);
    if (!g_image || address < base || address >= base + kImageSize) return false;
    module = "synth.dll";
    rva = static_cast<uint32_t>(address - base);
    return true;
}

struct Asm {
    std::vector<uint8_t> v;
    uint32_t start;
    explicit Asm(uint32_t at) : start(at) {}
    Asm& b(std::initializer_list<int> bytes) {
        for (int x : bytes) v.push_back(static_cast<uint8_t>(x));
        return *this;
    }
    uint32_t here() const { return start + static_cast<uint32_t>(v.size()); }
    // A rip-relative 32-bit displacement to 'target_rva'; 'tail' is the bytes of the instruction after it.
    Asm& rip(uint32_t target_rva, int tail = 0) {
        const int32_t d = static_cast<int32_t>(target_rva) - static_cast<int32_t>(here() + 4 + static_cast<uint32_t>(tail));
        for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(static_cast<uint32_t>(d) >> (8 * i)));
        return *this;
    }
};

struct Stub {
    uint32_t rva = 0;
    uint32_t after_call = 0;  // the return address of its call, as an RVA
    uint32_t fault = 0;       // the faulting instruction, as an RVA
    uint32_t end = 0;
    uint8_t* at() const { return g_image + rva; }
};

struct Builder {
    uint32_t next_code = kCodeStart, next_data = kDataStart;
    std::vector<RUNTIME_FUNCTION> table;
    uint32_t slot(uint64_t value) {
        const uint32_t rva = next_data;
        std::memcpy(g_image + rva, &value, 8);
        next_data += 8;
        return rva;
    }
    Stub finish(Asm& a, Stub s, bool with_unwind = true) {
        s.end = a.here();
        std::memcpy(g_image + a.start, a.v.data(), a.v.size());
        next_code = (s.end + 15) & ~15u;
        if (with_unwind) {
            RUNTIME_FUNCTION f{};
            f.BeginAddress = s.rva;
            f.EndAddress = s.end;
            f.UnwindData = kUnwindRva;
            table.push_back(f);
        }
        return s;
    }
    Stub call_slot(uint32_t fn_slot, std::initializer_list<int> setup = {}) {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28}).b(setup).b({0xFF, 0x15}).rip(fn_slot);
        s.after_call = a.here();
        a.b({0x48, 0x83, 0xC4, 0x28, 0xC3});
        return finish(a, s);
    }
    Stub waiter(uint32_t event_slot, uint32_t fn_slot) {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28, 0x48, 0x8B, 0x0D}).rip(event_slot).b({0xBA, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x15}).rip(fn_slot);
        s.after_call = a.here();
        a.b({0x48, 0x83, 0xC4, 0x28, 0xC3});
        return finish(a, s);
    }
    Stub spinner(uint32_t flag_slot) {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28, 0x80, 0x3D}).rip(flag_slot, 1).b({0x00, 0x74, 0xF7, 0x48, 0x83, 0xC4, 0x28, 0xC3});
        return finish(a, s);
    }
    Stub null_call() {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28, 0x31, 0xC0, 0xFF, 0xD0});  // xor eax,eax; call rax
        s.after_call = a.here();
        a.b({0x48, 0x83, 0xC4, 0x28, 0xC3});
        return finish(a, s);
    }
    Stub null_read() {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28, 0x31, 0xC9});
        s.fault = a.here();
        a.b({0x48, 0x8B, 0x41, 0x18, 0x48, 0x83, 0xC4, 0x28, 0xC3});  // mov rax,[rcx+0x18]
        return finish(a, s);
    }
    Stub ret7() {
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0xB8, 0x07, 0x00, 0x00, 0x00, 0xC3});
        return finish(a, s, false);
    }
    // A caller of a thunk that jumps through a slot: call rel32 -> jmp [rip+disp32].
    Stub via_thunk(uint32_t fn_slot) {
        Stub thunk;
        thunk.rva = next_code;
        Asm t(thunk.rva);
        t.b({0xFF, 0x25}).rip(fn_slot);
        finish(t, thunk, false);
        Stub s;
        s.rva = next_code;
        Asm a(s.rva);
        a.b({0x48, 0x83, 0xEC, 0x28, 0xE8});
        a.v.insert(a.v.end(), 4, 0);
        const int32_t d = static_cast<int32_t>(thunk.rva) - static_cast<int32_t>(a.here());
        std::memcpy(a.v.data() + a.v.size() - 4, &d, 4);
        s.after_call = a.here();
        a.b({0x48, 0x83, 0xC4, 0x28, 0xC3});
        return finish(a, s);
    }
};

void shim_marker() {}

uint64_t proc(const char* dll, const char* name) { return reinterpret_cast<uint64_t>(GetProcAddress(GetModuleHandleA(dll), name)); }

// ---- running stubs --------------------------------------------------------------------------------------------------

__declspec(noinline) bool run_catching(const Stub& s) {
    __try {
        reinterpret_cast<void (*)()>(s.at())();
        return false;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return true;
    }
}

struct Runner {
    Stub stub;
    const char* name;
    std::atomic<bool> up{false};
    HANDLE thread = nullptr;
};

DWORD WINAPI runner_main(void* p) {
    auto* r = static_cast<Runner*>(p);
    if (r->name) {  // the old way to name a thread, as engines do
        struct {
            DWORD type;
            DWORD pad;
            const char* name;
            DWORD id;
            DWORD flags;
        } info = {0x1000, 0, r->name, static_cast<DWORD>(-1), 0};
        __try {
            RaiseException(kThreadNameException, 0, 3, reinterpret_cast<const ULONG_PTR*>(&info));
        } __except (EXCEPTION_EXECUTE_HANDLER) {
        }
    }
    r->up = true;
    reinterpret_cast<void (*)()>(r->stub.at())();
    return 0;
}

void start_runner(Runner& r) {
    r.thread = CreateThread(nullptr, 0, runner_main, &r, 0, nullptr);
    while (!r.up) Sleep(5);
    Sleep(50);  // inside the stub
}

bool wait_for(const std::function<bool()>& done, int ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) {
        if (done()) return true;
        Sleep(20);
    }
    return done();
}

// ---- the image shared by the checks -------------------------------------------------------------------------------

struct World {
    Builder b;
    HANDLE event = nullptr;
    uint32_t flag = 0, wait_slot = 0, setevent_slot = 0;
    Stub waiter, waiter_wrong_slot, spinner, null_call, null_read, host_fault, shim_call, ret7, ret7_other, thunk_call;
};

World make_world() {
    World w;
    w.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);  // never set: the waiter blocks
    const uint32_t event_slot = w.b.slot(reinterpret_cast<uint64_t>(w.event));
    w.flag = w.b.slot(0);
    w.wait_slot = w.b.slot(proc("kernel32.dll", "WaitForSingleObject"));
    w.setevent_slot = w.b.slot(proc("kernel32.dll", "SetEvent"));
    const uint32_t init_slot = w.b.slot(proc("ntdll.dll", "RtlInitUnicodeString"));
    const uint32_t shim_slot = w.b.slot(reinterpret_cast<uint64_t>(&shim_marker));
    w.waiter = w.b.waiter(event_slot, w.wait_slot);
    w.waiter_wrong_slot = w.b.waiter(event_slot, w.setevent_slot);  // same code, the slot holds SetEvent
    w.spinner = w.b.spinner(w.flag);
    w.null_call = w.b.null_call();
    w.null_read = w.b.null_read();
    w.host_fault = w.b.call_slot(init_slot, {0x31, 0xC9, 0x31, 0xD2});  // xor ecx,ecx; xor edx,edx
    w.shim_call = w.b.call_slot(shim_slot);
    w.ret7 = w.b.ret7();
    w.ret7_other = w.b.ret7();
    w.thunk_call = w.b.via_thunk(w.wait_slot);
    return w;
}

// ---- the checks -----------------------------------------------------------------------------------------------------

void test_callee_names(World& w) {
    const auto base = reinterpret_cast<uint64_t>(g_image);
    const std::string waits = callee_of_call_ending_at(base + w.waiter.after_call);
    check(has(waits, "!WaitForSingleObject"), "call [rip+slot] holding kernel32!WaitForSingleObject names it (got '" + waits + "')");
    const std::string wrong = callee_of_call_ending_at(base + w.waiter_wrong_slot.after_call);
    check(has(wrong, "!SetEvent") && !has(wrong, "WaitForSingleObject"),
          "the same code with SetEvent in the slot names SetEvent, not the wait (got '" + wrong + "')");
    check(callee_of_call_ending_at(base + w.shim_call.after_call) == "synth.dll!Marker", "a slot holding a registered shim names the registry entry");
    const std::string thunk = callee_of_call_ending_at(base + w.thunk_call.after_call);
    check(has(thunk, "!WaitForSingleObject"), "call rel32 to a jmp [rip+slot] thunk is followed to the import (got '" + thunk + "')");
    check(callee_of_call_ending_at(base + w.ret7.rva + 5).empty(), "an address that follows no call instruction names nothing");
    check(follows_call(base + w.waiter.after_call) && follows_call(base + w.null_call.after_call) && !follows_call(base + w.ret7.rva + 5),
          "follows_call accepts call [rip+disp] and call reg, and rejects other bytes");
}

void test_scan_buffer() {
    uint64_t stack[6] = {0x1111, 0x500010, 0x500020, 0x9000000, 0x500030, 0};
    const auto hits = scan_stack(reinterpret_cast<uint8_t*>(stack), sizeof stack,
                                 [](uint64_t v) { return v >= 0x500000 && v < 0x600000; }, [](uint64_t v) { return v != 0x500020; });
    check(hits.size() == 2 && hits[0].address == 0x500010 && hits[0].slot_offset == 8 && hits[1].address == 0x500030 && hits[1].slot_offset == 32,
          "the stack scan lists the slots in the image that follow a call, with their offsets, and nothing else");
}

void test_names(World& w) {
    Runner r{w.waiter, "SynthName"};
    // The thread proc raises the old exception; the handler records the name.
    start_runner(r);
    check(thread_name(GetThreadId(r.thread)) == "SynthName", "the 0x406D1388 exception names the thread");
    SetEvent(w.event);
    WaitForSingleObject(r.thread, 5000);
    ResetEvent(w.event);
    CloseHandle(r.thread);
    check(thread_name(0xFFFFFF00u).empty(), "an unknown thread has no name");
    HANDLE self = GetCurrentThread();
    using SetDesc = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    auto set_desc = reinterpret_cast<SetDesc>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernelbase.dll"), "SetThreadDescription")));
    if (set_desc && SUCCEEDED(set_desc(self, L"MainDesc"))) check(thread_name(GetCurrentThreadId()) == "MainDesc", "SetThreadDescription names a thread too");
}

void test_faults(World& w) {
    const auto base = reinterpret_cast<uint64_t>(g_image);
    (void)base;
    clear_out();
    check(run_catching(w.null_call), "the null call faults");
    std::string t = all_text();
    check_in(has(t, "first-chance fault: access violation") && has(t, "call through a null pointer") && has(t, "called from " + hex_rva(w.null_call.after_call)),
             "a null call names the caller as the return address, " + hex_rva(w.null_call.after_call), t);
    check(!has(t, "null object"), "a null call is not described as a null object");

    clear_out();
    check(run_catching(w.null_read), "the null read faults");
    t = all_text();
    check(has(t, "at " + hex_rva(w.null_read.fault)) && has(t, "null pointer plus the field offset 0x18") && !has(t, "call through"),
          "a read of [0+0x18] names the instruction " + hex_rva(w.null_read.fault) + " and the field offset");
    check(has(t, "rcx=0 "), "the report carries the registers");

    clear_out();
    check(run_catching(w.host_fault), "the host fault happens");
    t = all_text();
    check(has(t, "the fault is in host code; image code " + hex_rva(w.host_fault.after_call) + " called ntdll.dll!RtlInitUnicodeString"),
          "a fault inside a host function names the import and its caller (" + hex_rva(w.host_fault.after_call) + ")");

    // The budget: LEONIDA_FAULTS=2 shows two and says so once.
    remove_fault_handlers();
    FaultConfig two;
    two.fault_reports = 2;
    install_fault_handlers(two);
    clear_out();
    for (int i = 0; i < 4; ++i) run_catching(w.null_read);
    t = all_text();
    check(count_of(t, "first-chance fault:") == 2 && count_of(t, "more than 2 first-chance faults") == 1, "only the first 2 faults are shown, and one line says more came");
    remove_fault_handlers();
    FaultConfig off;
    off.fault_reports = 0;
    install_fault_handlers(off);
    clear_out();
    run_catching(w.null_read);
    check(all_text().empty(), "LEONIDA_FAULTS=0 shows nothing");
    remove_fault_handlers();
}

struct PlainError : std::exception {};
struct Widget {};

__declspec(noinline) void throw_and_catch() {
    try {
        throw std::runtime_error("boom");
    } catch (const std::exception&) {
    }
    try {
        throw Widget();
    } catch (const Widget&) {
    }
}

void test_cpp_exceptions() {
    check(readable_type_name(".?AVruntime_error@std@@") == "std::runtime_error", "a decorated class name is made readable");
    check(readable_type_name(".?AUWidget@@") == "Widget", "a struct in the global namespace is made readable");
    check(readable_type_name(".?AUWidget@?A0xDA5B7332@@") == "(anonymous namespace)::Widget", "an anonymous namespace is named");
    check(readable_type_name(".?AV?$vector@H@std@@") == ".?AV?$vector@H@std@@", "a template name is left as it is");
    check(readable_type_name("not a type") == "not a type", "text that is not a decorated name is left as it is");
    FaultConfig fc;
    fc.first_chance_limit = 3;
    remove_fault_handlers();
    install_fault_handlers(fc);
    clear_out();
    throw_and_catch();
    throw_and_catch();
    const std::string t = all_text();
    check_in(has(t, "C++ throw of std::runtime_error (.?AVruntime_error@std@@)"), "a C++ throw is logged with the type read from its throw info", t);
    check_in(has(t, "C++ throw of (anonymous namespace)::Widget (.?AUWidget@?A0x"), "a second throw type is read too", t);
    check(count_of(t, "first-chance exception") == 3, "LEONIDA_FIRST_CHANCE=3 logs three of the four throws");
    remove_fault_handlers();
    clear_out();
    throw_and_catch();
    check(all_text().empty(), "with first-chance logging off a handled throw says nothing");
}

void test_breakpoint_restore_failure() {
    auto* page = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    check(page != nullptr, "allocate a page for the breakpoint restore failure");
    if (!page) return;
    *page = 0x90;
    const bool patched = add_breakpoint(page, "restore-failure");
    check(patched, "install a breakpoint before decommitting its page");
    if (!patched) {
        VirtualFree(page, 0, MEM_RELEASE);
        return;
    }
    const bool decommitted = VirtualFree(page, 4096, MEM_DECOMMIT) != FALSE;
    check(decommitted, "decommit the page to make VirtualProtect fail");
    if (!decommitted) return;
    // Keep the address reserved until process exit since the breakpoint table retains it.
    EXCEPTION_RECORD record{};
    record.ExceptionCode = EXCEPTION_BREAKPOINT;
    record.ExceptionAddress = page;
    CONTEXT context{};
    EXCEPTION_POINTERS info{&record, &context};
    clear_out();
    const LONG first = detail::veh_entry(&info);
    const LONG next = detail::veh_entry(&info);
    detail::rearm = nullptr;
    check(first == EXCEPTION_CONTINUE_SEARCH && next == EXCEPTION_CONTINUE_SEARCH &&
              !(context.EFlags & 0x100) && breakpoint_hits(page) == 0,
          "failed byte restoration passes the exception on without stepping or counting a hit");
    check(count_of(all_text(), "restore-failure disabled") == 1, "the breakpoint failure is logged once");
}

void test_handler_error_state() {
    EXCEPTION_RECORD record{};
    record.ExceptionCode = kThreadNameException;
    record.NumberParameters = 3;
    record.ExceptionInformation[0] = 0x1000;
    record.ExceptionInformation[1] = 1;  // unreadable name
    record.ExceptionInformation[2] = static_cast<DWORD>(-1);
    CONTEXT context{};
    EXCEPTION_POINTERS info{&record, &context};
    constexpr DWORD sentinel = 0x12345678;
    errno = EDOM;
    SetLastError(sentinel);
    const LONG result = detail::veh_entry(&info);
    const DWORD last_error = GetLastError();
    const int crt_error = errno;
    check(result == EXCEPTION_CONTINUE_SEARCH && last_error == sentinel && crt_error == EDOM,
          "a failed diagnostic read preserves the guest thread error state");

    bool reported = false;
    set_sink([](void* data, const std::string&) {
        *static_cast<bool*>(data) = true;
        errno = ERANGE;
        SetLastError(ERROR_INVALID_DATA);
    }, &reported);
    record.ExceptionCode = EXCEPTION_ACCESS_VIOLATION;
    record.NumberParameters = 2;
    errno = EDOM;
    SetLastError(sentinel);
    detail::veh_entry(&info);
    const DWORD reported_last_error = GetLastError();
    const int reported_errno = errno;
    set_sink(&collect, nullptr);
    check(reported && reported_last_error == sentinel && reported_errno == EDOM,
          "fault reporting preserves Win32 and CRT errors even if the sink changes both");
}

void test_environment_numbers() {
    struct Variable {
        const char* name;
        const char* invalid;
        const char* valid;
        std::vector<char> saved;
    };
    Variable vars[] = {
        {"LEONIDA_DIAG", "nan", "0", {}},
        {"LEONIDA_FAULTS", "2147483648", "2147483647", {}},
        {"LEONIDA_FIRST_CHANCE", "1.5", "12", {}},
        {"LEONIDA_WATCHDOG", "inf", "0.25", {}},
        {"LEONIDA_THREAD_DUMP", "18446744073709551616", "18446744073709551615", {}},
    };
    for (auto& var : vars) {
        const DWORD size = GetEnvironmentVariableA(var.name, nullptr, 0);
        var.saved.resize(size);
        if (size) GetEnvironmentVariableA(var.name, var.saved.data(), size);
        SetEnvironmentVariableA(var.name, var.invalid);
    }
    const auto invalid = settings_from_environment();
    check(invalid.problems.size() == 5 && invalid.enabled && invalid.faults.fault_reports == 8 &&
              invalid.faults.first_chance_limit == 0 && invalid.watchdog.timeout_seconds == 0 &&
              invalid.watchdog.dump_at_progress == 0,
          "non-finite, fractional and overflowing values are diagnosed and leave defaults");

    for (const auto& var : vars) SetEnvironmentVariableA(var.name, var.valid);
    const auto valid = settings_from_environment();
    check(valid.problems.empty() && !valid.enabled && valid.faults.fault_reports == INT_MAX &&
              valid.faults.first_chance_limit == 12 && valid.watchdog.timeout_seconds == 0.25 &&
              valid.watchdog.dump_at_progress == UINT64_MAX,
          "valid limits and fractional watchdog seconds are accepted exactly");

    SetEnvironmentVariableA("LEONIDA_FAULTS", "12tail");
    SetEnvironmentVariableA("LEONIDA_THREAD_DUMP", " -1");
    const auto malformed = settings_from_environment();
    check(malformed.problems.size() == 2 && malformed.faults.fault_reports == 8 &&
              malformed.watchdog.dump_at_progress == 0,
          "trailing characters and negative frame counts are diagnosed");
    for (const auto& var : vars)
        SetEnvironmentVariableA(var.name, var.saved.empty() ? nullptr : var.saved.data());
}

#ifdef LEONIDA_TEST_PRESENT_PROGRESS
void test_present_progress() {
    rt::d3d12::XboxPresentation presentation(std::make_unique<rt::d3d12::WindowPresenter>(nullptr, nullptr));
    const auto before = detail::wd().progress.load();
    check(presentation.PresentX(1, nullptr, nullptr) == E_INVALIDARG && detail::wd().progress.load() == before,
          "a rejected PresentX does not advance the watchdog");
    blank_present_result = E_FAIL;
    check(presentation.PresentX(0, nullptr, nullptr) == E_FAIL && detail::wd().progress.load() == before,
          "a failed backend present does not advance the watchdog");
    blank_present_result = S_FALSE;
    check(presentation.PresentX(0, nullptr, nullptr) == S_FALSE && detail::wd().progress.load() == before + 1,
          "a successful backend present advances the watchdog exactly once");
}
#endif

void test_breakpoints(World& w) {
    FaultConfig fc;
    install_fault_handlers(fc);
    const auto resolve = [](const std::string& module, uint8_t*& base, uint32_t& size) {
        if (!module.empty() && module != "synth.dll") return false;
        base = g_image;
        size = kImageSize;
        return true;
    };
    clear_out();
    const std::string list = format("%x,synth.dll+%x,zz,nosuch.dll+10,%x", w.ret7.rva, w.ret7.rva + 0, kImageSize + 4);
    const auto problems = set_breakpoints_from_text(list, resolve);
    check(problems.size() == 3, format("bad breakpoint items are each reported (%zu of 3)", problems.size()));
    const auto fn = reinterpret_cast<int (*)()>(w.ret7.at());
    int sum = 0;
    for (int i = 0; i < 3; ++i) sum += fn();
    check(sum == 21, "the function still returns 7 each time it hits a breakpoint");
    check(breakpoint_hits(w.ret7.at()) == 3, "three calls are three hits");
    std::string t = all_text();
    check(count_of(t, "breakpoint ") == 3 && has(t, "rax=") && has(t, "return address host:"), "each hit logs the registers and the return address");
    check(breakpoint_hits(w.ret7_other.at()) == -1 && !has(t, hex_rva(w.ret7_other.rva)), "the identical function at another RVA has no breakpoint");
    clear_out();
    for (int i = 0; i < 45; ++i) sum += fn();
    t = all_text();
    check(count_of(t, "hit ") == 37 && breakpoint_hits(w.ret7.at()) == 48, "after 40 hits the rest are counted, not logged (40 logged in all)");
    check(has(t, "the limit"), "the line of the 40th hit says it is the last");
    check(w.ret7_other.at()[0] == 0xB8, "a byte that was not patched stays as it was");
    remove_fault_handlers();
}

void test_watchdog(World& w) {
    Runner waiter{w.waiter, "SynthWaiter"};
    Runner spinner{w.spinner, "SynthSpinner"};
    start_runner(waiter);
    start_runner(spinner);

    clear_out();
    WatchdogConfig cfg;
    cfg.timeout_seconds = 0.6;
    cfg.dumps_per_stall = 2;
    check(start_watchdog(cfg), "the watchdog starts");
    check(wait_for([] { return watchdog_dump_count() >= 2; }, 15000), "two dumps arrive without progress");
    Sleep(1800);
    check(watchdog_dump_count() == 2, "and no third one");
    std::string t = all_text();
    check(has(t, "(dump 1 of 2)") && has(t, "(dump 2 of 2)") && !has(t, "dump 3"), "the dumps are numbered");
    const std::string waiter_line = format("thread %lu \"SynthWaiter\" WAITING", GetThreadId(waiter.thread));
    const std::string spinner_line = format("thread %lu \"SynthSpinner\" RUNNING", GetThreadId(spinner.thread));
    check(has(t, waiter_line), "the blocked thread is WAITING and carries its name");
    check(has(t, spinner_line), "the spinning thread is RUNNING and carries its name");
    check(has(t, "image code " + hex_rva(w.waiter.after_call) + " called kernelbase.dll!WaitForSingleObject") ||
              has(t, "image code " + hex_rva(w.waiter.after_call) + " called kernel32.dll!WaitForSingleObject"),
          "the waiter's line names the import and the caller, " + hex_rva(w.waiter.after_call));
    check(has(t, "running image code at synth.dll+0x"), "the spinner's line says it is in image code");
    check_in(has(t, "started at host:boot_diag_test.exe+0x"), "a thread's start address is given as module+RVA", t);
    stop_watchdog();

    // Progress keeps it quiet.
    cfg.timeout_seconds = 0.5;
    clear_out();
    start_watchdog(cfg);
    for (int i = 0; i < 16; ++i) {
        progress();
        Sleep(100);
    }
    check(watchdog_dump_count() == 0 && all_text().empty(), "a game that keeps presenting is never dumped");
    check(wait_for([] { return watchdog_dump_count() >= 1; }, 15000), "when the frames stop the dump comes");
    stop_watchdog();

    // The dump at a chosen frame.
    WatchdogConfig at;
    at.dump_at_progress = 3;
    clear_out();
    start_watchdog(at);
    progress();
    progress();
    Sleep(500);
    check(watchdog_dump_count() == 0, "no dump before the chosen frame");
    progress();
    check(wait_for([] { return watchdog_dump_count() >= 1; }, 15000), "the dump comes at the chosen frame");
    progress();
    progress();
    Sleep(800);
    check(watchdog_dump_count() == 1 && has(all_text(), "frame 3 reached"), "and only once");
    stop_watchdog();

    // Setting the flag ends the spinner; setting the event ends the waiter.
    uint8_t one = 1;
    std::memcpy(g_image + w.flag, &one, 1);
    SetEvent(w.event);
    WaitForSingleObject(spinner.thread, 5000);
    WaitForSingleObject(waiter.thread, 5000);
    CloseHandle(spinner.thread);
    CloseHandle(waiter.thread);
}

// The child of the last check: a thread blocks in the image, the main thread faults and nobody handles it.
int unhandled_child(World& w) {
    SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
    install_fault_handlers(FaultConfig{});
    Runner waiter{w.waiter, "ChildWaiter"};
    start_runner(waiter);
    reinterpret_cast<void (*)()>(w.null_read.at())();
    return 0;
}

std::string run_child_capturing(const std::wstring& command, DWORD& exit_code) {
    HANDLE read = nullptr, write = nullptr;
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    CreatePipe(&read, &write, &sa, 0);
    SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = si.hStdError = write;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::wstring cmd = command;
    exit_code = 0xDEAD;
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return {};
    CloseHandle(write);
    std::string out;
    // Polled, not read blocking: Wine's debugger (started for an unhandled fault, and waiting for ever with a display)
    // inherits the pipe, so it would never close.
    const auto drain = [&] {
        char buf[4096];
        DWORD avail = 0, got = 0;
        while (PeekNamedPipe(read, nullptr, 0, nullptr, &avail, nullptr) && avail && ReadFile(read, buf, sizeof buf, &got, nullptr) && got)
            out.append(buf, got);
    };
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (WaitForSingleObject(pi.hProcess, 100) == WAIT_TIMEOUT && std::chrono::steady_clock::now() < end) drain();
    if (WaitForSingleObject(pi.hProcess, 0) == WAIT_TIMEOUT) {
        drain();
        TerminateProcess(pi.hProcess, 0xBAD);
        WaitForSingleObject(pi.hProcess, 5000);
    }
    drain();
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(read);
    return out;
}

void test_unhandled() {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    DWORD code = 0;
    const std::string out = run_child_capturing(std::wstring(L"\"") + exe + L"\" --unhandled", code);
    check(has(out, "[diag] UNHANDLED access violation (0xc0000005) at synth.dll+0x"), "the unhandled fault is reported by the top-level filter");
    check(has(out, "null pointer plus the field offset 0x18"), "with what kind of fault it was");
    check(has(out, "thread dump: an exception nobody handled") && has(out, "\"ChildWaiter\" WAITING") && has(out, "WaitForSingleObject"),
          "and every other thread is dumped with its name and the import it waits in");
    if (code != 0xBAD)  // 0xBAD: Wine's debugger kept the child, which the test ended
        check(code == 0xC0000005, format("the process still ends with the fault code (0x%lx)", code));
    else
        std::printf("note  the child was held by Wine's debugger and ended by the test; its exit code was not checked\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a hang or a kill must not lose the lines already printed
    g_image = static_cast<uint8_t*>(VirtualAlloc(nullptr, kImageSize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!g_image) return 2;
    const uint8_t unwind[8] = {0x01, 0x04, 0x01, 0x00, 0x04, 0x42, 0x00, 0x00};  // version 1, prolog 4, 1 code: alloc 0x28
    std::memcpy(g_image + kUnwindRva, unwind, sizeof unwind);
    rt::shims::add_module_lookup(nullptr, &lookup);
    static const rt::shims::Entry entries[] = {{"synth.dll", "Marker", reinterpret_cast<void*>(&shim_marker)}};
    static const rt::shims::Table table(entries, 1);
    World w = make_world();
    if (!RtlAddFunctionTable(w.b.table.data(), static_cast<DWORD>(w.b.table.size()), reinterpret_cast<DWORD64>(g_image))) return 2;

    if (argc > 1 && std::strcmp(argv[1], "--unhandled") == 0) return unhandled_child(w);

    set_sink(&collect, nullptr);
    FaultConfig fc;
    install_fault_handlers(fc);  // the handler also reads thread names
    test_callee_names(w);
    test_scan_buffer();
    test_names(w);
    test_faults(w);
    test_cpp_exceptions();
    remove_fault_handlers();
    install_fault_handlers(fc);
    test_handler_error_state();
    test_breakpoint_restore_failure();
    test_environment_numbers();
#ifdef LEONIDA_TEST_PRESENT_PROGRESS
    test_present_progress();
#endif
    remove_fault_handlers();
    test_breakpoints(w);
    remove_fault_handlers();
    install_fault_handlers(fc);
    test_watchdog(w);
    remove_fault_handlers();
    set_sink(nullptr, nullptr);
    test_unhandled();

    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}

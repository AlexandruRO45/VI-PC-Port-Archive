// First DLL of the lifecycle fixture, built with the static MSVC runtime. Its startup is the real one:
// _DllMainCRTStartup runs the global constructor before DllMain(DLL_PROCESS_ATTACH) and the global
// destructor after DllMain(DLL_PROCESS_DETACH). Nothing here calls the constructor or destructor by hand.
// NO_CTOR removes the global constructor (a control: the state stays zero and its event is missing).
#include <windows.h>

extern "C" __declspec(dllimport) void sink_log(const char* token);

namespace {
int g_value;
unsigned g_ctor_runs;
unsigned long long g_frames[16];  // return addresses on the stack inside the constructor
unsigned g_frame_count;

// Return addresses of the callers of the caller, found with the documented unwind functions. RtlCaptureStackBackTrace
// is not used: it stops after one frame when the image was not mapped by the loader, while these functions use
// the unwind data that RtlAddFunctionTable registered.
unsigned capture_callers(unsigned long long* out, unsigned capacity) {
    CONTEXT context;
    RtlCaptureContext(&context);
    unsigned count = 0;
    while (count < capacity) {
        DWORD64 image_base = 0;
        RUNTIME_FUNCTION* function = RtlLookupFunctionEntry(context.Rip, &image_base, nullptr);
        if (!function) break;
        void* handler_data = nullptr;
        DWORD64 establisher = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, context.Rip, function, &context, &handler_data, &establisher,
                         nullptr);
        if (!context.Rip) break;
        out[count++] = context.Rip;
    }
    return count;
}

struct State {
#ifndef NO_CTOR
    State() {
        ++g_ctor_runs;
        g_value = 7;
        g_frame_count = capture_callers(g_frames, 16);
        sink_log("A:ctor");
    }
#endif
    ~State() { sink_log("A:dtor"); }
} g_state;
}  // namespace

extern "C" __declspec(dllexport) int a_value() { return g_value; }
extern "C" __declspec(dllexport) void a_set(int value) { g_value = value; }
extern "C" __declspec(dllexport) unsigned a_ctor_runs() { return g_ctor_runs; }
// Copies the constructor's return addresses into out (16 entries) and returns how many there are.
extern "C" __declspec(dllexport) unsigned a_ctor_frames(unsigned long long* out) {
    for (unsigned i = 0; i < g_frame_count; ++i) out[i] = g_frames[i];
    return g_frame_count;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    // Only a call into the sink, which is already loaded and takes no lock: nothing that needs the loader.
    if (reason == DLL_PROCESS_ATTACH) sink_log("A:attach");
    else if (reason == DLL_PROCESS_DETACH) sink_log("A:detach");
    return TRUE;
}

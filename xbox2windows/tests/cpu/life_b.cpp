// Second DLL of the lifecycle fixture, with the static MSVC runtime. It imports A. Its global constructor
// reads A's state, so it shows whether A was initialized first; its destructor reads A's state again, so it
// shows whether A is still alive when B is destroyed. NO_DTOR removes the global destructor (a control).
#include <windows.h>

extern "C" __declspec(dllimport) void sink_log(const char* token);
extern "C" __declspec(dllimport) int a_value();

namespace {
int g_value;
unsigned long long g_frames[16];
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
    State() {
        g_value = a_value() * 2;
        g_frame_count = capture_callers(g_frames, 16);
        sink_log("B:ctor");
    }
#ifndef NO_DTOR
    ~State() {
        char token[] = "B:dtor a=?";
        token[9] = static_cast<char>('0' + (a_value() % 10 + 10) % 10);  // A's state, read from B's destructor
        sink_log(token);
    }
#endif
} g_state;
}  // namespace

extern "C" __declspec(dllexport) int b_value() { return g_value; }
extern "C" __declspec(dllexport) unsigned b_ctor_frames(unsigned long long* out) {
    for (unsigned i = 0; i < g_frame_count; ++i) out[i] = g_frames[i];
    return g_frame_count;
}
// Depends on A's state at the time of the call, not only on what B captured when it was constructed.
extern "C" __declspec(dllexport) int b_via_a() { return a_value() + g_value; }

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) sink_log("B:attach");
    else if (reason == DLL_PROCESS_DETACH) sink_log("B:detach");
    return TRUE;
}

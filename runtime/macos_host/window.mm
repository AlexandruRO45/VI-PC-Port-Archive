// The user32 subset a Vulkan guest needs for one window: a Cocoa window whose content view is backed by a
// CAMetalLayer (vulkan.cpp turns the Win32 surface into a Metal surface on it), and a message pump that runs AppKit.
// ponytail: no window procedure messages (WM_SIZE, input); closing the window posts WM_QUIT. AppKit only runs on the
// process's main thread, so the guest runs on another one (run_main_loop) and every AppKit call is sent to it.
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include "host.h"

#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace host {
namespace {

struct Window {
    NSWindow* window;
    CAMetalLayer* layer;
    uint32_t thread;       // the guest thread that created it
    bool visible = false;  // shown by ShowWindow or SetWindowPos, as Win32 counts it
};
bool g_quit = false;
std::mutex g_windows_lock;
std::vector<Window*> g_windows;  // top-level windows in creation order, for EnumWindows

void on_main(void (^block)()) {
    if (pthread_main_np()) block();
    else dispatch_sync(dispatch_get_main_queue(), block);
}

void start_app() {
    static bool started = false;
    if (started) return;
    started = true;
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp finishLaunching];
    [NSApp activateIgnoringOtherApps:YES];
}

std::string narrow(const char16_t* s) {
    std::string out;  // ponytail: ASCII titles only
    for (; s && *s; ++s) out.push_back(static_cast<char>(*s));
    return out;
}

uint16_t RegisterClassExW(const void*) { return 1; }  // the class only carries the window procedure, unused
uint16_t RegisterClassExA(const void*) { return 1; }
void* create_window(const std::string& title, int32_t w, int32_t h) {
    if (w <= 0 || w == INT32_MIN) w = 1280;  // CW_USEDEFAULT
    if (h <= 0 || h == INT32_MIN) h = 720;
    __block Window* made = nullptr;
    on_main(^{ @autoreleasepool {
        start_app();
        // Win32 sizes are pixels; AppKit's are points, two pixels each on a Retina screen.
        CGFloat pts = [[NSScreen mainScreen] backingScaleFactor];
        NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(200, 200, w / pts, h / pts)
                                                    styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                              NSWindowStyleMaskMiniaturizable
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
        [win setReleasedWhenClosed:NO];
        [win setTitle:[NSString stringWithUTF8String:title.c_str()]];
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.contentsScale = [win backingScaleFactor];
        layer.drawableSize = CGSizeMake(w, h);
        NSView* view = [win contentView];
        [view setLayer:layer];
        [view setWantsLayer:YES];
        [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification
                                                          object:win
                                                           queue:nil
                                                      usingBlock:^(NSNotification*) { g_quit = true; }];
        made = new Window{win, layer, 0};
    }});
    made->thread = thread_id();
    std::lock_guard<std::mutex> lock(g_windows_lock);
    g_windows.push_back(made);
    return made;
}
void* CreateWindowExW(uint32_t, const char16_t*, const char16_t* title, uint32_t, int32_t, int32_t, int32_t w,
                      int32_t h, void*, void*, void*, void*) {
    return create_window(narrow(title), w, h);
}
void* CreateWindowExA(uint32_t, const char*, const char* title, uint32_t, int32_t, int32_t, int32_t w, int32_t h,
                      void*, void*, void*, void*) {
    return create_window(title ? title : "", w, h);
}
int ShowWindow(void* hwnd, int32_t cmd) {
    static_cast<Window*>(hwnd)->visible = cmd != 0;  // SW_HIDE
    on_main(^{ [static_cast<Window*>(hwnd)->window makeKeyAndOrderFront:nil]; });
    return 0;
}
int DestroyWindow(void* hwnd) {
    auto* w = static_cast<Window*>(hwnd);
    on_main(^{ [w->window orderOut:nil]; });
    std::lock_guard<std::mutex> lock(g_windows_lock);
    g_windows.erase(std::remove(g_windows.begin(), g_windows.end(), w), g_windows.end());
    return 1;
}
struct Msg {  // MSG
    void* hwnd;
    uint32_t message;
    uint64_t wparam;
    int64_t lparam;
    uint32_t time;
    int32_t x, y;
    uint32_t private_;
};
int PeekMessageW(Msg* msg, void*, uint32_t, uint32_t, uint32_t) {
    on_main(^{ @autoreleasepool {
        start_app();
        while (NSEvent* e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:[NSDate distantPast]
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES])
            [NSApp sendEvent:e];
    }});
    if (!g_quit) return 0;
    *msg = {};
    msg->message = 0x0012;  // WM_QUIT
    return 1;
}
void* LoadCursorW(void*, const char16_t*) { return reinterpret_cast<void*>(1); }  // the cursor is AppKit's
int TranslateMessage(const Msg*) { return 0; }
int64_t DispatchMessageW(const Msg*) { return 0; }
int64_t DefWindowProcW(void*, uint32_t, uint64_t, int64_t) { return 0; }
// The ANSI forms differ only in string arguments, which these never read.
int PeekMessageA(Msg* msg, void* hwnd, uint32_t lo, uint32_t hi, uint32_t remove) { return PeekMessageW(msg, hwnd, lo, hi, remove); }
int64_t DispatchMessageA(const Msg*) { return 0; }
int64_t DefWindowProcA(void*, uint32_t, uint64_t, int64_t) { return 0; }
int GetClientRect(void* hwnd, int32_t* rect) {
    CGSize px = static_cast<Window*>(hwnd)->layer.drawableSize;  // pixels, as Win32 counts
    rect[0] = rect[1] = 0;
    rect[2] = static_cast<int32_t>(px.width);
    rect[3] = static_cast<int32_t>(px.height);
    return 1;
}

// Window and client rect are the same here: CreateWindowExW and SetWindowPos size the content view, AppKit adds
// the title bar outside it.
uint16_t RegisterClassW(const void*) { return 1; }
int UnregisterClassW(const char16_t*, void*) { return 1; }
int AdjustWindowRect(int32_t*, uint32_t, int) { return 1; }
int AdjustWindowRectEx(int32_t*, uint32_t, int, uint32_t) { return 1; }
int32_t GetWindowLongW(void*, int32_t index) { return index == -16 ? 0x00CF0000 : 0; }  // GWL_STYLE: WS_OVERLAPPEDWINDOW
int SetWindowPos(void* hwnd, void*, int32_t, int32_t, int32_t cx, int32_t cy, uint32_t flags) {
    auto* w = static_cast<Window*>(hwnd);
    on_main(^{ @autoreleasepool {
        if (!(flags & 0x1)) {  // SWP_NOSIZE
            CGFloat scale = [w->window backingScaleFactor];
            [w->window setContentSize:NSMakeSize(cx / scale, cy / scale)];
            w->layer.drawableSize = CGSizeMake(cx, cy);
        }
        if (flags & 0x40) w->visible = true, [w->window makeKeyAndOrderFront:nil];  // SWP_SHOWWINDOW
    }});
    return 1;
}
int UpdateWindow(void*) { return 1; }
int IsIconic(void* hwnd) {
    __block int iconic = 0;
    on_main(^{ iconic = [static_cast<Window*>(hwnd)->window isMiniaturized]; });
    return iconic;
}
// ponytail: no foreground window, so tests skip reading the screen back through GDI (GetPixel) and DXGI outputs.
void* GetForegroundWindow() { return nullptr; }
void* GetProcessWindowStation() { return nullptr; }  // callers treat "unknown" as interactive
void* MonitorFromWindow(void*, uint32_t) { return reinterpret_cast<void*>(1); }
// What finding a process's main window reads: no owners, no tool windows, every window this process's.
int EnumWindows(int(MSABI* proc)(void*, int64_t), int64_t data) {
    std::vector<Window*> windows;
    {
        std::lock_guard<std::mutex> lock(g_windows_lock);
        windows = g_windows;
    }
    for (Window* w : windows)
        if (!call_guest(proc, static_cast<void*>(w), data)) break;
    return 1;
}
uint32_t GetWindowThreadProcessId(void* hwnd, uint32_t* pid) {
    if (pid) *pid = static_cast<uint32_t>(getpid());
    return static_cast<Window*>(hwnd)->thread;
}
int IsWindowVisible(void* hwnd) { return static_cast<Window*>(hwnd)->visible; }
void* GetWindow(void*, uint32_t) { return nullptr; }
int64_t GetWindowLongPtrW(void* hwnd, int32_t index) { return GetWindowLongW(hwnd, index); }

// One monitor, the main screen, its rect in pixels.
int EnumDisplayMonitors(void*, const int32_t*, int(MSABI* proc)(void*, void*, int32_t*, int64_t), int64_t data) {
    __block CGSize px;
    on_main(^{
        NSScreen* screen = [NSScreen mainScreen];
        px = CGSizeMake(screen.frame.size.width * screen.backingScaleFactor,
                        screen.frame.size.height * screen.backingScaleFactor);
    });
    int32_t rect[4] = {0, 0, static_cast<int32_t>(px.width), static_cast<int32_t>(px.height)};
    call_guest(proc, reinterpret_cast<void*>(1), static_cast<void*>(nullptr), rect, data);
    return 1;
}
int GetMonitorInfoW(void*, void*) { return 0; }  // no monitor details: the SDR white level falls back to its default

}  // namespace

// The main thread waits here for AppKit work from guest threads; the process ends when the guest exits.
[[noreturn]] void run_main_loop() {
    CFRunLoopAddTimer(CFRunLoopGetMain(), CFRunLoopTimerCreate(nullptr, 1e18, 0, 0, 0, nullptr, nullptr),
                      kCFRunLoopCommonModes);  // a run loop with no source returns at once
    for (;;) CFRunLoopRun();
}
void* metal_layer_for_window(void* hwnd) { return hwnd ? (__bridge void*)static_cast<Window*>(hwnd)->layer : nullptr; }

#define U(name) EXPORT("user32.dll", name)
Exports user32_exports() {
    return {U(RegisterClassExW), U(CreateWindowExW), U(ShowWindow), U(DestroyWindow), U(PeekMessageW),
            U(TranslateMessage), U(DispatchMessageW), U(DefWindowProcW), U(GetClientRect), U(LoadCursorW),
            U(RegisterClassW), U(UnregisterClassW), U(AdjustWindowRect), U(AdjustWindowRectEx), U(GetWindowLongW),
            U(SetWindowPos), U(UpdateWindow), U(IsIconic), U(GetForegroundWindow), U(GetProcessWindowStation),
            U(MonitorFromWindow), U(GetMonitorInfoW), U(RegisterClassExA), U(CreateWindowExA), U(PeekMessageA),
            U(DispatchMessageA), U(DefWindowProcA), U(EnumDisplayMonitors), U(EnumWindows),
            U(GetWindowThreadProcessId), U(IsWindowVisible), U(GetWindow), U(GetWindowLongPtrW)};
}

}  // namespace host

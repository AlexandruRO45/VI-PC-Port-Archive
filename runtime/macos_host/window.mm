// The user32 subset a Vulkan guest needs for one window: a Cocoa window whose content view is backed by a
// CAMetalLayer (vulkan.cpp turns the Win32 surface into a Metal surface on it), and a message pump that runs AppKit.
// ponytail: no window procedure messages (WM_SIZE, input); closing the window posts WM_QUIT. Main thread only.
#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#include "host.h"

namespace host {
namespace {

struct Window {
    NSWindow* window;
    CAMetalLayer* layer;
};
bool g_quit = false;

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
void* CreateWindowExW(uint32_t, const char16_t*, const char16_t* title, uint32_t, int32_t, int32_t, int32_t w,
                      int32_t h, void*, void*, void*, void*) {
    @autoreleasepool {
        start_app();
        if (w <= 0 || w == INT32_MIN) w = 1280;  // CW_USEDEFAULT
        if (h <= 0 || h == INT32_MIN) h = 720;
        NSWindow* win = [[NSWindow alloc] initWithContentRect:NSMakeRect(200, 200, w, h)
                                                    styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                                              NSWindowStyleMaskMiniaturizable
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
        [win setReleasedWhenClosed:NO];
        [win setTitle:[NSString stringWithUTF8String:narrow(title).c_str()]];
        CAMetalLayer* layer = [CAMetalLayer layer];
        CGFloat scale = [win backingScaleFactor];
        layer.contentsScale = scale;
        layer.drawableSize = CGSizeMake(w * scale, h * scale);
        NSView* view = [win contentView];
        [view setLayer:layer];
        [view setWantsLayer:YES];
        [[NSNotificationCenter defaultCenter] addObserverForName:NSWindowWillCloseNotification
                                                          object:win
                                                           queue:nil
                                                      usingBlock:^(NSNotification*) { g_quit = true; }];
        return new Window{win, layer};
    }
}
int ShowWindow(void* hwnd, int32_t) {
    [static_cast<Window*>(hwnd)->window makeKeyAndOrderFront:nil];
    return 0;
}
int DestroyWindow(void* hwnd) {
    auto* w = static_cast<Window*>(hwnd);
    [w->window orderOut:nil];
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
    @autoreleasepool {
        start_app();
        while (NSEvent* e = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:[NSDate distantPast]
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES])
            [NSApp sendEvent:e];
    }
    if (!g_quit) return 0;
    *msg = {};
    msg->message = 0x0012;  // WM_QUIT
    return 1;
}
int TranslateMessage(const Msg*) { return 0; }
int64_t DispatchMessageW(const Msg*) { return 0; }
int64_t DefWindowProcW(void*, uint32_t, uint64_t, int64_t) { return 0; }
int GetClientRect(void* hwnd, int32_t* rect) {
    NSRect r = [[static_cast<Window*>(hwnd)->window contentView] bounds];
    rect[0] = rect[1] = 0;
    rect[2] = static_cast<int32_t>(r.size.width);
    rect[3] = static_cast<int32_t>(r.size.height);
    return 1;
}

}  // namespace

void* metal_layer_for_window(void* hwnd) { return hwnd ? (__bridge void*)static_cast<Window*>(hwnd)->layer : nullptr; }

#define U(name) EXPORT("user32.dll", name)
Exports user32_exports() {
    return {U(RegisterClassExW), U(CreateWindowExW), U(ShowWindow), U(DestroyWindow), U(PeekMessageW),
            U(TranslateMessage), U(DispatchMessageW), U(DefWindowProcW), U(GetClientRect)};
}

}  // namespace host

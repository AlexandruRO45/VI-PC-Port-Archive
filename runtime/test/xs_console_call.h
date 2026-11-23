// Calls on the Xbox-ABI proxies by console slot (runtime/shims/d3d12_xs_console_slots.h), for tests that check the
// layer behind a proxy. Past device slot 42 and queue slot 10 the proxies follow the console order, so a call through
// the PC header's method would reach the console method at that index instead.
#pragma once

#include "../shims/d3d12_xs_console_slots.h"
#include "../shims/d3d12_xs_device.h"

#include <windows.h>

namespace xs_test {

// The method at console slot "slot" of a proxy, called with the arguments as the console title calls it.
template <class R = HRESULT, class... A>
R console_call(void* object, unsigned slot, A... args) {
    using Fn = R(WINAPI*)(void*, A...);
    return reinterpret_cast<Fn>((*static_cast<void***>(object))[slot])(object, args...);
}

// The console slot of a public ID3D12Device1..8 method, from its PC slot.
constexpr unsigned device_slot(unsigned pc_slot) { return rt::d3d12_xs::console::device_console_slot(pc_slot); }

// An ID3D12Device1..8 method by its PC slot: called there on a real device, at its console slot on a device proxy.
template <class... A>
HRESULT device_method(void* device, unsigned pc_slot, A... args) {
    const bool proxy = rt::d3d12_xs::device_proxy_real_device(device) != nullptr;
    return console_call(device, proxy ? device_slot(pc_slot) : pc_slot, args...);
}

// The object a queue or device proxy wraps, or the object itself when it is no proxy: for the PC methods the console
// does not have (the queue's GetTimestampFrequency, GetClockCalibration; ID3D12Device9 and later).
inline ID3D12CommandQueue* pc_queue(ID3D12CommandQueue* queue) {
    ID3D12CommandQueue* real = rt::d3d12_xs::queue_proxy_real_queue(queue);
    return real ? real : queue;
}

inline ID3D12Device* pc_device(ID3D12Device* device) {
    ID3D12Device* real = rt::d3d12_xs::device_proxy_real_device(device);
    return real ? real : device;
}

}  // namespace xs_test

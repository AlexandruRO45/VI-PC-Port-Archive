// The console order of the D3D12.X device and command-queue vtables (Xbox Series), as far as it is known.
//
// The Xbox device keeps the public ID3D12Device order for slots 0-43, then has its own methods at 44-82, and only
// then the public ID3D12Device1..5 (83-103) and ID3D12Device6..8 (105-112) methods; 104 is a console method. A PC
// device has Device1..14 at 44-82, so a proxy that forwards every slot in PC order sends the console methods to the
// wrong PC methods (#6498 shows slot 78 reaching CreateReservedResource2).
//
// Sources, and how far each one goes:
//   - Measured by Leonida on the package's own DLLs (docs/d3d12_vulkan.md section 7, d3d12_command_objects.md
//     section 3.5a): device slots 1, 14, 15, 16, 18, 19, 27, 38 in PC order; 62 (compute pipeline with two extra
//     words), 67 (compute pipeline from native code) and 76 (resource creation) are console methods. All three lie
//     inside 44-82, as the console block predicts, and none of the package's calls lands on a PC Device1+ method.
//   - sammwyy/xvalkyrie (GPL-3.0) at commit 53af7df77c10, src/runtime/modules/d3d12x/device.cpp:4196-4262 and
//     docs/research/d3d12x.md "Method order": the names and argument shapes of 43, 44, 48, 49-54, 55, 58, 60, 61,
//     68, 69, 72-75, 78-81, the Device1..8 positions, and the two queue layouts (device.cpp:2036-2186). Measured there
//     on other titles and on kit headers, not on this game. Only facts are taken; this table is Leonida's own.
//   - XCL's frame.rs and ebiten's ID3D12Device binding (PR #6522) number 78-81 the same way, independently.
// Where these disagree with Leonida's measurements, Leonida's are kept: 67 is ours (xvalkyrie leaves it unknown).
//
// 48, 73 and 78-80 are answered by the public HLE routes of the device proxy (#6935) before this table is read; the
// table refuses them only when LEONIDA_XS_*_SLOT has moved a route elsewhere.
//
// A slot the table does not name is reported with its index, its caller and its first argument words, and refused
// with E_NOTIMPL; it never reaches a PC method.
#pragma once

#include <cstdint>

namespace rt::d3d12_xs::console {

inline constexpr unsigned kNoPcSlot = ~0u;

enum class DeviceMethod : std::uint8_t {
    pc,        // the public method at pc_slot, forwarded
    handled,   // a console method the proxy implements
    accepted,  // a console method with no effect on a PC or Vulkan device (driver hints, callbacks): S_OK, counted
    refused,   // a named console method Leonida does not implement: E_NOTIMPL, reported
    unknown,   // no source names it: E_NOTIMPL, reported
};

struct DeviceSlot {
    const char* name;
    DeviceMethod kind;
    unsigned pc_slot;  // for kind == pc
};

// One past the last device slot the console order is known for (ID3D12Device8's last method).
inline constexpr unsigned kKnownDeviceSlots = 113;

constexpr DeviceSlot device_slot(unsigned slot) {
    using M = DeviceMethod;
    if (slot < 43) return {"public ID3D12Device method", M::pc, slot};
    if (slot >= 83 && slot <= 103) return {"public ID3D12Device1-5 method", M::pc, slot - 39};
    if (slot >= 105 && slot <= 112) return {"public ID3D12Device6-8 method", M::pc, slot - 40};
    switch (slot) {
        case 43: return {"GetAdapterLuid (LUID in RAX)", M::handled, kNoPcSlot};
        case 44: return {"CreatePlacedResourceX", M::handled, kNoPcSlot};
        case 48: return {"CreateCommandQueueX", M::refused, kNoPcSlot};
        case 49: case 50: case 51: case 52: case 53: case 54: case 58: case 69: case 72:
            return {"console driver hint or debug callback", M::accepted, kNoPcSlot};
        case 55: return {"CreateDerivedGraphicsPipelineStateX", M::refused, kNoPcSlot};
        case 60: return {"CreateGraphicsPipelineStateX", M::handled, kNoPcSlot};
        case 61: return {"CreateSamplerX", M::handled, kNoPcSlot};
        case 62: return {"CreateComputePipelineStateX", M::handled, kNoPcSlot};
        case 67: return {"DeserializeComputePipelineStateX", M::handled, kNoPcSlot};
        case 68: return {"GetGpuHardwareConfigurationX", M::handled, kNoPcSlot};
        case 73: return {"CreateCommandListX", M::refused, kNoPcSlot};
        case 74: return {"RegisterPagePoolX", M::handled, kNoPcSlot};
        case 75: return {"UnregisterPagePoolX", M::handled, kNoPcSlot};
        case 76: return {"CreateCommittedResourceX", M::handled, kNoPcSlot};
        case 78: return {"SetFrameIntervalX", M::refused, kNoPcSlot};
        case 79: return {"ScheduleFrameEventX", M::refused, kNoPcSlot};
        case 80: return {"WaitFrameEventX", M::refused, kNoPcSlot};
        case 81: return {"GetFrameStatisticsX", M::handled, kNoPcSlot};
        default: break;
    }
    if (slot < kKnownDeviceSlots) return {"console method no source names", M::unknown, kNoPcSlot};
    return {"past the known console order (ID3D12Device8)", M::unknown, kNoPcSlot};
}

// The console slot of a public device method (the inverse of device_slot for kind pc), or kNoPcSlot when the console
// order does not have it (GetAdapterLuid, which the console returns differently, and ID3D12Device9 and later).
constexpr unsigned device_console_slot(unsigned pc_slot) {
    if (pc_slot < 43) return pc_slot;
    if (pc_slot >= 44 && pc_slot <= 64) return pc_slot + 39;
    if (pc_slot >= 65 && pc_slot <= 72) return pc_slot + 40;
    return kNoPcSlot;
}

// ---- Command queue ----------------------------------------------------------------------------------------------
//
// Two layouts are known. Both keep the public order up to ExecuteCommandLists (10).
//   kit2020: SetMarker, BeginEvent, EndEvent, Signal (14), Wait (15) as on the PC; then two PIX capture methods
//            (16, 17), five console methods among which SuspendX and ResumeX (18-22, order not known), SignalX (23:
//            a fence, a value and flush flags), 24 and 25, GetDesc (26), three page-mapping methods (27-29), PresentX
//            (30) and SetHDRToneMapperX (31).
//   later:   two more methods before Signal (which two is not known), so 11-15 are markers or those two, and every
//            method from Signal on is two slots later: Signal 16, Wait 17, SignalX 25, GetDesc 28, PresentX 32.
// The PC's GetTimestampFrequency, GetClockCalibration and GetDesc at 16-18 are in neither layout.
//
// The proxy tells the two apart from the calls: Signal, Wait and SignalX take a fence, and the device proxy hands out
// every fence as a fence proxy, so a fence proxy as the first argument at 14, 15 or 23 means kit2020 and at 16, 17 or
// 25 means later. A call at 32 or 33 means later (kit2020 has nothing there). The public PresentX positions agree
// (30 in the older runtime, 32 in the newer: PR #6522, from XCL).
enum class QueueLayout : int { undecided = 0, kit2020 = 1, later = 2 };

enum class QueueMethod : std::uint8_t {
    pc,         // the public method at pc_slot, forwarded
    signal_x,   // SignalX(fence, value, flags): the public Signal at pc_slot
    get_desc,   // GetDesc: the public GetDesc at pc_slot
    present_x,  // PresentX(plane count, planes, params)
    accepted,   // no effect here: S_OK, counted
    refused,    // named, not implemented: E_NOTIMPL, reported
    unknown,    // not named: E_NOTIMPL, reported
    undecided,  // means different things in the two layouts, and the layout is not known yet: S_OK, reported
};

struct QueueSlot {
    const char* name;
    QueueMethod kind;
    unsigned pc_slot;
};

inline constexpr unsigned kQueueSignalKit2020 = 14, kQueueWaitKit2020 = 15, kQueueSignalXKit2020 = 23;
inline constexpr unsigned kQueueLaterShift = 2;

constexpr QueueSlot queue_slot_kit2020(unsigned slot) {
    using M = QueueMethod;
    if (slot <= 15) return {"public ID3D12CommandQueue method", M::pc, slot};
    if (slot == 16 || slot == 17) return {"PIX capture method", M::accepted, kNoPcSlot};
    if (slot >= 18 && slot <= 22) return {"SuspendX, ResumeX or another console method", M::accepted, kNoPcSlot};
    if (slot == 23) return {"SignalX", M::signal_x, 14};
    if (slot == 24 || slot == 25) return {"console queue method", M::accepted, kNoPcSlot};
    if (slot == 26) return {"GetDesc", M::get_desc, 18};
    if (slot >= 27 && slot <= 29) return {"page-mapping method", M::refused, kNoPcSlot};
    if (slot == 30) return {"PresentX", M::present_x, kNoPcSlot};
    if (slot == 31) return {"SetHDRToneMapperX", M::accepted, kNoPcSlot};
    return {"queue method past the known layout", M::unknown, kNoPcSlot};
}

constexpr QueueSlot queue_slot(QueueLayout layout, unsigned slot) {
    using M = QueueMethod;
    if (slot <= 10) return {"public ID3D12CommandQueue method", M::pc, slot};
    if (layout == QueueLayout::kit2020) return queue_slot_kit2020(slot);
    if (layout == QueueLayout::later) {
        if (slot <= 15) return {"event marker or one of the later kit's two new methods", M::accepted, kNoPcSlot};
        return queue_slot_kit2020(slot - kQueueLaterShift);
    }
    if (slot >= 34) return {"queue method past the known layouts", M::unknown, kNoPcSlot};
    return {"slot whose meaning depends on the queue layout", M::undecided, kNoPcSlot};
}

// The layout a call proves when its first argument is a fence proxy (undecided for any other slot).
constexpr QueueLayout layout_from_fence_call(unsigned slot) {
    if (slot == kQueueSignalKit2020 || slot == kQueueWaitKit2020 || slot == kQueueSignalXKit2020)
        return QueueLayout::kit2020;
    if (slot == kQueueSignalKit2020 + kQueueLaterShift || slot == kQueueWaitKit2020 + kQueueLaterShift ||
        slot == kQueueSignalXKit2020 + kQueueLaterShift)
        return QueueLayout::later;
    return QueueLayout::undecided;
}

// The layout a call at this slot proves whatever its arguments: only the later layout has methods at 32 and 33.
constexpr QueueLayout layout_from_slot(unsigned slot) {
    return slot == 32 || slot == 33 ? QueueLayout::later : QueueLayout::undecided;
}

}  // namespace rt::d3d12_xs::console

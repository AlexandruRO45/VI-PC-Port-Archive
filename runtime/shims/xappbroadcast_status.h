#pragma once

struct XAppBroadcastStatus {
    bool canStartBroadcast;
    bool isAnyAppBroadcasting;
    bool isCaptureResourceUnavailable;
    bool isGameStreamInProgress;
    bool isGpuConstrained;
    bool isAppInactive;
    bool isBlockedForApp;
    bool isDisabledByUser;
    bool isDisabledBySystem;
};

static_assert(sizeof(XAppBroadcastStatus) == 9, "XAppBroadcastStatus is nine bools");

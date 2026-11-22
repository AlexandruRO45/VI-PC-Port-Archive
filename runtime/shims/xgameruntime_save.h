// XGameSaveImpl: the GDK's game-save runtime class, offline and disk-backed (xgameruntime_save.cpp).
#pragma once

#include <windows.h>

namespace rt::xgr {

// QueryApiImpl's answer for XGameSaveImpl. Returns false when class_id is not XGameSaveImpl's (QueryApiImpl goes on
// with its other classes); otherwise true, with *result the QueryInterface answer for interface_id and *out the object
// or null.
bool query_save(const GUID* class_id, REFIID interface_id, void** out, HRESULT* result);

}  // namespace rt::xgr

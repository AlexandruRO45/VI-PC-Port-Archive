// Calls function(x, y, trace) and returns the code of the structured exception it raises, or 0 after
// storing its return value. The CPU tests use it to compare the fault of an original function with
// the fault of its translation.
#include <windows.h>

#include <cstdint>

using word = std::uint64_t;

extern "C" __declspec(dllexport) std::uint32_t guarded_call(word (*function)(word, word, word*), word x, word y,
                                                           word* trace, word* result) {
    __try {
        *result = function(x, y, trace);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return GetExceptionCode();
    }
    return 0;
}

namespace {
int capture(EXCEPTION_POINTERS* pointers, word* detail) {
    const EXCEPTION_RECORD* record = pointers->ExceptionRecord;
    detail[0] = record->NumberParameters >= 1 ? record->ExceptionInformation[0] : ~0ull;
    detail[1] = record->NumberParameters >= 2 ? record->ExceptionInformation[1] : ~0ull;
    return EXCEPTION_EXECUTE_HANDLER;
}
}  // namespace

// As guarded_call, and for an exception also stores its first two parameters in detail: for an access
// violation, the access type (0 read, 1 write, 8 execute) and the inaccessible address.
extern "C" __declspec(dllexport) std::uint32_t guarded_call_detail(word (*function)(word, word, word*), word x,
                                                                  word y, word* trace, word* result, word* detail) {
    detail[0] = detail[1] = 0;
    __try {
        *result = function(x, y, trace);
    } __except (capture(GetExceptionInformation(), detail)) {
        return GetExceptionCode();
    }
    return 0;
}

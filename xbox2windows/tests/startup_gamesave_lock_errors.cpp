// Experimental (docs/startup-runtime.md): which failed Win32 lock requests mean "held elsewhere, try
// again later" (busy) and which are real errors. The mapping is plain integer logic, so it runs on
// every platform with the codes a Windows node reports. Synthetic data only.
#include <startup_stubs/file_lock.h>

#include <cstdio>

namespace {
int failures;
void expect(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main()
{
    using namespace x2w_save;
    expect(win_lock_error_is_busy(33), "ERROR_LOCK_VIOLATION means held elsewhere");
    expect(win_lock_error_is_busy(32), "ERROR_SHARING_VIOLATION means held elsewhere");
    expect(win_lock_error_is_busy(997), "ERROR_IO_PENDING (not granted at once) means held elsewhere");
    expect(!win_lock_error_is_busy(0), "no error is not busy");
    expect(!win_lock_error_is_busy(5), "ERROR_ACCESS_DENIED is a real error");
    expect(!win_lock_error_is_busy(6), "ERROR_INVALID_HANDLE is a real error");
    expect(!win_lock_error_is_busy(87), "ERROR_INVALID_PARAMETER is a real error");
    if (failures) return 1;
    std::puts("startup_gamesave_lock_errors: ok");
    return 0;
}

// LEONIDA_SKIP_ASSERTS=N: step over the first N `int3` asserts raised by guest code (opt-in, off by default).
//
// A console devkit lets a title continue past a failed assert; the retail title has some asserts that are tuning
// checks, not faults (docs/retail_boot_chain.md). With the variable set to a positive number the runtime installs a
// vectored exception handler that, for a breakpoint at an address outside every host-loaded module (the guest's
// mapped images and its own allocations), whose byte is 0xCC, logs the site and resumes one byte later. Anything
// else continues the exception search untouched.
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace rt::guest_asserts {

// True when a breakpoint whose byte is `byte` is to be stepped over. `remaining` is decremented once per guest int3
// considered and the skip happens while the result is not negative, so N allows exactly N skips.
inline bool should_skip(std::atomic<long>& remaining, bool in_host_module, unsigned char byte) {
    if (in_host_module || byte != 0xCC) return false;
    return remaining.fetch_sub(1) - 1 >= 0;
}

// Parses the variable's value: a positive count, else 0 (off).
inline long parse_count(const char* text) {
    if (!text) return 0;
    const long n = std::strtol(text, nullptr, 10);
    return n > 0 ? n : 0;
}

// Installs the handler with the given budget; a count of 0 installs nothing. Returns whether it was installed.
bool install(long count);

// The handler itself, exposed for the test. Uses the budget set by install().
LONG CALLBACK handler(EXCEPTION_POINTERS* info);

}  // namespace rt::guest_asserts

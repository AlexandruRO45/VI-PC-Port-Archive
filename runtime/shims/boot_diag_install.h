// Turns the boot diagnostics on from the environment (docs/boot.md, "Diagnosing a hang or a crash"):
//
//   LEONIDA_DIAG=0           everything below off
//   LEONIDA_FAULTS=N         first-chance fault reports shown (default 8; 0 off); the unhandled one is always shown
//   LEONIDA_FIRST_CHANCE=N   also log the first N non-fault exceptions (C++ throws with their type name)
//   LEONIDA_BREAK=LIST       int3 at image offsets: "1c4a0,client.dll+3f20", 40 hits each
//   LEONIDA_WATCHDOG=SEC     dump every thread when no frame has been presented for SEC seconds (three times)
//   LEONIDA_THREAD_DUMP=N    dump every thread once when frame N is presented
//
// Reading the environment is separate from installing so tests can install with explicit values.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "boot_faults.h"

namespace rt::boot_diag {

struct Settings {
    bool enabled = true;
    FaultConfig faults;
    WatchdogConfig watchdog;
    std::string breakpoints;
    std::vector<std::string> problems;  // variables that were set but could not be read
};

namespace detail {
inline bool env_text(const char* name, std::string& out, std::vector<std::string>* problems = nullptr) {
    char buf[1024];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof buf);
    if (n >= sizeof buf) {
        if (problems) problems->push_back(std::string(name) + " is too long; ignored");
        return false;
    }
    if (n == 0) return false;
    out.assign(buf, n);
    return true;
}
template <typename T>
inline bool env_number(const char* name, T& out, std::vector<std::string>& problems) {
    std::string text;
    if (!env_text(name, text, &problems)) return false;
    const char* start = text.c_str();
    char* stop = nullptr;
    bool valid = false;
    T value{};
    errno = 0;
    if constexpr (std::is_same_v<T, double>) {
        value = std::strtod(start, &stop);
        valid = std::isfinite(value) && value >= 0;
    } else if constexpr (std::is_unsigned_v<T>) {
        const auto parsed = std::strtoull(start, &stop, 10);
        const auto sign = text.find_first_not_of(" \t\n\r\f\v");
        valid = sign != std::string::npos && text[sign] != '-' && parsed <= (std::numeric_limits<T>::max)();
        if (valid) value = static_cast<T>(parsed);
    } else {
        const auto parsed = std::strtoll(start, &stop, 10);
        valid = parsed >= 0 && parsed <= (std::numeric_limits<T>::max)();
        if (valid) value = static_cast<T>(parsed);
    }
    if (!valid || errno == ERANGE || stop == start || *stop != '\0') {
        problems.push_back(std::string(name) + "='" + text + "' is invalid or out of range; ignored");
        return false;
    }
    out = value;
    return true;
}
}  // namespace detail

inline Settings settings_from_environment() {
    Settings s;
    double v = 0;
    if (detail::env_number("LEONIDA_DIAG", v, s.problems) && v == 0) s.enabled = false;
    detail::env_number("LEONIDA_FAULTS", s.faults.fault_reports, s.problems);
    detail::env_number("LEONIDA_FIRST_CHANCE", s.faults.first_chance_limit, s.problems);
    detail::env_number("LEONIDA_WATCHDOG", s.watchdog.timeout_seconds, s.problems);
    detail::env_number("LEONIDA_THREAD_DUMP", s.watchdog.dump_at_progress, s.problems);
    detail::env_text("LEONIDA_BREAK", s.breakpoints);
    return s;
}

// Installs what the settings ask for. Call after the executable is mapped (breakpoints need its base) and before any
// image code runs. Prints what it turned on, and what it could not, to stderr.
inline void install(const Settings& s, const ImageResolver& resolve) {
    for (const auto& p : s.problems) report("diag_setup", {"LEONIDA diagnostics: " + p});
    if (!s.enabled) return;
    install_fault_handlers(s.faults);
    if (!s.breakpoints.empty())
        for (const auto& p : set_breakpoints_from_text(s.breakpoints, resolve)) report("diag_setup", {p});
    if (start_watchdog(s.watchdog))
        report("diag_setup", {format("watchdog on: dump every thread after %.1f s without a frame (3 times), thread dump at frame %llu", s.watchdog.timeout_seconds,
                                     static_cast<unsigned long long>(s.watchdog.dump_at_progress))});
}

}  // namespace rt::boot_diag

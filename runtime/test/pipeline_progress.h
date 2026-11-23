// vk_pipeline_check's time-based progress line: how far through the corpus the run is and a rough ETA.
//
// The corpus is not uniform. On an RTX 4080 SUPER (Windows) a 4.6 MB compute module took 139 s of CPU and a 6.3 MB one
// 248 s (exponent about 1.8 in size), while a module under 64 KB takes well under a second, so a module count says
// little: the old "N/M" line, printed every 1000 modules, was silent for hours while the multi-MB modules compiled.
// A module's weight is bytes^1.5 (rough on purpose, between the measured exponents of 1.1 for pixel shaders and 1.8 for
// compute). The rate is measured over a sliding window so that a resumed run, whose first part is pipeline-cache hits
// and takes seconds, does not make the rest look instant.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <utility>

namespace leonida {

inline double module_weight(uint64_t spirv_bytes) { return std::pow(static_cast<double>(spirv_bytes), 1.5); }

class ProgressEta {
public:
    explicit ProgressEta(double window_seconds = 600.0, double min_span_seconds = 30.0)
        : window_(window_seconds), min_span_(min_span_seconds) {}

    // Record that done_weight was reached at now_seconds (monotonic, non-decreasing).
    void sample(double now_seconds, double done_weight) {
        samples_.emplace_back(now_seconds, done_weight);
        // Keep one sample older than the window as the anchor.
        while (samples_.size() > 2 && samples_[1].first <= now_seconds - window_) samples_.pop_front();
    }

    // Seconds left at the rate measured over the window, or -1 when it can't be told yet (too little time has passed or
    // nothing finished in it).
    double eta_seconds(double total_weight) const {
        if (samples_.size() < 2) return -1;
        const auto& a = samples_.front();
        const auto& b = samples_.back();
        const double span = b.first - a.first, gained = b.second - a.second;
        if (span < min_span_ || gained <= 0) return -1;
        const double left = total_weight - b.second;
        return left <= 0 ? 0 : left / (gained / span);
    }

private:
    double window_, min_span_;
    std::deque<std::pair<double, double>> samples_;
};

inline std::string format_duration(double seconds) {
    if (seconds < 0) return "unknown";
    const long long s = static_cast<long long>(seconds + 0.5);
    char buf[48];
    if (s >= 3600) std::snprintf(buf, sizeof buf, "%lldh%02lldm", s / 3600, (s % 3600) / 60);
    else if (s >= 60) std::snprintf(buf, sizeof buf, "%lldm%02llds", s / 60, s % 60);
    else std::snprintf(buf, sizeof buf, "%llds", s);
    return buf;
}

}  // namespace leonida

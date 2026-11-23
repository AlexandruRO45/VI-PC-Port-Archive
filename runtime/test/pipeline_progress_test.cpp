// vk_pipeline_check's progress ETA (pipeline_progress.h), without a Vulkan device.
#include "pipeline_progress.h"

#include <cmath>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

using leonida::ProgressEta;

static int weights_and_format() {
    CHECK(leonida::module_weight(0) == 0);
    // 4x the bytes is 8x the weight: a big module counts for more than its share of the bytes.
    CHECK(std::fabs(leonida::module_weight(4000000) / leonida::module_weight(1000000) - 8.0) < 1e-9);
    CHECK(leonida::format_duration(-1) == "unknown");
    CHECK(leonida::format_duration(59.6) == "1m00s");
    CHECK(leonida::format_duration(61) == "1m01s");
    CHECK(leonida::format_duration(3 * 3600 + 5 * 60 + 29) == "3h05m");
    return 0;
}

static int steady_rate() {
    ProgressEta eta;
    CHECK(eta.eta_seconds(100) < 0);            // no samples
    eta.sample(0, 0);
    CHECK(eta.eta_seconds(100) < 0);            // one sample
    eta.sample(10, 10);
    CHECK(eta.eta_seconds(100) < 0);            // under the 30 s minimum span
    eta.sample(60, 60);                         // 1 weight per second
    CHECK(std::fabs(eta.eta_seconds(100) - 40) < 1e-9);
    eta.sample(160, 160);
    CHECK(eta.eta_seconds(100) == 0);           // everything done
    return 0;
}

// A resumed run: the first 90 % is pipeline-cache hits, done in 5 s; the rest compiles at 1 per second. Over the whole
// run the rate looks like 900 per 5 s; the window forgets it once it is older than window_seconds.
static int cache_hits_forgotten() {
    ProgressEta eta(600.0, 30.0);
    eta.sample(0, 0);
    eta.sample(5, 900);
    for (int t = 10; t <= 1300; t += 10) eta.sample(t, 900 + (t - 5));
    const double w = 900 + (1300 - 5);
    // 100 weight left at 1 per second, not "instant" as the early burst would suggest.
    CHECK(std::fabs(eta.eta_seconds(w + 100) - 100) < 1.0);
    return 0;
}

static int stalled() {
    ProgressEta eta;
    eta.sample(0, 5);
    eta.sample(100, 5);                         // nothing finished in 100 s
    CHECK(eta.eta_seconds(50) < 0);
    return 0;
}

// Workers finish modules of one size while a sampler (the watchdog) reads: every snapshot must hold a whole number of
// modules, i.e. bytes, weight and the large count all agree with done. If done were bumped before the rest was
// published, a sample taken between the two would show done ahead of bytes.
static int snapshots_are_consistent() {
    constexpr uint64_t kBytes = 1000;
    constexpr size_t kModules = 200000;
    leonida::ProgressCounters counters;
    std::atomic<bool> stop{false};
    std::atomic<size_t> torn{0}, samples{0};
    std::thread sampler([&] {
        while (!stop.load()) {
            const auto s = counters.snapshot();
            samples++;
            if (s.bytes != static_cast<double>(s.done) * kBytes || s.large != s.done ||
                std::fabs(s.weight - static_cast<double>(s.done) * leonida::module_weight(kBytes)) > 1e-3 * (s.done + 1))
                torn++;
        }
    });
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; t++)
        workers.emplace_back([&] { for (size_t i = 0; i < kModules / 4; i++) counters.complete(kBytes, true); });
    for (auto& w : workers) w.join();
    stop = true;
    sampler.join();
    const auto end = counters.snapshot();
    CHECK(end.done == kModules);
    CHECK(end.bytes == static_cast<double>(kModules) * kBytes);
    CHECK(end.large == kModules);
    CHECK(samples.load() > 0);
    CHECK(torn.load() == 0);
    return 0;
}

int main() {
    if (int rc = snapshots_are_consistent()) return rc;
    if (int rc = weights_and_format()) return rc;
    if (int rc = steady_rate()) return rc;
    if (int rc = cache_hits_forgotten()) return rc;
    if (int rc = stalled()) return rc;
    std::puts("pipeline_progress_test: ok");
    return 0;
}

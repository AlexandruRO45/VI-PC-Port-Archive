// vk_pipeline_check's progress ETA (pipeline_progress.h), without a Vulkan device.
#include "pipeline_progress.h"

#include <cmath>
#include <cstdio>

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

int main() {
    if (int rc = weights_and_format()) return rc;
    if (int rc = steady_rate()) return rc;
    if (int rc = cache_hits_forgotten()) return rc;
    if (int rc = stalled()) return rc;
    std::puts("pipeline_progress_test: ok");
    return 0;
}

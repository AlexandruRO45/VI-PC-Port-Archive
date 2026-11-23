// CPU unit tests for XboxPresentation's background SDR-white worker lifecycle, wakeup, and finite setters
// (shims/xbox_present.{h,cpp}) over a stub WindowPresenter: no Vulkan device, no window, no game files.
#include "../shims/xbox_present.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace {

struct MockPresenterState {
    std::atomic<bool> hdr_active{false};
    std::atomic<float> white_nits{203.0f};
    std::atomic<std::uint32_t> query_count{0};
    std::mutex mu;
    std::condition_variable cv;

    // Optional gate for Subtest 6: blocks inside display_sdr_white_nits() until released.
    bool gate_next = false;
    bool in_gate = false;
    bool release_gate = false;
    float gated_nits = 300.0f;

    leonida::PresentPlanes last_composed{};
    std::uint64_t composed_count = 0;
    std::uint64_t plain_count = 0;
    std::function<bool()> abort_fn;
};

struct GateReleaseGuard {
    explicit GateReleaseGuard(MockPresenterState& s) : state(s) {}
    ~GateReleaseGuard() {
        {
            std::lock_guard lock(state.mu);
            state.gate_next = false;
            state.release_gate = true;
        }
        state.cv.notify_all();
    }
    GateReleaseGuard(const GateReleaseGuard&) = delete;
    GateReleaseGuard& operator=(const GateReleaseGuard&) = delete;
    MockPresenterState& state;
};

class FakeResource final : public ID3D12Resource {
public:
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void** out) override {
        if (out) *out = nullptr;
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE Map(UINT, const D3D12_RANGE*, void**) override { return E_NOTIMPL; }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = 320;
        d.Height = 240;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
        d.SampleDesc.Count = 1;
        return d;
    }
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return 0; }
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT, const D3D12_BOX*, const void*, UINT, UINT) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void*, UINT, UINT, UINT, const D3D12_BOX*) override {
        return E_NOTIMPL;
    }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES* props, D3D12_HEAP_FLAGS* flags) override {
        if (props) *props = D3D12_HEAP_PROPERTIES{D3D12_HEAP_TYPE_DEFAULT};
        if (flags) *flags = D3D12_HEAP_FLAG_ALLOW_DISPLAY;
        return S_OK;
    }
};

}  // namespace

namespace rt::d3d12 {

WindowPresenter::WindowPresenter(IUnknown* device, std::unique_ptr<leonida::VulkanPresenter> presenter)
    : device_(device), presenter_(std::move(presenter)) {}

WindowPresenter::~WindowPresenter() = default;

void WindowPresenter::set_abort(std::function<bool()> abort) {
    reinterpret_cast<MockPresenterState*>(device_)->abort_fn = std::move(abort);
}

bool WindowPresenter::hdr_active() const {
    return reinterpret_cast<MockPresenterState*>(device_)->hdr_active.load();
}

float WindowPresenter::display_sdr_white_nits() const {
    auto* state = reinterpret_cast<MockPresenterState*>(device_);
    bool gated = false;
    float ret = 0.0f;
    {
        std::unique_lock lock(state->mu);
        if (state->gate_next) {
            state->gate_next = false;
            state->in_gate = true;
            state->cv.notify_all();
            state->cv.wait_for(lock, std::chrono::seconds(2), [&] { return state->release_gate; });
            state->in_gate = false;
            gated = true;
            ret = state->gated_nits;
        }
    }
    if (!gated) ret = state->white_nits.load();
    {
        std::lock_guard lock(state->mu);
        state->query_count.fetch_add(1);
    }
    state->cv.notify_all();
    return ret;
}

bool WindowPresenter::tear_supported() const { return false; }
bool WindowPresenter::last_present_tore() const { return false; }

HRESULT WindowPresenter::present_blank(UINT) { return S_OK; }

HRESULT WindowPresenter::present_planes(const leonida::PresentPlanes& planes, UINT, bool) {
    auto* state = reinterpret_cast<MockPresenterState*>(device_);
    state->last_composed = planes;
    ++state->composed_count;
    const bool hdr = planes.plane[0].hdr10 || (planes.count > 1 && planes.plane[1].hdr10);
    state->hdr_active.store(hdr);
    return S_OK;
}

HRESULT WindowPresenter::present(ID3D12Resource*, UINT, const D3D12_RECT*, bool) {
    auto* state = reinterpret_cast<MockPresenterState*>(device_);
    ++state->plain_count;
    state->hdr_active.store(false);
    return S_OK;
}

}  // namespace rt::d3d12

namespace {

constexpr UINT kPeriodUs = rt::d3d12::xbox_frame_interval_60hz_us;

void init_pacer(rt::d3d12::XboxPresentation& xbox) {
    xbox.SetFrameIntervalX(nullptr, kPeriodUs, 1, 0);
    xbox.ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0);
}

std::uint64_t next_token(rt::d3d12::XboxPresentation& xbox, leonida::ManualFrameClock& clock) {
    clock.advance_us(4 * kPeriodUs);
    std::uint64_t token = 0;
    const HRESULT hr = xbox.WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token);
    return SUCCEEDED(hr) ? token : 0;
}

HRESULT present_hdr10(rt::d3d12::XboxPresentation& xbox, leonida::ManualFrameClock& clock, ID3D12Resource* res) {
    const std::uint64_t token = next_token(xbox, clock);
    rt::d3d12::XboxPresentPlaneParameters plane{};
    plane.Token = token;
    plane.ResourceCount = 1;
    plane.ppResources = &res;
    plane.ColorSpace = rt::d3d12::xbox_color_space_g2084_p2020;
    return xbox.PresentX(1, &plane, nullptr);
}

HRESULT present_sdr_composed(rt::d3d12::XboxPresentation& xbox, leonida::ManualFrameClock& clock, ID3D12Resource* res) {
    const std::uint64_t token = next_token(xbox, clock);
    rt::d3d12::XboxPresentPlaneParameters plane{};
    plane.Token = token;
    plane.ResourceCount = 1;
    plane.ppResources = &res;
    plane.ColorSpace = rt::d3d12::xbox_color_space_g22_p709;
    plane.ScaleFilter = rt::d3d12::xbox_scale_filter_point;
    return xbox.PresentX(1, &plane, nullptr);
}

HRESULT present_sdr_plain(rt::d3d12::XboxPresentation& xbox, leonida::ManualFrameClock& clock, ID3D12Resource* res) {
    const std::uint64_t token = next_token(xbox, clock);
    rt::d3d12::XboxPresentPlaneParameters plane{};
    plane.Token = token;
    plane.ResourceCount = 1;
    plane.ppResources = &res;
    plane.ColorSpace = rt::d3d12::xbox_color_space_g22_p709;
    return xbox.PresentX(1, &plane, nullptr);
}

bool wait_query_count(MockPresenterState& state, std::uint32_t min_count, std::chrono::milliseconds timeout) {
    std::unique_lock lock(state.mu);
    return state.cv.wait_for(lock, timeout, [&] { return state.query_count.load() >= min_count; });
}

bool wait_composed_white(rt::d3d12::XboxPresentation& xbox, leonida::ManualFrameClock& clock, FakeResource& res,
                         MockPresenterState& state, float expected_nits, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (present_hdr10(xbox, clock, &res) != S_OK) return false;
        if (std::fabs(state.last_composed.sdr_white_nits - expected_nits) < 1e-3f) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

std::unique_ptr<rt::d3d12::XboxPresentation> make_xbox(MockPresenterState& state, leonida::ManualFrameClock& clock) {
    auto presenter = std::make_unique<rt::d3d12::WindowPresenter>(reinterpret_cast<IUnknown*>(&state), nullptr);
    auto xbox = std::make_unique<rt::d3d12::XboxPresentation>(std::move(presenter), &clock);
    init_pacer(*xbox);
    return xbox;
}

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s line %d: %s\n", __func__, __LINE__, #cond); \
        return false; \
    } \
} while (0)

bool subtest_immediate_wakeup_when_hdr_already_active() {
    MockPresenterState state;
    state.hdr_active = true;
    state.white_nits = 320.0f;
    leonida::ManualFrameClock clock;
    FakeResource res;
    auto xbox = make_xbox(state, clock);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(state.query_count.load() == 0);

    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 320.0f, std::chrono::milliseconds(250)));
    std::printf("subtest 1 (immediate wakeup when HDR10 active): PASS\n");
    return true;
}

bool subtest_first_frame_hdr_activation() {
    MockPresenterState state;
    state.hdr_active = false;
    state.white_nits = 280.0f;
    leonida::ManualFrameClock clock;
    FakeResource res;
    auto xbox = make_xbox(state, clock);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(state.query_count.load() == 0);

    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(std::fabs(state.last_composed.sdr_white_nits - 203.0f) < 1e-3f);
    CHECK(state.hdr_active.load());
    CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 280.0f, std::chrono::milliseconds(250)));
    std::printf("subtest 2 (first-frame HDR10 swapchain transition wakes worker): PASS\n");
    return true;
}

bool subtest_sdr_stops_polling_and_reentry_wakes_immediately() {
    MockPresenterState state;
    state.hdr_active = false;
    state.white_nits = 280.0f;
    leonida::ManualFrameClock clock;
    FakeResource res;
    auto xbox = make_xbox(state, clock);

    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 280.0f, std::chrono::milliseconds(250)));

    // Plain blit SDR present stops background polling.
    CHECK(present_sdr_plain(*xbox, clock, &res) == S_OK);
    CHECK(!state.hdr_active.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const std::uint32_t count_after_plain = state.query_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    CHECK(state.query_count.load() == count_after_plain);

    // Re-entering HDR10 wakes immediately and picks up 360 nits.
    state.white_nits = 360.0f;
    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, count_after_plain + 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 360.0f, std::chrono::milliseconds(250)));

    // Composed SDR present also stops background polling.
    CHECK(present_sdr_composed(*xbox, clock, &res) == S_OK);
    CHECK(!state.hdr_active.load());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const std::uint32_t count_after_composed = state.query_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    CHECK(state.query_count.load() == count_after_composed);

    // Re-entering HDR10 wakes immediately again.
    state.white_nits = 390.0f;
    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, count_after_composed + 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 390.0f, std::chrono::milliseconds(250)));
    std::printf("subtest 3 (SDR stops polling and HDR10 re-entry wakes immediately): PASS\n");
    return true;
}

bool subtest_override_and_suspend_pause_polling() {
    MockPresenterState state;
    state.hdr_active = true;
    state.white_nits = 260.0f;
    leonida::ManualFrameClock clock;
    FakeResource res;
    auto xbox = make_xbox(state, clock);

    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 260.0f, std::chrono::milliseconds(250)));

    // Explicit positive override takes effect immediately and pauses background polling.
    xbox->set_sdr_white_nits(240.0f);
    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(std::fabs(state.last_composed.sdr_white_nits - 240.0f) < 1e-3f);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const std::uint32_t count_override = state.query_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    CHECK(state.query_count.load() == count_override);

    // Clearing override resumes immediate wakeup on next HDR10 present.
    state.white_nits = 420.0f;
    xbox->set_sdr_white_nits(0.0f);
    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, count_override + 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 420.0f, std::chrono::milliseconds(250)));

    // SuspendX pauses polling.
    CHECK(xbox->SuspendX(0) == S_OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const std::uint32_t count_suspend = state.query_count.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    CHECK(state.query_count.load() == count_suspend);

    // ResumeX + re-register + HDR10 present wakes immediately.
    state.white_nits = 340.0f;
    CHECK(xbox->ResumeX() == S_OK);
    init_pacer(*xbox);
    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, count_suspend + 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 340.0f, std::chrono::milliseconds(250)));
    std::printf("subtest 4 (override and SuspendX/ResumeX polling control): PASS\n");
    return true;
}

bool subtest_failed_query_and_non_finite_setters() {
    MockPresenterState state;
    state.hdr_active = false;
    state.white_nits = 420.0f;
    leonida::ManualFrameClock clock;
    FakeResource res;
    auto xbox = make_xbox(state, clock);

    CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
    CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
    CHECK(wait_composed_white(*xbox, clock, res, state, 420.0f, std::chrono::milliseconds(250)));

    // Failed/non-finite query results (0, +Inf, NaN) preserve cached 420.0f.
    const float bad_queries[] = {
        0.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    std::uint32_t expected_queries = state.query_count.load();
    for (float bad : bad_queries) {
        CHECK(present_sdr_plain(*xbox, clock, &res) == S_OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        state.white_nits = bad;
        expected_queries = state.query_count.load() + 1;
        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
        CHECK(wait_query_count(state, expected_queries, std::chrono::milliseconds(250)));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
        CHECK(std::isfinite(state.last_composed.sdr_white_nits));
        CHECK(std::fabs(state.last_composed.sdr_white_nits - 420.0f) < 1e-3f);
    }

    // Non-finite or non-positive setter inputs must reset/ignore without poisoning PresentPlanes.
    const float bad_setters[] = {
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        -10.0f,
        0.0f,
    };
    for (float bad : bad_setters) {
        xbox->set_sdr_white_nits(bad);
        xbox->set_hdr_peak_nits(bad);
        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
        CHECK(std::isfinite(state.last_composed.sdr_white_nits));
        CHECK(state.last_composed.sdr_white_nits > 0.0f);
        CHECK(std::fabs(state.last_composed.sdr_white_nits - 420.0f) < 1e-3f);
        CHECK(std::isfinite(state.last_composed.hdr_peak_nits));
        CHECK(std::fabs(state.last_composed.hdr_peak_nits - 1000.0f) < 1e-3f);
    }

    std::printf("subtest 5 (failed queries & non-finite setters guarded): PASS\n");
    return true;
}

bool subtest_rerequest_during_inflight_query_and_prompt_shutdown() {
    {
        MockPresenterState state;
        state.hdr_active = false;
        {
            std::lock_guard lock(state.mu);
            state.gate_next = true;
            state.gated_nits = 300.0f;
        }
        leonida::ManualFrameClock clock;
        FakeResource res;
        auto xbox = make_xbox(state, clock);
        GateReleaseGuard gate_guard(state);  // Destroyed before `xbox` so `~XboxPresentation` cannot hang on fail.

        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
        {
            std::unique_lock lock(state.mu);
            CHECK(state.cv.wait_for(lock, std::chrono::milliseconds(250), [&] { return state.in_gate; }));
        }

        // While query #1 is in-flight, leave to SDR then re-enter HDR10 with updated 450 nits.
        CHECK(present_sdr_plain(*xbox, clock, &res) == S_OK);
        state.white_nits = 450.0f;
        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);

        {
            std::lock_guard lock(state.mu);
            state.release_gate = true;
        }
        state.cv.notify_all();

        CHECK(wait_query_count(state, 2, std::chrono::milliseconds(250)));
        CHECK(wait_composed_white(*xbox, clock, res, state, 450.0f, std::chrono::milliseconds(250)));
    }

    // Prompt destructor join while HDR10-active (worker in wait_for).
    {
        MockPresenterState state;
        state.hdr_active = false;
        state.white_nits = 310.0f;
        leonida::ManualFrameClock clock;
        FakeResource res;
        auto xbox = make_xbox(state, clock);
        CHECK(present_hdr10(*xbox, clock, &res) == S_OK);
        CHECK(wait_query_count(state, 1, std::chrono::milliseconds(250)));
        const auto t0 = std::chrono::steady_clock::now();
        xbox.reset();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ms < 250);
    }

    // Prompt destructor join while SDR-idle (worker in wait).
    {
        MockPresenterState state;
        state.hdr_active = false;
        leonida::ManualFrameClock clock;
        FakeResource res;
        auto xbox = make_xbox(state, clock);
        CHECK(present_sdr_plain(*xbox, clock, &res) == S_OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const auto t0 = std::chrono::steady_clock::now();
        xbox.reset();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ms < 250);
    }

    std::printf("subtest 6 (re-request during in-flight query and prompt shutdown): PASS\n");
    return true;
}

}  // namespace

int main() {
    int failures = 0;
    if (!subtest_immediate_wakeup_when_hdr_already_active()) ++failures;
    if (!subtest_first_frame_hdr_activation()) ++failures;
    if (!subtest_sdr_stops_polling_and_reentry_wakes_immediately()) ++failures;
    if (!subtest_override_and_suspend_pause_polling()) ++failures;
    if (!subtest_failed_query_and_non_finite_setters()) ++failures;
    if (!subtest_rerequest_during_inflight_query_and_prompt_shutdown()) ++failures;

    std::printf("xbox_present_white_worker_test: %d/6 passed\n", 6 - failures);
    return failures == 0 ? 0 : 1;
}

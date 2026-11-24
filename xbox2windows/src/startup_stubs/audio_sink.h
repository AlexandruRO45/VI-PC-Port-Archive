// Desktop render sink for startup audio experiments (WASAPI shared mode).
// The first half is pure (no platform headers) so its arithmetic can be built and run anywhere.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace x2w_audio {

enum class SampleKind { Float32, Pcm16 };

struct BlockFormat {
    SampleKind kind = SampleKind::Pcm16;
    uint32_t channels = 1;
    uint32_t rate = 48000;
    uint32_t frame_bytes() const { return channels * (kind == SampleKind::Float32 ? 4u : 2u); }
};

inline float read_sample(const BlockFormat& f, const void* src, size_t i)
{
    if (f.kind == SampleKind::Float32) {
        float v;
        std::memcpy(&v, static_cast<const uint8_t*>(src) + i * 4, 4);
        return v;
    }
    int16_t v;
    std::memcpy(&v, static_cast<const uint8_t*>(src) + i * 2, 2);
    return static_cast<float>(v) / 32768.0f;
}

inline void write_sample(const BlockFormat& f, void* dst, size_t i, float s)
{
    if (f.kind == SampleKind::Float32) {
        std::memcpy(static_cast<uint8_t*>(dst) + i * 4, &s, 4);
        return;
    }
    const float scaled = std::clamp(std::nearbyint(s * 32768.0f), -32768.0f, 32767.0f);
    const int16_t v = static_cast<int16_t>(scaled);
    std::memcpy(static_cast<uint8_t*>(dst) + i * 2, &v, 2);
}

// Largest absolute sample in 0..1. Exactly 0 means every sample is zero: a valid silent block.
inline float block_peak(const BlockFormat& f, const void* data, uint32_t frames)
{
    float peak = 0.0f;
    for (size_t i = 0; i < size_t(frames) * f.channels; ++i) peak = std::max(peak, std::fabs(read_sample(f, data, i)));
    return peak;
}

// Converts a submitted block to the endpoint format. Same rate required; mono is copied to every
// channel, otherwise channel counts must match. Returns false for any other combination.
inline bool convert_block(const BlockFormat& in, const void* src, uint32_t frames, const BlockFormat& out,
                          std::vector<uint8_t>& dst)
{
    if (in.rate != out.rate || in.channels == 0 || out.channels == 0) return false;
    if (in.channels != out.channels && in.channels != 1) return false;
    dst.assign(size_t(frames) * out.frame_bytes(), 0);
    for (size_t f = 0; f < frames; ++f)
        for (size_t c = 0; c < out.channels; ++c)
            write_sample(out, dst.data(), f * out.channels + c, read_sample(in, src, f * in.channels + (in.channels == 1 ? 0 : c)));
    return true;
}

inline double now_ms()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Drained interval: time the playing queue sat empty while the producer withheld data.
// Recovery: time from the producer resuming to consumption advancing again. Both come from
// timestamps and frame counters only, and are independent of any glitch count.
struct StarvationTimer {
    double drained_start = -1, drained_ms = 0, resume_at = -1, recovery_ms = -1;
    uint64_t consumed_at_resume = 0;

    void sample_queue(double now, uint32_t padding, bool withholding)
    {
        if (withholding && padding == 0 && drained_start < 0) drained_start = now;
    }
    void resume(double now, uint64_t consumed)
    {
        if (drained_start >= 0) drained_ms += now - drained_start;
        drained_start = -1;
        resume_at = now;
        consumed_at_resume = consumed;
        recovery_ms = -1;
    }
    void sample_consumed(double now, uint64_t consumed)
    {
        if (resume_at >= 0 && recovery_ms < 0 && consumed > consumed_at_resume) recovery_ms = now - resume_at;
    }
};

struct Report {
    int64_t first_block = -1;         // index of the submit() call whose block was accepted first
    BlockFormat submitted{};          // format of the first accepted block
    std::vector<uint32_t> frames;     // frame count of every accepted block, in order
    std::vector<float> peaks;         // peak amplitude of every accepted block (0 = silent)
    uint64_t submitted_frames = 0;    // frames written to the endpoint
    uint64_t consumed_frames = 0;     // submitted frames minus frames still queued
    uint64_t clock_frames = 0;        // IAudioClock position converted to frames
    uint32_t unsupported = 0, rejected_full = 0;
    uint32_t padding_errors = 0;      // GetCurrentPadding failed (e.g. the endpoint went away): not a measurement
    double drained_ms = 0, recovery_ms = -1;  // recovery_ms < 0: not measured
    // Glitches: the render API exposes no discontinuity flag, so this stays "not available".
    bool glitches_available = false;
};

enum class Submit { Accepted, Full, Unsupported, NotOpen };

#ifdef _WIN32
class AudioSink {
public:
    AudioSink();
    ~AudioSink();
    AudioSink(const AudioSink&) = delete;
    AudioSink& operator=(const AudioSink&) = delete;
    // Empty string on success; otherwise why there is no usable endpoint (a capability skip).
    std::string open();
    Submit submit(const BlockFormat& in, uint32_t frames, const void* bytes);
    // Samples padding and the clock. withholding = the producer is deliberately not submitting.
    void observe(bool withholding);
    // Frames still queued, or padding_failed (and padding_errors counts it) when the query fails.
    static constexpr uint32_t padding_failed = 0xFFFFFFFFu;
    uint32_t padding();
    Report report() const;
    const std::string& endpoint() const { return endpoint_; }
    const BlockFormat& mix_format() const { return mix_; }
    uint32_t buffer_frames() const { return buffer_frames_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::string endpoint_;
    BlockFormat mix_{};
    uint32_t buffer_frames_ = 0;
    uint32_t calls_ = 0;
    bool started_ = false;
    Report r_;
    StarvationTimer timer_;
};
#endif

}  // namespace x2w_audio

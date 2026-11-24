// Platform-independent checks for src/startup_stubs/audio_sink.h; they also build with g++ on Linux.
#pragma once
#include "startup_stubs/audio_sink.h"

#include <cstdio>

namespace x2w_audio {

// Generated mono PCM16 sine block; amplitude 0 gives an all-zero (silent) block.
inline std::vector<int16_t> make_tone(uint32_t rate, uint32_t frames, double hz, double amplitude)
{
    std::vector<int16_t> pcm(frames);
    for (uint32_t i = 0; i < frames; ++i)
        pcm[i] = static_cast<int16_t>(std::lround(amplitude * 32767.0 * std::sin(2.0 * 3.14159265358979 * hz * i / rate)));
    return pcm;
}

inline int check_pure()
{
    const BlockFormat mono16{SampleKind::Pcm16, 1, 48000};
    const auto tone = make_tone(48000, 480, 440.0, 0.5);
    const auto silent = make_tone(48000, 480, 440.0, 0.0);
    if (block_peak(mono16, silent.data(), 480) != 0.0f || block_peak(mono16, tone.data(), 480) < 0.45f) return 10;

    std::vector<uint8_t> out;
    const BlockFormat stereo_f{SampleKind::Float32, 2, 48000};
    if (!convert_block(mono16, tone.data(), 480, stereo_f, out) || out.size() != 480 * 8) return 11;
    if (std::fabs(block_peak(stereo_f, out.data(), 480) - block_peak(mono16, tone.data(), 480)) > 1e-6f) return 12;
    if (!convert_block(mono16, silent.data(), 480, stereo_f, out) || block_peak(stereo_f, out.data(), 480) != 0.0f) return 13;
    const BlockFormat stereo16{SampleKind::Pcm16, 2, 48000};
    if (!convert_block(mono16, tone.data(), 480, stereo16, out) || out.size() != 480 * 4) return 14;
    std::vector<uint8_t> back;  // PCM16 -> float -> PCM16 must be lossless
    const BlockFormat mono_f{SampleKind::Float32, 1, 48000};
    if (!convert_block(mono16, tone.data(), 480, mono_f, out) || !convert_block(mono_f, out.data(), 480, mono16, back) ||
        std::memcmp(back.data(), tone.data(), 960) != 0) return 15;
    if (convert_block(mono16, tone.data(), 480, {SampleKind::Pcm16, 1, 44100}, out)) return 16;     // rate mismatch
    if (convert_block(stereo16, out.data(), 120, {SampleKind::Pcm16, 6, 48000}, back)) return 17;   // channel mismatch

    StarvationTimer t;  // queue empties at 100 ms, producer resumes at 400 ms, data moves at 430 ms
    t.sample_queue(50, 10, true);
    if (t.drained_start >= 0) return 20;                    // padding > 0: not drained
    t.sample_queue(100, 0, false);
    if (t.drained_start >= 0) return 21;                    // producer not withholding: not starvation
    t.sample_queue(100, 0, true);
    t.sample_queue(250, 0, true);
    t.resume(400, 960);
    t.sample_consumed(410, 960);
    t.sample_consumed(430, 1000);
    t.sample_consumed(500, 2000);
    if (t.drained_ms != 300 || t.recovery_ms != 30) return 22;
    std::puts("pure checks: conversion, silence detection and drained/recovery arithmetic ok");
    return 0;
}

}  // namespace x2w_audio

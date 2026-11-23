// Portable contract tests for runtime/src/chad_dsp.h (no Windows or GPU dependency).
#include "../src/chad_dsp.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using namespace leonida::chad;
    DspEngine engine;

    std::uint8_t backing[4096] = {};
    void* ctx = nullptr;

    // Parameter validation on connect.
    expect(engine.connect(backing, sizeof(backing), 2, 128, 1, nullptr) == Status::pointer,
           "connect(null out) -> pointer");
    expect(engine.connect(nullptr, sizeof(backing), 2, 128, 1, &ctx) == Status::invalid_arg && !ctx,
           "connect(null buffer) -> invalid_arg");
    expect(engine.connect(backing, 0, 2, 128, 1, &ctx) == Status::invalid_arg && !ctx,
           "connect(zero size) -> invalid_arg");
    expect(engine.connect(backing, sizeof(backing), 0, 128, 1, &ctx) == Status::invalid_arg && !ctx,
           "connect(zero max_streams) -> invalid_arg");
    expect(engine.connect(backing, sizeof(backing), 2, 0, 1, &ctx) == Status::invalid_arg && !ctx,
           "connect(zero max_commands) -> invalid_arg");

    // Successful connect.
    expect(engine.connect(backing, sizeof(backing), 2, 128, 1, &ctx) == Status::ok && ctx != nullptr,
           "connect succeeds with non-null context");
    expect(engine.active_context_count() == 1, "active_context_count == 1");

    // Stream activation + max stream limit enforcement.
    DspActivateDesc desc{};
    desc.kind = 1;
    desc.mode_flags = 2;
    desc.block_size = 0x400;
    desc.channel_count = 2;
    desc.frame_count = 480;

    void* state1 = nullptr;
    void* stream1 = nullptr;
    expect(engine.activate(ctx, &desc, &state1, nullptr) == Status::pointer,
           "activate(null out_stream) -> pointer");
    expect(engine.activate(nullptr, &desc, &state1, &stream1) == Status::invalid_arg && !stream1,
           "activate(null ctx) -> invalid_arg");
    expect(engine.activate(ctx, nullptr, &state1, &stream1) == Status::invalid_arg && !stream1,
           "activate(null desc) -> invalid_arg");

    expect(engine.activate(ctx, &desc, &state1, &stream1) == Status::ok && state1 != nullptr && stream1 != nullptr,
           "first activate succeeds");
    const auto* st1 = static_cast<const DspStreamState*>(state1);
    expect(st1->kind == 1 && st1->channel_count == 2 && st1->block_size == 0x400,
           "stream state reflects activate descriptor");

    void* state2 = nullptr;
    void* stream2 = nullptr;
    desc.kind = 2;
    desc.channel_count = 1;
    expect(engine.activate(ctx, &desc, &state2, &stream2) == Status::ok && stream2 != nullptr && stream2 != stream1,
           "second activate succeeds");
    expect(engine.active_stream_count() == 2, "active_stream_count == 2");

    // Third stream exceeds max_streams == 2.
    void* state3 = reinterpret_cast<void*>(1);
    void* stream3 = reinterpret_cast<void*>(1);
    expect(engine.activate(ctx, &desc, &state3, &stream3) == Status::out_of_memory && !state3 && !stream3,
           "third activate hits max_streams limit");

    // Submit commands -> monotonic non-zero completion tokens.
    DspCommand cmd{};
    cmd.opcode = 1;
    cmd.input_buffer = backing;
    cmd.output_buffer = backing + 1024;
    std::uint32_t tok1 = 0, tok2 = 0;
    expect(engine.submit_command(nullptr, &cmd, &tok1) == Status::invalid_arg, "submit(null stream) -> invalid_arg");
    expect(engine.submit_command(stream1, nullptr, &tok1) == Status::invalid_arg, "submit(null cmd) -> invalid_arg");
    expect(engine.submit_command(stream1, &cmd, &tok1) == Status::ok && tok1 == 1, "first submit -> token 1");
    expect(engine.submit_command(stream2, &cmd, &tok2) == Status::ok && tok2 == 2, "second submit -> token 2");
    expect(engine.submit_command(stream1, &cmd, nullptr) == Status::ok, "submit(null out_token) succeeds");
    expect(engine.stream_commands_submitted(stream1) == 2 && engine.stream_commands_submitted(stream2) == 1,
           "per-stream command counts tracked");

    // Deactivate frees a slot; double deactivate fails cleanly.
    expect(engine.deactivate(stream1) == Status::ok, "deactivate stream1 succeeds");
    expect(engine.deactivate(stream1) == Status::invalid_arg, "double deactivate rejected");
    expect(engine.submit_command(stream1, &cmd, &tok1) == Status::invalid_arg,
           "submit on deactivated stream rejected");
    expect(engine.active_stream_count() == 1, "active_stream_count == 1 after deactivate");

    // Re-activating reuses the freed slot.
    expect(engine.activate(ctx, &desc, nullptr, &stream3) == Status::ok && stream3 != nullptr,
           "activate after deactivate succeeds (null out_state allowed)");
    expect(engine.active_stream_count() == 2, "active_stream_count == 2 again");

    // Disconnect tears down remaining active streams cleanly without leaking.
    expect(engine.disconnect(ctx) == Status::ok, "disconnect succeeds");
    expect(engine.disconnect(ctx) == Status::invalid_arg, "double disconnect rejected");
    expect(engine.active_context_count() == 0 && engine.active_stream_count() == 0,
           "disconnect cleans up context and orphaned streams");
    expect(engine.deactivate(stream2) == Status::invalid_arg, "orphaned stream rejected after disconnect");

    if (g_failures) {
        std::fprintf(stderr, "chad_dsp_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("chad_dsp: connect, stream limits, command tokens, deactivate, and disconnect cleanup passed\n");
    return 0;
}

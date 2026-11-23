#pragma once

// Portable Xbox Series chad.dll audio DSP session, stream, and command model (no Windows dependency).
//
// Call shapes measured statically from the 5 chad.dll imports:
//   - XDspConnectWithMaximumStreamLimit(pBuffer, bufferSize, maxStreams, maxCommands, flags, ppContext) -> HRESULT
//   - XDspDisconnect(pContext) -> HRESULT
//   - XDspActivate(pContext, pDesc /* 0x1C bytes */, ppOutState, ppOutStream) -> HRESULT
//   - XDspDeactivate(pStream) -> HRESULT
//   - XDspSubmitCommand(pStream, pCommand /* 0x20 bytes */, pOutToken) -> HRESULT
//
// Every live context and stream handle is tracked in-engine under a mutex so stale, double-freed, or foreign handles
// are rejected cleanly without dereferencing invalid pointers.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace leonida::chad {

enum class Status : std::uint32_t {
    ok = 0,
    invalid_arg = 1,
    pointer = 2,
    out_of_memory = 3,
};

// Matches the 0x1C-byte stream activation descriptor passed to XDspActivate.
struct DspActivateDesc {
    std::uint32_t kind = 0;            // +0x00 (e.g. 1 or 2)
    std::uint32_t mode_flags = 0;      // +0x04
    std::uint32_t block_size = 0;      // +0x08 (e.g. 0x400)
    std::uint32_t channel_count = 0;   // +0x0C (e.g. 1 or 2)
    std::uint32_t frame_count = 0;     // +0x10
    std::uint32_t user_context_lo = 0; // +0x14 (8-byte pointer/token, 4-byte aligned)
    std::uint32_t user_context_hi = 0; // +0x18
};
static_assert(sizeof(DspActivateDesc) == 0x1C);
static_assert(offsetof(DspActivateDesc, block_size) == 0x08);
static_assert(offsetof(DspActivateDesc, channel_count) == 0x0C);
static_assert(offsetof(DspActivateDesc, frame_count) == 0x10);
static_assert(offsetof(DspActivateDesc, user_context_lo) == 0x14);

// Matches the 0x20-byte command descriptor passed to XDspSubmitCommand.
struct DspCommand {
    std::uint32_t opcode = 0;         // +0x00
    std::uint32_t param = 0;          // +0x04
    void* input_buffer = nullptr;     // +0x08
    void* output_buffer = nullptr;    // +0x10
    std::uint64_t reserved = 0;       // +0x18
};
static_assert(sizeof(DspCommand) == 0x20);
static_assert(offsetof(DspCommand, input_buffer) == 0x08);
static_assert(offsetof(DspCommand, output_buffer) == 0x10);
static_assert(offsetof(DspCommand, reserved) == 0x18);

// Owned zero-initialized state block handed back through XDspActivate's third parameter.
struct DspStreamState {
    std::uint32_t version = 1;
    std::uint32_t kind = 0;
    std::uint32_t channel_count = 0;
    std::uint32_t block_size = 0;
    std::uint64_t commands_submitted = 0;
    std::uint64_t reserved[5] = {};
};
static_assert(sizeof(DspStreamState) == 0x40);

class DspEngine {
public:
    Status connect(void* memory_buffer, std::uint32_t memory_size, std::uint32_t max_streams,
                   std::uint32_t max_commands, std::uint32_t flags, void** out_context) {
        if (!out_context) return Status::pointer;
        *out_context = nullptr;
        if (!memory_buffer || memory_size == 0 || max_streams == 0 || max_commands == 0) {
            return Status::invalid_arg;
        }
        auto ctx = std::make_unique<Context>();
        ctx->memory_buffer = memory_buffer;
        ctx->memory_size = memory_size;
        ctx->max_streams = max_streams;
        ctx->max_commands = max_commands;
        ctx->flags = flags;

        Context* raw = ctx.get();
        std::lock_guard<std::mutex> lock(mu_);
        contexts_.emplace(raw, std::move(ctx));
        *out_context = raw;
        return Status::ok;
    }

    Status disconnect(void* context_handle) {
        if (!context_handle) return Status::invalid_arg;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = contexts_.find(context_handle);
        if (it == contexts_.end()) return Status::invalid_arg;
        // Detach and destroy any streams still belonging to this context.
        for (auto sit = streams_.begin(); sit != streams_.end();) {
            if (sit->second->owner == it->second.get()) {
                sit = streams_.erase(sit);
            } else {
                ++sit;
            }
        }
        contexts_.erase(it);
        return Status::ok;
    }

    Status activate(void* context_handle, const DspActivateDesc* desc, void** out_state, void** out_stream) {
        if (out_state) *out_state = nullptr;
        if (!out_stream) return Status::pointer;
        *out_stream = nullptr;
        if (!context_handle || !desc) return Status::invalid_arg;

        std::lock_guard<std::mutex> lock(mu_);
        auto cit = contexts_.find(context_handle);
        if (cit == contexts_.end()) return Status::invalid_arg;
        Context* ctx = cit->second.get();
        if (ctx->active_streams >= ctx->max_streams) return Status::out_of_memory;

        auto stream = std::make_unique<Stream>();
        stream->owner = ctx;
        stream->desc = *desc;
        stream->state.kind = desc->kind;
        stream->state.channel_count = desc->channel_count;
        stream->state.block_size = desc->block_size;

        Stream* raw = stream.get();
        ++ctx->active_streams;
        streams_.emplace(raw, std::move(stream));
        if (out_state) *out_state = &raw->state;
        *out_stream = raw;
        return Status::ok;
    }

    Status deactivate(void* stream_handle) {
        if (!stream_handle) return Status::invalid_arg;
        std::lock_guard<std::mutex> lock(mu_);
        auto sit = streams_.find(stream_handle);
        if (sit == streams_.end()) return Status::invalid_arg;
        Context* owner = sit->second->owner;
        if (owner && owner->active_streams > 0) --owner->active_streams;
        streams_.erase(sit);
        return Status::ok;
    }

    Status submit_command(void* stream_handle, const DspCommand* command, std::uint32_t* out_token) {
        if (!stream_handle || !command) return Status::invalid_arg;
        std::lock_guard<std::mutex> lock(mu_);
        auto sit = streams_.find(stream_handle);
        if (sit == streams_.end()) return Status::invalid_arg;
        Stream* stream = sit->second.get();
        Context* owner = stream->owner;
        if (!owner) return Status::invalid_arg;

        std::uint32_t token = owner->next_token++;
        if (token == 0) token = owner->next_token++;  // keep completion tokens non-zero on wrap
        ++stream->state.commands_submitted;
        stream->last_token = token;
        if (out_token) *out_token = token;
        return Status::ok;
    }

    std::size_t active_context_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return contexts_.size();
    }

    std::size_t active_stream_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return streams_.size();
    }

    std::uint64_t stream_commands_submitted(void* stream_handle) const {
        if (!stream_handle) return 0;
        std::lock_guard<std::mutex> lock(mu_);
        auto sit = streams_.find(stream_handle);
        return sit == streams_.end() ? 0 : sit->second->state.commands_submitted;
    }

private:
    struct Context {
        void* memory_buffer = nullptr;
        std::uint32_t memory_size = 0;
        std::uint32_t max_streams = 0;
        std::uint32_t max_commands = 0;
        std::uint32_t flags = 0;
        std::uint32_t active_streams = 0;
        std::uint32_t next_token = 1;
    };

    struct Stream {
        Context* owner = nullptr;
        DspActivateDesc desc{};
        DspStreamState state{};
        std::uint32_t last_token = 0;
    };

    mutable std::mutex mu_;
    std::unordered_map<void*, std::unique_ptr<Context>> contexts_;
    std::unordered_map<void*, std::unique_ptr<Stream>> streams_;
};

}  // namespace leonida::chad

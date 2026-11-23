#pragma once

#include <windows.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace leonida::discord_presence {

// The public XSAPI ABI, not the game's private presence-operation object.
// https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/live/xsapi-c/presence_c/structs/xblpresencerichpresenceids
struct XboxPresenceIds {
    char scid[40];
    const char* presence_id;
    const char** token_ids;
    std::size_t token_count;
};
static_assert(sizeof(XboxPresenceIds) == 64 && offsetof(XboxPresenceIds, presence_id) == 40);

struct Snapshot {
    bool active = false;
    std::string scid, presence_id;
    std::vector<std::string> token_ids;
};

struct Strings {
    std::map<std::string, std::string> presences, tokens;
};

struct Config {
    std::string application_id, image_sha256;
    uint32_t call_rva = 0, setter_rva = 0;
    std::map<std::string, Strings> strings;
};

bool parse_config(std::string_view json, Config& config, std::string& error);
Config load_config(const std::filesystem::path& path);
bool capture(bool active, const XboxPresenceIds* ids, Snapshot& snapshot) noexcept;
std::optional<std::string> resolve(const Config& config, const Snapshot& snapshot);
std::string activity_payload(const std::optional<std::string>& details, uint32_t pid, uint64_t nonce);

// Framed Discord IPC only: no TCP listener, web service, user token or game-service authentication.
class Rpc {
public:
    explicit Rpc(std::wstring pipe_prefix = L"\\\\?\\pipe\\discord-ipc-");
    ~Rpc();
    Rpc(const Rpc&) = delete;
    Rpc& operator=(const Rpc&) = delete;
    bool connect(const std::string& application_id, HANDLE stop);
    bool publish(const std::optional<std::string>& details, HANDLE stop);
    bool poll(HANDLE stop);
    void disconnect();
private:
    bool transfer(bool write, void* bytes, DWORD size, HANDLE stop, DWORD timeout = 1000);
    bool send(uint32_t opcode, std::string_view json, HANDLE stop);
    bool receive(std::string& json, HANDLE stop);
    HANDLE pipe_ = INVALID_HANDLE_VALUE;
    std::wstring pipe_prefix_;
    uint64_t nonce_ = 0;
    ULONGLONG deadline_ = 0;
};

class Bridge {
public:
    explicit Bridge(Config config);
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;
    // Caller owns the image and stops all mapped-code threads before destroying it, as Loader requires.
    void install(uint8_t* base, std::size_t size, const std::filesystem::path& image);
    void stop() noexcept;
private:
    struct State;
    std::shared_ptr<State> state_;
    static void observe(uint64_t context, uint64_t active, uint64_t ids, uint64_t async) noexcept;
};

}  // namespace leonida::discord_presence

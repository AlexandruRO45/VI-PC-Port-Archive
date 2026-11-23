// Synthetic presence identifiers and a local fake Discord; no game files or Xbox/Rockstar connections.
#include "../src/discord_presence.h"
#include "../src/shader_library.h"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <fstream>
#include <functional>
#include <thread>

using namespace leonida::discord_presence;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL %s\n", message); }
}
const char* config_text = R"({
  "application_id":"123456789012345678",
  "image_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "call_rva":"0x1010", "setter_rva":"0x1100",
  "strings":{"synthetic-service":{
    "presences":{"explore":"Exploring {0}","menu":"In the menu","repeat":"{0} and {0}","emoji":"Exploring 🌴"},
    "tokens":{"island":"Synthetic Island","quote":"A \"quoted\" place"}
  }}
})";

bool transfer(HANDLE pipe, bool write, void* data, DWORD size) {
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    bool ok = event != nullptr;
    for (DWORD done = 0; ok && done < size;) {
        OVERLAPPED io{}; io.hEvent = event; ResetEvent(event);
        DWORD count = 0;
        BOOL immediate = write ? WriteFile(pipe, static_cast<char*>(data) + done, size - done, &count, &io)
                               : ReadFile(pipe, static_cast<char*>(data) + done, size - done, &count, &io);
        if (!immediate) {
            if (GetLastError() != ERROR_IO_PENDING || WaitForSingleObject(event, 2000) != WAIT_OBJECT_0) {
                CancelIoEx(pipe, &io); GetOverlappedResult(pipe, &io, &count, TRUE); ok = false; break;
            }
            ok = GetOverlappedResult(pipe, &io, &count, FALSE) != FALSE;
        }
        if (!count) ok = false;
        done += count;
    }
    if (event) CloseHandle(event);
    return ok;
}
bool send_frame(HANDLE pipe, uint32_t opcode, std::string json) {
    uint32_t header[] = {opcode, uint32_t(json.size())};
    // Deliberately fragment both header and body across writes.
    return transfer(pipe, true, header, 3) && transfer(pipe, true, reinterpret_cast<char*>(header) + 3, 5) &&
           transfer(pipe, true, json.data(), DWORD(json.size()));
}
bool read_frame(HANDLE pipe, uint32_t opcode, std::string& json) {
    uint32_t header[2]{};
    if (!transfer(pipe, false, header, 8) || header[0] != opcode || header[1] > 65536) return false;
    json.resize(header[1]);
    return transfer(pipe, false, json.data(), header[1]);
}
class FakeDiscord {
public:
    std::wstring prefix;
    HANDLE pipe;
    std::thread thread;
    bool ok = false;
    explicit FakeDiscord(std::function<bool(HANDLE)> script) {
        static int sequence = 0;
        prefix = L"\\\\?\\pipe\\leonida-presence-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                 std::to_wstring(++sequence) + L"-";
        pipe = CreateNamedPipeW((prefix + L"0").c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                               PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 4096, 4096, 0, nullptr);
        if (pipe == INVALID_HANDLE_VALUE) return;
        thread = std::thread([this, script] {
            HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            OVERLAPPED io{}; io.hEvent = event;
            bool connected = ConnectNamedPipe(pipe, &io) != FALSE;
            if (!connected) {
                const DWORD error = GetLastError();
                DWORD count = 0;
                connected = error == ERROR_PIPE_CONNECTED || (error == ERROR_IO_PENDING &&
                    WaitForSingleObject(event, 2000) == WAIT_OBJECT_0 && GetOverlappedResult(pipe, &io, &count, FALSE));
                if (!connected) { CancelIoEx(pipe, &io); GetOverlappedResult(pipe, &io, &count, TRUE); }
            }
            CloseHandle(event);
            if (connected) ok = script(pipe);
        });
    }
    bool finish() { if (thread.joinable()) thread.join(); return ok; }
    ~FakeDiscord() { finish(); if (pipe != INVALID_HANDLE_VALUE) CloseHandle(pipe); }
};
bool ready(HANDLE pipe) {
    std::string json;
    return read_frame(pipe, 0, json) && json.find("123456789012345678") != std::string::npos &&
           send_frame(pipe, 1, R"({"evt":"READY"})");
}
bool acknowledge(HANDLE pipe, std::string& text, bool error = false) {
    if (!read_frame(pipe, 1, text)) return false;
    leonida::Json root;
    if (!leonida::JsonReader(text).parse(root) || !root.find("nonce")) return false;
    return send_frame(pipe, 1, "{\"cmd\":\"SET_ACTIVITY\",\"nonce\":\"" + root.find("nonce")->string +
                      "\",\"evt\":" + (error ? "\"ERROR\"" : "null") + "}");
}
void rpc_tests() {
    FakeDiscord server([](HANDLE pipe) {
        std::string text;
        if (!read_frame(pipe, 0, text) || !send_frame(pipe, 3, "ping") || !read_frame(pipe, 4, text) || text != "ping" ||
            !send_frame(pipe, 1, R"({"evt":"READY"})")) return false;
        if (!send_frame(pipe, 1, R"({"cmd":"SET_ACTIVITY","nonce":"unrelated","evt":null})") ||
            !acknowledge(pipe, text) || text.find("Exploring Synthetic Island") == std::string::npos) return false;
        // The client must not grant an arbitrary pipe server impersonation rights.
        if (!ImpersonateNamedPipeClient(pipe)) return false;
        HANDLE token = nullptr;
        SECURITY_IMPERSONATION_LEVEL level{}; DWORD size = 0;
        const bool safe = OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &token) &&
            GetTokenInformation(token, TokenImpersonationLevel, &level, sizeof(level), &size) && level == SecurityIdentification;
        if (token) CloseHandle(token);
        RevertToSelf();
        return safe && acknowledge(pipe, text) && text.find("\"activity\":null") != std::string::npos;
    });
    Rpc rpc(server.prefix);
    check(rpc.connect("123456789012345678", nullptr), "local IPC handshake waits for READY");
    check(rpc.publish(std::string("Exploring Synthetic Island"), nullptr), "fragmented frames, PING/PONG and matching acknowledgement work");
    check(rpc.publish(std::nullopt, nullptr), "clear is acknowledged");
    rpc.disconnect();
    check(server.finish(), "fake Discord received correct details, clear and safe impersonation level");

    FakeDiscord rejected([](HANDLE pipe) { std::string text; return ready(pipe) && acknowledge(pipe, text, true); });
    Rpc retry(rejected.prefix);
    check(retry.connect("123456789012345678", nullptr) && !retry.publish(std::string("Rejected presence"), nullptr),
          "Discord ERROR is not treated as a published activity");
    retry.disconnect(); check(rejected.finish(), "error response was exercised");

    FakeDiscord reconnect([](HANDLE pipe) { std::string text; return ready(pipe) && acknowledge(pipe, text); });
    Rpc next(reconnect.prefix);
    check(next.connect("123456789012345678", nullptr) && next.publish(std::string("Rejected presence"), nullptr),
          "latest activity can be resent after reconnecting");
    next.disconnect(); check(reconnect.finish(), "reconnected server received the activity");

    FakeDiscord oversized([](HANDLE pipe) {
        std::string text; uint32_t header[] = {1, 65537};
        return read_frame(pipe, 0, text) && transfer(pipe, true, header, 8);
    });
    Rpc invalid(oversized.prefix);
    check(!invalid.connect("123456789012345678", nullptr), "oversized frame is rejected before allocation");
    check(oversized.finish(), "oversized frame was exercised");

    HANDLE stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    FakeDiscord stalled([](HANDLE pipe) {
        std::string text; uint32_t header[] = {1, 20}; char byte = '{';
        return read_frame(pipe, 0, text) && transfer(pipe, true, header, 8) && transfer(pipe, true, &byte, 1);
    });
    Rpc cancelled(stalled.prefix);
    std::thread signal([stop] { Sleep(100); SetEvent(stop); });
    const auto start = GetTickCount64();
    check(!cancelled.connect("123456789012345678", stop), "stop cancels an incomplete frame");
    check(GetTickCount64() - start < 1000, "cancelled I/O is drained promptly without a dangling buffer");
    signal.join(); check(stalled.finish(), "stalled frame was exercised"); CloseHandle(stop);
}

uint64_t observed[4]{};
uint64_t original(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    observed[0] = a; observed[1] = b; observed[2] = c; observed[3] = d;
    return 0xFEDCBA9876543210;
}
void hook_tests(Config config) {
    auto* base = static_cast<uint8_t*>(VirtualAlloc(nullptr, 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    check(base != nullptr, "synthetic image allocation"); if (!base) return;
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base); dos->e_magic = IMAGE_DOS_SIGNATURE; dos->e_lfanew = 0x80;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(base + 0x80);
    nt->Signature = IMAGE_NT_SIGNATURE; nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64); nt->FileHeader.NumberOfSections = 1;
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC; nt->OptionalHeader.SizeOfImage = 8192;
    auto* section = IMAGE_FIRST_SECTION(nt); section->VirtualAddress = 0x1000; section->Misc.VirtualSize = 4096;
    section->Characteristics = IMAGE_SCN_MEM_EXECUTE;
    std::memset(base + 0x1000, 0x90, 0x100);
    const uint8_t prologue[] = {0x48,0x83,0xEC,0x28}; std::memcpy(base + 0x1000, prologue, 4);
    base[0x1010] = 0xE8; int32_t rel = 0x1100 - 0x1015; std::memcpy(base + 0x1011, &rel, 4);
    const uint8_t epilogue[] = {0x48,0x83,0xC4,0x28,0xC3}; std::memcpy(base + 0x1015, epilogue, 5);
    base[0x1100] = 0x48; base[0x1101] = 0xB8;
    auto address = reinterpret_cast<uint64_t>(&original); std::memcpy(base + 0x1102, &address, 8);
    base[0x110A] = 0xFF; base[0x110B] = 0xE0;
    const auto image = std::filesystem::temp_directory_path() / ("leonida_presence_" + std::to_string(GetCurrentProcessId()) + ".bin");
    { std::ofstream file(image, std::ios::binary); file.write(reinterpret_cast<char*>(base), 8192); }
    config.image_sha256 = leonida::sha256_hex(std::span(base, 8192));
    DWORD old = 0; VirtualProtect(base, 8192, PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), base, 8192);
    const auto unchanged = leonida::sha256_hex(std::span(base, 8192));
    auto rejects = [&](Config bad, const char* message) {
        bool rejected = false;
        try { Bridge bridge(bad); bridge.install(base, 8192, image); } catch (const std::exception&) { rejected = true; }
        check(rejected && leonida::sha256_hex(std::span(base, 8192)) == unchanged, message);
    };
    Config bad = config; bad.image_sha256[0] = bad.image_sha256[0] == '0' ? '1' : '0'; rejects(bad, "hash mismatch leaves image untouched");
    bad = config; ++bad.call_rva; rejects(bad, "invalid call boundary leaves image untouched");
    bad = config; ++bad.setter_rva; rejects(bad, "wrong original target leaves image untouched");
    bad = config; bad.call_rva = 8190; rejects(bad, "out-of-range instruction leaves image untouched");
    {
        Bridge bridge(config); bridge.install(base, 8192, image);
        using Function = uint64_t (*)(uint64_t,uint64_t,uint64_t,uint64_t);
        const uint64_t args[] = {0x1234567800000001,0xABCDEFAB00000000,0,0xFFEEDDCC00000004};
        check(reinterpret_cast<Function>(base + 0x1000)(args[0],args[1],args[2],args[3]) == 0xFEDCBA9876543210,
              "observer preserves original 64-bit result");
        check(std::memcmp(observed, args, sizeof(args)) == 0, "observer preserves all four full-width arguments");
        const auto start = GetTickCount64(); bridge.stop(); bridge.stop();
        check(GetTickCount64() - start < 1500, "stop is bounded and idempotent");
    }
    check(leonida::sha256_hex(std::span(base, 8192)) == unchanged, "stop restores the exact original call");
    VirtualFree(base, 0, MEM_RELEASE); std::filesystem::remove(image);
}
}
int main() {
    Config config;
    std::string error;
    check(parse_config(config_text, config, error), "valid local configuration parses");
    Snapshot s{true, "synthetic-service", "explore", {"island"}};
    check(resolve(config, s) == "Exploring Synthetic Island", "IDs resolve using the matching SCID catalogue");
    s.presence_id = "repeat";
    check(resolve(config, s) == "Synthetic Island and Synthetic Island", "repeated positional tokens resolve");
    s.presence_id = "emoji";
    check(resolve(config, s) == "Exploring 🌴", "literal UTF-8 is preserved");
    s.active = false;
    check(!resolve(config, s), "inactive presence clears activity");
    s.active = true; s.scid = "unknown-service";
    check(!resolve(config, s), "unknown SCID is not published as guessed text");
    s.scid = "synthetic-service"; s.presence_id = "unknown";
    check(!resolve(config, s), "unknown presence ID is not published");
    s.presence_id = "explore"; s.token_ids = {"unknown"};
    check(!resolve(config, s), "unknown token is not published");
    s.token_ids.clear();
    check(!resolve(config, s), "missing positional token is not published");
    Config bad;
    std::string application_id;
    check(parse_settings(R"({"richpresence_application_id":""})", application_id, error) && application_id.empty(),
          "empty user ID disables presence without requiring a hook profile");
    check(parse_settings("{}", application_id, error) && application_id.empty(), "missing user ID is optional");
    check(parse_settings(R"({"richpresence_application_id":"123456789012345679"})", application_id, error) &&
          application_id == "123456789012345679", "single user Application ID parses");
    check(!parse_settings(R"({"richpresence_application_id":"invalid"})", application_id, error), "invalid ID is rejected");
    check(!parse_settings(R"({"richpresence_application_id":123456789012345678})", application_id, error),
          "ID is a string to avoid numeric precision loss");
    check(!parse_settings("malformed JSON", application_id, error), "malformed user settings are rejected");
    const auto settings_dir = std::filesystem::temp_directory_path() /
        ("leonida_presence_settings_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(settings_dir / "leonida");
    const auto settings_path = settings_dir / "discord_presence.json";
    { std::ofstream file(settings_path); file << R"({"richpresence_application_id":""})"; }
    check(load_config(settings_path).application_id.empty(), "blank ID loads without any internal profile");
    { std::ofstream file(settings_path); file << "{}"; }
    check(load_config(settings_path).application_id.empty(), "absent ID loads without any internal profile");
    { std::ofstream file(settings_dir / "leonida" / "presence_profile.json"); file << config_text; }
    { std::ofstream file(settings_path); file << R"({"richpresence_application_id":"123456789012345679"})"; }
    const auto combined = load_config(settings_path);
    check(combined.application_id == "123456789012345679", "user ID overrides the internal profile's legacy ID");
    check(resolve(combined, Snapshot{true,"synthetic-service","explore",{"island"}}) == "Exploring Synthetic Island",
          "separate profile still resolves Xbox presence");
    std::filesystem::remove_all(settings_dir);
    check(parse_config(R"({"application_id":"","_comment":["Enter your own public Application ID"]})", bad, error) &&
          bad.application_id.empty(), "generated blank Application ID disables optional presence");
    check(!parse_config("{}", bad, error), "incomplete configuration is rejected");
    std::string changed = config_text;
    changed.replace(changed.find("123456789012345678"), 18, "not-an-app-id");
    check(!parse_config(changed, bad, error), "Application ID must be a decimal snowflake");
    changed = config_text;
    changed.replace(changed.find("0x1010"), 6, "0x100000000");
    check(!parse_config(changed, bad, error), "overflowing RVA is rejected");
    changed = config_text;
    changed.replace(changed.find("🌴"), std::strlen("🌴"), "\\ud83c\\udf34");
    check(parse_config(changed, bad, error) && resolve(bad, Snapshot{true,"synthetic-service","emoji",{}}) == "Exploring 🌴",
          "JSON surrogate pair normalizes to UTF-8");
    changed.replace(changed.find("\\ud83c\\udf34"), 12, "\\ud83c");
    check(!parse_config(changed, bad, error), "lone surrogate is rejected");
    const char* token = "island";
    XboxPresenceIds ids{};
    strcpy_s(ids.scid, "synthetic-service"); ids.presence_id = "explore";
    ids.token_ids = &token; ids.token_count = 1;
    Snapshot copied;
    check(capture(true, &ids, copied) && resolve(config, copied) == "Exploring Synthetic Island", "guest input is copied");
    token = "unknown";
    check(copied.token_ids[0] == "island", "snapshot owns token strings after caller mutation");
    check(!capture(true, reinterpret_cast<const XboxPresenceIds*>(1), copied), "invalid guest pointer does not fault");
    ids.token_count = 17;
    check(!capture(true, &ids, copied), "oversized token array is rejected");
    check(capture(false, nullptr, copied) && !copied.active, "null optional presence can clear activity");
    const std::string payload = activity_payload(std::string("A \"quoted\" \\ place"), 123, 7);
    leonida::Json root;
    check(leonida::JsonReader(payload).parse(root), "RPC payload is valid escaped JSON");
    const auto* args = root.find("args");
    const auto* activity = args ? args->find("activity") : nullptr;
    check(activity && activity->object.size() == 1, "only the details activity field is supplied");
    check(activity && activity->find("details") && activity->find("details")->string == "A \"quoted\" \\ place", "details is resolved text");
    check(activity && !activity->find("name") && !activity->find("type") && !activity->find("assets") &&
          !activity->find("buttons"), "Discord application defaults are not overridden");
    check(activity_payload(std::nullopt, 123, 8).find("\"activity\":null") != std::string::npos, "clear sends a null activity");
    rpc_tests();
    hook_tests(config);
    std::printf("discord_presence_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

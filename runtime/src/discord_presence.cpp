#include "discord_presence.h"

#include "code_hooks.h"
#include "shader_library.h"

#include <atomic>
#include <charconv>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace leonida::discord_presence {
namespace {
constexpr std::size_t max_tokens = 16, max_id = 256, max_frame = 65536;

std::string quote(std::string_view text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += char(c);
    }
    return out + '"';
}

// JsonReader emits UTF-8 code units for JSON surrogate escapes; combine pairs at this boundary.
bool unicode(std::string& text, std::size_t* count = nullptr) {
    std::wstring wide;
    for (std::size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i++]);
        uint32_t cp = c;
        int n = 0;
        if (c >= 0xC2 && c <= 0xDF) { cp = c & 31; n = 1; }
        else if (c >= 0xE0 && c <= 0xEF) { cp = c & 15; n = 2; }
        else if (c >= 0xF0 && c <= 0xF4) { cp = c & 7; n = 3; }
        else if (c >= 0x80) return false;
        for (int j = 0; j < n; ++j) {
            if (i == text.size() || (static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (static_cast<unsigned char>(text[i++]) & 63);
        }
        if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) ||
            cp > 0x10FFFF || cp < 32 || cp == 127) return false;
        if (cp > 0xFFFF) {
            cp -= 0x10000;
            wide += wchar_t(0xD800 + (cp >> 10)); wide += wchar_t(0xDC00 + (cp & 1023));
        } else wide += wchar_t(cp);
    }
    std::size_t points = 0;
    for (std::size_t i = 0; i < wide.size(); ++i, ++points) {
        if (wide[i] >= 0xD800 && wide[i] <= 0xDBFF) {
            if (++i == wide.size() || wide[i] < 0xDC00 || wide[i] > 0xDFFF) return false;
        } else if (wide[i] >= 0xDC00 && wide[i] <= 0xDFFF) return false;
    }
    const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), int(wide.size()),
                                         nullptr, 0, nullptr, nullptr);
    if (!bytes && !wide.empty()) return false;
    text.resize(bytes);
    if (bytes) WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), int(wide.size()),
                                   text.data(), bytes, nullptr, nullptr);
    if (count) *count = points;
    return true;
}

bool string_field(const Json& object, const char* key, std::string& value) {
    const Json* v = object.find(key);
    if (!v || v->kind != Json::Kind::string) return false;
    value = v->string;
    return unicode(value);
}
bool rva_field(const Json& object, const char* key, uint32_t& value) {
    std::string s;
    if (!string_field(object, key, s) || !s.starts_with("0x")) return false;
    auto [end, ec] = std::from_chars(s.data() + 2, s.data() + s.size(), value, 16);
    return ec == std::errc() && end == s.data() + s.size() && value != 0;
}
bool catalogue(const Json* object, std::map<std::string, std::string>& values) {
    if (!object || object->kind != Json::Kind::object) return false;
    for (const auto& entry : object->object) {
        std::string key = entry.key, value = entry.value.string;
        if (entry.value.kind != Json::Kind::string || key.empty() || key.size() >= max_id ||
            value.size() > 2048 || !unicode(key) || !unicode(value) || !values.emplace(key, value).second) return false;
    }
    return true;
}

struct RawSnapshot {
    char scid[40], presence[max_id], tokens[max_tokens][max_id];
    std::size_t count;
};
bool copy_string(char* out, std::size_t size, const char* in) {
    if (!in) return false;
    for (std::size_t i = 0; i < size; ++i) if (!(out[i] = in[i])) return true;
    return false;
}
// No C++ objects with destructors inside the SEH region (MSVC C2712).
bool copy_guest(const XboxPresenceIds* ids, RawSnapshot* out) noexcept {
    __try {
        if (!copy_string(out->scid, sizeof(out->scid), ids->scid) ||
            !copy_string(out->presence, max_id, ids->presence_id) || ids->token_count > max_tokens) return false;
        out->count = ids->token_count;
        for (std::size_t i = 0; i < out->count; ++i)
            if (!copy_string(out->tokens[i], max_id, ids->token_ids[i])) return false;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) { return false; }
}

std::vector<uint8_t> read_file(const std::filesystem::path& path, std::size_t limit) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || uint64_t(f.tellg()) > limit) throw std::runtime_error("presence: unreadable or oversized file");
    std::vector<uint8_t> bytes(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    if (!bytes.empty() && !f.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("presence: cannot read file");
    return bytes;
}
}

bool parse_config(std::string_view json, Config& config, std::string& error) {
    Config next;
    Json root;
    error = "presence: invalid configuration (see docs/discord_presence.md)";
    if (json.size() > 4 * 1024 * 1024 || !JsonReader(json).parse(root) || root.kind != Json::Kind::object ||
        !string_field(root, "application_id", next.application_id)) return false;
    if (next.application_id.empty()) {
        config = std::move(next); error.clear();
        return true;  // The generated setup file leaves the optional bridge disabled until an ID is supplied.
    }
    if (next.application_id.size() < 16 ||
        next.application_id.size() > 20 || next.application_id.find_first_not_of("0123456789") != std::string::npos ||
        !string_field(root, "image_sha256", next.image_sha256) || next.image_sha256.size() != 64 ||
        next.image_sha256.find_first_not_of("0123456789abcdef") != std::string::npos ||
        !rva_field(root, "call_rva", next.call_rva) || !rva_field(root, "setter_rva", next.setter_rva)) return false;
    uint64_t snowflake = 0;
    if (std::from_chars(next.application_id.data(), next.application_id.data() + next.application_id.size(), snowflake).ec !=
        std::errc() || snowflake == 0) return false;
    const Json* strings = root.find("strings");
    if (!strings || strings->kind != Json::Kind::object) return false;
    for (const auto& entry : strings->object) {
        std::string scid = entry.key;
        Strings values;
        if (scid.empty() || scid.size() >= 40 || !unicode(scid) || entry.value.kind != Json::Kind::object ||
            !catalogue(entry.value.find("presences"), values.presences) ||
            !catalogue(entry.value.find("tokens"), values.tokens) || !next.strings.emplace(scid, std::move(values)).second)
            return false;
    }
    config = std::move(next); error.clear();
    return true;
}
Config load_config(const std::filesystem::path& path) {
    const auto bytes = read_file(path, 4 * 1024 * 1024);
    Config config; std::string error;
    if (!parse_config(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()), config, error))
        throw std::runtime_error(error);
    return config;
}
bool capture(bool active, const XboxPresenceIds* ids, Snapshot& snapshot) noexcept {
    try {
        Snapshot next;
        next.active = active && ids;
        if (next.active) {
            RawSnapshot raw{};
            if (!copy_guest(ids, &raw)) return false;
            next.scid = raw.scid; next.presence_id = raw.presence;
            for (std::size_t i = 0; i < raw.count; ++i) next.token_ids.emplace_back(raw.tokens[i]);
        }
        snapshot = std::move(next);
        return true;
    } catch (...) { return false; }
}
std::optional<std::string> resolve(const Config& config, const Snapshot& snapshot) {
    if (!snapshot.active) return std::nullopt;
    const auto service = config.strings.find(snapshot.scid);
    if (service == config.strings.end()) return std::nullopt;
    const auto presence = service->second.presences.find(snapshot.presence_id);
    if (presence == service->second.presences.end()) return std::nullopt;
    std::string out;
    const std::string& pattern = presence->second;
    for (std::size_t i = 0; i < pattern.size();) {
        const char c = pattern[i++];
        if ((c == '{' || c == '}') && i < pattern.size() && pattern[i] == c) { out += c; ++i; }
        else if (c == '{') {
            const auto end = pattern.find('}', i);
            std::size_t index = 0;
            if (end == std::string::npos) return std::nullopt;
            const auto result = std::from_chars(pattern.data() + i, pattern.data() + end, index);
            if (result.ec != std::errc() || result.ptr != pattern.data() + end || index >= snapshot.token_ids.size())
                return std::nullopt;
            const auto token = service->second.tokens.find(snapshot.token_ids[index]);
            if (token == service->second.tokens.end()) return std::nullopt;
            out += token->second; i = end + 1;
        } else if (c == '}') return std::nullopt;
        else out += c;
        if (out.size() > 512) return std::nullopt;
    }
    std::size_t points = 0;
    if (!unicode(out, &points) || points < 2 || points > 128) return std::nullopt;
    return out;
}
std::string activity_payload(const std::optional<std::string>& details, uint32_t pid, uint64_t nonce) {
    const std::string activity = details ?
        "{\"type\":0,\"name\":\"Grand Theft Auto VI\",\"details\":" + quote(*details) +
        ",\"assets\":{\"large_image\":\"xbs_logo_480\",\"large_text\":\"Grand Theft Auto VI\"},"
        "\"buttons\":[{\"label\":\"Leonida\",\"url\":\"https://github.com/ItsNotRexed/Leonida\"}]}" : "null";
    return "{\"cmd\":\"SET_ACTIVITY\",\"args\":{\"pid\":" + std::to_string(pid) +
           ",\"activity\":" + activity + "},\"nonce\":" + quote(std::to_string(nonce)) + "}";
}

Rpc::Rpc(std::wstring pipe_prefix) : pipe_prefix_(std::move(pipe_prefix)) {}
Rpc::~Rpc() { disconnect(); }
void Rpc::disconnect() { if (pipe_ != INVALID_HANDLE_VALUE) CloseHandle(pipe_); pipe_ = INVALID_HANDLE_VALUE; }
bool Rpc::transfer(bool write, void* bytes, DWORD size, HANDLE stop, DWORD timeout) {
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!event) return false;
    bool ok = true;
    DWORD done = 0;
    const ULONGLONG end = (std::min)(deadline_, GetTickCount64() + timeout);
    while (done < size) {
        if ((stop && WaitForSingleObject(stop, 0) == WAIT_OBJECT_0) || GetTickCount64() >= end) { ok = false; break; }
        OVERLAPPED io{}; io.hEvent = event;
        ResetEvent(event);
        DWORD moved = 0;
        const BOOL immediate = write ? WriteFile(pipe_, static_cast<char*>(bytes) + done, size - done, &moved, &io)
                                     : ReadFile(pipe_, static_cast<char*>(bytes) + done, size - done, &moved, &io);
        if (!immediate) {
            if (GetLastError() != ERROR_IO_PENDING) { ok = false; break; }
            HANDLE waits[] = {event, stop};
            const DWORD wait = WaitForMultipleObjects(stop ? 2 : 1, waits, FALSE,
                DWORD((std::min)(end > GetTickCount64() ? end - GetTickCount64() : 0, ULONGLONG(1000))));
            if (wait != WAIT_OBJECT_0) {
                CancelIoEx(pipe_, &io);
                // OVERLAPPED and its buffer must outlive cancellation completion.
                GetOverlappedResult(pipe_, &io, &moved, TRUE);
                ok = false; break;
            }
            if (!GetOverlappedResult(pipe_, &io, &moved, FALSE)) { ok = false; break; }
        }
        if (!moved) { ok = false; break; }
        done += moved;
    }
    CloseHandle(event);
    return ok;
}
bool Rpc::send(uint32_t opcode, std::string_view json, HANDLE stop) {
    uint32_t header[] = {opcode, uint32_t(json.size())};
    return json.size() <= max_frame && transfer(true, header, sizeof(header), stop) &&
           transfer(true, const_cast<char*>(json.data()), DWORD(json.size()), stop);
}
bool Rpc::receive(std::string& json, HANDLE stop) {
    for (int i = 0; i < 16; ++i) {
        uint32_t header[2]{};
        if (!transfer(false, header, sizeof(header), stop) || header[1] > max_frame) return false;
        json.resize(header[1]);
        if (!transfer(false, json.data(), header[1], stop)) return false;
        if (header[0] == 1) return true;
        if (header[0] != 3 || !send(4, json, stop)) return false;
    }
    return false;
}
bool Rpc::connect(const std::string& application_id, HANDLE stop) {
    disconnect();
    deadline_ = GetTickCount64() + 1000;
    for (int i = 0; i < 10; ++i) {
        pipe_ = CreateFileW((pipe_prefix_ + std::to_wstring(i)).c_str(), GENERIC_READ | GENERIC_WRITE,
            0, nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (pipe_ == INVALID_HANDLE_VALUE) continue;
        std::string reply; Json root;
        if (send(0, "{\"v\":1,\"client_id\":" + quote(application_id) + "}", stop) && receive(reply, stop) &&
            JsonReader(reply).parse(root)) {
            const Json* evt = root.find("evt");
            if (evt && evt->kind == Json::Kind::string && evt->string == "READY") return true;
        }
        disconnect();
    }
    return false;
}
bool Rpc::publish(const std::optional<std::string>& details, HANDLE stop) {
    if (pipe_ == INVALID_HANDLE_VALUE) return false;
    deadline_ = GetTickCount64() + 1000;
    const auto nonce = std::to_string(++nonce_);
    if (!send(1, activity_payload(details, GetCurrentProcessId(), nonce_), stop)) return false;
    for (int i = 0; i < 16; ++i) {
        std::string reply; Json root;
        if (!receive(reply, stop) || !JsonReader(reply).parse(root)) return false;
        const auto* n = root.find("nonce");
        if (!n || n->kind != Json::Kind::string || n->string != nonce) continue;
        const auto* cmd = root.find("cmd"); const auto* evt = root.find("evt");
        return cmd && cmd->string == "SET_ACTIVITY" && (!evt || evt->kind == Json::Kind::null);
    }
    return false;
}
bool Rpc::poll(HANDLE stop) {
    DWORD available = 0;
    if (!PeekNamedPipe(pipe_, nullptr, 0, nullptr, &available, nullptr)) return false;
    if (!available) return true;
    deadline_ = GetTickCount64() + 1000;
    uint32_t header[2]{};
    if (!transfer(false, header, 8, stop) || header[1] > max_frame) return false;
    std::string body(header[1], '\0');
    if (!transfer(false, body.data(), header[1], stop)) return false;
    if (header[0] == 3) return send(4, body, stop);
    if (header[0] != 1) return false;
    Json root;
    if (!JsonReader(body).parse(root)) return false;
    const Json* evt = root.find("evt");
    return !evt || evt->string != "ERROR";
}

struct Bridge::State {
    inline static std::atomic<std::shared_ptr<State>> active;
    Config config;
    HANDLE stopped = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    std::mutex mutex;
    Snapshot latest;
    std::thread worker;
    uint8_t* call = nullptr;
    uint8_t* thunk = nullptr;
    uint8_t original[5]{};
    explicit State(Config c) : config(std::move(c)) {
        if (!stopped) throw std::runtime_error("presence: cannot create stop event");
    }
    ~State() { if (thunk) VirtualFree(thunk, 0, MEM_RELEASE); CloseHandle(stopped); }
    void work() noexcept {
        try {
            Rpc rpc;
            bool connected = false, acknowledged = false;
            std::optional<std::string> sent;
            do {
                Snapshot snapshot;
                { std::lock_guard lock(mutex); snapshot = latest; }
                auto details = resolve(config, snapshot);
                if (connected && !rpc.poll(stopped)) { rpc.disconnect(); connected = false; }
                if (!connected) { connected = rpc.connect(config.application_id, stopped); acknowledged = false; }
                if (connected && (!acknowledged || details != sent)) {
                    if (rpc.publish(details, stopped)) { sent = std::move(details); acknowledged = true; }
                    else { rpc.disconnect(); connected = false; }
                }
            } while (WaitForSingleObject(stopped, 5000) == WAIT_TIMEOUT);
            if (connected) rpc.publish(std::nullopt, nullptr);  // best-effort clear, bounded to one second
        } catch (...) { std::fprintf(stderr, "[rt] presence: publisher stopped after an error\n"); }
    }
};
Bridge::Bridge(Config config) : state_(std::make_shared<State>(std::move(config))) {}
Bridge::~Bridge() { stop(); }
void Bridge::observe(uint64_t, uint64_t active, uint64_t ids, uint64_t) noexcept {
    try {
        const auto state = State::active.load();
        if (!state) return;
        Snapshot snapshot;
        // Invalid input must also clear a previously published activity.
        capture((active & 255) != 0, reinterpret_cast<const XboxPresenceIds*>(ids), snapshot);
        std::lock_guard lock(state->mutex);
        state->latest = std::move(snapshot);
    } catch (...) {}
}
void Bridge::install(uint8_t* base, std::size_t size, const std::filesystem::path& image) {
    if (state_->call || state_->worker.joinable()) throw std::runtime_error("presence: bridge already installed");
    const auto bytes = read_file(image, std::size_t(2) << 30);
    if (sha256_hex(bytes) != state_->config.image_sha256) throw std::runtime_error("presence: executable SHA-256 mismatch");
    const uint32_t call = state_->config.call_rva, setter = state_->config.setter_rva;
    auto fail = [] { throw std::runtime_error("presence: invalid PE, executable RVAs or direct call target"); };
    if (size < sizeof(IMAGE_DOS_HEADER)) fail();
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0 ||
        uint64_t(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > size) fail();
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64 ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC || nt->OptionalHeader.SizeOfImage != size ||
        uint64_t(dos->e_lfanew) + 24 + nt->FileHeader.SizeOfOptionalHeader +
            uint64_t(nt->FileHeader.NumberOfSections) * sizeof(IMAGE_SECTION_HEADER) > size) fail();
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    auto executable = [&](uint32_t rva, uint32_t length) {
        if (uint64_t(rva) + length > size) return false;
        for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i)
            if ((sections[i].Characteristics & IMAGE_SCN_MEM_EXECUTE) && rva >= sections[i].VirtualAddress &&
                uint64_t(rva) + length <= uint64_t(sections[i].VirtualAddress) + sections[i].Misc.VirtualSize) return true;
        return false;
    };
    if (!executable(call, 5) || !executable(setter, 1) || base[call] != 0xE8) fail();
    int32_t rel = 0; std::memcpy(&rel, base + call + 1, 4);
    if (int64_t(call) + 5 + rel != setter) fail();
    auto& s = *state_;
    s.thunk = hooks::alloc_near(base + call, 4096);
    if (!s.thunk) throw std::runtime_error("presence: cannot allocate observer thunk");
    uint8_t* p = s.thunk;
    auto put = [&](std::initializer_list<uint8_t> code) { for (auto b : code) *p++ = b; };
    auto put64 = [&](uint64_t value) { std::memcpy(p, &value, 8); p += 8; };
    // Four pushes + 0x68 align RSP, reserve shadow space, and preserve all four 64-bit arguments and XMM0-3.
    put({0x51,0x52,0x41,0x50,0x41,0x51,0x48,0x83,0xEC,0x68});
    for (uint8_t i = 0; i < 4; ++i) put({0xF3,0x0F,0x7F,uint8_t(0x44 + 8*i),0x24,uint8_t(0x20 + 16*i)});
    put({0x48,0xB8}); put64(reinterpret_cast<uint64_t>(&Bridge::observe)); put({0xFF,0xD0});
    for (uint8_t i = 0; i < 4; ++i) put({0xF3,0x0F,0x6F,uint8_t(0x44 + 8*i),0x24,uint8_t(0x20 + 16*i)});
    put({0x48,0x83,0xC4,0x68,0x41,0x59,0x41,0x58,0x5A,0x59,0x48,0xB8});
    put64(reinterpret_cast<uint64_t>(base + setter)); put({0xFF,0xE0});
    DWORD old = 0;
    if (!VirtualProtect(s.thunk, 4096, PAGE_EXECUTE_READ, &old) ||
        !FlushInstructionCache(GetCurrentProcess(), s.thunk, p - s.thunk)) throw std::runtime_error("presence: cannot protect thunk");
    const intptr_t displacement = reinterpret_cast<intptr_t>(s.thunk) - reinterpret_cast<intptr_t>(base + call + 5);
    if (displacement < INT32_MIN || displacement > INT32_MAX) throw std::runtime_error("presence: thunk out of range");
    std::shared_ptr<State> empty;
    if (!State::active.compare_exchange_strong(empty, state_)) throw std::runtime_error("presence: another bridge is installed");
    try {
        s.worker = std::thread([state = state_] { state->work(); });
        if (!VirtualProtect(base + call, 5, PAGE_EXECUTE_READWRITE, &old)) throw std::runtime_error("presence: cannot patch call");
        std::memcpy(s.original, base + call, 5);
        rel = int32_t(displacement); std::memcpy(base + call + 1, &rel, 4);
        s.call = base + call;
        DWORD ignored = 0;
        if (!VirtualProtect(base + call, 5, old, &ignored) || !FlushInstructionCache(GetCurrentProcess(), base + call, 5)) {
            throw std::runtime_error("presence: cannot finalise call patch");
        }
    } catch (...) { stop(); throw; }
}
void Bridge::stop() noexcept {
    auto expected = state_;
    State::active.compare_exchange_strong(expected, {});
    SetEvent(state_->stopped);
    if (state_->worker.joinable()) state_->worker.join();
    if (state_->call) {
        DWORD old = 0, ignored = 0;
        if (VirtualProtect(state_->call, 5, PAGE_EXECUTE_READWRITE, &old)) {
            std::memcpy(state_->call, state_->original, 5);
            VirtualProtect(state_->call, 5, old, &ignored);
            FlushInstructionCache(GetCurrentProcess(), state_->call, 5);
        } else {
            // Keep the inert forwarding thunk alive if the original call could not be restored.
            state_->thunk = nullptr;
        }
        state_->call = nullptr;
    }
}
}  // namespace leonida::discord_presence

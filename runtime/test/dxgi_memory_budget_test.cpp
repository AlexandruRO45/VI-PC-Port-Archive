// Budget notifications with synthetic Vulkan heap properties and real Win32 events. No Vulkan device is opened.
#include "../shims/dxgi_com.h"
#include "../shims/dxgi_memory_budget.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct Event {
    HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
    bool signaled(DWORD timeout = 0) const { return WaitForSingleObject(handle, timeout) == WAIT_OBJECT_0; }
    void set() const { SetEvent(handle); }
    void reset() const { ResetEvent(handle); }
};

constexpr DWORD timeout = 4000;

struct Source {
    std::atomic<unsigned> phase{0};
    std::atomic<unsigned> reads{0};
    std::array<Event, 6> observed;
    Event finish_read;
};

VKAPI_ATTR VkResult VKAPI_CALL extensions(VkPhysicalDevice, const char*, std::uint32_t* count,
                                         VkExtensionProperties* out) {
    *count = 1;
    if (out) std::memcpy(out[0].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
                         sizeof(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME));
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL read_budget(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties2* out) {
    auto& source = *reinterpret_cast<Source*>(physical);
    const unsigned phase = source.phase.load();
    ++source.reads;
    auto& memory = out->memoryProperties;
    memory.memoryHeapCount = 4;
    memory.memoryHeaps[0] = {16384, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    memory.memoryHeaps[1] = {512, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};  // a smaller CPU-visible BAR heap
    memory.memoryHeaps[2] = {8192, 0};
    memory.memoryHeaps[3] = {8192, 0};
    auto& budget = *static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(out->pNext);
    budget.heapBudget[0] = phase >= 4 ? 4096 : phase >= 2 ? 6144 : 8192;
    budget.heapUsage[0] = phase ? 400 : 100;
    budget.heapBudget[1] = phase ? 256 : 512;  // changing only this heap does not change the LOCAL budget
    budget.heapBudget[2] = phase >= 3 ? 3072 : 4096;
    budget.heapUsage[2] = 200;
    budget.heapBudget[3] = 1024;
    budget.heapUsage[3] = 300;
    source.observed[phase].set();
    if (phase == 5) source.finish_read.signaled(timeout);  // keep one Vulkan query in flight during shutdown
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get(VkInstance, const char* name) {
    if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&extensions);
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(&read_budget);
    return nullptr;
}

}  // namespace

int main() {
    Source source;
    Event local, other, stopped, stopping;
    CHECK(local.handle && other.handle && stopped.handle && stopping.handle && source.finish_read.handle);
    for (const auto& event : source.observed) CHECK(event.handle);
    auto monitor = rt::dxgi::make_video_memory_budget_monitor(
        mock_get, VK_NULL_HANDLE, reinterpret_cast<VkPhysicalDevice>(&source));

    int a = 0, b = 0;
    DWORD first = 99, second = 0;
    CHECK(source.reads == 0);
    CHECK(monitor->subscribe(&a, nullptr, &first) == E_INVALIDARG && first == 0);
    CHECK(monitor->subscribe(&a, local.handle, nullptr) == E_INVALIDARG);

    // The subscriber may close its copy after registering. The monitor still holds the same event object.
    HANDLE duplicate = nullptr;
    CHECK(DuplicateHandle(GetCurrentProcess(), local.handle, GetCurrentProcess(), &duplicate, 0, FALSE,
                          DUPLICATE_SAME_ACCESS));
    const HRESULT registered = monitor->subscribe(&a, duplicate, &first);
    CloseHandle(duplicate);
    CHECK(registered == S_OK && first != 0);
    CHECK(monitor->subscribe(&b, other.handle, &second) == S_OK && second != first && second != 0);
    CHECK(!local.signaled() && !other.signaled());

    source.phase = 1;
    CHECK(source.observed[1].signaled(timeout));
    const auto usage_changed = monitor->query();  // also waits for the sample holding the monitor lock to finish
    CHECK(usage_changed.local.budget == 8192 && usage_changed.local.usage == 400);
    CHECK(usage_changed.non_local.budget == 5120 && usage_changed.non_local.usage == 500);
    CHECK(!local.signaled() && !other.signaled());

    source.phase = 2;
    CHECK(local.signaled(timeout) && other.signaled(timeout));
    local.reset();
    other.reset();

    monitor->unsubscribe(&b, first);  // a cookie belongs to the adapter that registered it
    source.phase = 3;
    CHECK(local.signaled(timeout) && other.signaled(timeout));
    local.reset();
    other.reset();

    monitor->unsubscribe(&a, first);
    source.phase = 4;
    CHECK(other.signaled(timeout) && !local.signaled());
    monitor->unsubscribe_all(&b);
    other.reset();
    CHECK(monitor->subscribe(&a, local.handle, &first) == S_OK && first != 0);
    CHECK(!local.signaled() && !other.signaled());

    // Once shutdown returns, the backing VkInstance can be destroyed even if an adapter still holds the monitor.
    source.phase = 5;
    CHECK(source.observed[5].signaled(timeout));
    std::thread closer([&] {
        stopping.set();
        monitor->shutdown();
        stopped.set();
    });
    const bool waited_for_read = stopping.signaled(timeout) && !stopped.signaled();
    source.finish_read.set();
    closer.join();
    CHECK(waited_for_read && stopped.signaled());
    const auto reads = source.reads.load();
    CHECK(monitor->query().local.budget == 0 && source.reads == reads);
    CHECK(monitor->subscribe(&a, local.handle, &first) == DXGI_ERROR_DEVICE_REMOVED && first == 0);

    rt::dxgi::VideoMemoryBudgetMonitor fixed_budget({});
    CHECK(fixed_budget.subscribe(&b, other.handle, &second) == S_OK && second && !other.signaled());
    std::printf("DXGI budget notifications: synthetic heap changes and teardown passed\n");
    return 0;
}

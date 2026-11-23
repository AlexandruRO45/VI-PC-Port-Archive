#pragma once

#include <windows.h>
#include <dxgi1_4.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

namespace rt::dxgi {

struct VideoMemorySegment {
    UINT64 budget = 0;
    UINT64 usage = 0;
};

struct VideoMemoryBudget {
    VideoMemorySegment local, non_local;
};

// Vulkan reports current estimates but has no budget-change event. The device shuts down the monitor
// before it destroys the VkInstance, even if callers still hold a DXGI factory or adapter.
class VideoMemoryBudgetMonitor {
public:
    using ReadBudget = std::function<VideoMemoryBudget()>;
    explicit VideoMemoryBudgetMonitor(ReadBudget read) : read_(std::move(read)) {}
    ~VideoMemoryBudgetMonitor() { shutdown(); }

    VideoMemoryBudget query() {
        std::lock_guard lock(mutex_);
        return !closed_ && read_ ? read_() : VideoMemoryBudget{};
    }

    HRESULT subscribe(const void* owner, HANDLE event, DWORD* cookie) {
        if (!cookie) return DXGI_ERROR_INVALID_CALL;
        *cookie = 0;
        if (!event || event == INVALID_HANDLE_VALUE) return DXGI_ERROR_INVALID_CALL;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), event, GetCurrentProcess(), &duplicate, 0, FALSE,
                             DUPLICATE_SAME_ACCESS)) {
            const DWORD error = GetLastError();
            return error == ERROR_INVALID_HANDLE ? DXGI_ERROR_INVALID_CALL : HRESULT_FROM_WIN32(error);
        }
        Event held(duplicate);
        if (!can_signal_event(held.get())) return DXGI_ERROR_INVALID_CALL;

        std::lock_guard lock(mutex_);
        if (closed_) return DXGI_ERROR_DEVICE_REMOVED;
        if (read_ && !timer_) {
            timer_ = CreateThreadpoolTimer(poll, this, nullptr);
            if (!timer_) return HRESULT_FROM_WIN32(GetLastError());
        }
        DWORD next;
        do {
            next = next_cookie_++;
        } while (!next || subscribers_.contains(next));
        const bool first = subscribers_.empty();
        try {
            subscribers_.emplace(next, Subscriber{owner, std::move(held)});
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        }
        if (first && timer_) {
            last_ = read_();
            schedule_sample();
        }
        *cookie = next;
        return S_OK;
    }

    void unsubscribe(const void* owner, DWORD cookie) {
        std::lock_guard lock(mutex_);
        const auto found = subscribers_.find(cookie);
        if (found != subscribers_.end() && found->second.owner == owner) subscribers_.erase(found);
        pause_if_empty();
    }

    void unsubscribe_all(const void* owner) {
        std::lock_guard lock(mutex_);
        std::erase_if(subscribers_, [owner](const auto& item) { return item.second.owner == owner; });
        pause_if_empty();
    }

    void shutdown() {
        PTP_TIMER timer;
        {
            // The query and any event signals are finished before we acquire this lock.
            std::lock_guard lock(mutex_);
            if (closed_) return;
            closed_ = true;
            subscribers_.clear();
            timer = std::exchange(timer_, nullptr);
            if (timer) SetThreadpoolTimer(timer, nullptr, 0, 0);
        }
        if (timer) {
            WaitForThreadpoolTimerCallbacks(timer, TRUE);
            CloseThreadpoolTimer(timer);
        }
    }

private:
    static bool can_signal_event(HANDLE event);

    struct CloseEvent {
        void operator()(HANDLE handle) const { CloseHandle(handle); }
    };
    using Event = std::unique_ptr<void, CloseEvent>;
    struct Subscriber {
        const void* owner;
        Event event;
    };

    void pause_if_empty() {
        if (timer_ && subscribers_.empty()) SetThreadpoolTimer(timer_, nullptr, 0, 0);
    }

    // One shot, rearmed after the sample, so a slow Vulkan query cannot queue extra polls.
    void schedule_sample() {
        LARGE_INTEGER interval{};
        interval.QuadPart = -2'500'000;  // 250 ms relative to now, in 100 ns units
        FILETIME due{interval.LowPart, static_cast<DWORD>(interval.HighPart)};
        SetThreadpoolTimer(timer_, &due, 0, 0);
    }

    static void CALLBACK poll(PTP_CALLBACK_INSTANCE, void* context, PTP_TIMER) {
        auto& self = *static_cast<VideoMemoryBudgetMonitor*>(context);
        std::lock_guard lock(self.mutex_);
        if (self.closed_ || self.subscribers_.empty()) return;
        const auto current = self.read_();
        if (current.local.budget != self.last_.local.budget ||
            current.non_local.budget != self.last_.non_local.budget) {
            for (const auto& [cookie, subscriber] : self.subscribers_) SetEvent(subscriber.event.get());
        }
        self.last_ = current;
        self.schedule_sample();
    }

    ReadBudget read_;
    std::mutex mutex_;
    std::map<DWORD, Subscriber> subscribers_;
    VideoMemoryBudget last_{};
    PTP_TIMER timer_ = nullptr;
    DWORD next_cookie_ = 1;
    bool closed_ = false;
};

}  // namespace rt::dxgi

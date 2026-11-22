#pragma once
#include <atomic>
#include <array>
#include <cstddef>

template <typename T, size_t Capacity>
class LockFreeRingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
public:
    bool push(T&& item) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) & (Capacity - 1);
        if (next == tail_.load(std::memory_order_acquire)) return false;
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool pop(T& out) noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;
        out = std::move(buffer_[tail]);
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return true;
    }
private:
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T, Capacity> buffer_;
};

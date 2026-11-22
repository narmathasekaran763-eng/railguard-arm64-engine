#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <new>

class CacheAlignedArena {
public:
    explicit CacheAlignedArena(size_t total_bytes) : capacity_(total_bytes), offset_(0) {
        if (posix_memalign(reinterpret_cast<void**>(&buffer_), 64, total_bytes) != 0 || !buffer_) {
            throw std::bad_alloc();
        }
    }
    ~CacheAlignedArena() { std::free(buffer_); }

    void* allocate(size_t bytes, size_t alignment = 64) {
        size_t aligned = (offset_ + (alignment - 1)) & ~(alignment - 1);
        if (aligned + bytes > capacity_) throw std::runtime_error("Arena full");
        void* ptr = buffer_ + aligned;
        offset_ = aligned + bytes;
        return ptr;
    }
    void reset() noexcept { offset_ = 0; }
private:
    uint8_t* buffer_ = nullptr;
    size_t capacity_, offset_;
};

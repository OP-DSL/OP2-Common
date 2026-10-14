#pragma once

// Device memory, as a backend provides it to code that is otherwise backend
// neutral, such as the hierarchical plan cache.

#include <cstddef>

namespace op {

// Each operation is complete, as far as the host and later device work can
// tell, when it returns.
struct Device {
    void *(*malloc)(std::size_t bytes);
    void (*free)(void *pointer);
    // Between any two of host and device memory.
    void (*copy)(void *destination, const void *source, std::size_t bytes);
    void (*memset)(void *pointer, int value, std::size_t bytes);
};

// Register the backend's device at initialisation.  Until one is, the host
// stands in, so code that only moves plan data runs without a device.
void set_device(const Device& device);
const Device& device();

} // namespace op

#pragma once

// A plan's host arrays copied to the device for its kernels to read, and
// freed there.  OP2 selects one device at initialisation and never switches,
// so an array is freed on the device it was allocated on.

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>

#include <vector>

namespace op::rt::gpu {

// A device copy of source in destination; none when source is empty.
template<typename T>
void upload(const std::vector<T>& source, T *&destination) {
    if (source.empty())
        return;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&destination), source.size() * sizeof(T)));
    GPU_CHECK(gpuMemcpy(destination, source.data(), source.size() * sizeof(T),
                        gpuMemcpyHostToDevice));
}

// Free a device array, if any.
template<typename T>
void release(T *&pointer) {
    if (pointer != nullptr)
        GPU_CHECK(gpuFree(pointer));
    pointer = nullptr;
}

} // namespace op::rt::gpu

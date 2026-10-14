#include <op_device.h>

#include <cstdlib>
#include <cstring>

namespace op {
namespace {

void *host_malloc(std::size_t bytes) { return std::malloc(bytes); }
void host_free(void *pointer) { std::free(pointer); }
void host_copy(void *destination, const void *source, std::size_t bytes) {
    std::memcpy(destination, source, bytes);
}
void host_memset(void *pointer, int value, std::size_t bytes) {
    std::memset(pointer, value, bytes);
}

Device current{host_malloc, host_free, host_copy, host_memset};

} // namespace

void set_device(const Device& device) { current = device; }
const Device& device() { return current; }

} // namespace op

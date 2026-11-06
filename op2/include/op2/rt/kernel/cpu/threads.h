#pragma once

// The host's threads as the generated code's OpenMP sees them, which a loop
// descriptor hands the library (op2/rt/abi/cpu/descriptor.h): the library
// itself is built without OpenMP.

#include <omp.h>

namespace op::rt::kernel::cpu {

// The threads a phase may use.
inline int max_threads() { return omp_get_max_threads(); }

} // namespace op::rt::kernel::cpu

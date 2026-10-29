#pragma once

// What the generated code tells the library about a loop, whatever the
// backend: the arguments a staged kernel groups, a parameter's type and
// source, and a compiled kernel's figures.

#include <span>

namespace op::rt::abi {

// The indirect arguments a staged kernel stages, grouped by dat as it lays
// its regions out.  Generated with the kernel they describe.
struct ArgGroup {
    int arg;
    int group;
};

struct ArgGroups {
    std::span<const ArgGroup> args;
    int groups = 0;
};

// A parameter's element type.
enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
};

// Where a parameter's value comes from: the application, the call, or the
// strategy.
enum class ParamSource {
    external,    // a constant the application sets
    scalar_arg,  // a scalar READ global argument
    array_arg,   // a READ global argument of any dimension, read in place
    dat_stride,  // a dat's SoA stride
    dat_dim,     // a dat's run-time dimension
    strategy,    // a value the strategy fixes, in its backend's slot
};

// A kernel's resources as its compiler left them: registers and local
// memory per thread, static shared memory per block, and the most threads a
// block can have with them.
struct KernelFigures {
    int registers = 0;
    int local_bytes = 0;
    int shared_bytes = 0;
    int max_threads = 0;
};

} // namespace op::rt::abi

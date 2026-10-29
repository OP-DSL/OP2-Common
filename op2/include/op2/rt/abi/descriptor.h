#pragma once

// What the generated code tells the library about a loop, whatever the
// backend: the arguments a staged kernel groups, the parameters its kernels
// may be specialised on, and a compiled kernel's figures.

#include <cstddef>
#include <cstdint>
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

template<typename T> struct ParamTypes {};
template<> struct ParamTypes<int> { static constexpr ParamType value = ParamType::i32; };
template<> struct ParamTypes<long long> { static constexpr ParamType value = ParamType::i64; };
template<> struct ParamTypes<long> { static constexpr ParamType value = ParamType::i64; };
template<> struct ParamTypes<float> { static constexpr ParamType value = ParamType::f32; };
template<> struct ParamTypes<double> { static constexpr ParamType value = ParamType::f64; };
template<> struct ParamTypes<bool> { static constexpr ParamType value = ParamType::logical; };

// A parameter a loop's kernels may be specialised on, as the generated code
// registers it: built by the functions below, one per source.
struct Param {
    const char *name;
    ParamType type;
    std::size_t elem_size;
    ParamSource source;
    // The value, read in place; none for an array_arg, whose data is each
    // call's.
    void *data;
    // Its elements: one for a scalar, the call's dimension for an array_arg.
    std::size_t elems;
    bool array;
    // The argument its value comes from, or -1.
    int index;
    // Whether a specialised kernel can have it at run time.
    bool runtime;
    // Whether every kernel has the argument by value, needing no device copy.
    bool by_value;
    // The copy an offline kernel reads, a __constant__ symbol, if any, and
    // the hash of the value there, shared by every loop that reads it.
    const void *symbol;
    std::uint64_t *shared_hash;
};

// A constant the application sets, of elems elements when an array.
template<typename T>
constexpr Param constant(const char *name, T *data, const void *symbol,
                         std::uint64_t *shared_hash) {
    return {name, ParamTypes<T>::value, sizeof(T), ParamSource::external, data, 1, false,
            -1, true, false, symbol, shared_hash};
}

template<typename T>
constexpr Param constant(const char *name, T *data, std::size_t elems,
                         const void *symbol, std::uint64_t *shared_hash) {
    return {name, ParamTypes<T>::value, sizeof(T), ParamSource::external, data, elems,
            true, -1, true, false, symbol, shared_hash};
}

// A scalar READ global argument, which every kernel has by value; data is
// the generated code's copy of it.
template<typename T>
constexpr Param scalar_global(const char *name, T *data, int arg) {
    return {name, ParamTypes<T>::value, sizeof(T), ParamSource::scalar_arg, data, 1,
            false, arg, true, true, nullptr, nullptr};
}

// Any other READ global argument: by value, or else read by an offline
// kernel from its device copy.
template<typename T>
constexpr Param global(const char *name, int arg, bool by_value) {
    return {name, ParamTypes<T>::value, sizeof(T), ParamSource::array_arg, nullptr, 0,
            true, arg, true, by_value, nullptr, nullptr};
}

// A dat's SoA stride, through the argument arg that reaches it.
constexpr Param dat_stride(const char *name, int *data, int arg, const void *symbol) {
    return {name, ParamType::i32, sizeof(int), ParamSource::dat_stride, data, 1, false, arg,
            true, false, symbol, nullptr};
}

// A dat's run-time dimension, which a specialised kernel always bakes in.
constexpr Param dat_dim(const char *name, int *data, int arg) {
    return {name, ParamType::i32, sizeof(int), ParamSource::dat_dim, data, 1, false, arg,
            false, false, nullptr, nullptr};
}

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

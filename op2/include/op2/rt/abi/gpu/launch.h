#pragma once

// What one launch of a loop's kernel is given: its launch record.  A range
// of elements for the plain, atomics and colouring strategies, a batch of
// chunks for the chunked ones; the library fills it, and a kernel takes its
// fields, in the order listed, as its first parameters, ahead of the loop's
// own arguments.  As separate parameters, a pointer one __restrict, a kernel
// compiles as it would with each declared by hand, which a record passed by
// value does not.  Compiled on the host and for the device, by NVRTC too, so
// it includes only the staging layout.

#include <op2/rt/abi/staging.h>

namespace op::rt::abi::gpu {

// The block size a strategy runs unless OP_BLOCK_SIZES says otherwise, and
// the offline kernels' launch bounds; a specialised kernel has its own block
// size, op2_block_size_d, as launch bounds.
constexpr int default_block_size = 128;

// A record's fields, X(type, name, qualifier) each: the qualifier a kernel
// takes it with, const or __restrict.  Every record starts with the stride
// of the per-element arrays and of the per-thread globals, and which
// optional arguments are active, which the generated binding fills.  An
// offline kernel reads the global stride from here; a specialised one bakes
// it in as op2_stride_gbl_d.
#define OP_ABI_GPU_COMMON_FIELDS(X)                                                      \
    X(int, stride, const)                                                                \
    X(int, global_stride, const)                                                         \
    X(unsigned, opt_flags, const)

// A range of elements, [start, end), or under a set colouring the positions
// of order.
#define OP_ABI_GPU_RANGE_FIELDS(X)                                                       \
    OP_ABI_GPU_COMMON_FIELDS(X)                                                          \
    X(int, start, const)                                                                 \
    X(int, end, const)                                                                   \
    X(const int *, order, __restrict)

// A batch of chunks, [chunk_begin, chunk_end) of the plan's positions, the
// plan's arrays on the device, and under a wavefront the section's position
// counter and the call's epoch.  A region's slot counts come from its
// offsets: a kernel reads its counts up front and its offsets only when it
// flushes, so no offset stays in a register through the user kernel.  The
// strategy's switches are a specialised kernel's constants,
// op2_unstaged_singles_d and op2_wavefront_d.
#define OP_ABI_GPU_CHUNK_FIELDS(X)                                                       \
    OP_ABI_GPU_COMMON_FIELDS(X)                                                          \
    X(int, chunk_begin, const)                                                           \
    X(int, chunk_end, const)                                                             \
    OP_ABI_GPU_CHUNK_ARRAYS(X)                                                           \
    X(int *, wave_ticket, )                                                              \
    X(int, wave_epoch, const)

// The chunk plan's arrays, which the library uploads once per plan and every
// chunk launch's record carries.
#define OP_ABI_GPU_CHUNK_ARRAYS(X)                                                       \
    X(const int *, source_offsets, __restrict)                                           \
    X(const ::op::rt::abi::StageWord *, stage_words, __restrict)                         \
    X(const int *, stage_counts, __restrict)                                             \
    X(const int *, slot_offsets, __restrict)                                             \
    X(const int *, slot_targets, __restrict)                                             \
    X(const int *, chunk_order, __restrict)                                              \
    X(const unsigned char *, thread_colours, __restrict)                                 \
    X(const int *, chunk_thread_colours, __restrict)                                     \
    X(const int *, wave_pred_offsets, __restrict)                                        \
    X(const int *, wave_preds, __restrict)                                               \
    X(int *, wave_done, )

#define OP_ABI_GPU_FIELD(type, name, qualifier) type name{};

struct Range {
    OP_ABI_GPU_RANGE_FIELDS(OP_ABI_GPU_FIELD)
};

struct Chunk {
    OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_FIELD)
};

#undef OP_ABI_GPU_FIELD

// A record field as a kernel's parameter, its list's comma included: a
// kernel opens its parameters with OP_ABI_GPU_RANGE_FIELDS(OP_ABI_GPU_PARAM)
// or OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM).
#define OP_ABI_GPU_PARAM(type, name, qualifier) [[maybe_unused]] type qualifier name,

} // namespace op::rt::abi::gpu

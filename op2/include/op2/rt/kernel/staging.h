#pragma once

// The chunked strategies' staging helpers, on the host and on the device: a
// chunk's regions carved out of its scratch, cleared or seeded, accumulated
// into, and stored or flushed back to their targets, laid out as
// op2/rt/abi/staging.h says.

#include <op2/rt/abi/staging.h>
#include <op2/rt/kernel/inc.h>
#include <op2/rt/kernel/prelude.h>

namespace op::rt::kernel {

/* Scratch staging helpers.

   A staged kernel carves one dense region per staged dat out of its dynamic
   shared allocation, accumulates into it, then writes each slot back to its
   target.  A region keeps each slot's components together,
   region[slot * dim + c], so with a JIT kernel's constant dim a component's
   offset is an immediate rather than a register.

   The AoS/SoA distinction never reaches them: a caller passes the global
   element as a Ptr, whose stride already encodes the layout (1 for AoS, the
   dat's SoA stride otherwise), exactly as the accumulation path does.

   Thread indices are parameters rather than reads of threadIdx, which keeps
   the staging helpers compilable for the host. */

// Carve the next region out of the shared buffer, advancing the cursor
// (alignof == sizeof for every supported scalar type).  Pad the pointer, not
// an integer round trip of it: that hides that it points into shared memory,
// so every increment through it becomes a generic atomic.  An empty region
// takes no padding, as the host sizes none for it.
template<typename T>
OP_KERNEL_DEVICE inline T *stage_region(char *&cursor, IndexType count, IndexType dim) {
    if (count == 0)
        return (T *)cursor;
    constexpr size_t alignment = sizeof(T);
    cursor += (alignment - ((size_t)cursor & (alignment - 1))) &
              (alignment - 1);

    T *base = (T *)cursor;
    cursor += (size_t)count * (size_t)dim * sizeof(T);
    return base;
}

// Cooperatively zero a region before accumulating into it.
template<typename T>
OP_KERNEL_DEVICE inline void stage_clear(T *region, IndexType elements, IndexType lane,
                                         IndexType lanes) {
    for (IndexType i = lane; i < elements; i += lanes)
        region[i] = 0;
}

// Seed a slot from its target's current value.
template<typename T, typename Source>
OP_KERNEL_DEVICE inline void stage_seed(Ptr<Source> target, T *region, IndexType slot,
                                        IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        region[slot * dim + c] = target[c];
}

// Add one element's per-thread increments into its staged slot.
template<typename T>
OP_KERNEL_DEVICE inline void stage_add(Ptr<T> slot, const T *increment, IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        slot[c] += increment[c];
}

// Store a slot back to its target, which no other chunk running alongside
// this one reaches.
template<typename T>
OP_KERNEL_DEVICE inline void stage_store(Ptr<T> target, const T *region, IndexType slot,
                                         IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        target[c] = region[slot * dim + c];
}

// Add a slot's accumulated increments into its target, which other chunks
// of the section may be adding to as well.
template<typename T>
OP_KERNEL_DEVICE inline void stage_flush(Ptr<T> target, const T *region, IndexType slot,
                                         IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        atomic_inc(target[c], region[slot * dim + c]);
}

} // namespace op::rt::kernel

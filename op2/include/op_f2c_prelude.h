#pragma once

#if defined(__CUDACC__) && !defined(__CUDACC_RTC__)
#include <cassert>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DEVICE __device__

#define H_MIN ::min
#define H_MAX ::max
#else
#include <cmath>
#include <cassert>

#define DEVICE

#define H_MIN std::min
#define H_MAX std::max
#endif

namespace op::f2c {

constexpr int round32(int x) { return (x + 31) & ~31; }
constexpr size_t round32(size_t x) { return (x + 31) & ~31; }

DEVICE inline void trap() {
#ifdef __HIPCC__
    __builtin_trap();
#else
    assert(false);
#endif
}

/* Hierarchical shared-memory atomics control word, one per staged argument
   and source element:

     bits  0..29  block-local shared-memory slot
     bit      30  owner, the reference that flushes the slot
     bit      31  exclusive, owner may flush without a global atomic

   The host planner writes these and both the offline and JIT staged wrappers
   read them, so they are defined here where all three can see one copy. */
using HierSmemStageWord = unsigned int;

constexpr HierSmemStageWord hier_smem_slot_mask = 0x3fffffffu;
constexpr HierSmemStageWord hier_smem_owner_bit = 0x40000000u;
constexpr HierSmemStageWord hier_smem_exclusive_bit = 0x80000000u;

// Encode a validated slot and its flush metadata into one device word.
DEVICE constexpr HierSmemStageWord
hier_smem_pack_stage_word(unsigned int slot, bool owner,
                          bool exclusive = false) {
    return slot | (owner ? hier_smem_owner_bit : 0u) |
           (exclusive ? hier_smem_exclusive_bit : 0u);
}

// Extract the block-local slot from a packed device word.
DEVICE constexpr unsigned int hier_smem_stage_slot(HierSmemStageWord word) {
    return word & hier_smem_slot_mask;
}

// Test whether this source/argument reference owns the slot flush.
DEVICE constexpr bool hier_smem_stage_owner(HierSmemStageWord word) {
    return (word & hier_smem_owner_bit) != 0;
}

// Test whether an owner may flush without a global atomic.
DEVICE constexpr bool hier_smem_stage_exclusive(HierSmemStageWord word) {
    return (word & hier_smem_exclusive_bit) != 0;
}

/* Span (+ extent) raw pointer wrappers with Fortran-style indexing */
using int64_t = long long int;
using IndexType = int;

// Avoids <type_traits> (not reliably available under NVRTC JIT compilation)
template<typename U> struct is_const_type { static constexpr bool value = false; };
template<typename U> struct is_const_type<const U> { static constexpr bool value = true; };

template<bool B, typename R = void> struct enable_if {};
template<typename R> struct enable_if<true, R> { using type = R; };

template<typename T>
struct Ptr {
    T* data;
    IndexType stride = 1;

    constexpr Ptr(T* data) : data{data} {}
    constexpr Ptr(T* data, IndexType stride) : data{data}, stride{stride} {}

    // Excluded when T is already const, else this is a conversion to its own type (nvcc #554-D)
    template<typename U = T, typename = typename enable_if<!is_const_type<U>::value>::type>
    constexpr operator Ptr<const T>() const { return Ptr<const T>{data, stride}; }
};

/* Hierarchical shared-memory staging helpers.

   A staged wrapper carves one dense region per staged dat out of its dynamic
   shared allocation, accumulates into it, then flushes each region's owners
   back to global memory.  These keep that arithmetic in one reviewable place
   instead of emitting it per dat and per argument from the template.

   The AoS/SoA distinction never reaches them: a caller passes the global
   element as a Ptr, whose stride already encodes the layout (1 for AoS, the
   dat's SoA stride otherwise), exactly as the accumulation path does.

   Thread indices are parameters rather than reads of threadIdx so that this
   header stays compilable for the host, where the planner includes it. */

// Carve the next region out of the shared buffer, advancing the cursor.
// Every supported scalar type has alignof == sizeof, so one value does both.
template<typename T>
DEVICE inline T *hier_smem_region(char *&cursor, IndexType count,
                                  IndexType dim) {
    constexpr size_t alignment = sizeof(T);
    cursor = (char *)(((size_t)cursor + (alignment - 1)) &
                      ~(size_t)(alignment - 1));

    T *base = (T *)cursor;
    cursor += (size_t)count * (size_t)dim * sizeof(T);
    return base;
}

// Cooperatively zero a region before accumulating into it.
template<typename T>
DEVICE inline void hier_smem_clear(T *region, IndexType elements,
                                   IndexType lane, IndexType lanes) {
    for (IndexType i = lane; i < elements; i += lanes)
        region[i] = 0;
}

// Seed an exclusive owner's slot from its target's current value, so the
// flush below can store rather than read-modify-write.
template<typename T>
DEVICE inline void hier_smem_seed(HierSmemStageWord word, Ptr<T> target,
                                  T *region, IndexType count, IndexType dim) {
    IndexType slot = (IndexType)hier_smem_stage_slot(word);
    for (IndexType c = 0; c < dim; ++c)
        region[slot + c * count] = target.data[c * target.stride];
}

#if defined(__CUDACC__) || defined(__HIPCC__)
// Flush one owner's slot back to its target.  An exclusive owner is the only
// reference to that target in its schedule section, and its slot was seeded
// with the target's previous value, so it already holds the total and can be
// stored outright; everything else has to add into whatever other blocks in
// the section are contributing.
template<typename T>
DEVICE inline void hier_smem_flush(HierSmemStageWord word, Ptr<T> target,
                                   const T *region, IndexType count,
                                   IndexType dim) {
    IndexType slot = (IndexType)hier_smem_stage_slot(word);

    if (hier_smem_stage_exclusive(word)) {
        for (IndexType c = 0; c < dim; ++c)
            target.data[c * target.stride] = region[slot + c * count];
    } else {
        for (IndexType c = 0; c < dim; ++c)
            atomicAdd(&target.data[c * target.stride],
                      region[slot + c * count]);
    }
}
#endif

struct Extent {
    const IndexType lower;
    const IndexType upper;

    constexpr Extent(IndexType lower, IndexType upper) : lower{lower}, upper{upper} {}
    constexpr IndexType size() const { return upper - lower + 1; }
};

template<typename T, unsigned N>
class Slice;

template<typename T, unsigned N>
class Span {
private:
    const Ptr<T> m_data;
    const Extent m_extents[N];

    constexpr Slice<T, N> slice_all(auto... extents) const {
        if constexpr (sizeof...(extents) == N)
            return slice(extents...);
        else
            return slice_all(m_extents[N - sizeof...(extents) - 1], extents...);
    }

public:
    constexpr Span(Ptr<T> data, auto... extents)
        : m_data{data}, m_extents{extents...} {}

    constexpr T& operator()(auto... indices) const {
        static_assert(sizeof...(indices) == N);
        IndexType indicies[sizeof...(indices)] = {indices...};

        IndexType offset = indicies[0] - m_extents[0].lower;
        IndexType next_index_stride = m_extents[0].size();

        for (unsigned i = 1; i < sizeof...(indices); ++i) {
            offset += (indicies[i] - m_extents[i].lower) * next_index_stride;
            next_index_stride *= m_extents[i].size();
        }

        return m_data.data[offset * m_data.stride];
    }

    constexpr Ptr<T> ptr_at(auto... indices) const {
        auto& elem = operator()(indices...);
        return Ptr{&elem, m_data.stride};
    }

    constexpr Slice<T, N> slice(auto... es) const {
        static_assert(sizeof...(es) == N);
        return Slice(*this, es...);
    }

    constexpr Span operator=(const T& v) const {
        slice_all() = v;
        return *this;
    }

    template<typename S>
    constexpr operator Ptr<S>() const { return m_data; }
};

/* - Deduction guides not working with current ROCm (6.2) -
template<typename T, typename... Es>
Span(Ptr<T>, Es...) -> Span<T, sizeof...(Es)>;

template<typename T, typename... Es>
Span(Ptr<const T>, Es...) -> Span<const T, sizeof...(Es)>;
*/

template<typename T, unsigned N>
class Slice {
private:
    const Span<T, N>& m_span;
    const Extent m_extents[N];

    constexpr void set(const T& v, auto... is) const {
        if constexpr (sizeof...(is) == N) {
            m_span(is...) = v;
        } else {
            auto& extent = m_extents[N - sizeof...(is) - 1];
            for (IndexType i = extent.lower; i <= extent.upper; ++i)
                set(v, i, is...);
        }
    }

public:
    constexpr Slice(const Span<T, N>& span, auto... extents)
        : m_span{span}, m_extents{extents...} {}

    constexpr Slice operator=(const T& v) const {
        set(v);
        return *this;
    }
};

/* Fortran intrinsics */

template<typename T>
inline constexpr T pow(T x, int e) {
    if (e < 0)  return 0;
    if (e == 0) return 1;

    T r = x;
    for (int i = 1; i < e; ++i)
        r *= x;

    return r;
}

inline constexpr float pow(float x, float e) { return powf(x, e); }
inline constexpr double pow(double x, double e) { return ::pow(x, e); }

inline constexpr double pow(float x, double e) { return ::pow((double) x, e); }
inline constexpr double pow(double x, float e) { return ::pow(x, (double) e); }

inline constexpr float pow(int x, float e) { return powf((float) x, e); }
inline constexpr double pow(int x, double e) { return ::pow((double) x, e); }

DEVICE inline int abs(int x) { return ::abs(x); }
DEVICE inline int64_t abs(int64_t x) { return ::abs(x); }
inline constexpr float abs(float x) { return fabsf(x); }
inline constexpr double abs(double x) { return fabs(x); }

inline constexpr double dble(int x) { return (double)x; }
inline constexpr double dble(int64_t x) { return (double)x; }
inline constexpr double dble(float x) { return (double)x; }
inline constexpr double dble(double x) { return x; }

inline constexpr int int_(int x) { return x; }
inline constexpr int int_(int64_t x) { return (int)x; }
inline constexpr int int_(float x) { return (int)x; }
inline constexpr int int_(double x) { return (int)x; }

DEVICE inline int min(int x0, int x1) { return H_MIN(x0, x1); }
DEVICE inline int min(int x0, int x1, int x2) { return H_MIN(H_MIN(x0, x1), x2); }
DEVICE inline int min(int x0, int x1, int x2, int x3) { return H_MIN(H_MIN(x0, x1), H_MIN(x2, x3)); }

DEVICE inline int64_t min(int64_t x0, int64_t x1) { return H_MIN(x0, x1); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2) { return H_MIN(H_MIN(x0, x1), x2); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return H_MIN(H_MIN(x0, x1), H_MIN(x2, x3)); }

inline constexpr float min(float x0, float x1) { return fminf(x0, x1); }
inline constexpr float min(float x0, float x1, float x2) { return fminf(fminf(x0, x1), x2); }
inline constexpr float min(float x0, float x1, float x2, float x3) { return fminf(fminf(x0, x1), fminf(x2, x3)); }

inline constexpr double min(double x0, double x1) { return fmin(x0, x1); }
inline constexpr double min(double x0, double x1, double x2) { return fmin(fmin(x0, x1), x2); }
inline constexpr double min(double x0, double x1, double x2, double x3) { return fmin(fmin(x0, x1), fmin(x2, x3)); }

DEVICE inline int max(int x0, int x1) { return H_MAX(x0, x1); }
DEVICE inline int max(int x0, int x1, int x2) { return H_MAX(H_MAX(x0, x1), x2); }
DEVICE inline int max(int x0, int x1, int x2, int x3) { return H_MAX(H_MAX(x0, x1), H_MAX(x2, x3)); }

DEVICE inline int64_t max(int64_t x0, int64_t x1) { return H_MAX(x0, x1); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2) { return H_MAX(H_MAX(x0, x1), x2); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return H_MAX(H_MAX(x0, x1), H_MAX(x2, x3)); }

inline constexpr float max(float x0, float x1) { return fmaxf(x0, x1); }
inline constexpr float max(float x0, float x1, float x2) { return fmaxf(fmaxf(x0, x1), x2); }
inline constexpr float max(float x0, float x1, float x2, float x3) { return fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)); }

inline constexpr double max(double x0, double x1) { return fmax(x0, x1); }
inline constexpr double max(double x0, double x1, double x2) { return fmax(fmax(x0, x1), x2); }
inline constexpr double max(double x0, double x1, double x2, double x3) { return fmax(fmax(x0, x1), fmax(x2, x3)); }

inline constexpr int mod(int a, int p) { return a % p; }
inline constexpr int64_t mod(int64_t a, int64_t p) { return a % p; }
inline constexpr float mod(float a, float p) { return fmodf(a, p); }
inline constexpr double mod(double a, double p) { return fmod(a, p); }

inline constexpr int nint(float x) { return lroundf(x); }
inline constexpr int nint(double x) { return lround(x); }

DEVICE inline int copysign(int x, int y) { return y >= 0 ? abs(x) : -abs(x); }
DEVICE inline int64_t copysign(int64_t x, int64_t y) { return y >= 0 ? abs(x) : -abs(x); }
inline constexpr float copysign(float x, float y) { return copysignf(x, y); }
inline constexpr double copysign(double x, double y) { return ::copysign(x, y); }

// ----------

inline constexpr float acos(float x) { return acosf(x); }
inline constexpr double acos(double x) { return ::acos(x); }

inline constexpr float asin(float x) { return asinf(x); }
inline constexpr double asin(double x) { return ::asin(x); }

inline constexpr float atan(float x) { return atanf(x); }
inline constexpr double atan(double x) { return ::atan(x); }

inline constexpr float atan2(float x, float y) { return atan2f(x, y); }
inline constexpr double atan2(double x, double y) { return ::atan2(x, y); }

inline constexpr float cos(float x) { return cosf(x); }
inline constexpr double cos(double x) { return ::cos(x); }

inline constexpr float cosh(float x) { return coshf(x); }
inline constexpr double cosh(double x) { return ::cosh(x); }

inline constexpr float exp(float x) { return expf(x); }
inline constexpr double exp(double x) { return ::exp(x); }

inline constexpr float log(float x) { return logf(x); }
inline constexpr double log(double x) { return ::log(x); }

inline constexpr float log10(float x) { return log10f(x); }
inline constexpr double log10(double x) { return ::log10(x); }

inline constexpr float sin(float x) { return sinf(x); }
inline constexpr double sin(double x) { return ::sin(x); }

inline constexpr float sinh(float x) { return sinhf(x); }
inline constexpr double sinh(double x) { return ::sinh(x); }

inline constexpr float sqrt(float x) { return sqrtf(x); }
inline constexpr double sqrt(double x) { return ::sqrt(x); }

inline constexpr float tan(float x) { return tanf(x); }
inline constexpr double tan(double x) { return ::tan(x); }

inline constexpr float tanh(float x) { return tanhf(x); }
inline constexpr double tanh(double x) { return ::tanh(x); }

} // namespace op::prelude

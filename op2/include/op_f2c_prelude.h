#pragma once

#ifdef __CUDACC__
#define DEVICE __device__
#define MIN ::min
#define MAX ::max
#else
#include <math.h>
#define DEVICE
#define MIN std::min
#define MAX std::max
#endif

namespace op::f2c {

/* Span (+ extent) raw pointer wrappers with Fortran-style indexing */
using int64_t = long long int;
using IndexType = int;

template<typename T>
struct Ptr {
    T* data;
    constexpr Ptr(T* data) : data{data} {}
};

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
    const IndexType m_stride = 1;

    constexpr Slice<T, N> slice_all(auto... extents) const {
        if constexpr (sizeof...(extents) == N)
            return slice(extents...);
        else
            return slice_all(m_extents[N - sizeof...(extents) - 1], extents...);
    }

public:
    constexpr Span(Ptr<T> data, auto... extents)
        : m_data{data}, m_extents{extents...} {}

    constexpr Span(IndexType stride, Ptr<T> data, auto... extents)
        : m_data{data}, m_extents{extents...}, m_stride{stride} {}

    constexpr T& operator()(auto... indices) const {
        static_assert(sizeof...(indices) == N);
        IndexType indicies[sizeof...(indices)] = {indices...};

        IndexType offset = indicies[0] - m_extents[0].lower;
        IndexType next_index_stride = m_extents[0].size();

        for (unsigned i = 1; i < sizeof...(indices); ++i) {
            offset += (indicies[i] - m_extents[i].lower) * next_index_stride;
            next_index_stride *= m_extents[i].size();
        }

        return m_data.data[offset * m_stride];
    }

    constexpr Slice<T, N> splice(auto... es) const {
        static_assert(sizeof...(es) == N);
        return Slice(*this, es...);
    }

    constexpr Span operator=(const T& v) const {
        slice_all() = v;
        return *this;
    }

    constexpr Slice<T, N> slice(auto... es) const { return Slice(*this, es...); }

    template<typename S>
    constexpr operator Ptr<S>() const { return m_data.data; }
};

template<typename T, typename... Es>
Span(Ptr<T>, Es...) -> Span<T, sizeof...(Es)>;

template<typename T, typename... Es>
Span(IndexType, Ptr<T>, Es...) -> Span<T, sizeof...(Es)>;

template<typename T, typename... Es>
Span(Ptr<const T>, Es...) -> Span<const T, sizeof...(Es)>;

template<typename T, typename... Es>
Span(IndexType, Ptr<const T>, Es...) -> Span<const T, sizeof...(Es)>;

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

DEVICE inline int min(int x0, int x1) { return MIN(x0, x1); }
DEVICE inline int min(int x0, int x1, int x2) { return MIN(MIN(x0, x1), x2); }
DEVICE inline int min(int x0, int x1, int x2, int x3) { return MIN(MIN(x0, x1), MIN(x2, x3)); }

DEVICE inline int64_t min(int64_t x0, int64_t x1) { return MIN(x0, x1); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2) { return MIN(MIN(x0, x1), x2); }
DEVICE inline int64_t min(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return MIN(MIN(x0, x1), MIN(x2, x3)); }

inline constexpr float min(float x0, float x1) { return fminf(x0, x1); }
inline constexpr float min(float x0, float x1, float x2) { return fminf(fminf(x0, x1), x2); }
inline constexpr float min(float x0, float x1, float x2, float x3) { return fminf(fminf(x0, x1), fminf(x2, x3)); }

inline constexpr double min(double x0, double x1) { return fmin(x0, x1); }
inline constexpr double min(double x0, double x1, double x2) { return fmin(fmin(x0, x1), x2); }
inline constexpr double min(double x0, double x1, double x2, double x3) { return fmin(fmin(x0, x1), fmin(x2, x3)); }

DEVICE inline int max(int x0, int x1) { return MAX(x0, x1); }
DEVICE inline int max(int x0, int x1, int x2) { return MAX(MAX(x0, x1), x2); }
DEVICE inline int max(int x0, int x1, int x2, int x3) { return MAX(MAX(x0, x1), MAX(x2, x3)); }

DEVICE inline int64_t max(int64_t x0, int64_t x1) { return MAX(x0, x1); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2) { return MAX(MAX(x0, x1), x2); }
DEVICE inline int64_t max(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return MAX(MAX(x0, x1), MAX(x2, x3)); }

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

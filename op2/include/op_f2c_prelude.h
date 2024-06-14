#pragma once

namespace op::f2c {

typedef long long int int64_t;

/* Span (+ extent) raw pointer wrappers with Fortran-style indexing */
using index_type = int;

struct extent {
    index_type lower;
    index_type upper;

    constexpr extent(index_type upper) : lower{1}, upper{upper} {}
    constexpr extent(index_type lower, index_type upper) : lower{lower}, upper{upper} {}

    constexpr index_type size() { return upper - lower + 1; }
};

template<typename T, unsigned N>
struct span {
    T* data;
    extent extents[N];
    index_type stride;

    template<typename... Es>
    constexpr span(T* data, Es... extents)
        : data{data}, extents{extents...}, stride{1} {}

    template<typename... Es>
    constexpr span(index_type stride, T* data, Es... extents)
        : data{data}, extents{extents...}, stride{stride} {}

    template<typename... Is>
    constexpr T& operator()(Is... is) {
        static_assert(sizeof...(is) == N);
        index_type indicies[sizeof...(is)] = {is...};

        index_type offset = indicies[0] - extents[0].lower;
        index_type next_index_stride = extents[0].size();

        for (auto i = 1; i < sizeof...(is); ++i) {
            offset += (indicies[i] - extents[i].lower) * next_index_stride;
            next_index_stride *= extents[i].size();
        }

        return data[offset * stride];
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

/*
template<typename T> inline constexpr T abs(T x) { return ::abs(x); }
template<> inline constexpr float abs(float x) { return fabsf(x); }
template<> inline constexpr double abs(double x) { return fabs(x); }
*/

__device__ inline int abs(int x) { return ::abs(x); }
__device__ inline int64_t abs(int64_t x) { return ::abs(x); }
inline constexpr float abs(float x) { return fabsf(x); }
inline constexpr double abs(double x) { return fabs(x); }

/*
template<typename T> inline constexpr double dble(T x) { return (double)x; }
template<typename T> inline constexpr int int_(T x) { return (int)x; }
*/

inline constexpr int dble(int x) { return x; }
inline constexpr int dble(int64_t x) { return (int)x; }
inline constexpr int dble(float x) { return (int)x; }
inline constexpr int dble(double x) { return (int)x; }

inline constexpr double int_(int x) { return (double)x; }
inline constexpr double int_(int64_t x) { return (double)x; }
inline constexpr double int_(float x) { return (double)x; }
inline constexpr double int_(double x) { return x; }

/*
template<typename T, typename... Ts>
inline constexpr T min(T x0, T x1, Ts... xs) {
    if constexpr (sizeof...(xs) == 0) {
        return x0 < x1 ? x0 : x1;
    } else {
        return min(min(x0, x1), xs...);
    }
}
*/

__device__ inline int min(int x0, int x1) { return ::min(x0, x1); }
__device__ inline int min(int x0, int x1, int x2) { return ::min(::min(x0, x1), x2); }
__device__ inline int min(int x0, int x1, int x2, int x3) { return ::min(::min(x0, x1), ::min(x2, x3)); }

__device__ inline int64_t min(int64_t x0, int64_t x1) { return ::min(x0, x1); }
__device__ inline int64_t min(int64_t x0, int64_t x1, int64_t x2) { return ::min(::min(x0, x1), x2); }
__device__ inline int64_t min(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return ::min(::min(x0, x1), ::min(x2, x3)); }

inline constexpr float min(float x0, float x1) { return fminf(x0, x1); }
inline constexpr float min(float x0, float x1, float x2) { return fminf(fminf(x0, x1), x2); }
inline constexpr float min(float x0, float x1, float x2, float x3) { return fminf(fminf(x0, x1), fminf(x2, x3)); }

inline constexpr double min(double x0, double x1) { return fmin(x0, x1); }
inline constexpr double min(double x0, double x1, double x2) { return fmin(fmin(x0, x1), x2); }
inline constexpr double min(double x0, double x1, double x2, double x3) { return fmin(fmin(x0, x1), fmin(x2, x3)); }

/*
template<typename T, typename... Ts>
inline constexpr T max(T x0, T x1, Ts... xs) {
    if constexpr (sizeof...(xs) == 0) {
        return x0 > x1 ? x0 : x1;
    } else {
        return max(max(x0, x1), xs...);
    }
}
*/

__device__ inline int max(int x0, int x1) { return ::max(x0, x1); }
__device__ inline int max(int x0, int x1, int x2) { return ::max(::max(x0, x1), x2); }
__device__ inline int max(int x0, int x1, int x2, int x3) { return ::max(::max(x0, x1), ::max(x2, x3)); }

__device__ inline int64_t max(int64_t x0, int64_t x1) { return ::max(x0, x1); }
__device__ inline int64_t max(int64_t x0, int64_t x1, int64_t x2) { return ::max(::max(x0, x1), x2); }
__device__ inline int64_t max(int64_t x0, int64_t x1, int64_t x2, int64_t x3) { return ::max(::max(x0, x1), ::max(x2, x3)); }

inline constexpr float max(float x0, float x1) { return fmaxf(x0, x1); }
inline constexpr float max(float x0, float x1, float x2) { return fmaxf(fmaxf(x0, x1), x2); }
inline constexpr float max(float x0, float x1, float x2, float x3) { return fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)); }

inline constexpr double max(double x0, double x1) { return fmax(x0, x1); }
inline constexpr double max(double x0, double x1, double x2) { return fmax(fmax(x0, x1), x2); }
inline constexpr double max(double x0, double x1, double x2, double x3) { return fmax(fmax(x0, x1), fmax(x2, x3)); }

/*
template<typename T> inline constexpr T mod(T a, T p) { return a % p; }
template<> inline constexpr float mod(float a, float p) { return fmodf(a, p); }
template<> inline constexpr double mod(double a, double p) { return fmod(a, p); }
*/

inline constexpr int mod(int a, int p) { return a % p; }
inline constexpr int64_t mod(int64_t a, int64_t p) { return a % p; }
inline constexpr float mod(float a, float p) { return fmodf(a, p); }
inline constexpr double mod(double a, double p) { return fmod(a, p); }

inline constexpr int nint(float x) { return lroundf(x); }
inline constexpr int nint(double x) { return lround(x); }

/*
template<typename T> inline constexpr T copysign(T x, T y) { return y >= 0 ? abs(x) : -abs(x); }
template<> inline constexpr float copysign(float x, float y) { return copysignf(x, y); }
template<> inline constexpr double copysign(double x, double y) { return ::copysign(x, y); }
*/

__device__ inline int copysign(int x, int y) { return y >= 0 ? abs(x) : -abs(x); }
__device__ inline int64_t copysign(int64_t x, int64_t y) { return y >= 0 ? abs(x) : -abs(x); }
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

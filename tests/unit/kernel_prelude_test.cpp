// The translated kernels' intrinsics on the host: an integer one takes
// whichever type int64_t is, a real one keeps its own overload, and MIN and
// MAX take any number of arguments, of mixed real kinds too.

#include <op2/rt/kernel/inc.h>
#include <op2/rt/kernel/prelude.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../test_support.h"

namespace k = op::rt::kernel;

namespace {

template<typename T, typename U> constexpr bool same = false;
template<typename T> constexpr bool same<T, T> = true;

} // namespace

int main() {
    int64_t a = -5;
    int64_t b = 3;
    long long c = -7;

    static_assert(same<decltype(k::abs(a)), int64_t>);
    static_assert(same<decltype(k::abs(c)), long long>);
    static_assert(same<decltype(k::abs(-2.5f)), float>);
    static_assert(same<decltype(k::min(a, b, a)), int64_t>);
    static_assert(same<decltype(k::min(1.0, 2.0f, 3.0)), double>);

    CHECK(k::abs(a) == 5 && k::abs(c) == 7 && k::abs(-3) == 3 && k::abs(-2.5) == 2.5);
    CHECK(k::min(a, b) == -5 && k::max(a, b) == 3);
    CHECK(k::mod(a, b) == -2 && k::copysign(b, a) == -3);
    CHECK(k::dble(a) == -5.0);
    CHECK(k::pow(a, 2) == 25 && k::pow(int64_t{2}, -1) == 0);
    CHECK(k::min(5, 4, 3, 2, 1) == 1 && k::max(1.0, 5.0, 2.0, 4.0, 3.0) == 5.0);
    CHECK(k::min(2.0f, 1.0, 3.0f) == 1.0 && k::max(2.0, 4.0f) == 4.0);

    int64_t sum = 0;
    k::atomic_inc(sum, int64_t{-4});
    k::atomic_inc(sum, 1);
    CHECK(sum == -3);

    std::printf("kernel prelude tests passed\n");
    return EXIT_SUCCESS;
}

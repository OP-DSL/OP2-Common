// The JIT's backend-neutral half on the host: the build cache's policy over
// fake kernels and jobs, the decision to specialise, and the parameters'
// values, keys and C++ constants.

#include <loop/jit.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace loop = op::loop;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

constexpr auto later = loop::Urgency::later;
constexpr auto now = loop::Urgency::now;
constexpr auto wait = loop::Urgency::wait;
constexpr auto ready = loop::Status::ready;

struct FakeKernel {
    int id;
};

using Cache = loop::JitCache<FakeKernel>;

// The builds started, and a gate each build waits at while it is closed.
struct Builds {
    std::atomic_int started = 0;
    std::atomic_bool open = true;

    // A make_job for kernel id, or for a build that fails.
    auto job(int id, bool fail = false) {
        return [this, id, fail] {
            ++started;
            return [this, id, fail]() -> std::unique_ptr<FakeKernel> {
                while (!open.load())
                    std::this_thread::yield();
                if (fail)
                    return nullptr;
                return std::make_unique<FakeKernel>(FakeKernel{id});
            };
        };
    }
};

// Read the policy from these variables alone.
void policy(std::initializer_list<std::pair<const char *, const char *>> variables) {
    for (const char *name : {"OP_JIT_ENABLE", "OP_JIT_FORCE", "OP_JIT_SEQ_COMPILE",
                             "OP_JIT_DEBUG", "OP_JIT_MIN_CALLS", "OP_JIT_MAX_THREADS"})
        unsetenv(name);
    for (const auto& [name, value] : variables)
        setenv(name, value, 1);
    loop::init_jit_policy();
}

// A key builds once asked for OP_JIT_MIN_CALLS times; until then the
// offline kernel runs, or nothing does.
void test_min_calls() {
    policy({{"OP_JIT_MIN_CALLS", "3"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    Cache cache{"kernel", 4};
    for (int call = 0; call < 2; ++call) {
        auto offline = cache.acquire(1, later, true, builds.job(1));
        CHECK(offline.status == ready && offline.kernel == nullptr);
        auto pending = cache.acquire(2, later, false, builds.job(2));
        CHECK(pending.status == loop::Status::pending && pending.reason == "compiling");
    }
    CHECK(builds.started == 0 && cache.state(1) == loop::BuildState::seen);

    auto first = cache.acquire(1, later, true, builds.job(1));
    CHECK(first.status == ready && first.kernel != nullptr && first.kernel->id == 1);
    auto second = cache.acquire(2, later, false, builds.job(2));
    CHECK(second.status == ready && second.kernel->id == 2);
    CHECK(cache.acquire(1, later, true, builds.job(1)).kernel == first.kernel);
    CHECK(builds.started == 2);
}

// now starts a build on a thread at once, the offline kernel running
// meanwhile; wait builds here when nothing else can run, and joins a build
// already running.
void test_urgency() {
    policy({{"OP_JIT_MIN_CALLS", "100"}});
    Builds builds;
    Cache cache{"kernel", 4};
    builds.open = false;
    auto acquired = cache.acquire(1, now, true, builds.job(1));
    CHECK(acquired.status == ready && acquired.kernel == nullptr);
    CHECK(cache.state(1) == loop::BuildState::building && builds.started == 1);
    acquired = cache.acquire(1, wait, true, builds.job(1));
    CHECK(acquired.status == ready && acquired.kernel == nullptr);
    acquired = cache.acquire(3, now, false, builds.job(3));
    CHECK(acquired.status == loop::Status::pending);

    builds.open = true;
    acquired = cache.acquire(1, wait, false, builds.job(1));
    CHECK(acquired.status == ready && acquired.kernel->id == 1);
    acquired = cache.acquire(2, wait, false, builds.job(2));
    CHECK(acquired.status == ready && acquired.kernel->id == 2);
    CHECK(builds.started == 3);

    cache.join();
    CHECK(cache.state(3) == loop::BuildState::ready);
    CHECK(cache.acquire(3, later, false, builds.job(3)).kernel->id == 3);
    CHECK(builds.started == 3);
}

// A failed build is never retried: the offline kernel runs, else the
// strategy is unavailable.
void test_failed_build() {
    policy({{"OP_JIT_MIN_CALLS", "1"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    Cache cache{"kernel", 4};
    for (int call = 0; call < 2; ++call) {
        auto failed = cache.acquire(1, later, false, builds.job(1, true));
        CHECK(failed.status == loop::Status::unavailable && failed.reason == "build_failed");
    }
    auto offline = cache.acquire(1, wait, true, builds.job(1, true));
    CHECK(offline.status == ready && offline.kernel == nullptr);
    CHECK(builds.started == 1 && cache.state(1) == loop::BuildState::failed);
}

// Past 256 keys the least recently asked-for half of those not started go;
// started ones stay.
void test_eviction() {
    policy({{"OP_JIT_MIN_CALLS", "1000"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    Cache cache{"kernel", 4};
    CHECK(cache.acquire(1000, now, false, builds.job(1000)).kernel != nullptr);
    for (std::uint64_t key = 1; key <= 300; ++key)
        cache.acquire(key, later, true, builds.job(1));
    CHECK(builds.started == 1);
    CHECK(cache.state(1) == loop::BuildState::unseen);
    CHECK(cache.state(300) == loop::BuildState::seen);
    CHECK(cache.state(1000) == loop::BuildState::ready);
    CHECK(cache.state(5000) == loop::BuildState::unseen);
}

// Builds on threads share OP_JIT_MAX_THREADS slots, the cache's default
// when it is unset; a build that is waited for takes none.
void test_build_slots() {
    for (bool variable : {true, false}) {
        if (variable)
            policy({{"OP_JIT_MIN_CALLS", "100"}, {"OP_JIT_MAX_THREADS", "1"}});
        else
            policy({{"OP_JIT_MIN_CALLS", "100"}});
        Builds builds;
        Cache cache{"kernel", variable ? 8 : 1};
        builds.open = false;
        cache.acquire(1, now, true, builds.job(1));
        cache.acquire(2, now, true, builds.job(2));
        CHECK(cache.state(1) == loop::BuildState::building);
        CHECK(cache.state(2) == loop::BuildState::seen && builds.started == 1);

        Builds waited;
        CHECK(cache.acquire(3, wait, false, waited.job(3)).kernel->id == 3);

        builds.open = true;
        cache.join();
        cache.acquire(2, now, true, builds.job(2));
        cache.join();
        CHECK(cache.state(2) == loop::BuildState::ready && builds.started == 2);
    }
}

// Specialise when the JIT is on and no offline kernel can run, the run
// forces it, or the mechanism's prior says a build pays.
void test_specialise() {
    policy({});
    const auto& jit = loop::jit_policy();
    CHECK(jit.specialise(false, false));
    CHECK(!jit.specialise(true, false));
    CHECK(jit.specialise(true, true));
    policy({{"OP_JIT_FORCE", "1"}});
    CHECK(jit.specialise(true, false));
    policy({{"OP_JIT_ENABLE", "0"}, {"OP_JIT_FORCE", "1"}});
    CHECK(!jit.specialise(false, true));
}

// The parameters take the call's and the strategy's values, key a build by
// them, and format as C++ constants exactly.
void test_params() {
    op_set_core set{};
    set.size = 100;
    op_dat_core dat{};
    dat.set = &set;
    double global = 2.5;
    op_arg args[3]{};
    args[0].opt = 1;
    args[0].argtype = OP_ARG_GBL;
    args[0].data = reinterpret_cast<char *>(&global);
    args[1].opt = 1;
    args[1].argtype = OP_ARG_DAT;
    args[1].dat = &dat;
    args[1].dim = 3;
    args[2].opt = 0;
    args[2].argtype = OP_ARG_GBL;

    double constant = 1.0;
    double table[3] = {1.0, std::numeric_limits<double>::quiet_NaN(),
                       -std::numeric_limits<double>::infinity()};
    double gbl0 = 0.0;
    double gbl2 = 7.0;
    int stride = 0;
    int dim = 0;
    bool flag = true;
    int64_t big = int64_t{1} << 40;

    loop::ParamSet params;
    params.add("c", &constant, 1, loop::ParamSource::external);
    params.add("table", table, 3, loop::ParamSource::external);
    params.add("g0", &gbl0, 1, loop::ParamSource::scalar_arg, 0);
    params.add("g2", &gbl2, 1, loop::ParamSource::scalar_arg, 2);
    params.add("stride", &stride, 1, loop::ParamSource::dat_stride, 1);
    params.add("dim", &dim, 1, loop::ParamSource::dat_dim, 1);
    params.add("block", static_cast<int *>(nullptr), 1, loop::ParamSource::strategy, 1);
    params.add("flag", &flag, 1, loop::ParamSource::external);
    params.add("big", &big, 1, loop::ParamSource::external);

    params.update(args, 3);
    CHECK(gbl0 == 2.5 && gbl2 == 7.0 && stride == 128 && dim == 3);
    std::array<int, 2> slots{0, 64};
    params.apply(slots);
    auto key = params.hash();

    global = 3.5;
    params.update(args, 3);
    CHECK(params.hash() != key);
    global = 2.5;
    params.update(args, 3);
    CHECK(params.hash() == key);
    slots[1] = 128;
    params.apply(slots);
    CHECK(params.hash() != key);
    slots[1] = 64;
    params.apply(slots);

    CHECK(loop::format_cxx(params) ==
          "static constexpr double c = 0x1p+0;\n"
          "static constexpr double table[3] = { 0x1p+0, "
          "__builtin_bit_cast(double, 0x7ff8000000000000ull), "
          "__builtin_bit_cast(double, 0xfff0000000000000ull) };\n"
          "static constexpr double g0 = 0x1.4p+1;\n"
          "static constexpr double g2 = 0x1.cp+2;\n"
          "static constexpr int stride = 128;\n"
          "static constexpr int dim = 3;\n"
          "static constexpr int block = 64;\n"
          "static constexpr bool flag = true;\n"
          "static constexpr int64_t big = 1099511627776;\n");
}

} // namespace

int main() {
    try {
        test_min_calls();
        test_urgency();
        test_failed_build();
        test_eviction();
        test_build_slots();
        test_specialise();
        test_params();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop JIT tests passed\n");
    return EXIT_SUCCESS;
}

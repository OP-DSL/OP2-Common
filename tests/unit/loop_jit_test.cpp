// The JIT's backend-neutral half on the host: the build cache's policy over
// fake kernels and jobs, the decision to specialise, the parameters' values,
// keys and C++ constants, and the modes and statistics that move them.

#include <loop/jit.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../test_support.h"

namespace loop = op::loop;

namespace {

constexpr auto later = loop::Urgency::later;
constexpr auto now = loop::Urgency::now;
constexpr auto wait = loop::Urgency::wait;
constexpr auto ready = loop::Status::ready;
constexpr auto baked = loop::ParamMode::baked;
constexpr auto runtime = loop::ParamMode::runtime;

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
    for (const char *name :
         {"OP_JIT_LEVEL", "OP_JIT_FORCE", "OP_JIT_SEQ_COMPILE", "OP_JIT_DEBUG",
          "OP_JIT_MIN_CALLS", "OP_JIT_MAX_THREADS", "OP_JIT_VARYING",
          "OP_JIT_MAX_GBL_BYTES", "OP_JIT_MAX_CONST_BYTES", "OP_JIT_RUNTIME_PARAMS",
          "OP_JIT_CONSTANT_PARAMS"})
        unsetenv(name);
    for (const auto& [name, value] : variables)
        setenv(name, value, 1);
    loop::init_jit_options();
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

// A strategy with the estimates it is given, for prepare to choose between,
// keeping the kind it chose.
struct Estimates : loop::Strategy {
    loop::Estimate offline;
    loop::Estimate jit;
    loop::KernelKind chosen = loop::KernelKind::offline;

    Estimates(loop::Estimate offline_, loop::Estimate jit_)
        : offline{offline_}, jit{jit_} {}
    std::string_view name() const override { return "estimates"; }
    loop::Estimate estimate(const loop::Call&,
                            loop::KernelKind kind) const override {
        return kind == loop::KernelKind::offline ? offline : jit;
    }
    loop::Prepared prepare_kind(const loop::Call&, loop::Urgency,
                                loop::KernelKind kind) override {
        chosen = kind;
        return {};
    }
    std::optional<float> run(const loop::Prepared&, const loop::Launch&) override {
        return std::nullopt;
    }
};

// A specialised kernel when the JIT is on and one can run, and the offline
// one cannot, OP_JIT_FORCE is set, or the backend expects it to be faster;
// a tie or a missing estimate keeps the offline kernel.
void test_choose_kernel() {
    op_set_core set{};
    loop::Call call{&set, {}, 0};
    auto choose = [&](loop::Estimate offline, loop::Estimate jit) {
        Estimates strategy{offline, jit};
        strategy.prepare(call, loop::Urgency::later);
        return strategy.chosen;
    };
    constexpr auto offline = loop::KernelKind::offline;
    constexpr auto jit = loop::KernelKind::jit;
    const loop::Estimate none{};
    const loop::Estimate unknown{true, std::nullopt};

    policy({});
    CHECK(choose({true, 1.0f}, {true, 0.9f}) == jit);
    CHECK(choose({true, 1.0f}, {true, 1.0f}) == offline);
    CHECK(choose({true, 1.0f}, {true, 1.1f}) == offline);
    CHECK(choose({true, 1.0f}, unknown) == offline);
    CHECK(choose(unknown, {true, 0.9f}) == offline);
    CHECK(choose(none, {true, 1.1f}) == jit);
    CHECK(choose({true, 1.0f}, none) == offline);
    CHECK(choose(none, none) == offline);
    policy({{"OP_JIT_FORCE", "1"}});
    CHECK(choose({true, 1.0f}, {true, 1.1f}) == jit);
    CHECK(choose({true, 1.0f}, none) == offline);
    policy({{"OP_JIT_LEVEL", "0"}, {"OP_JIT_FORCE", "1"}});
    CHECK(choose(none, {true, 0.9f}) == offline);
}

// The parameters take the call's and the strategy's values, key a build by
// them, and format as C++ constants exactly.
void test_params() {
    policy({});
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
    params.add("c", &constant, loop::ParamSource::external);
    params.add_array("table", table, 3, loop::ParamSource::external);
    params.add("g0", &gbl0, loop::ParamSource::scalar_arg, 0);
    params.add("g2", &gbl2, loop::ParamSource::scalar_arg, 2);
    params.add("stride", &stride, loop::ParamSource::dat_stride, 1);
    params.add("dim", &dim, loop::ParamSource::dat_dim, 1);
    params.add("block", static_cast<int *>(nullptr), loop::ParamSource::strategy, 1);
    params.add("flag", &flag, loop::ParamSource::external);
    params.add("big", &big, loop::ParamSource::external);

    loop::Call call{&set, {args, 3}, 100};
    params.update(args, 3);
    CHECK(gbl0 == 2.5 && gbl2 == 7.0 && stride == 128 && dim == 3);
    std::array<int, 2> slots{0, 64};
    auto key = params.key(call, slots);
    CHECK(params.key(call, slots) == key);

    global = 3.5;
    params.update(args, 3);
    CHECK(params.key(call, slots) != key);
    global = 2.5;
    params.update(args, 3);
    CHECK(params.key(call, slots) == key);
    slots[1] = 128;
    CHECK(params.key(call, slots) != key);
    slots[1] = 64;
    CHECK(params.key(call, slots) == key);

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

// A key tells a baked zero from a run-time parameter, and keeps each of more
// parameters than it hashes at a time.
void test_key_words() {
    op_set_core set{};
    loop::Call call{&set, {}, 0};
    std::array<double, 70> constants{};
    auto key = [&](const char *runtime) {
        if (runtime != nullptr)
            policy({{"OP_JIT_RUNTIME_PARAMS", runtime}});
        else
            policy({});
        loop::ParamSet params{"flux"};
        for (std::size_t i = 0; i < constants.size(); ++i)
            params.add("c" + std::to_string(i), &constants[i], loop::ParamSource::external,
                       -1, true);
        return params.key(call, {});
    };

    auto baked = key(nullptr);
    CHECK(key("c0") != baked && key("c69") != baked && key("c0") != key("c69"));
    constants[69] = 1.0;
    CHECK(key(nullptr) != baked);
    constants[69] = 0.0;
    CHECK(key(nullptr) == baked);
}

// The initial modes: structural parameters baked; constants baked and
// moving; READ globals at run time, moving at level 2; a cap or a name in
// the variables fixing either, unless the mechanism cannot give it at run
// time.
void test_modes() {
    double constant = 1.0;
    double table[3] = {1.0, 2.0, 3.0};
    double global = 0.0;
    int stride = 0;
    auto modes = [&] {
        loop::ParamSet params{"flux"};
        params.add("c", &constant, loop::ParamSource::external, -1, true);
        params.add_array("table", table, 3, loop::ParamSource::external, -1, true);
        params.add("g", &global, loop::ParamSource::scalar_arg, 0, true);
        params.add("stride", &stride, loop::ParamSource::dat_stride, 1, true);
        params.add("fixed", &constant, loop::ParamSource::external);
        std::string out;
        for (std::size_t i = 0; i < params.size(); ++i) {
            out += params[i].initial() == baked ? "b" : "r";
            out += params[i].fixed() ? " " : "* ";
        }
        return out;
    };

    policy({});
    CHECK(modes() == "b* b* r* b b ");
    policy({{"OP_JIT_LEVEL", "1"}});
    CHECK(modes() == "b* b* r b b ");
    policy({{"OP_JIT_MAX_CONST_BYTES", "8"}, {"OP_JIT_MAX_GBL_BYTES", "4"}});
    CHECK(modes() == "b* r r b b ");
    policy({{"OP_JIT_RUNTIME_PARAMS", "c,flux:stride,other:table,fixed"},
            {"OP_JIT_CONSTANT_PARAMS", "g"}});
    CHECK(modes() == "r b* b r b ");
}

// A constant that keeps changing moves to run time, and a global most calls
// agree on is baked in, each once, by its configuration's statistics; the
// key follows the modes.
void test_statistics() {
    policy({});
    std::array<op_set_core, 5> sets{};
    for (int i = 0; i < 5; ++i) {
        sets[i].index = i;
        sets[i].size = 10 * (i + 1);
        sets[i].core_size = sets[i].size;
    }
    double value = 0.0;
    op_arg args[1]{};
    args[0].opt = 1;
    args[0].argtype = OP_ARG_GBL;
    args[0].acc = OP_READ;
    args[0].dim = 1;
    args[0].data = reinterpret_cast<char *>(&value);

    double constant = 0.0;
    double global = 0.0;
    loop::ParamSet params{"flux"};
    params.add("c", &constant, loop::ParamSource::external, -1, true);
    params.add("g", &global, loop::ParamSource::scalar_arg, 0, true);
    auto next = [&](int set) {
        params.update(args, 1);
        loop::Call call{&sets[set], {args, 1}, sets[set].size};
        return params.key(call, {});
    };

    // Nine changes in 32 calls move the constant to run time, for good.
    for (int i = 0; i < 9; ++i) {
        constant = i;
        next(0);
        CHECK(params[0].mode() == baked);
    }
    constant = 9.0;
    auto key = next(0);
    CHECK(params[0].mode() == runtime && params[1].mode() == runtime);
    // A run-time value is no part of the key.
    for (int i = 0; i < 40; ++i) {
        value = i;
        CHECK(next(0) == key);
    }
    CHECK(params[0].mode() == runtime && params[1].mode() == runtime);

    // A global invariant for 32 calls of another configuration is baked in,
    // the constant there still baked; varying, it moves back for good.
    value = 5.0;
    for (int i = 0; i < 31; ++i) {
        next(1);
        CHECK(params[1].mode() == runtime && params[0].mode() == baked);
    }
    key = next(1);
    CHECK(params[1].mode() == baked);
    value = 6.0;
    CHECK(next(1) != key);
    for (int i = 0; i < 7; ++i) {
        value = 7.0 + i;
        next(1);
        CHECK(params[1].mode() == baked);
    }
    value = 100.0;
    next(1);
    CHECK(params[1].mode() == runtime);
    value = 5.0;
    for (int i = 0; i < 40; ++i)
        next(1);
    CHECK(params[1].mode() == runtime);
    next(0);
    CHECK(params[0].mode() == runtime);

    // Nine calls in ten agreeing is a majority; two in three is not.
    for (int i = 0; i < 32; ++i) {
        value = i % 10 == 9 ? 1.0 : 2.0;
        next(2);
    }
    CHECK(params[1].mode() == baked);
    for (int i = 0; i < 64; ++i) {
        value = i % 3 == 2 ? 1.0 : 2.0;
        next(3);
    }
    CHECK(params[1].mode() == runtime);

    // A change stops counting as it leaves the 32-call window: a change in
    // every four calls keeps a constant baked, one in every three does not.
    for (int i = 0; i < 128; ++i) {
        constant = i / 4;
        next(4);
        CHECK(params[0].mode() == baked);
    }
    for (int i = 0; i < 32; ++i) {
        constant = -1.0 - i / 3;
        next(4);
    }
    CHECK(params[0].mode() == runtime);
}

// A READ global of any dimension is read in place with the call's
// dimension, at run time when inactive or above OP_JIT_MAX_GBL_BYTES (and
// then never hashed), and otherwise moves like a scalar global; an array of
// one element is still an array.
void test_array_globals() {
    policy({});
    op_set_core set{};
    set.size = 10;
    set.core_size = 10;
    double g5[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double one = 1.0;
    std::vector<double> big(1000, 0.0);
    op_arg args[4]{};
    for (auto& arg : args) {
        arg.opt = 1;
        arg.argtype = OP_ARG_GBL;
        arg.acc = OP_READ;
    }
    args[0].data = reinterpret_cast<char *>(g5);
    args[0].dim = 5;
    args[1].opt = 0;
    args[2].data = reinterpret_cast<char *>(&one);
    args[2].dim = 1;
    args[3].data = reinterpret_cast<char *>(big.data());
    args[3].dim = 1000;

    loop::ParamSet params{"flux"};
    for (int i = 0; i < 4; ++i)
        params.add_array(std::string{"g"} + std::to_string(i),
                         static_cast<double *>(nullptr), 0,
                         loop::ParamSource::array_arg, i, true);
    loop::Call call{&set, {args, 4}, 10};
    auto next = [&] {
        params.update(args, 4);
        return params.key(call, {});
    };

    auto key = next();
    CHECK(params[0].data() == g5 && params[0].elems() == 5 && params[0].array());
    CHECK(params[1].elems() == 0 && params[2].elems() == 1 && params[2].array());
    for (int i = 0; i < 4; ++i)
        CHECK(params[i].mode() == runtime);
    CHECK(params[0].last_hash() != 0 && params[3].last_hash() == 0);
    g5[0] = 10.0;
    CHECK(next() == key);

    // The new value leads from the third call, so 32 calls later it is in.
    for (int i = 0; i < 32; ++i)
        next();
    CHECK(params[0].mode() == baked && params[2].mode() == baked);
    CHECK(params[1].mode() == runtime && params[3].mode() == runtime);
    key = next();
    g5[1] = 20.0;
    CHECK(next() != key);
    CHECK(loop::format_cxx(params) ==
          "static constexpr double g0[5] = { 0x1.4p+3, 0x1.4p+4, 0x1.8p+1, 0x1p+2, 0x1.4p+2 };\n"
          "static constexpr double g2[1] = { 0x1p+0 };\n");

    policy({{"OP_JIT_MAX_GBL_BYTES", "16"}});
    loop::ParamSet capped{"flux"};
    capped.add_array("g0", static_cast<double *>(nullptr), 0,
                     loop::ParamSource::array_arg, 0, true);
    for (int i = 0; i < 40; ++i) {
        capped.update(args, 4);
        capped.key(call, {});
    }
    CHECK(capped[0].mode() == runtime);
}

// A call observed without any key still counts: a global invariant for 32
// such calls is baked in when the first key is asked for, and that key's
// call is not counted twice.
void test_observe_every_call() {
    policy({});
    op_set_core set{};
    set.size = 10;
    set.core_size = 10;
    double value = 5.0;
    op_arg args[1]{};
    args[0].opt = 1;
    args[0].argtype = OP_ARG_GBL;
    args[0].acc = OP_READ;
    args[0].dim = 1;
    args[0].data = reinterpret_cast<char *>(&value);
    double global = 0.0;
    loop::ParamSet params{"flux"};
    params.add("g", &global, loop::ParamSource::scalar_arg, 0, true);
    CHECK(params.moving());

    loop::Call call{&set, {args, 1}, 10};
    for (int i = 0; i < 31; ++i) {
        params.update(args, 1);
        params.observe_call(call);
    }
    CHECK(params[0].mode() == runtime);
    params.update(args, 1);
    params.observe_call(call);
    auto key = params.key(call, {});
    CHECK(params[0].mode() == baked && params.key(call, {}) == key);
}

} // namespace

int main() {
    try {
        test_min_calls();
        test_urgency();
        test_failed_build();
        test_eviction();
        test_build_slots();
        test_choose_kernel();
        test_params();
        test_key_words();
        test_modes();
        test_statistics();
        test_array_globals();
        test_observe_every_call();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop JIT tests passed\n");
    return EXIT_SUCCESS;
}

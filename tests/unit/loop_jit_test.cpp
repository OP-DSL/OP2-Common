// The JIT's backend-neutral half on the host: the build cache's policy over
// fake kernels and jobs, the parameters' values, keys and C++ constants, and
// the modes and statistics that move them.

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/gpu/layout.h>
#include <op2/rt/jit/cache.h>
#include <op2/rt/jit/options.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>

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

namespace abi = op::rt::abi;
namespace jit = op::rt::jit;
namespace loop = op::rt::loop;

namespace {

// A parameter row as generated code describes one: one T at data, or an
// array of elems; runtime is whether a specialised kernel can have it at run
// time.
template<typename T>
abi::Param row(const char *name, T *data, abi::ParamSource source, int index = -1,
               bool runtime = false) {
    return {.name = name, .type = abi::ElementTypes<T>::value, .elem_size = sizeof(T),
            .source = source, .data = data, .elems = 1, .array = false, .index = index,
            .runtime = runtime};
}

template<typename T>
abi::Param array_row(const char *name, T *data, std::size_t elems, abi::ParamSource source,
                     int index = -1, bool runtime = false) {
    auto array = row(name, data, source, index, runtime);
    array.elems = elems;
    array.array = true;
    return array;
}

constexpr auto later = loop::Urgency::later;
constexpr auto now = loop::Urgency::now;
constexpr auto wait = loop::Urgency::wait;
constexpr auto ready = loop::Status::ready;
constexpr auto baked = jit::ParamMode::baked;
constexpr auto runtime = jit::ParamMode::runtime;

struct FakeKernel {
    int id;
};

using Cache = jit::Cache<FakeKernel>;

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

// The policy every cache and parameter set here follows.
jit::Options options;

// Read the policy from these variables alone.
void policy(std::initializer_list<std::pair<const char *, const char *>> variables) {
    for (const char *name :
         {"OP_JIT_LEVEL", "OP_JIT_FORCE", "OP_JIT_SEQ_COMPILE", "OP_JIT_DEBUG",
          "OP_JIT_MIN_CALLS", "OP_JIT_MAX_THREADS", "OP_JIT_MAX_VALUES",
          "OP_JIT_MAX_GBL_BYTES", "OP_JIT_MAX_CONST_BYTES", "OP_JIT_RUNTIME_PARAMS",
          "OP_JIT_CONSTANT_PARAMS"})
        unsetenv(name);
    for (const auto& [name, value] : variables)
        setenv(name, value, 1);
    options = jit::read_options();
}

// A key builds once asked for OP_JIT_MIN_CALLS times, pending until then.
void test_min_calls() {
    policy({{"OP_JIT_MIN_CALLS", "3"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    jit::BuildPool pool;
    pool.set_limit(4);
    Cache cache{"kernel", options, pool};
    for (int call = 0; call < 2; ++call) {
        auto pending = cache.acquire(1, later, builds.job(1));
        CHECK(pending.status == loop::Status::pending && pending.reason == "compiling");
    }
    CHECK(builds.started == 0 && cache.counts().waiting == 1);

    auto first = cache.acquire(1, later, builds.job(1));
    CHECK(first.status == ready && first.kernel != nullptr && first.kernel->id == 1);
    CHECK(cache.acquire(1, later, builds.job(1)).kernel == first.kernel);
    CHECK(builds.started == 1 && cache.counts().built == 1);
}

// now starts a build on the pool at once, pending meanwhile; wait builds
// here, and waits for a build under way.
void test_urgency() {
    policy({{"OP_JIT_MIN_CALLS", "100"}});
    Builds builds;
    jit::BuildPool pool;
    pool.set_limit(4);
    Cache cache{"kernel", options, pool};
    builds.open = false;
    auto acquired = cache.acquire(1, now, builds.job(1));
    CHECK(acquired.status == loop::Status::pending);
    CHECK(cache.counts().building == 1 && builds.started == 1);
    acquired = cache.acquire(3, now, builds.job(3));
    CHECK(acquired.status == loop::Status::pending);

    builds.open = true;
    acquired = cache.acquire(1, wait, builds.job(1));
    CHECK(acquired.status == ready && acquired.kernel->id == 1);
    acquired = cache.acquire(2, wait, builds.job(2));
    CHECK(acquired.status == ready && acquired.kernel->id == 2);
    CHECK(builds.started == 3);

    acquired = cache.acquire(3, wait, builds.job(3));
    CHECK(acquired.status == ready && acquired.kernel->id == 3);
    CHECK(builds.started == 3);
}

// A failed build is never retried: its kernel is unavailable.
void test_failed_build() {
    policy({{"OP_JIT_MIN_CALLS", "1"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    jit::BuildPool pool;
    pool.set_limit(4);
    Cache cache{"kernel", options, pool};
    for (auto urgency : {later, later, wait}) {
        auto failed = cache.acquire(1, urgency, builds.job(1, true));
        CHECK(failed.status == loop::Status::unavailable && failed.reason == "build_failed");
    }
    CHECK(builds.started == 1 && cache.counts().failed == 1);
}

// Past 256 keys the least recently asked-for half of those not started go;
// started ones stay.
void test_eviction() {
    policy({{"OP_JIT_MIN_CALLS", "1000"}, {"OP_JIT_SEQ_COMPILE", "1"}});
    Builds builds;
    jit::BuildPool pool;
    pool.set_limit(4);
    Cache cache{"kernel", options, pool};
    CHECK(cache.acquire(1000, now, builds.job(1000)).kernel != nullptr);
    for (std::uint64_t key = 1; key <= 300; ++key)
        cache.acquire(key, later, builds.job(1));
    // The 257th entry evicted keys 1 to 128.
    auto counts = cache.counts();
    CHECK(builds.started == 1 && counts.built == 1 && counts.waiting == 172);
    cache.acquire(300, later, builds.job(1));
    CHECK(cache.counts().waiting == 172);
    cache.acquire(1, later, builds.job(1));
    CHECK(cache.counts().waiting == 173);
}

// Builds on the pool start while fewer than its limit are under way, across
// every cache, a later call starting one that could not; a build that is
// waited for runs here, taking no place.
void test_pool() {
    policy({{"OP_JIT_MIN_CALLS", "100"}});
    Builds builds;
    jit::BuildPool pool;
    pool.set_limit(1);
    Cache cache{"kernel", options, pool};
    Cache other{"other", options, pool};
    builds.open = false;
    cache.acquire(1, now, builds.job(1));
    cache.acquire(2, now, builds.job(2));
    other.acquire(1, now, builds.job(1));
    CHECK(cache.counts().building == 1 && cache.counts().waiting == 1);
    CHECK(other.counts().waiting == 1 && builds.started == 1);

    Builds waited;
    CHECK(cache.acquire(3, wait, waited.job(3)).kernel->id == 3);

    // Once the build under way is published its place is free: the next
    // starts.
    builds.open = true;
    CHECK(cache.acquire(1, wait, builds.job(1)).kernel->id == 1);
    cache.acquire(2, now, builds.job(2));
    CHECK(builds.started == 2);
    CHECK(cache.acquire(2, wait, builds.job(2)).kernel->id == 2 && builds.started == 2);
}

// The parameters take the call's and the strategy's values, key a build by
// them, and format as C++ constants exactly.
void test_params() {
    policy({});
    op_set_core set{};
    set.name = "set";
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

    jit::ParamSet params{"flux", options};
    params.set_dat_stride(&op::rt::gpu::dat_stride);
    params.add(row("c", &constant, abi::ParamSource::external));
    params.add(array_row("table", table, 3, abi::ParamSource::external));
    params.add(row("g0", &gbl0, abi::ParamSource::scalar_arg, 0));
    params.add(row("g2", &gbl2, abi::ParamSource::scalar_arg, 2));
    params.add(row("stride", &stride, abi::ParamSource::dat_stride, 1));
    params.add(row("dim", &dim, abi::ParamSource::dat_dim, 1));
    params.add(row("block", static_cast<int *>(nullptr), abi::ParamSource::strategy, 1));
    params.add(row("flag", &flag, abi::ParamSource::external));
    params.add(row("big", &big, abi::ParamSource::external));

    loop::Call call{&set, {args, 3}, 100, false};
    params.update(args);
    CHECK(gbl0 == 2.5 && gbl2 == 7.0 && stride == 128 && dim == 3);
    std::array<int, 2> slots{0, 64};
    auto key = params.key(call, slots);
    CHECK(params.key(call, slots) == key);

    global = 3.5;
    params.update(args);
    CHECK(params.key(call, slots) != key);
    global = 2.5;
    params.update(args);
    CHECK(params.key(call, slots) == key);
    slots[1] = 128;
    CHECK(params.key(call, slots) != key);
    slots[1] = 64;
    CHECK(params.key(call, slots) == key);

    CHECK(jit::format_cxx(params, slots) ==
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
    set.name = "set";
    loop::Call call{&set, {}, 0, false};
    std::array<double, 70> constants{};
    std::vector<std::string> names;
    for (std::size_t i = 0; i < constants.size(); ++i)
        names.push_back("c" + std::to_string(i));
    auto key = [&](const char *runtime) {
        if (runtime != nullptr)
            policy({{"OP_JIT_RUNTIME_PARAMS", runtime}});
        else
            policy({});
        jit::ParamSet params{"flux", options};
        for (std::size_t i = 0; i < constants.size(); ++i)
            params.add(abi::constant(names[i].c_str(), &constants[i]));
        params.update({});
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
        jit::ParamSet params{"flux", options};
        params.add(row("c", &constant, abi::ParamSource::external, -1, true));
        params.add(array_row("table", table, 3, abi::ParamSource::external, -1, true));
        params.add(row("g", &global, abi::ParamSource::scalar_arg, 0, true));
        params.add(row("stride", &stride, abi::ParamSource::dat_stride, 1, true));
        params.add(row("fixed", &constant, abi::ParamSource::external));
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

// A constant that takes more than OP_JIT_MAX_VALUES values moves to run
// time, and a global that takes no more in 32 calls is baked in, each once,
// by its configuration's statistics; the key follows the modes.
void test_statistics() {
    policy({});
    std::array<op_set_core, 6> sets{};
    for (int i = 0; i < 6; ++i) {
        sets[i].name = "set";
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
    jit::ParamSet params{"flux", options};
    params.add(row("c", &constant, abi::ParamSource::external, -1, true));
    params.add(row("g", &global, abi::ParamSource::scalar_arg, 0, true));
    auto next = [&](int set) {
        params.update(args);
        loop::Call call{&sets[set], {args, 1}, sets[set].size, false};
        return params.key(call, {});
    };

    // A ninth value moves the constant to run time, for good.
    for (int i = 0; i < 8; ++i) {
        constant = i;
        next(0);
        CHECK(params[0].mode() == baked);
    }
    constant = 8.0;
    auto key = next(0);
    CHECK(params[0].mode() == runtime && params[1].mode() == runtime);
    // A run-time value is no part of the key.
    for (int i = 0; i < 40; ++i) {
        value = i;
        CHECK(next(0) == key);
    }
    CHECK(params[0].mode() == runtime && params[1].mode() == runtime);

    // A global invariant for 32 calls of another configuration is baked in,
    // the constant there still baked; taking a ninth value since, it moves
    // back for good.
    value = 5.0;
    for (int i = 0; i < 31; ++i) {
        next(1);
        CHECK(params[1].mode() == runtime && params[0].mode() == baked);
    }
    key = next(1);
    CHECK(params[1].mode() == baked);
    value = 6.0;
    CHECK(next(1) != key);
    for (int i = 0; i < 6; ++i) {
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

    // Values are counted, not shares: a flag set in one call in ten, or in
    // one in three, is baked in alike.
    for (int i = 0; i < 32; ++i) {
        value = i % 10 == 9 ? 1.0 : 2.0;
        next(2);
    }
    CHECK(params[1].mode() == baked);
    for (int i = 0; i < 32; ++i) {
        value = i % 3 == 2 ? 1.0 : 2.0;
        next(3);
    }
    CHECK(params[1].mode() == baked);

    // Values are counted, not changes: a constant cycling through eight
    // values every call stays baked, however long; a ninth value moves it.
    for (int i = 0; i < 128; ++i) {
        constant = i % 8;
        next(4);
        CHECK(params[0].mode() == baked);
    }
    constant = 100.0;
    next(4);
    CHECK(params[0].mode() == runtime);

    // A global that varies, then settles, is baked in once it has: its values
    // are counted afresh every 32 calls.
    for (int i = 0; i < 40; ++i) {
        value = 10.0 + i;
        next(5);
        CHECK(params[1].mode() == runtime);
    }
    value = 3.0;
    for (int i = 0; i < 64; ++i)
        next(5);
    CHECK(params[1].mode() == baked);

    // Each configuration's modes as the reports read them: c moved to run
    // time on sets 0 and 4, g baked in on sets 2 to 5 and moved on 1 to 5.
    std::size_t c_moved = 0, g_baked = 0, g_moved = 0;
    auto configurations = params.configurations();
    CHECK(configurations.size() == 6);
    for (const auto& configuration : configurations) {
        CHECK(configuration.params.size() == 2);
        for (const auto& outcome : configuration.params) {
            if (outcome.param == &params[0]) {
                c_moved += outcome.mode == runtime && outcome.moved;
            } else {
                g_baked += outcome.mode == baked;
                g_moved += outcome.moved;
            }
        }
    }
    CHECK(c_moved == 2 && g_baked == 4 && g_moved == 5);
}

// A READ global of any dimension is read in place with the call's
// dimension, at run time when inactive or above OP_JIT_MAX_GBL_BYTES (and
// then never hashed), and otherwise moves like a scalar global; an array of
// one element is still an array.
void test_array_globals() {
    policy({});
    op_set_core set{};
    set.name = "set";
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

    jit::ParamSet params{"flux", options};
    const char *names[] = {"g0", "g1", "g2", "g3"};
    for (int i = 0; i < 4; ++i)
        params.add(abi::global<double>(names[i], i));
    loop::Call call{&set, {args, 4}, 10, false};
    auto next = [&] {
        params.update(args);
        return params.key(call, {});
    };

    auto key = next();
    CHECK(params[0].data() == g5 && params[0].elems() == 5 && params[0].array());
    CHECK(params[1].elems() == 0 && params[2].elems() == 1 && params[2].array());
    for (int i = 0; i < 4; ++i)
        CHECK(params[i].mode() == runtime);
    CHECK(params[0].hash() != 0 && params[3].hash() == 0);
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
    CHECK(jit::format_cxx(params, {}) ==
          "static constexpr double g0[5] = { 0x1.4p+3, 0x1.4p+4, 0x1.8p+1, 0x1p+2, 0x1.4p+2 };\n"
          "static constexpr double g2[1] = { 0x1p+0 };\n");

    policy({{"OP_JIT_MAX_GBL_BYTES", "16"}});
    jit::ParamSet capped{"flux", options};
    capped.add(abi::global<double>("g0", 0));
    for (int i = 0; i < 40; ++i) {
        capped.update(args);
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
    set.name = "set";
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
    jit::ParamSet params{"flux", options};
    params.add(row("g", &global, abi::ParamSource::scalar_arg, 0, true));
    CHECK(params.moving());

    loop::Call call{&set, {args, 1}, 10, false};
    for (int i = 0; i < 31; ++i) {
        params.update(args);
        params.observe_call(call);
    }
    CHECK(params[0].mode() == runtime);
    params.update(args);
    params.observe_call(call);
    auto key = params.key(call, {});
    CHECK(params[0].mode() == baked && params.key(call, {}) == key);
}

// A configuration's modes are settled once the statistics have seen a window
// of its calls; at once with nothing that can move, or with the JIT off.
void test_settled() {
    policy({});
    op_set_core set{};
    set.name = "set";
    set.size = 10;
    set.core_size = 10;
    double value = 5.0;
    op_arg args[1]{};
    args[0].opt = 1;
    args[0].argtype = OP_ARG_GBL;
    args[0].acc = OP_READ;
    args[0].dim = 1;
    args[0].data = reinterpret_cast<char *>(&value);
    loop::Call call{&set, {args, 1}, 10, false};

    double global = 0.0;
    jit::ParamSet params{"flux", options};
    params.add(row("g", &global, abi::ParamSource::scalar_arg, 0, true));
    for (std::uint64_t i = 0; i < jit::statistics_window; ++i) {
        CHECK(!params.settled(call));
        params.update(args);
        params.observe_call(call);
    }
    CHECK(params.settled(call));

    int stride = 0;
    jit::ParamSet fixed{"flux", options};
    fixed.add(row("stride", &stride, abi::ParamSource::dat_stride, 0, true));
    CHECK(!fixed.moving() && fixed.settled(call));

    jit::ParamSet unobserved{"flux", options};
    unobserved.add(row("g", &global, abi::ParamSource::scalar_arg, 0, true));
    policy({{"OP_JIT_LEVEL", "0"}});
    CHECK(unobserved.settled(call));
    policy({});
}

} // namespace

int main() {
    try {
        test_min_calls();
        test_urgency();
        test_failed_build();
        test_eviction();
        test_pool();
        test_params();
        test_key_words();
        test_modes();
        test_statistics();
        test_array_globals();
        test_observe_every_call();
        test_settled();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop JIT tests passed\n");
    return EXIT_SUCCESS;
}

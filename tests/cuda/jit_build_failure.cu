// A specialised kernel whose compile fails: its strategy is unavailable and
// the ladder runs the next, or its offline kernel runs instead, and the run
// goes on.

#define OP_F2C_JIT_TAG jit_build_failure_test
#include <kernel/jit_embed.h>

#include <gpu/loop.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

extern "C" __global__ void build_failure_offline(int *result, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        result[0] = value;
}

namespace {

namespace gpu = op::gpu;

constexpr char broken_source[] = R"op2(
extern "C" __global__ void build_failure_offline(int *result, int value) {
    this does not compile;
}
)op2";

int *output_d = nullptr;
int value = 0;

void bind(gpu::LaunchContext&, op_arg *, std::vector<void *>& arguments) {
    arguments = {&output_d, &value};
}

bool env_enabled(const char *name) {
    const char *text = std::getenv(name);
    return text != nullptr && text[0] != '\0' && text[0] != '0';
}

gpu::JitHeaders headers() {
    return {{OP_F2C_PRELUDE_DATA, OP_STAGING_LAYOUT_DATA, OP_STRATEGY_KERNEL_DATA},
            {"kernel/prelude.h", "plan/staging_layout.h", "kernel/strategy.h"}};
}

// The kernel's result after one call of loop.
int run(gpu::Loop& loop, op_set_core& set, std::array<op_arg, 1>& args, int call) {
    value = call;
    loop.invoke(&set, args.data(), static_cast<int>(args.size()));
    int output = -1;
    GPU_CHECK(gpuMemcpy(&output, output_d, sizeof(output), gpuMemcpyDeviceToHost));
    return output;
}

bool check(bool condition, const char *what) {
    if (!condition)
        std::fprintf(stderr, "jit build failure: %s\n", what);
    return condition;
}

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);
    setenv("OP_JIT_MIN_CALLS", "1", 1);
    op::loop::init();

    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&output_d), sizeof(int)));

    op_set_core set{};
    set.size = 64;
    set.core_size = 64;
    set.name = "set";
    std::array<op_arg, 1> args{};
    args[0].opt = 0;
    args[0].argtype = OP_ARG_GBL;

    bool jit = env_enabled("OP_EXPECT_JIT");
    bool passed = true;
    {
        // A strategy with only the broken source, then a working floor.
        gpu::Loop loop("build_failure", "c_CUDA", "Direct", {true, false}, headers());
        loop.add<gpu::Simple>("broken", op::loop::Default::enabled,
                              gpu::KernelSource{"build_failure_offline", nullptr,
                                                broken_source},
                              &bind);
        loop.add<gpu::Simple>(
            "plain", op::loop::Default::enabled,
            gpu::KernelSource{"build_failure_offline",
                              reinterpret_cast<const void *>(build_failure_offline),
                              broken_source},
            &bind);

        for (int call = 1; call <= 3; ++call)
            passed &= check(run(loop, set, args, call) == call, "result");

        op::loop::Configuration configuration{&set, args, set.size};
        auto broken = loop.ladder()[0].prepare(configuration, op::loop::Urgency::now);
        passed &= check(broken.status == op::loop::Status::unavailable,
                        "the broken strategy is available");
        std::string_view expected = jit ? "build_failed" : "kernel_unavailable";
        passed &= check(broken.reason == expected, "the broken strategy's reason");
        auto selection = loop.ladder().select(configuration);
        passed &= check(selection.strategy->family() == "plain" &&
                            selection.kernel == nullptr,
                        "the floor runs its offline kernel");
    }

    GPU_CHECK(gpuFree(output_d));
    op_exit();
    if (!passed)
        return EXIT_FAILURE;
    std::printf("JIT build failure %s passed\n", jit ? "fallback" : "offline");
    return EXIT_SUCCESS;
}

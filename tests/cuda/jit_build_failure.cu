// A specialised kernel whose compile fails: its strategy is unavailable and
// the ladder runs the next, its offline kernel, and the run goes on.

#define OP_KERNEL_JIT_TAG jit_build_failure_test
#include <op2/rt/kernel/gpu/jit_embed.h>

#include <op2/rt/gpu/loop.h>
#include <op2/rt/lifecycle.h>
#include <op2/rt/gpu/strategy.h>
#include <op2/rt/kernel/gpu/offline.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" __global__ void build_failure_offline(int *result, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        result[0] = value;
}

namespace {

namespace gpu = op::rt::gpu;

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

gpu::JitHeaders headers() { return OP_KERNEL_JIT_HEADERS; }

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
    op::rt::init();
    op::rt::gpu::init();
    // A broken family, atomics' plain strategy, ahead of the working plain one.
    const op::rt::loop::Family families[] = {{"atomics", gpu::Simple::axes()},
                                         {"plain", gpu::Simple::axes()}};
    op::rt::loop::declare_families(families);

    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&output_d), sizeof(int)));

    op_set_core set{};
    set.size = 64;
    set.core_size = 64;
    set.name = "set";
    std::array<op_arg, 1> args{};
    args[0].opt = 0;
    args[0].argtype = OP_ARG_GBL;

    bool passed = true;
    {
        // A strategy with only the broken source, then a working floor.
        gpu::Loop loop("build_failure", "cuda", "Direct", true, false, headers());
        loop.add("atomics", gpu::KernelSource{"build_failure_offline", {}, broken_source},
                 &bind);
        loop.add("plain",
                 gpu::KernelSource{"build_failure_offline",
                                   op::rt::kernel::gpu::offline(
                                       reinterpret_cast<const void *>(build_failure_offline)),
                                   broken_source},
                 &bind);

        for (int call = 1; call <= 3; ++call)
            passed &= check(run(loop, set, args, call) == call, "result");

        op::rt::loop::Call call{&set, args, set.size};
        auto broken = loop.ladder()[0].prepare(call, op::rt::loop::Urgency::now);
        passed &= check(broken.status == op::rt::loop::Status::unavailable,
                        "the broken strategy is available");
        passed &= check(broken.reason == "build_failed", "the broken strategy's reason");
        auto selection = loop.ladder().select(call);
        passed &= check(selection.candidate->name.starts_with("plain=") &&
                            selection.kernel == nullptr,
                        "the floor runs its offline kernel");
    }

    GPU_CHECK(gpuFree(output_d));
    op_exit();
    if (!passed)
        return EXIT_FAILURE;
    std::printf("JIT build failure passed\n");
    return EXIT_SUCCESS;
}

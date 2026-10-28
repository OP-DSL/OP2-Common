// A kernel with more registers than a 1024-thread block leaves: compiled
// without launch bounds its specialised kernel cannot launch that block and
// is unavailable (thread_limit); with them it runs, with two blocks a
// multiprocessor its registers are capped to fit both, and with
// shared-memory spilling (CUDA 13) it runs too, while a minimum of blocks or
// spilling without launch bounds is unavailable (needs_launch_bounds).

#define OP_KERNEL_JIT_TAG jit_thread_limit_test
#include <kernel/jit_embed.h>

#include <gpu/loop.h>
#include <op_lib_c.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace gpu = op::gpu;
namespace loop = op::loop;

constexpr int width = 64;
constexpr int elements = 4096;

constexpr char source[] = R"op2(
extern "C" __global__ OP2_JIT_LAUNCH_BOUNDS
void thread_limit_kernel(double *out, const double *in, int start, int end) {
    OP2_JIT_SMEM_SPILLING
    for (int e = start + threadIdx.x + blockIdx.x * blockDim.x; e < end;
         e += blockDim.x * gridDim.x) {
        double a[64];
#pragma unroll
        for (int i = 0; i < 64; ++i)
            a[i] = in[e + i * 4096];
#pragma unroll 1
        for (int r = 0; r < 16; ++r) {
#pragma unroll
            for (int i = 0; i < 64; ++i)
                a[i] = a[i] * 0.999 + a[(i + 1) % 64] * 0.001;
        }
        double s = 0.0;
#pragma unroll
        for (int i = 0; i < 64; ++i)
            s += a[i];
        out[e] = s;
    }
}
)op2";

double *out_d = nullptr;
double *in_d = nullptr;

void bind(gpu::LaunchContext& context, op_arg *, std::vector<void *>& arguments) {
    arguments = {&out_d, &in_d, &context.start, &context.end};
}

gpu::JitHeaders headers() {
    return {{OP_KERNEL_PRELUDE_DATA, OP_STAGING_LAYOUT_DATA, OP_STRATEGY_KERNEL_DATA},
            {"kernel/prelude.h", "plan/staging_layout.h", "kernel/strategy.h"}};
}

std::vector<double> reference(const std::vector<double>& in) {
    std::vector<double> out(elements);
    for (int e = 0; e < elements; ++e) {
        std::array<double, width> a;
        for (int i = 0; i < width; ++i)
            a[i] = in[e + i * elements];
        for (int r = 0; r < 16; ++r)
            for (int i = 0; i < width; ++i)
                a[i] = a[i] * 0.999 + a[(i + 1) % width] * 0.001;
        double s = 0.0;
        for (int i = 0; i < width; ++i)
            s += a[i];
        out[e] = s;
    }
    return out;
}

// Whether every expectation so far held.
bool passed = true;

void expect(bool condition, const std::string& what) {
    if (!condition)
        std::fprintf(stderr, "jit thread limit: %s\n", what.c_str());
    passed &= condition;
}

// Declare plain at one value of each switch, and read the JIT variables.
void declare(const char *launch_bounds, const char *smem_spilling,
             const char *min_blocks = "1") {
    setenv("OP_LAUNCH_BOUNDS", launch_bounds, 1);
    setenv("OP_MIN_BLOCKS", min_blocks, 1);
    if (gpu::smem_spilling_supported)
        setenv("OP_SMEM_SPILLING", smem_spilling, 1);
    loop::init();
    gpu::init();
    const loop::Family families[] = {{"plain", gpu::Simple::axes()}};
    loop::declare_families(families);
}

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);
    setenv("OP_BLOCK_SIZES", "1024", 1);

    std::vector<double> in(static_cast<std::size_t>(width) * elements);
    for (std::size_t i = 0; i < in.size(); ++i)
        in[i] = 1.0 + static_cast<double>(i % 97) * 1e-3;
    auto expected = reference(in);
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&in_d), in.size() * sizeof(double)));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&out_d), elements * sizeof(double)));
    GPU_CHECK(gpuMemcpy(in_d, in.data(), in.size() * sizeof(double), gpuMemcpyHostToDevice));

    op_set_core set{};
    set.size = elements;
    set.core_size = elements;
    set.name = "set";
    std::array<op_arg, 1> args{};
    args[0].opt = 0;
    args[0].argtype = OP_ARG_GBL;
    loop::Call call{&set, args, set.size};

    // The loop's one candidate, as prepared: its specialised kernel, or the
    // offline candidate kept to refuse a point with no kernel.
    auto prepare = [&](gpu::Loop& jit_loop) {
        expect(jit_loop.ladder().size() == 1, "one candidate");
        return jit_loop.ladder()[0].prepare(call, loop::Urgency::wait);
    };
    // A call of the loop, compared with the host.
    auto runs = [&](gpu::Loop& jit_loop, const std::string& what) {
        GPU_CHECK(gpuMemset(out_d, 0, elements * sizeof(double)));
        jit_loop.invoke(&set, args.data(), static_cast<int>(args.size()));
        std::vector<double> out(elements);
        GPU_CHECK(gpuMemcpy(out.data(), out_d, elements * sizeof(double),
                            gpuMemcpyDeviceToHost));
        double error = 0.0;
        for (int e = 0; e < elements; ++e)
            error = std::max(error, std::abs(out[e] - expected[e]));
        auto prepared = prepare(jit_loop);
        auto statistics = prepared.candidate->strategy->statistics(prepared);
        std::printf("%s: %s, max error %g\n", what.c_str(), statistics.c_str(), error);
        expect(error < 1e-9, what + " computes the result");
        return statistics;
    };
    auto make = [&](gpu::Loop& jit_loop) {
        jit_loop.add_launch_param("op2_block_size_d", gpu::LaunchValue::block_size);
        jit_loop.add<gpu::Simple>(
            "plain", gpu::KernelSource{"thread_limit_kernel", nullptr, source}, &bind);
    };

    {
        declare("off", "off");
        gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
        make(jit_loop);
        auto prepared = prepare(jit_loop);
        expect(prepared.status == loop::Status::unavailable &&
                         prepared.reason == "thread_limit",
                     "without launch bounds: thread_limit, not " +
                         std::string{prepared.reason});
    }
    {
        declare("on", "off");
        gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
        make(jit_loop);
        runs(jit_loop, "launch bounds");
    }
    {
        declare("off", "off", "2");
        gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
        make(jit_loop);
        auto prepared = prepare(jit_loop);
        expect(prepared.status == loop::Status::unavailable &&
                   prepared.reason == "needs_launch_bounds",
               "a minimum of blocks without launch bounds: needs_launch_bounds");
    }
    {
        // Two blocks of 1024 threads leave each thread 32 registers.
        declare("on", "off", "2");
        gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
        make(jit_loop);
        auto statistics = runs(jit_loop, "launch bounds for two blocks");
        expect(std::atoi(statistics.c_str() + statistics.find(", ") + 2) <= 32,
               "two blocks cap the registers at 32");
    }
    if (gpu::smem_spilling_supported) {
        {
            declare("off", "on");
            gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
            make(jit_loop);
            auto prepared = prepare(jit_loop);
            expect(prepared.status == loop::Status::unavailable &&
                             prepared.reason == "needs_launch_bounds",
                         "spilling without launch bounds: needs_launch_bounds");
        }
        {
            declare("on", "on");
            gpu::Loop jit_loop("thread_limit", "cuda", "Direct", {true}, headers());
            make(jit_loop);
            auto statistics = runs(jit_loop, "launch bounds and shared-memory spilling");
            expect(statistics.find(" B shared") != std::string::npos,
                         "shared-memory spilling takes shared memory");
        }
    }

    GPU_CHECK(gpuFree(in_d));
    GPU_CHECK(gpuFree(out_d));
    op_exit();
    if (!passed)
        return EXIT_FAILURE;
    std::printf("JIT thread limit passed\n");
    return EXIT_SUCCESS;
}

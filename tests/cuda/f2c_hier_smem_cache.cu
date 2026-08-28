#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

#define OP_F2C_PRELUDE OP_F2C_PRELUDE_HIER_SMEM_CACHE_TEST
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_HIER_SMEM_CACHE_TEST_data
INCTXT(OP_F2C_PRELUDE, "op_f2c_prelude.h");

#include <op_f2c_helpers.h>
#include <op_f2c_prelude.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void f2c_hier_smem_cache_baseline(int *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}

extern "C" __global__ void f2c_hier_smem_cache_staged(
    int *result, const int *source_offsets, const unsigned int *stage_words,
    const int *stage_counts, int chunk_begin, int chunk_end, int set_stride,
    int last_shared_byte) {
    extern __shared__ unsigned char scratch[];

    if (blockIdx.x == 0 && threadIdx.x == 0)
        scratch[last_shared_byte] = 42;
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = source_offsets[chunk_begin];
        result[1] = source_offsets[chunk_end];
        result[2] = stage_counts[chunk_begin];
        result[3] = static_cast<int>(stage_words[0]);
        result[4] = static_cast<int>(stage_words[1]);
        result[5] = chunk_begin;
        result[6] = chunk_end;
        result[7] = set_stride;
        result[8] = scratch[last_shared_byte];
    }
}

namespace {

namespace f2c = op::f2c;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "check failed: %s at line %d\n",            \
                         #condition, __LINE__);                                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

constexpr char baseline_source[] = R"op2(
extern "C" __global__ void f2c_hier_smem_cache_baseline(int *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}
)op2";

constexpr char staged_source[] = R"op2(
extern "C" __global__ void f2c_hier_smem_cache_staged(
    int *result, const int *source_offsets, const unsigned int *stage_words,
    const int *stage_counts, int chunk_begin, int chunk_end, int set_stride,
    int last_shared_byte) {
    extern __shared__ unsigned char scratch[];

    if (blockIdx.x == 0 && threadIdx.x == 0)
        scratch[last_shared_byte] = 42;
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = source_offsets[chunk_begin];
        result[1] = source_offsets[chunk_end];
        result[2] = stage_counts[chunk_begin];
        result[3] = static_cast<int>(stage_words[0]);
        result[4] = static_cast<int>(stage_words[1]);
        result[5] = chunk_begin;
        result[6] = chunk_end;
        result[7] = set_stride;
        result[8] = scratch[last_shared_byte];
    }
}
)op2";

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

struct Fixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    op_dat_core wide_dat{};
    op_map_core map{};
    std::vector<int> map_values;
    std::array<op_arg, 2> args{};
    std::array<f2c::HierSmemArgDescriptor, 1> arg_descriptor{{{0, 0}}};
    std::array<f2c::HierSmemDatDescriptor, 1> dat_descriptor{{
        {f2c::HierSmemScalarType::f64},
    }};

    Fixture() : map_values(256) {
        source.index = 0;
        source.size = 256;
        source.core_size = 256;
        source.name = "source";

        target.index = 1;
        target.size = 4;
        target.core_size = 4;
        target.name = "target";

        dat.index = 0;
        dat.set = &target;
        dat.dim = 1;
        dat.size = sizeof(double);
        dat.type = "double";
        dat.name = "dat";

        wide_dat.index = 1;
        wide_dat.set = &target;
        wide_dat.dim = 2;
        wide_dat.size = 2 * sizeof(double);
        wide_dat.type = "double";
        wide_dat.name = "wide_dat";

        map.index = 0;
        map.from = &source;
        map.to = &target;
        map.dim = 1;
        map.map = map_values.data();
        map.name = "map";
        for (std::size_t i = 0; i < map_values.size(); ++i)
            map_values[i] = static_cast<int>((i / 2) % target.size);

        args[0].opt = 1;
        args[0].argtype = OP_ARG_DAT;
        args[0].dat = &dat;
        args[0].map = &map;
        args[0].dim = 1;
        args[0].idx = 0;
        args[0].size = sizeof(double);
        args[0].map_data = map_values.data();
        args[0].type = "double";
        args[0].acc = OP_INC;
        args[1] = args[0];
        args[1].opt = 0;
    }

    f2c::HierSmemStagingDescriptor descriptor() const {
        return {arg_descriptor, dat_descriptor, 256};
    }

    int shared_bytes() const {
        return target.size * args[0].dim * static_cast<int>(sizeof(double));
    }
};

struct Bindings {
    int *output_d;
    int last_shared_byte;

    template<f2c::KernelVariant Variant>
    auto make_arguments(f2c::LaunchContext& launch, op_arg *) {
        if constexpr (Variant == f2c::KernelVariant::baseline) {
            auto args = std::array<void *, 1>{&output_d};
            return f2c::KernelArguments{args, args};
        } else {
            auto args = std::array<void *, 8>{
                &output_d,
                &launch.staged.plan.source_offsets,
                &launch.staged.plan.stage_words,
                &launch.staged.plan.stage_counts,
                &launch.staged.chunk_begin,
                &launch.staged.chunk_end,
                &launch.set_stride,
                &last_shared_byte,
            };
            return f2c::KernelArguments{args, args};
        }
    }

    void init_globals(const f2c::GlobalInitContext&, op_arg *) {}
};

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    Fixture fixture;
    std::array<int, 9> output{};
    int *output_d = nullptr;
    CUDA_SAFE_CALL(gpuMalloc(reinterpret_cast<void **>(&output_d),
                             sizeof(output)));

    bool saw_jit = false;
    {
        f2c::KernelInfo info(
            "f2c_hier_smem_cache", "c_CUDA", "Atomics",
            f2c::ExecutionPolicy::atomics(false),
            "f2c_hier_smem_cache_baseline",
            reinterpret_cast<const void *>(f2c_hier_smem_cache_baseline),
            baseline_source);
        info.register_staged_variant(
            "f2c_hier_smem_cache_staged",
            reinterpret_cast<const void *>(f2c_hier_smem_cache_staged),
            staged_source, fixture.descriptor());

        CUDA_SAFE_CALL(gpuMemset(output_d, 0, sizeof(output)));
        auto dormant = info.invoke(
            &fixture.source, fixture.args.data(), fixture.args.size(),
            Bindings{output_d, fixture.shared_bytes() - 1});
        CUDA_SAFE_CALL(gpuDeviceSynchronize());
        CUDA_SAFE_CALL(gpuMemcpy(output.data(), output_d, sizeof(output),
                                 gpuMemcpyDeviceToHost));
        CHECK(dormant.variant == f2c::KernelVariant::baseline);
        CHECK(output[0] == -1);
        CHECK(info.hier_smem_plan_cache_statistics().entries == 0);

        auto invoke = [&]() {
            CUDA_SAFE_CALL(gpuMemset(output_d, 0, sizeof(output)));
            auto result = info.invoke(
                &fixture.source, fixture.args.data(), fixture.args.size(),
                Bindings{output_d, fixture.shared_bytes() - 1},
                f2c::KernelExecutionOptions::hierarchical_test());
            CUDA_SAFE_CALL(gpuDeviceSynchronize());
            CUDA_SAFE_CALL(gpuMemcpy(output.data(), output_d, sizeof(output),
                                     gpuMemcpyDeviceToHost));
            saw_jit |= result.used_jit;
            return result;
        };

        f2c::KernelInvocationResult result{};
        for (int invocation = 0; invocation < 10; ++invocation) {
            result = invoke();
            CHECK(result.variant == f2c::KernelVariant::staged);
            CHECK(result.hier_smem_reason ==
                  f2c::HierSmemFallbackReason::none);
            CHECK(result.block_size == 128);
            CHECK(result.max_blocks == 1);
        }

        CHECK(output[0] == 0);
        CHECK(output[1] == 256);
        CHECK(output[2] == 4);
        // This fixture is a single chunk, so its section reaches every target
        // exactly once and each owner flushes without a global atomic.
        CHECK(output[3] == static_cast<int>(f2c::hier_smem_owner_bit |
                                            f2c::hier_smem_exclusive_bit));
        CHECK(output[4] == 0);
        CHECK(output[5] == 0);
        CHECK(output[6] == 1);
        CHECK(output[7] == 256);
        CHECK(output[8] == 42);

        auto statistics = info.hier_smem_plan_cache_statistics();
        CHECK(statistics.entries == 1);
        CHECK(statistics.builds == 1);
        CHECK(statistics.uploads == 1);

        fixture.args[0].opt = 0;
        result = invoke();
        CHECK(result.variant == f2c::KernelVariant::baseline);
        CHECK(result.hier_smem_reason ==
              f2c::HierSmemFallbackReason::no_active_increment);
        CHECK(output[0] == -1);

        statistics = info.hier_smem_plan_cache_statistics();
        CHECK(statistics.entries == 2);
        CHECK(statistics.builds == 2);
        CHECK(statistics.uploads == 1);

        fixture.args[0].opt = 1;
        fixture.args[0].dat = &fixture.wide_dat;
        fixture.args[0].dim = 2;
        fixture.args[0].size = 2 * sizeof(double);
        result = invoke();
        CHECK(result.variant == f2c::KernelVariant::staged);
        CHECK(output[8] == 42);

        statistics = info.hier_smem_plan_cache_statistics();
        CHECK(statistics.entries == 3);
        CHECK(statistics.builds == 3);
        CHECK(statistics.uploads == 2);

        fixture.args[1].opt = 1;
        result = invoke();
        CHECK(result.variant == f2c::KernelVariant::baseline);
        CHECK(result.hier_smem_reason ==
              f2c::HierSmemFallbackReason::incompatible_argument);

        statistics = info.hier_smem_plan_cache_statistics();
        CHECK(statistics.entries == 4);
        CHECK(statistics.builds == 4);
        CHECK(statistics.uploads == 2);

        CUDA_SAFE_CALL(gpuFree(output_d));
        output_d = nullptr;
        op_exit();

        statistics = info.hier_smem_plan_cache_statistics();
        CHECK(statistics.entries == 0);
        CHECK(statistics.builds == 4);
        CHECK(statistics.uploads == 2);
    }

    bool expect_jit = env_enabled("OP_EXPECT_JIT");
    CHECK(saw_jit == expect_jit);
    std::printf("hierarchical smem cache %s test passed\n",
                saw_jit ? "JIT" : "offline");
    return EXIT_SUCCESS;
}

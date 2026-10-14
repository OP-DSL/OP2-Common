// The chunk_atomics strategy in a loop with atomics as its floor: disabled
// by default, forced through OP_STRATEGY, its partition cached per
// configuration and rejected for the reasons the planner names.

#define OP_F2C_JIT_TAG chunk_strategy_test
#include <kernel/jit_embed.h>

#include <gpu/loop.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" __global__ void chunk_strategy_baseline(int *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}

extern "C" __global__ void chunk_strategy_staged(
    int *result, const int *source_offsets, const unsigned short *stage_words,
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

namespace gpu = op::gpu;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "check failed: %s at line %d\n",            \
                         #condition, __LINE__);                                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

constexpr char baseline_source[] = R"op2(
extern "C" __global__ void chunk_strategy_baseline(int *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}
)op2";

constexpr char staged_source[] = R"op2(
extern "C" __global__ void chunk_strategy_staged(
    int *result, const int *source_offsets, const unsigned short *stage_words,
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

int *output_d = nullptr;
int last_shared_byte = 0;

void bind_baseline(gpu::LaunchContext&, op_arg *, std::vector<void *>& offline,
                   std::vector<void *>& jit) {
    offline = {&output_d};
    jit = offline;
}

void bind_staged(gpu::LaunchContext& launch, op_arg *,
                 std::vector<void *>& offline, std::vector<void *>& jit) {
    offline = {
        (void *)&output_d,
        (void *)&launch.chunk.plan->source_offsets,
        (void *)&launch.chunk.plan->stage_words,
        (void *)&launch.chunk.plan->stage_counts,
        (void *)&launch.chunk.chunk_begin,
        (void *)&launch.chunk.chunk_end,
        (void *)&launch.set_stride,
        (void *)&last_shared_byte,
    };
    jit = offline;
}

// 256 sources, each pair reaching one of 4 targets of a dim-1 double dat,
// with a second, inactive argument on the same dat.
struct Fixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    op_dat_core wide_dat{};
    op_map_core map{};
    std::vector<int> map_values;
    std::array<op_arg, 2> args{};
    std::array<op::chunk::ArgGroup, 1> group_args{{{0, 0}}};

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

    op::chunk::ArgGroups groups() const { return {group_args, 1}; }

    int shared_bytes() const {
        return target.size * args[0].dim * static_cast<int>(sizeof(double));
    }
};

gpu::JitHeaders headers() {
    return {{OP_F2C_PRELUDE_DATA, OP_CHUNK_LAYOUT_DATA, OP_STRATEGY_KERNEL_DATA},
            {"kernel/prelude.h", "loop/chunk_layout.h", "kernel/strategy.h"}};
}

void add_strategies(gpu::Loop& loop, const Fixture& fixture) {
    loop.add<gpu::ChunkAtomics>(
        op::loop::Default::disabled,
        gpu::KernelSource{"chunk_strategy_staged",
                          reinterpret_cast<const void *>(chunk_strategy_staged),
                          staged_source},
        &bind_staged, fixture.groups(), 256);
    loop.add<gpu::Simple>(
        op::loop::Default::enabled, "atomics",
        gpu::KernelSource{"chunk_strategy_baseline",
                          reinterpret_cast<const void *>(chunk_strategy_baseline),
                          baseline_source},
        &bind_baseline);
}

std::array<int, 9> output{};

// Select, run and read back; what ran and why chunk_atomics was passed over.
struct Outcome {
    std::string strategy;
    std::string chunk_reason;
    bool jit;
    int max_blocks;
};

Outcome run(gpu::Loop& loop, Fixture& fixture) {
    CUDA_SAFE_CALL(gpuMemset(output_d, 0, sizeof(output)));
    op::loop::Configuration configuration{&fixture.source, fixture.args};
    auto chunk = loop.ladder()[0].prepare(configuration, false);
    auto selection = loop.ladder().select(configuration);
    CHECK(loop.invoke(&fixture.source, fixture.args.data(),
                      static_cast<int>(fixture.args.size())) ==
          op::loop::Status::ready);
    CUDA_SAFE_CALL(gpuDeviceSynchronize());
    CUDA_SAFE_CALL(gpuMemcpy(output.data(), output_d, sizeof(output),
                             gpuMemcpyDeviceToHost));
    return {std::string{selection.strategy->name()}, std::string{chunk.reason},
            selection.prepared.kernel != nullptr,
            static_cast<gpu::StrategyBase *>(selection.strategy)->call().max_blocks};
}

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    Fixture fixture;
    CUDA_SAFE_CALL(gpuMalloc(reinterpret_cast<void **>(&output_d), sizeof(output)));
    last_shared_byte = fixture.shared_bytes() - 1;

    bool saw_jit = false;
    {
        // Disabled by default: atomics runs and no partition is built.
        gpu::Loop dormant("chunk_strategy", "c_CUDA", "Indirect", {false, false},
                          headers());
        add_strategies(dormant, fixture);
        auto outcome = run(dormant, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(output[0] == -1);
        CHECK(static_cast<gpu::ChunkBase&>(dormant.ladder()[0]).entries() == 1);
    }

    setenv("OP_STRATEGY", "chunk_atomics", 1);
    {
        gpu::Loop loop("chunk_strategy", "c_CUDA", "Indirect", {false, false},
                       headers());
        add_strategies(loop, fixture);
        auto& chunk = static_cast<gpu::ChunkBase&>(loop.ladder()[0]);

        Outcome outcome{};
        for (int invocation = 0; invocation < 10; ++invocation) {
            outcome = run(loop, fixture);
            CHECK(outcome.strategy == "chunk_atomics");
            CHECK(outcome.max_blocks == 1);
            saw_jit |= outcome.jit;
        }
        CHECK(output[0] == 0);
        CHECK(output[1] == 256);
        CHECK(output[2] == 4);
        // Elements 0 and 1 both reach target 0, the chunk's first slot.
        CHECK(output[3] == 0);
        CHECK(output[4] == 0);
        CHECK(output[5] == 0);
        CHECK(output[6] == 1);
        CHECK(output[7] == 256);
        CHECK(output[8] == 42);
        CHECK(chunk.entries() == 1);

        // The inactive argument: nothing to stage, and a second entry.
        fixture.args[0].opt = 0;
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(outcome.chunk_reason == "no_active_argument");
        CHECK(output[0] == -1);
        CHECK(chunk.entries() == 2);

        // A wider dat: a third entry with its own staging.
        fixture.args[0].opt = 1;
        fixture.args[0].dat = &fixture.wide_dat;
        fixture.args[0].dim = 2;
        fixture.args[0].size = 2 * sizeof(double);
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "chunk_atomics");
        CHECK(output[8] == 42);
        CHECK(chunk.entries() == 3);

        // Two active arguments on one dat, one of them ungrouped.
        fixture.args[1].opt = 1;
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(outcome.chunk_reason == "incompatible_argument");
        CHECK(chunk.entries() == 4);

        // Reduction scratch is sized from the largest capped grid of any one
        // section: with two sections of one chunk each, max_blocks stays 1.
        {
            Fixture split;
            split.source.core_size = 128;
            gpu::Loop split_loop("chunk_strategy_split", "c_CUDA", "Indirect",
                                 {false, false}, headers());
            add_strategies(split_loop, split);
            auto split_outcome = run(split_loop, split);
            CHECK(split_outcome.strategy == "chunk_atomics");
            // The last launch is the second section: chunk 1 of 2.
            CHECK(output[5] == 1);
            CHECK(output[6] == 2);
            CHECK(split_outcome.max_blocks == 1);
        }

        // Every reference reaches a different target, so staging combines
        // nothing: the partition is rejected below the default minimum
        // compression.
        {
            Fixture spread;
            spread.target.size = 256;
            spread.target.core_size = 256;
            for (std::size_t i = 0; i < spread.map_values.size(); ++i)
                spread.map_values[i] = static_cast<int>(i);
            gpu::Loop spread_loop("chunk_strategy_spread", "c_CUDA", "Indirect",
                                  {false, false}, headers());
            add_strategies(spread_loop, spread);
            auto spread_outcome = run(spread_loop, spread);
            CHECK(spread_outcome.strategy == "atomics");
            CHECK(spread_outcome.chunk_reason == "low_compression");
            CHECK(output[0] == -1);
        }

        CUDA_SAFE_CALL(gpuFree(output_d));
        output_d = nullptr;
        op_exit();
        CHECK(chunk.entries() == 0);
    }

    bool expect_jit = env_enabled("OP_EXPECT_JIT");
    CHECK(saw_jit == expect_jit);
    std::printf("chunk strategy %s test passed\n", saw_jit ? "JIT" : "offline");
    return EXIT_SUCCESS;
}

// The chunk_atomics strategy in a loop with atomics as its floor: disabled
// by default, forced through OP_STRATEGY, its plan cached per
// configuration and rejected for the reasons the planner names, and a
// launch staging more than the 48 KB of shared memory a kernel gets without
// opting in.  Run offline (OP_JIT_LEVEL=0) and with every kernel compiled.

#define OP_KERNEL_JIT_TAG chunk_strategy_test
#include <op2/rt/kernel/gpu/jit_embed.h>

#include <op2/gpu/check.h>
#include <op2/rt/context.h>
#include <op2/rt/gpu/executor.h>
#include <op2/rt/gpu/strategy.h>
#include <op2/rt/kernel/gpu/offline.h>
#include <op2/rt/lifecycle.h>
#include <op_lib_c.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include "../loop_parts.h"
#include "../test_support.h"

extern "C" __global__ void chunk_strategy_baseline(OP_ABI_GPU_RANGE_FIELDS(OP_ABI_GPU_PARAM)
                                                   int *result, int) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}

extern "C" __global__ void chunk_strategy_staged(OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                                 int *result, int last_shared_byte) {
    extern __shared__ unsigned char scratch[];

    if (blockIdx.x == 0 && threadIdx.x == 0)
        scratch[last_shared_byte] = 42;
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = chunk_offsets[chunk_begin];
        result[1] = chunk_offsets[chunk_end];
        result[2] = stage_counts[chunk_begin];
        result[3] = static_cast<int>(stage_words[0]);
        result[4] = static_cast<int>(stage_words[1]);
        result[5] = chunk_begin;
        result[6] = chunk_end;
        result[7] = stride;
        result[8] = scratch[last_shared_byte];
        result[9] = global_stride;
    }
}

namespace {

namespace gpu = op::rt::gpu;

constexpr char baseline_source[] = R"op2(
extern "C" __global__ void chunk_strategy_baseline(OP_ABI_GPU_RANGE_FIELDS(OP_ABI_GPU_PARAM)
                                                   int *result, int) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        result[0] = -1;
}
)op2";

constexpr char staged_source[] = R"op2(
extern "C" __global__ void chunk_strategy_staged(OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                                 int *result, int last_shared_byte) {
    extern __shared__ unsigned char scratch[];

    if (blockIdx.x == 0 && threadIdx.x == 0)
        scratch[last_shared_byte] = 42;
    __syncthreads();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        result[0] = chunk_offsets[chunk_begin];
        result[1] = chunk_offsets[chunk_end];
        result[2] = stage_counts[chunk_begin];
        result[3] = static_cast<int>(stage_words[0]);
        result[4] = static_cast<int>(stage_words[1]);
        result[5] = chunk_begin;
        result[6] = chunk_end;
        result[7] = stride;
        result[8] = scratch[last_shared_byte];
        result[9] = global_stride;
    }
}
)op2";

int *output_d = nullptr;
int last_shared_byte = 0;

int bind(op_arg *, unsigned&, std::span<void *> arguments) {
    arguments[0] = &output_d;
    arguments[1] = &last_shared_byte;
    return 2;
}

// 256 sources, each pair reaching one of 4 targets of a dim-1 double dat,
// with a second, inactive argument on the same dat.
// Sets and maps numbered across fixtures, as OP2 numbers them, since plans
// are shared between loops by the shapes of their calls.
int next_index = 0;

struct Fixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    op_dat_core wide_dat{};
    op_map_core map{};
    std::vector<int> map_values;
    std::array<op_arg, 2> args{};
    std::array<op::rt::abi::ArgGroup, 1> group_args{{{0, 0}}};

    Fixture() : map_values(256) {
        source.index = next_index++;
        source.size = 256;
        source.core_size = 256;
        source.name = "source";

        target.index = next_index++;
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

        map.index = next_index++;
        map.name = "map";
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

    op::rt::abi::ArgGroups groups() const { return {group_args, 1}; }

    int shared_bytes() const {
        return target.size * args[0].dim * static_cast<int>(sizeof(double));
    }
};

// The values the chunked strategies fix for their kernels.
const op::rt::abi::gpu::Param params[] = {
    op::rt::abi::gpu::launch_value("op2_block_size_d",
                                   op::rt::abi::gpu::LaunchValue::block_size),
    op::rt::abi::gpu::launch_value("op2_unstaged_singles_d",
                                   op::rt::abi::gpu::LaunchValue::unstaged_singles),
    op::rt::abi::gpu::launch_value("op2_wavefront_d", op::rt::abi::gpu::LaunchValue::wavefront),
};

// A loop of chunk_atomics and atomics over fixture's groups, its parts made
// from the descriptor kept here.
struct Described {
    std::array<op::rt::abi::gpu::Family, 2> families;
    op::rt::abi::gpu::Loop descriptor;
    LoopParts<gpu::Executor> loop;

    Described(const char *name, const Fixture& fixture)
        : families{{{{"chunk_atomics", fixture.groups()}, "chunk_strategy_staged",
                     reinterpret_cast<const void *>(chunk_strategy_staged), staged_source},
                    {{"atomics", {}}, "chunk_strategy_baseline",
                     reinterpret_cast<const void *>(chunk_strategy_baseline),
                     baseline_source}}},
          descriptor{{.name = name, .target = "cuda", .direct = false},
                     false, params, families, 2, &bind,
                     &op::rt::kernel::gpu::calls, OP_KERNEL_JIT_HEADERS},
          loop{descriptor, op::rt::context()} {}
};

std::array<int, 10> output{};

// Select, run and read back: what ran, chunk_atomics' plan entry and why it
// was passed over, and the blocks the reduction scratch was sized for (the
// record's global stride over the 256-thread block).
struct Outcome {
    std::string strategy;
    const void *chunk_entry;
    std::string chunk_reason;
    bool jit;
    int max_blocks;
};

Outcome run(LoopParts<gpu::Executor>& loop, Fixture& fixture) {
    GPU_CHECK(gpuMemset(output_d, 0, sizeof(output)));
    op::rt::loop::Call call{&fixture.source, fixture.args,
                            fixture.source.size + fixture.source.exec_size, false};
    auto chunk = loop.ladder[0].prepare(call, op::rt::loop::Urgency::later);
    auto selection = loop.run(call);
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpy(output.data(), output_d, sizeof(output),
                        gpuMemcpyDeviceToHost));
    const auto& name = selection.candidate->name;
    return {name.substr(0, name.find('=')), chunk.entry, std::string{chunk.reason},
            selection.kernel != nullptr, output[9] / 256};
}

} // namespace

int main(int argc, char **argv) {
    // Every source in one chunk: a block's elements.
    setenv("OP_CHUNK_ATOMICS_BLOCK_SIZES", "256", 1);
    op_init(argc, argv, 2);
    const bool expect_jit = op::rt::context().jit.level > 0;

    Fixture fixture;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&output_d), sizeof(output)));
    last_shared_byte = fixture.shared_bytes() - 1;

    bool saw_jit = false;
    {
        // Disabled by default: atomics runs, and the one plan is the one
        // run() prepares itself.
        Described dormant_described{"chunk_strategy", fixture};
        auto& dormant = dormant_described.loop;
        auto outcome = run(dormant, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(output[0] == -1);
        CHECK(outcome.chunk_entry != nullptr);
    }

    setenv("OP_STRATEGY", "chunk_atomics", 1);
    op::rt::init();
    op::rt::init_gpu();
    {
        Described loop_described{"chunk_strategy", fixture};
        auto& loop = loop_described.loop;

        // Each configuration's plan entry, kept across its calls.
        std::vector<const void *> entries;
        Outcome outcome{};
        for (int invocation = 0; invocation < 10; ++invocation) {
            outcome = run(loop, fixture);
            if (invocation == 0)
                entries.push_back(outcome.chunk_entry);
            CHECK(outcome.strategy == "chunk_atomics");
            CHECK(outcome.chunk_entry == entries[0]);
            CHECK(outcome.max_blocks == 1);
            saw_jit |= outcome.jit;
        }
        auto new_entry = [&](const void *entry) {
            bool fresh = std::find(entries.begin(), entries.end(), entry) == entries.end();
            entries.push_back(entry);
            return entry != nullptr && fresh;
        };
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

        // The inactive argument: nothing to stage, and a second entry.
        fixture.args[0].opt = 0;
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(outcome.chunk_reason == "no_active_argument");
        CHECK(output[0] == -1);
        CHECK(new_entry(outcome.chunk_entry));

        // A wider dat: a third entry with its own staging.
        fixture.args[0].opt = 1;
        fixture.args[0].dat = &fixture.wide_dat;
        fixture.args[0].dim = 2;
        fixture.args[0].size = 2 * sizeof(double);
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "chunk_atomics");
        CHECK(output[8] == 42);
        CHECK(new_entry(outcome.chunk_entry));

        // Two active arguments on one dat, one of them ungrouped.
        fixture.args[1].opt = 1;
        outcome = run(loop, fixture);
        CHECK(outcome.strategy == "atomics");
        CHECK(outcome.chunk_reason == "incompatible_argument");
        CHECK(new_entry(outcome.chunk_entry));

        // Reduction scratch is sized from the largest capped grid of any one
        // section: with two sections of one chunk each, max_blocks stays 1.
        {
            Fixture split;
            split.source.core_size = 128;
            Described split_loop_described{"chunk_strategy_split", split};
            auto& split_loop = split_loop_described.loop;
            auto split_outcome = run(split_loop, split);
            CHECK(split_outcome.strategy == "chunk_atomics");
            // The last launch is the second section: chunk 1 of 2.
            CHECK(output[5] == 1);
            CHECK(output[6] == 2);
            CHECK(split_outcome.max_blocks == 1);
        }

        // Every reference reaches a different target, so staging combines
        // nothing: the plan is rejected below the default minimum
        // compression.
        {
            Fixture spread;
            spread.target.size = 256;
            spread.target.core_size = 256;
            for (std::size_t i = 0; i < spread.map_values.size(); ++i)
                spread.map_values[i] = static_cast<int>(i);
            Described spread_loop_described{"chunk_strategy_spread", spread};
            auto& spread_loop = spread_loop_described.loop;
            auto spread_outcome = run(spread_loop, spread);
            CHECK(spread_outcome.strategy == "atomics");
            CHECK(spread_outcome.chunk_reason == "low_compression");
            CHECK(output[0] == -1);
        }

        // A dim-64 dat whose 128 targets need 64 KB of staging, more than a
        // launch gets without opting in.
        {
            Fixture large;
            large.target.size = 128;
            large.target.core_size = 128;
            large.dat.dim = 64;
            large.dat.size = 64 * sizeof(double);
            large.args[0].dim = 64;
            large.args[0].size = 64 * sizeof(double);
            for (std::size_t i = 0; i < large.map_values.size(); ++i)
                large.map_values[i] = static_cast<int>(i / 2);

            gpuDeviceProp_t props;
            int device = 0;
            GPU_CHECK(gpuGetDevice(&device));
            GPU_CHECK(gpuGetDeviceProperties(&props, device));
            if (props.sharedMemPerBlockOptin <
                static_cast<std::size_t>(large.shared_bytes())) {
                std::printf("device allows under %d bytes of shared memory\n",
                            large.shared_bytes());
            } else {
                last_shared_byte = large.shared_bytes() - 1;
                Described large_loop_described{"chunk_strategy_large", large};
                auto& large_loop = large_loop_described.loop;
                auto large_outcome = run(large_loop, large);
                CHECK(large_outcome.strategy == "chunk_atomics");
                CHECK(large_outcome.max_blocks == 1);
                CHECK(output[8] == 42);
            }
        }

        // Two loops whose calls have the same shape plan chunk_atomics alike:
        // one plan, made and uploaded once, which stays while either holds it.
        {
            Fixture alike;
            Described first_described{"chunk_strategy_alike_first", alike};
            Described second_described{"chunk_strategy_alike_second", alike};
            auto& first_loop = first_described.loop;
            auto& second_loop = second_described.loop;
            op::rt::loop::Call call{&alike.source, alike.args,
                                    alike.source.size + alike.source.exec_size, false};
            auto first = first_loop.ladder[0].prepare(call, op::rt::loop::Urgency::later);
            auto second = second_loop.ladder[0].prepare(call, op::rt::loop::Urgency::later);
            CHECK(first.entry != nullptr && first.entry == second.entry);
            // The second forgets it; the first still holds it, so the second
            // finds it again rather than planning anew.
            second_loop.ladder[0].strategy->forget(call.key());
            auto again = second_loop.ladder[0].prepare(call, op::rt::loop::Urgency::later);
            CHECK(again.entry == first.entry);
        }

        GPU_CHECK(gpuFree(output_d));
        output_d = nullptr;
        op_exit();
    }

    CHECK(saw_jit == expect_jit);
    std::printf("chunk strategy %s test passed\n", saw_jit ? "JIT" : "offline");
    return EXIT_SUCCESS;
}

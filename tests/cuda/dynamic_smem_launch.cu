// A chunk_atomics launch asking for more dynamic shared memory than a kernel
// gets without opting in, offline and compiled.

#define OP_F2C_JIT_TAG dynamic_smem_test
#include <kernel/jit_embed.h>

#include <gpu/loop.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

constexpr int result_value = 0x5a17;

extern "C" __global__ void dynamic_smem_baseline(int *result, int last_word) {
    if (threadIdx.x == 0)
        result[0] = last_word;
}

extern "C" __global__ void dynamic_smem_staged(int *result, int last_word) {
    extern __shared__ int scratch[];

    if (threadIdx.x == 0)
        scratch[last_word] = result_value;

    __syncthreads();

    if (threadIdx.x == 0)
        result[0] = scratch[last_word];
}

namespace {

namespace gpu = op::gpu;

constexpr char baseline_source[] = R"op2(
extern "C" __global__ void dynamic_smem_baseline(int *result, int last_word) {
    if (threadIdx.x == 0)
        result[0] = last_word;
}
)op2";

constexpr char staged_source[] = R"op2(
extern "C" __global__ void dynamic_smem_staged(int *result, int last_word) {
    extern __shared__ int scratch[];

    if (threadIdx.x == 0)
        scratch[last_word] = 0x5a17;

    __syncthreads();

    if (threadIdx.x == 0)
        result[0] = scratch[last_word];
}
)op2";

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int *output_d = nullptr;
int last_word = 0;

void bind(gpu::LaunchContext&, op_arg *, std::vector<void *>& offline,
          std::vector<void *>& jit) {
    offline = {&output_d, &last_word};
    jit = offline;
}

// One chunk of 256 elements, each pair reaching one target of a dim-64
// double dat: 128 * 64 * 8 bytes of staging, above the 48 KB a launch gets
// without opting in.
constexpr int sources = 256;
constexpr int targets = sources / 2;
constexpr int dim = 64;
constexpr int shared_bytes = targets * dim * static_cast<int>(sizeof(double));

struct Fixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    op_map_core map{};
    std::vector<int> map_values;
    std::array<op_arg, 1> args{};
    std::array<op::plan::ArgGroup, 1> group_args{{{0, 0}}};

    Fixture() : map_values(sources) {
        source.index = 0;
        source.size = sources;
        source.core_size = sources;
        source.name = "source";

        target.index = 1;
        target.size = targets;
        target.core_size = targets;
        target.name = "target";

        dat.index = 0;
        dat.set = &target;
        dat.dim = dim;
        dat.size = dim * sizeof(double);
        dat.type = "double";
        dat.name = "dat";

        map.index = 0;
        map.from = &source;
        map.to = &target;
        map.dim = 1;
        map.map = map_values.data();
        map.name = "map";
        for (int i = 0; i < sources; ++i)
            map_values[static_cast<std::size_t>(i)] = i / 2;

        args[0].opt = 1;
        args[0].argtype = OP_ARG_DAT;
        args[0].dat = &dat;
        args[0].map = &map;
        args[0].dim = dim;
        args[0].idx = 0;
        args[0].size = dim * sizeof(double);
        args[0].map_data = map_values.data();
        args[0].type = "double";
        args[0].acc = OP_INC;
    }

    op::plan::ArgGroups groups() const { return {group_args, 1}; }
};

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    gpuDeviceProp_t props;
    int device = 0;
    CUDA_SAFE_CALL(gpuGetDevice(&device));
    CUDA_SAFE_CALL(gpuGetDeviceProperties(&props, device));
    if (props.sharedMemPerBlockOptin < static_cast<std::size_t>(shared_bytes)) {
        std::printf("device allows under %d bytes of dynamic shared memory\n",
                    shared_bytes);
        op_exit();
        return EXIT_SUCCESS;
    }

    int output = -1;
    CUDA_SAFE_CALL(gpuMalloc(reinterpret_cast<void **>(&output_d), sizeof(output)));
    CUDA_SAFE_CALL(gpuMemcpy(output_d, &output, sizeof(output), gpuMemcpyHostToDevice));
    last_word = shared_bytes / static_cast<int>(sizeof(int)) - 1;

    op_profile_start("Dynamic shared-memory launch test");
    op_profile_enter("Launches");

    bool saw_jit = false;
    int launched_blocks = -1;
    {
        Fixture fixture;
        setenv("OP_STRATEGY", "chunk_atomics", 1);
        setenv("OP_CHUNK_ATOMICS_CHUNK_SIZES", std::to_string(sources).c_str(), 1);
        op::loop::init();
        gpu::Loop loop("dynamic_smem_launch", "c_CUDA", "Indirect", {false, false},
                       gpu::JitHeaders{{OP_F2C_PRELUDE_DATA, OP_STAGING_LAYOUT_DATA,
                                        OP_STRATEGY_KERNEL_DATA},
                                       {"kernel/prelude.h", "plan/staging_layout.h",
                                        "kernel/strategy.h"}});
        loop.add<gpu::ChunkAtomics>(
            "chunk_atomics", op::loop::Default::disabled,
            gpu::KernelSource{"dynamic_smem_staged",
                              reinterpret_cast<const void *>(dynamic_smem_staged),
                              staged_source},
            &bind, fixture.groups());
        loop.add<gpu::Simple>(
            "plain", op::loop::Default::enabled,
            gpu::KernelSource{"dynamic_smem_baseline",
                              reinterpret_cast<const void *>(dynamic_smem_baseline),
                              baseline_source},
            &bind);

        for (int invocation = 0; invocation < 10; ++invocation) {
            op::loop::Configuration configuration{&fixture.source, fixture.args};
            auto selection = loop.ladder().select(configuration);
            if (selection.strategy->family() != "chunk_atomics") {
                std::fprintf(stderr, "the plan was rejected: %s\n",
                             std::string{selection.prepared.reason}.c_str());
                return EXIT_FAILURE;
            }
            saw_jit |= selection.prepared.kernel != nullptr;
            launched_blocks =
                static_cast<gpu::StrategyBase *>(selection.strategy)->call().max_blocks;

            loop.invoke(&fixture.source, fixture.args.data(),
                        static_cast<int>(fixture.args.size()));
        }

        CUDA_SAFE_CALL(gpuDeviceSynchronize());
        CUDA_SAFE_CALL(gpuMemcpy(&output, output_d, sizeof(output),
                                 gpuMemcpyDeviceToHost));
    }

    op_profile_exit();
    op_profile_end();

    CUDA_SAFE_CALL(gpuFree(output_d));
    op_exit();

    bool expect_jit = env_enabled("OP_EXPECT_JIT");
    if (output != result_value || launched_blocks != 1) {
        std::fprintf(stderr,
                     "dynamic shared-memory result mismatch: %d != %d (blocks=%d)\n",
                     output, result_value, launched_blocks);
        return EXIT_FAILURE;
    }

    if (saw_jit != expect_jit) {
        std::fprintf(stderr, "unexpected launch path: saw_jit=%d, expected=%d\n",
                     saw_jit, expect_jit);
        return EXIT_FAILURE;
    }

    std::printf("dynamic shared-memory %s launch passed (%d bytes)\n",
                saw_jit ? "JIT" : "offline", shared_bytes);
    return EXIT_SUCCESS;
}

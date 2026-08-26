#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

#define OP_F2C_PRELUDE OP_F2C_PRELUDE_DYNAMIC_SMEM_TEST
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_DYNAMIC_SMEM_TEST_data
INCTXT(OP_F2C_PRELUDE, "op_f2c_prelude.h");

#include <op_f2c_prelude.h>
#include <op_f2c_helpers.h>
#include <op_lib_c.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

constexpr int result_value = 0x5a17;

extern "C" __global__ void f2c_dynamic_smem_baseline(
    int *result, int last_word) {
    if (threadIdx.x == 0)
        result[0] = last_word;
}

extern "C" __global__ void f2c_dynamic_smem_staged(
    int *result, int last_word) {
    extern __shared__ int scratch[];

    if (threadIdx.x == 0)
        scratch[last_word] = result_value;

    __syncthreads();

    if (threadIdx.x == 0)
        result[0] = scratch[last_word];
}

namespace {

constexpr char baseline_source[] = R"op2(
extern "C" __global__ void f2c_dynamic_smem_baseline(
    int *result, int last_word) {
    if (threadIdx.x == 0)
        result[0] = last_word;
}
)op2";

constexpr char staged_source[] = R"op2(
extern "C" __global__ void f2c_dynamic_smem_staged(
    int *result, int last_word) {
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

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    gpuDeviceProp_t props;
    int device = 0;
    CUDA_SAFE_CALL(gpuGetDevice(&device));
    CUDA_SAFE_CALL(gpuGetDeviceProperties(&props, device));

    constexpr int requested_shared_bytes = 64 * 1024;
    int shared_bytes = static_cast<int>(std::min(
        static_cast<std::size_t>(requested_shared_bytes),
        props.sharedMemPerBlockOptin));
    shared_bytes -= shared_bytes % static_cast<int>(sizeof(int));

    if (shared_bytes == 0) {
        std::fprintf(stderr, "device reports no dynamic shared memory\n");
        op_exit();
        return EXIT_FAILURE;
    }

    int output = -1;
    int *output_d = nullptr;
    CUDA_SAFE_CALL(gpuMalloc(reinterpret_cast<void **>(&output_d),
                             sizeof(output)));
    CUDA_SAFE_CALL(gpuMemcpy(output_d, &output, sizeof(output),
                             gpuMemcpyHostToDevice));

    op_profile_start("F2C dynamic shared-memory launch test");
    op_profile_enter("Launches");

    bool saw_jit = false;
    int launched_blocks = -1;
    {
        op_set_core set{};
        set.size = 1;
        set.core_size = 1;

        op::f2c::KernelInfo info(
            "f2c_dynamic_smem_baseline",
            reinterpret_cast<const void *>(f2c_dynamic_smem_baseline),
            baseline_source);
        info.register_smem_variant(
            "f2c_dynamic_smem_staged",
            reinterpret_cast<const void *>(f2c_dynamic_smem_staged),
            staged_source);

        int last_word = shared_bytes / static_cast<int>(sizeof(int)) - 1;
        void *kernel_args[] = {&output_d, &last_word};

        for (int invocation = 0; invocation < 10; ++invocation) {
            auto execution = info.prepare(
                nullptr, 0, op::f2c::ExecutionSections::direct(&set),
                op::f2c::KernelExecutionOptions{
                    op::f2c::KernelVariant::staged, shared_bytes});

            saw_jit |= execution.jit_kernel != nullptr;
            launched_blocks = execution.num_blocks(0);
            info.invoke(execution, launched_blocks, kernel_args, kernel_args);
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
    if (output != result_value) {
        std::fprintf(stderr,
                     "dynamic shared-memory result mismatch: %d != %d "
                     "(blocks=%d)\n",
                     output, result_value, launched_blocks);
        return EXIT_FAILURE;
    }

    if (saw_jit != expect_jit) {
        std::fprintf(stderr,
                     "unexpected launch path: saw_jit=%d, expected=%d\n",
                     saw_jit, expect_jit);
        return EXIT_FAILURE;
    }

    std::printf("dynamic shared-memory %s launch passed (%d bytes)\n",
                saw_jit ? "JIT" : "offline", shared_bytes);
    return EXIT_SUCCESS;
}

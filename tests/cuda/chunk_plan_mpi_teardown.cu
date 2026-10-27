// A device plan under the MPI CUDA library is released by op_exit through
// its loop's release, while the device is still up.

#include <gpu/chunk_encode.h>
#include <gpu/shims.h>
#include <loop/loop.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../test_support.h"

namespace {

// A loop whose release frees a device plan.
struct PlanOwner : op::loop::Loop {
    op::gpu::DevicePlan& plan;

    explicit PlanOwner(op::gpu::DevicePlan& plan_)
        : op::loop::Loop{"teardown", "cuda", "Direct", true}, plan{plan_} {}

    void release() override {
        op::loop::Loop::release();
        plan.release();
    }
};

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    op_set_core source{};
    source.index = 0;
    source.size = 128;
    source.core_size = 128;

    op_set_core target{};
    target.index = 1;
    target.size = 4;
    target.core_size = 4;

    op_dat_core dat{};
    dat.index = 0;
    dat.set = &target;
    dat.dim = 1;
    dat.size = sizeof(double);
    dat.type = "double";

    std::vector<int> map_values(source.size);
    for (int i = 0; i < source.size; ++i)
        map_values[static_cast<std::size_t>(i)] = (i / 2) % target.size;

    op_map_core map{};
    map.index = 0;
    map.from = &source;
    map.to = &target;
    map.dim = 1;
    map.map = map_values.data();

    op_arg arg{};
    arg.opt = 1;
    arg.argtype = OP_ARG_DAT;
    arg.dat = &dat;
    arg.map = &map;
    arg.dim = 1;
    arg.idx = 0;
    arg.size = sizeof(double);
    arg.map_data = map_values.data();
    arg.type = "double";
    arg.acc = OP_INC;

    std::array args{arg};
    std::array groups{op::plan::ArgGroup{0, 0}};

    int device = -1;
    gpuDeviceProp_t properties{};
    CHECK(gpuGetDevice(&device) == gpuSuccess);
    CHECK(gpuGetDeviceProperties(&properties, device) == gpuSuccess);
    auto result = op::gpu::plan_chunk_atomics(
        args, op::loop::Sections::direct(&source), op::plan::ArgGroups{groups, 1}, 128,
        op::gpu::ChunkLimits{128, properties.sharedMemPerBlock}, false, 1.5);
    CHECK(result);

    op::gpu::DevicePlan plan(*result.plan);
    CHECK(plan.source_offsets != nullptr && plan.stage_words != nullptr);
    {
        PlanOwner owner{plan};
        op_exit();
        CHECK(plan.source_offsets == nullptr && plan.stage_words == nullptr);
    }

    std::printf("MPI chunk plan teardown test passed\n");
    return EXIT_SUCCESS;
}

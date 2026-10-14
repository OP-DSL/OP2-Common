// A device plan under the MPI CUDA library is released by op_exit through
// the teardown registry, while the device is still up.

#include <gpu/chunk_encode.h>
#include <gpu/shims.h>
#include <op_lib_c.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "check failed: %s at line %d\n",            \
                         #condition, __LINE__);                                 \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

void release_plan(void *owner) {
    static_cast<op::gpu::DevicePlan *>(owner)->release();
}

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
    std::array groups{op::chunk::ArgGroup{0, 0}};

    int device = -1;
    gpuDeviceProp_t properties{};
    CHECK(gpuGetDevice(&device) == gpuSuccess);
    CHECK(gpuGetDeviceProperties(&properties, device) == gpuSuccess);
    op::chunk::Constraints constraints;
    constraints.lanes = 128;
    constraints.elements = 128;
    constraints.scratch_bytes = properties.sharedMemPerBlock;

    auto result = op::chunk::build_partition(
        op::chunk::Kind::staged, op::loop::Configuration{&source, args},
        op::loop::Sections::direct(&source), op::chunk::ArgGroups{groups, 1},
        constraints, op::chunk::Options{});
    CHECK(result);

    op::gpu::DevicePlan plan(*result.partition);
    CHECK(plan.source_offsets != nullptr && plan.stage_words != nullptr);
    op::loop::register_release(&plan, release_plan);

    op_exit();
    CHECK(plan.source_offsets == nullptr && plan.stage_words == nullptr);
    op::loop::unregister_release(&plan);

    std::printf("MPI chunk plan teardown test passed\n");
    return EXIT_SUCCESS;
}

// A device plan under the MPI CUDA library is released by op_exit through
// its loop's destruction, while the device is still up.  With no_exit the
// program skips op_exit, and nothing is torn down after the driver has.

#include <op2/rt/gpu/chunk_encode.h>
#include <op2/rt/gpu/layout.h>
#include <op2/gpu/shims.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_c.h>

#include <mpi.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <vector>

#include "../test_support.h"

namespace {

// A loop whose executor frees its descriptor's device plan when it goes.
struct Owned : op::rt::abi::Loop {
    std::optional<op::rt::gpu::DevicePlan> *plan;
};

struct PlanOwner : op::rt::loop::Executor {
    std::optional<op::rt::gpu::DevicePlan>& plan;

    PlanOwner(const Owned& owned, op::rt::jit::ParamSet&) : plan{*owned.plan} {}

    ~PlanOwner() override { plan.reset(); }

    void add_strategies(op::rt::loop::Ladder&) override {}
};

} // namespace

int main(int argc, char **argv) {
    bool no_exit = argc > 1 && std::string_view{argv[1]} == "no_exit";
    op_init(argc, argv, 2);

    op_set_core source{};
    source.name = "source";
    source.index = 0;
    source.size = 128;
    source.core_size = 128;

    op_set_core target{};
    target.name = "target";
    target.index = 1;
    target.size = 4;
    target.core_size = 4;

    op_dat_core dat{};
    dat.name = "dat";
    dat.index = 0;
    dat.set = &target;
    dat.dim = 1;
    dat.size = sizeof(double);
    dat.type = "double";

    std::vector<int> map_values(source.size);
    for (int i = 0; i < source.size; ++i)
        map_values[static_cast<std::size_t>(i)] = (i / 2) % target.size;

    op_map_core map{};
    map.name = "map";
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
    std::array groups{op::rt::abi::ArgGroup{0, 0}};

    int device = -1;
    gpuDeviceProp_t properties{};
    CHECK(gpuGetDevice(&device) == gpuSuccess);
    CHECK(gpuGetDeviceProperties(&properties, device) == gpuSuccess);
    auto result = op::rt::gpu::plan_chunk_atomics(
        args, op::rt::loop::Sections::direct(&source), op::rt::abi::ArgGroups{groups, 1},
        op::rt::gpu::ChunkSpec{128, properties.sharedMemPerBlock,
                               op::rt::gpu::element_stride(&source, true)},
        false, 1.5);
    CHECK(result);

    // The descriptor and the plan outlive main when op_exit is skipped.
    auto& plan = *new std::optional<op::rt::gpu::DevicePlan>(std::in_place, *result.plan);
    op::rt::abi::gpu::Chunk record;
    plan->fill(record, 0);
    CHECK(record.source_offsets != nullptr && record.stage_words != nullptr);
    static Owned owned{{"teardown", "cuda", "Direct", true}, &plan};
    op::rt::loop::runtime<PlanOwner>(owned);
    if (no_exit) {
        MPI_Finalize();
        std::printf("MPI chunk plan exit without op_exit test passed\n");
        return EXIT_SUCCESS;
    }
    op_exit();
    CHECK(!plan.has_value());
    CHECK(owned.runtime == nullptr);
    delete &plan;

    std::printf("MPI chunk plan teardown test passed\n");
    return EXIT_SUCCESS;
}

#include <op_hier_smem_cache.h>
#include <op_lib_c.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

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

void clear_cache(void *owner) {
    static_cast<f2c::detail::HierSmemPlanCache *>(owner)->clear();
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
    std::array sections{f2c::ExecutionSection{0, source.size}};
    std::array arg_descriptors{f2c::HierSmemArgDescriptor{0, 0}};
    std::array dat_descriptors{
        f2c::HierSmemDatDescriptor{f2c::HierSmemScalarType::f64}};
    f2c::HierSmemStagingDescriptor descriptor{
        arg_descriptors, dat_descriptors, 128};

    int device = -1;
    gpuDeviceProp_t properties{};
    CHECK(gpuGetDevice(&device) == gpuSuccess);
    CHECK(gpuGetDeviceProperties(&properties, device) == gpuSuccess);
    std::size_t shared_memory_limit = properties.sharedMemPerBlock;
#ifdef OP2_CUDA
    shared_memory_limit = std::max(
        shared_memory_limit,
        static_cast<std::size_t>(properties.sharedMemPerBlockOptin));
#endif

    f2c::HierSmemPlanOptions options{128, 128, shared_memory_limit};
    auto key = f2c::detail::make_hier_smem_plan_key(
        &source, args, static_cast<int>(sections.size()), descriptor,
        options);

    f2c::detail::HierSmemPlanCache cache;
    f2c::register_hier_smem_plan_owner(&cache, clear_cache);
    const auto& entry = cache.get_or_build(std::move(key), [&]() {
        return f2c::build_hier_smem_plan(
            &source, args, sections, descriptor, options);
    });

    CHECK(entry);
    CHECK(cache.statistics().entries == 1);
    CHECK(cache.statistics().uploads == 1);

    op_exit();
    CHECK(cache.statistics().entries == 0);
    f2c::unregister_hier_smem_plan_owner(&cache);

    std::printf("MPI hierarchical smem cache teardown test passed\n");
    return EXIT_SUCCESS;
}

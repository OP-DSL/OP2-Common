// The device plan against the host device: every array a staged kernel
// reads is uploaded, the wavefront state is zeroed, and release frees it.

#include <gpu/chunk_encode.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace chunk = op::chunk;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

int mallocs = 0;
int frees = 0;

void *counting_malloc(std::size_t bytes) {
    ++mallocs;
    return std::malloc(bytes);
}

void counting_free(void *pointer) {
    ++frees;
    std::free(pointer);
}

void host_copy(void *destination, const void *source, std::size_t bytes) {
    std::memcpy(destination, source, bytes);
}

void host_memset(void *pointer, int value, std::size_t bytes) {
    std::memset(pointer, value, bytes);
}

template<typename T>
bool same(const T *device, const std::vector<T>& host) {
    return std::memcmp(device, host.data(), host.size() * sizeof(T)) == 0;
}

// Edge i of a chain of 16 reaches nodes i and i+1; chunks of 4 lanes.
void test_encode() {
    op_set_core edges{};
    edges.size = 16;
    edges.core_size = 16;
    op_set_core nodes{};
    nodes.size = 17;
    nodes.core_size = 17;
    op_dat_core dat{};
    dat.set = &nodes;
    dat.dim = 1;
    dat.size = sizeof(int);
    dat.type = "int";

    std::vector<int> map_values;
    for (int i = 0; i < 16; ++i) {
        map_values.push_back(i);
        map_values.push_back(i + 1);
    }
    op_map_core map{};
    map.from = &edges;
    map.to = &nodes;
    map.dim = 2;
    map.map = map_values.data();

    std::array<op_arg, 2> args{};
    for (int idx = 0; idx < 2; ++idx) {
        args[idx].opt = 1;
        args[idx].argtype = OP_ARG_DAT;
        args[idx].dat = &dat;
        args[idx].map = &map;
        args[idx].dim = 1;
        args[idx].idx = idx;
        args[idx].size = sizeof(int);
        args[idx].map_data = map_values.data();
        args[idx].type = "int";
        args[idx].acc = OP_RW;
    }
    std::array<chunk::ArgGroup, 2> group_args{{{0, 0}, {1, 0}}};

    chunk::Constraints constraints;
    constraints.lanes = 4;
    constraints.elements = 4;
    constraints.scratch_bytes = 1 << 16;
    chunk::Options options;
    options.min_compression = 0.0;
    options.variants = chunk::variant_wavefront;
    options.window = 2;

    auto result = chunk::build_partition(
        chunk::Kind::coloured, op::loop::Configuration{&edges, args},
        op::loop::Sections::direct(&edges), chunk::ArgGroups{group_args, 1},
        constraints, options);
    CHECK(result);
    const auto& partition = *result.partition;
    CHECK(partition.colouring->wavefront);

    op::set_device({counting_malloc, counting_free, host_copy, host_memset});
    {
        op::gpu::DevicePlan plan(partition);
        CHECK(mallocs == 12);
        CHECK(same(plan.source_offsets, partition.source_offsets));
        CHECK(same(plan.stage_words, partition.staging.stage_words));
        CHECK(same(plan.stage_counts, partition.staging.stage_counts));
        CHECK(same(plan.slot_offsets, partition.staging.slot_offsets));
        CHECK(same(plan.slot_targets, partition.staging.slot_targets));
        CHECK(same(plan.chunk_order, partition.colouring->chunk_order));
        CHECK(same(plan.thread_colours, partition.colouring->thread_colours));
        CHECK(same(plan.chunk_thread_colours,
                   partition.colouring->chunk_thread_colours));
        CHECK(same(plan.wave_pred_offsets, partition.colouring->wave_pred_offsets));
        CHECK(same(plan.wave_preds, partition.colouring->wave_preds));
        CHECK(plan.wavefront());
        for (std::size_t i = 0; i < partition.num_chunks(); ++i)
            CHECK(plan.wave_done[i] == 0);
        CHECK(plan.wave_tickets[0] == 0);
        CHECK(plan.next_wave_epoch() == 1 && plan.next_wave_epoch() == 2);

        op::gpu::DevicePlan moved(std::move(plan));
        CHECK(plan.source_offsets == nullptr && moved.source_offsets != nullptr);
        CHECK(frees == 0);
    }
    CHECK(frees == 12);

    // A staged partition without colouring uploads five arrays.
    options.variants = 0;
    for (auto& arg : args)
        arg.acc = OP_INC;
    auto staged = chunk::build_partition(
        chunk::Kind::staged, op::loop::Configuration{&edges, args},
        op::loop::Sections::direct(&edges), chunk::ArgGroups{group_args, 1},
        constraints, options);
    CHECK(staged);
    mallocs = frees = 0;
    {
        op::gpu::DevicePlan plan(*staged.partition);
        CHECK(mallocs == 5);
        CHECK(!plan.wavefront());
        plan.release();
        CHECK(frees == 5 && plan.stage_words == nullptr);
    }
    CHECK(frees == 5);
}

} // namespace

int main() {
    try {
        test_encode();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("chunk encode tests passed\n");
    return EXIT_SUCCESS;
}

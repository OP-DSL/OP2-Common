// The device plan: every array a staged kernel reads is uploaded, the
// wavefront state is zeroed, and release frees it.

#include <gpu/chunk_encode.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace plan = op::plan;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

template<typename T>
std::vector<T> download(const T *device, std::size_t count) {
    std::vector<T> host(count);
    GPU_CHECK(gpuMemcpy(host.data(), device, count * sizeof(T),
                        gpuMemcpyDeviceToHost));
    return host;
}

template<typename T>
bool same(const T *device, const std::vector<T>& host) {
    return device != nullptr && download(device, host.size()) == host;
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
    std::array<plan::ArgGroup, 2> group_args{{{0, 0}, {1, 0}}};
    auto sections = op::loop::Sections::direct(&edges);
    op::gpu::ChunkLimits limits{4, 1 << 16};

    auto result = op::gpu::plan_chunk_colouring(
        args, sections, plan::ArgGroups{group_args, 1}, 4, limits, true, 2);
    CHECK(result);
    const auto& chunk_plan = *result.plan;
    CHECK(chunk_plan.wavefront);

    {
        op::gpu::DevicePlan plan(chunk_plan);
        CHECK(same(plan.source_offsets, chunk_plan.chunks.offsets));
        CHECK(same(plan.stage_words, chunk_plan.staging.stage_words));
        const auto& offsets = chunk_plan.staging.slot_offsets;
        std::vector<int> counts;
        for (std::size_t region = 0; region + 1 < offsets.size(); ++region)
            counts.push_back(offsets[region + 1] - offsets[region]);
        CHECK(same(plan.stage_counts, counts));
        CHECK(same(plan.slot_offsets, chunk_plan.staging.slot_offsets));
        CHECK(same(plan.slot_targets, chunk_plan.staging.slot_targets));
        CHECK(same(plan.chunk_order, chunk_plan.batches.order));
        CHECK(same(plan.thread_colours, chunk_plan.thread_colours));
        CHECK(same(plan.chunk_thread_colours, chunk_plan.chunk_thread_colours));
        CHECK(same(plan.wave_pred_offsets, chunk_plan.wave_pred_offsets));
        CHECK(same(plan.wave_preds, chunk_plan.wave_preds));
        CHECK(plan.wave_tickets != nullptr);
        auto chunks = static_cast<std::size_t>(chunk_plan.chunks.count());
        CHECK(download(plan.wave_done, chunks) == std::vector<int>(chunks, 0));
        CHECK(download(plan.wave_tickets, 1) == std::vector<int>{0});
        CHECK(plan.next_wave_epoch() == 1 && plan.next_wave_epoch() == 2);

        op::gpu::DevicePlan moved(std::move(plan));
        CHECK(plan.source_offsets == nullptr && moved.source_offsets != nullptr);
    }

    // A chunk_atomics plan uploads no colour or wavefront arrays.
    for (auto& arg : args)
        arg.acc = OP_INC;
    auto staged = op::gpu::plan_chunk_atomics(
        args, sections, plan::ArgGroups{group_args, 1}, 4, limits, false, 0.0);
    CHECK(staged);
    {
        op::gpu::DevicePlan plan(*staged.plan);
        CHECK(same(plan.stage_words, staged.plan->staging.stage_words));
        CHECK(plan.thread_colours == nullptr && plan.wave_tickets == nullptr);
        plan.release();
        CHECK(plan.stage_words == nullptr && plan.source_offsets == nullptr);
    }
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

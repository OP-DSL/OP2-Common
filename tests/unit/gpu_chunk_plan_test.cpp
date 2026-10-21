// The GPU's chunk plans: staging slot lists, chunk sizing against the
// shared memory and the stage word, chunk_colouring's invariants and its
// wavefront, and the rejections.

#include <gpu/chunk_plan.h>
#include <kernel/strategy.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu = op::gpu;
namespace plan = op::plan;
using op::loop::Sections;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

// chunk_atomics with no compression floor, so small fixtures are never
// rejected for it.
gpu::ChunkResult atomics(std::span<const op_arg> args, const Sections& sections,
                         const plan::ArgGroups& groups, int lanes, int size,
                         std::size_t shared_bytes, bool unstaged_singles = false,
                         double min_compression = 0.0) {
    return gpu::plan_chunk_atomics(args, sections, groups, size,
                                   gpu::ChunkLimits{lanes, shared_bytes},
                                   unstaged_singles, min_compression);
}

gpu::ChunkResult colouring(std::span<const op_arg> args, const Sections& sections,
                           const plan::ArgGroups& groups, int lanes, int size,
                           std::size_t shared_bytes, bool wavefront = false,
                           int window = 0) {
    return gpu::plan_chunk_colouring(args, sections, groups, size,
                                     gpu::ChunkLimits{lanes, shared_bytes},
                                     wavefront, window);
}

op_arg make_dat_arg(op_dat dat, op_map map, int dim, const char *type,
                    int scalar_size, int opt = 1,
                    op_access access = OP_INC) {
    op_arg arg{};
    arg.opt = opt;
    arg.argtype = OP_ARG_DAT;
    arg.dat = dat;
    arg.map = map;
    arg.dim = dim;
    arg.idx = 0;
    arg.size = dim * scalar_size;
    arg.map_data = map->map;
    arg.type = type;
    arg.acc = access;
    return arg;
}

void initialize_set(op_set_core& set, int size, int core_size = -1,
                    int exec_size = 0, int nonexec_size = 0) {
    set = {};
    set.size = size;
    set.core_size = core_size < 0 ? size : core_size;
    set.exec_size = exec_size;
    set.nonexec_size = nonexec_size;
}

void initialize_dat(op_dat_core& dat, op_set set, int dim, int scalar_size,
                    const char *type) {
    dat = {};
    dat.set = set;
    dat.dim = dim;
    dat.size = dim * scalar_size;
    dat.type = type;
}

void initialize_map(op_map_core& map, op_set from, op_set to,
                    std::vector<int>& values) {
    map = {};
    map.from = from;
    map.to = to;
    map.dim = 1;
    map.map = values.data();
}

// Three dats of different scalar types over a set of 6 elements, 2 core:
// two sections {0,2} and {2,6}.
struct MixedFixture {
    op_set_core source{};
    op_set_core target_a{};
    op_set_core target_b{};
    op_set_core target_c{};

    op_dat_core dat_a{};
    op_dat_core dat_b{};
    op_dat_core dat_c{};

    std::vector<int> map_a0_values{0, 0, 1, 1, 2, 2};
    std::vector<int> map_a1_values{0, 1, 1, 2, 2, 3};
    std::vector<int> map_b_values{0, 1, 0, 1, 2, 2};
    std::vector<int> map_c_values{0, 0, 0, 1, 1, 1};

    op_map_core map_a0{};
    op_map_core map_a1{};
    op_map_core map_b{};
    op_map_core map_c{};

    std::vector<op_arg> args;
    std::array<plan::ArgGroup, 4> group_args{{{0, 0}, {1, 0}, {2, 1}, {3, 2}}};

    MixedFixture() {
        initialize_set(source, 6, 2);
        initialize_set(target_a, 4);
        initialize_set(target_b, 3);
        initialize_set(target_c, 2);

        initialize_dat(dat_a, &target_a, 2, sizeof(double), "double");
        initialize_dat(dat_b, &target_b, 3, sizeof(int), "int");
        initialize_dat(dat_c, &target_c, 1, sizeof(float), "float");

        initialize_map(map_a0, &source, &target_a, map_a0_values);
        initialize_map(map_a1, &source, &target_a, map_a1_values);
        initialize_map(map_b, &source, &target_b, map_b_values);
        initialize_map(map_c, &source, &target_c, map_c_values);

        args.push_back(make_dat_arg(&dat_a, &map_a0, 2, "double",
                                    sizeof(double)));
        args.push_back(make_dat_arg(&dat_a, &map_a1, 2, "real(8)",
                                    sizeof(double)));
        args.push_back(
            make_dat_arg(&dat_b, &map_b, 3, "integer(4)", sizeof(int)));
        args.push_back(
            make_dat_arg(&dat_c, &map_c, 1, "real(4)", sizeof(float), 0));
    }

    plan::ArgGroups groups() const { return {group_args, 3}; }
    Sections sections() { return Sections::indirect(&source, false); }

    gpu::ChunkResult build(bool coloured = false) {
        return coloured ? colouring(args, sections(), groups(), 2, 4, 1024)
                        : atomics(args, sections(), groups(), 2, 4, 1024);
    }
};

// One dat reached through an identity map over 1024 elements.
struct ChunkFixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    std::vector<int> map_values;
    op_map_core map{};
    std::array<op_arg, 1> args{};
    std::array<plan::ArgGroup, 1> group_args{{{0, 0}}};

    ChunkFixture() : map_values(1024) {
        initialize_set(source, 1024);
        initialize_set(target, 1024);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        for (int i = 0; i < 1024; ++i)
            map_values[static_cast<std::size_t>(i)] = i;
        initialize_map(map, &source, &target, map_values);
        args[0] = make_dat_arg(&dat, &map, 1, "double", sizeof(double));
    }

    plan::ArgGroups groups() const { return {group_args, 1}; }

    gpu::ChunkResult build(int requested, std::size_t scratch, int lanes = 128) {
        return atomics(args, Sections::direct(&source), groups(), lanes, requested,
                       scratch);
    }
};

void expect_rejected(const gpu::ChunkResult& result, std::string_view reason) {
    CHECK(!result);
    CHECK(result.rejected == reason);
}

std::size_t num_chunks(const gpu::ChunkPlan& chunk_plan) {
    return static_cast<std::size_t>(chunk_plan.chunks.count());
}

std::size_t max_chunk_scratch_bytes(const gpu::ChunkPlan& chunk_plan, int begin,
                                    int end) {
    std::size_t bytes = 0;
    for (int position = begin; position < end; ++position)
        bytes = std::max(bytes, chunk_plan.staging.chunk_scratch_bytes[
                                    static_cast<std::size_t>(
                                        chunk_plan.batches.chunk(position))]);
    return bytes;
}

plan::StageWord stage_word(const gpu::ChunkPlan& chunk_plan,
                            std::size_t staged_arg, int source_element) {
    return chunk_plan.staging.stage_words[
        staged_arg * static_cast<std::size_t>(chunk_plan.staging.stride) +
        static_cast<std::size_t>(source_element)];
}

// Each region's slot count, from the offsets.
std::vector<int> slot_counts(const plan::Staging& staging) {
    std::vector<int> counts;
    for (std::size_t region = 0; region + 1 < staging.slot_offsets.size(); ++region)
        counts.push_back(staging.slot_offsets[region + 1] - staging.slot_offsets[region]);
    return counts;
}

int stage_count(const gpu::ChunkPlan& chunk_plan, std::size_t ndats,
                std::size_t chunk, std::size_t staged_dat) {
    const auto& offsets = chunk_plan.staging.slot_offsets;
    auto region = chunk * ndats + staged_dat;
    return offsets[region + 1] - offsets[region];
}

// Check the staging every kernel relies on: each region lists its chunk's
// distinct targets in ascending order, a slot each, and every active staged
// reference's word names its target's slot.
void check_slot_lists(const gpu::ChunkPlan& chunk_plan,
                      std::span<const op_arg> args,
                      const plan::ArgGroups& groups) {
    const auto& staging = chunk_plan.staging;
    const auto ndats = static_cast<std::size_t>(groups.groups);
    CHECK(staging.slot_offsets.size() == num_chunks(chunk_plan) * ndats + 1);
    CHECK(staging.slot_offsets.front() == 0);
    CHECK(static_cast<std::size_t>(staging.slot_offsets.back()) ==
          staging.slot_targets.size());

    std::size_t slots = 0;
    for (std::size_t region = 0; region + 1 < staging.slot_offsets.size(); ++region) {
        auto first = static_cast<std::size_t>(staging.slot_offsets[region]);
        CHECK(staging.slot_offsets[region + 1] >= staging.slot_offsets[region]);
        auto count = static_cast<std::size_t>(staging.slot_offsets[region + 1]) - first;
        for (std::size_t s = 1; s < count; ++s)
            CHECK(staging.slot_targets[first + s - 1] <
                  staging.slot_targets[first + s]);
        slots += count;
    }
    CHECK(slots + chunk_plan.staging.unstaged_singles ==
          chunk_plan.distinct_targets);

    for (std::size_t chunk = 0; chunk < num_chunks(chunk_plan); ++chunk) {
        for (int source = chunk_plan.chunks.offsets[chunk];
             source < chunk_plan.chunks.offsets[chunk + 1]; ++source) {
            for (std::size_t i = 0; i < groups.args.size(); ++i) {
                const auto& group = groups.args[i];
                const op_arg& arg = args[static_cast<std::size_t>(group.arg)];
                if (arg.opt == 0)
                    continue;

                auto region = chunk * ndats + static_cast<std::size_t>(group.group);
                int slot = stage_word(chunk_plan, i, source);
                if (slot == plan::unstaged_word)
                    continue;
                CHECK(slot < staging.slot_offsets[region + 1] - staging.slot_offsets[region]);
                CHECK(staging.slot_targets[static_cast<std::size_t>(
                          staging.slot_offsets[region] + slot)] ==
                      arg.map_data[source * arg.map->dim + arg.idx]);
            }
        }
    }
}

// A chunk that spanned two sections would run by whichever section came
// first, so its later half would run before the halo wait or after the
// globals had already been processed.
void check_chunks_within_sections(const gpu::ChunkPlan& chunk_plan,
                                  const Sections& sections) {
    CHECK(chunk_plan.chunks.section_offsets.size() ==
          static_cast<std::size_t>(sections.size()) + 1);
    CHECK(chunk_plan.chunks.section_offsets.front() == 0);
    CHECK(chunk_plan.chunks.section_offsets.back() ==
          static_cast<int>(num_chunks(chunk_plan)));

    for (int section = 0; section < sections.size(); ++section) {
        int first = chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section)];
        int last = chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section) + 1];
        CHECK(first <= last);

        for (int c = first; c < last; ++c) {
            int begin = chunk_plan.chunks.offsets[static_cast<std::size_t>(c)];
            int end = chunk_plan.chunks.offsets[static_cast<std::size_t>(c) + 1];
            CHECK(begin < end);
            CHECK(begin >= sections[section].begin);
            CHECK(end <= sections[section].end);
        }

        if (first < last) {
            CHECK(chunk_plan.chunks.offsets[static_cast<std::size_t>(first)] ==
                  sections[section].begin);
            CHECK(chunk_plan.chunks.offsets[static_cast<std::size_t>(last)] ==
                  sections[section].end);
        } else {
            CHECK(sections[section].size() == 0);
        }
    }
}

void test_mixed_plan() {
    MixedFixture fixture;
    auto result = fixture.build();
    CHECK(result);

    const auto& chunk_plan = *result.plan;
    CHECK(chunk_plan.chunks.size == 4);
    CHECK(chunk_plan.staging.stride == 32);
    CHECK(num_chunks(chunk_plan) == 2);
    CHECK((chunk_plan.chunks.offsets == std::vector<int>{0, 2, 6}));
    CHECK((chunk_plan.chunks.section_offsets == std::vector<int>{0, 1, 2}));
    // One batch per section, sized by its chunk.
    CHECK((chunk_plan.batches.section_batches == std::vector<int>{0, 1, 2}));
    CHECK(chunk_plan.batches.batch_offsets == chunk_plan.chunks.section_offsets);
    CHECK(chunk_plan.batches.batches(0) == 1 && chunk_plan.batches.batches(1) == 1);
    CHECK(chunk_plan.batches.positions(1, 0) == std::make_pair(1, 2));
    CHECK(chunk_plan.scratch_bytes(1, 0) ==
          chunk_plan.staging.chunk_scratch_bytes[1]);
    CHECK((slot_counts(chunk_plan.staging) == std::vector<int>{2, 2, 0, 3, 3, 0}));
    CHECK(stage_count(chunk_plan, 3, 0, 0) == 2);
    CHECK(stage_count(chunk_plan, 3, 0, 1) == 2);
    CHECK((chunk_plan.staging.chunk_scratch_bytes ==
           std::vector<std::size_t>{56, 84}));
    CHECK(chunk_plan.batch_scratch_bytes == chunk_plan.staging.chunk_scratch_bytes);

    CHECK(stage_word(chunk_plan, 0, 0) == 0);
    CHECK(stage_word(chunk_plan, 1, 0) == 0);
    CHECK(stage_word(chunk_plan, 0, 1) == 0);
    CHECK(stage_word(chunk_plan, 1, 1) == 1);
    CHECK(stage_word(chunk_plan, 2, 0) == 0);
    CHECK(stage_word(chunk_plan, 2, 1) == 1);
    CHECK(stage_word(chunk_plan, 1, 5) == 2);
    CHECK(stage_word(chunk_plan, 2, 4) == 2);
    CHECK((chunk_plan.staging.slot_offsets == std::vector<int>{0, 2, 4, 4, 7, 10, 10}));
    CHECK((chunk_plan.staging.slot_targets ==
           std::vector<int>{0, 1, 0, 1, 1, 2, 3, 0, 1, 2}));

    // The inactive argument's words stay untouched.
    for (int source = 0; source < 6; ++source)
        CHECK(stage_word(chunk_plan, 3, source) == 0);
    check_slot_lists(chunk_plan, fixture.args, fixture.groups());

    CHECK(chunk_plan.references == 18);
    CHECK(chunk_plan.distinct_targets == 10);
    check_chunks_within_sections(chunk_plan, fixture.sections());

    // The default floor rejects a plan below 1.5x compression.
    auto rejected = atomics(fixture.args, fixture.sections(), fixture.groups(), 2, 4,
                            1024, false, 1.5);
    CHECK(rejected);
    fixture.map_a0_values = {0, 1, 0, 1, 2, 3};
    fixture.map_a1_values = {2, 3, 1, 2, 3, 0};
    fixture.map_b_values = {0, 1, 0, 1, 2, 0};
    rejected = atomics(fixture.args, fixture.sections(), fixture.groups(), 2, 4, 1024,
                       false, 1.5);
    expect_rejected(rejected, "low_compression");
    CHECK(rejected.plan == nullptr && rejected.compression > 0.0 &&
          rejected.compression < 1.5);
}

void test_optional_and_alignment() {
    MixedFixture fixture;
    CHECK(fixture.build());

    fixture.args[3].opt = 1;
    auto active = fixture.build();
    CHECK(active);
    CHECK((slot_counts(active.plan->staging) == std::vector<int>{2, 2, 1, 3, 3, 2}));
    CHECK((active.plan->batch_scratch_bytes ==
           std::vector<std::size_t>{60, 92}));
    CHECK(active.plan->references == 24);
    CHECK(active.plan->distinct_targets == 13);

    // A float region before a double one pads the double region's start.
    fixture.args[1].opt = 0;
    fixture.args[2].opt = 0;
    std::array<plan::ArgGroup, 2> group_args{{{3, 0}, {0, 1}}};
    auto aligned = atomics(fixture.args, fixture.sections(),
                           plan::ArgGroups{group_args, 2}, 2, 4, 1024);
    CHECK(aligned);
    CHECK((aligned.plan->batch_scratch_bytes ==
           std::vector<std::size_t>{24, 40}));
}

// The kernel's region walk over each chunk's counts ends within the scratch
// the plan sized it, an inactive double group between two active ones
// included.
void test_region_walk() {
    MixedFixture fixture;
    fixture.args[0].opt = 0;
    fixture.args[1].opt = 0;
    fixture.args[3].opt = 1;
    std::array<plan::ArgGroup, 4> group_args{{{3, 0}, {0, 1}, {1, 1}, {2, 2}}};
    auto result = atomics(fixture.args, fixture.sections(),
                          plan::ArgGroups{group_args, 3}, 2, 4, 1024);
    CHECK(result);
    const auto& staging = result.plan->staging;
    const auto all_counts = slot_counts(staging);
    alignas(16) static char base[1 << 12];
    for (int chunk = 0; chunk < result.plan->chunks.count(); ++chunk) {
        const int *counts = &all_counts[static_cast<std::size_t>(chunk) * 3];
        char *cursor = base;
        op::f2c::stage_region<float>(cursor, counts[0], 1);
        op::f2c::stage_region<double>(cursor, counts[1], 2);
        op::f2c::stage_region<int>(cursor, counts[2], 3);
        CHECK(static_cast<std::size_t>(cursor - base) <=
              staging.chunk_scratch_bytes[static_cast<std::size_t>(chunk)]);
    }
}

void test_chunk_sizes_and_clamping() {
    ChunkFixture fixture;
    for (int chunk_size : {128, 256, 512}) {
        auto result = fixture.build(chunk_size, 1024 * sizeof(double));
        CHECK(result);
        CHECK(result.plan->chunks.size == chunk_size);
        CHECK(num_chunks(*result.plan) ==
              static_cast<std::size_t>(1024 / chunk_size));
        CHECK(result.plan->chunks.offsets.front() == 0);
        CHECK(result.plan->chunks.offsets.back() == 1024);
        CHECK(result.plan->batch_scratch_bytes[0] ==
              static_cast<std::size_t>(chunk_size) * sizeof(double));
    }

    // A request beyond the scratch clamps to what fits.
    auto clamped = fixture.build(512, 2048);
    CHECK(clamped);
    CHECK(clamped.plan->chunks.size == 256);
    CHECK(num_chunks(*clamped.plan) == 4);

    auto saturated = fixture.build(INT_MAX, 1024 * sizeof(double));
    CHECK(saturated);
    CHECK(saturated.plan->chunks.size == 1024);

    expect_rejected(fixture.build(512, 1023), "insufficient_scratch");

    // A region's slots must fit a 16-bit stage word, however much scratch
    // there is.
    auto bounded = fixture.build(INT_MAX, std::size_t{1} << 20);
    CHECK(bounded);
    CHECK(bounded.plan->chunks.size == plan::max_slots / 128 * 128);
    static_assert(plan::max_slots - 1 <=
                  std::numeric_limits<plan::StageWord>::max());
}

void test_runtime_rejections() {
    {
        MixedFixture fixture;
        fixture.args.push_back(make_dat_arg(&fixture.dat_a, &fixture.map_a0,
                                             2, "double", sizeof(double), 1,
                                             OP_READ));
        expect_rejected(fixture.build(), "incompatible_argument");
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_rejected(fixture.build(), "no_active_argument");
    }
    {
        // Two groups resolving to one dat.
        MixedFixture fixture;
        std::array<plan::ArgGroup, 4> group_args{{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
        auto result = atomics(fixture.args, fixture.sections(),
                              plan::ArgGroups{group_args, 4}, 2, 4, 1024);
        expect_rejected(result, "incompatible_argument");
    }
}

// One dat reached through one map, whose first references in each chunk
// come in descending target order:
//
//   source  0 1 2 3 | 4 5 6 7      chunk 0 = [0,4), chunk 1 = [4,8)
//   target  3 1 3 0 | 2 1 2 0
struct OrderFixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    std::vector<int> map_values{3, 1, 3, 0, 2, 1, 2, 0};
    op_map_core map{};
    std::array<op_arg, 1> args{};
    std::array<plan::ArgGroup, 1> group_args{{{0, 0}}};

    OrderFixture() {
        initialize_set(source, 8);
        initialize_set(target, 4);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        initialize_map(map, &source, &target, map_values);
        args[0] = make_dat_arg(&dat, &map, 1, "double", sizeof(double));
    }

    gpu::ChunkResult build(bool unstaged_singles = false) {
        return atomics(args, Sections::direct(&source), plan::ArgGroups{group_args, 1},
                       4, 4, 1024, unstaged_singles);
    }
};

std::vector<int> words_of(const gpu::ChunkPlan& chunk_plan, int elements) {
    std::vector<int> words;
    for (int element = 0; element < elements; ++element)
        words.push_back(stage_word(chunk_plan, 0, element));
    return words;
}

void test_slots_in_target_order() {
    OrderFixture fixture;
    auto result = fixture.build();
    CHECK(result);
    const auto& chunk_plan = *result.plan;
    CHECK(num_chunks(chunk_plan) == 2);
    CHECK((slot_counts(chunk_plan.staging) == std::vector<int>{3, 3}));
    CHECK((chunk_plan.staging.slot_targets == std::vector<int>{0, 1, 3, 0, 1, 2}));
    CHECK((words_of(chunk_plan, 8) == std::vector<int>{2, 1, 2, 0, 2, 1, 2, 0}));
    CHECK(chunk_plan.references == 8);
    CHECK(chunk_plan.distinct_targets == 6);
    check_slot_lists(chunk_plan, fixture.args, plan::ArgGroups{fixture.group_args, 1});
}

// Targets 1 and 0 of the first chunk, and 1 and 0 of the second, are
// referenced once there, so they stay in global memory.
void test_unstaged_singles() {
    OrderFixture fixture;
    auto result = fixture.build(true);
    CHECK(result);
    const auto& chunk_plan = *result.plan;
    constexpr int u = plan::unstaged_word;
    CHECK((slot_counts(chunk_plan.staging) == std::vector<int>{1, 1}));
    CHECK((chunk_plan.staging.slot_targets == std::vector<int>{3, 2}));
    CHECK((words_of(chunk_plan, 8) == std::vector<int>{0, u, 0, u, 0, u, 0, u}));
    CHECK(chunk_plan.references == 8);
    CHECK(chunk_plan.distinct_targets == 6);
    CHECK(chunk_plan.staging.unstaged_singles == 4);
    check_slot_lists(chunk_plan, fixture.args, plan::ArgGroups{fixture.group_args, 1});
}

// Check every colouring invariant the kernel relies on: batches divide
// each section's chunks, elements of one chunk and thread colour reach
// distinct targets, and chunks of one batch reach distinct targets.
void check_colouring(const gpu::ChunkPlan& chunk_plan,
                     std::span<const op_arg> args,
                     const plan::ArgGroups& groups, const Sections& sections) {
    CHECK(!chunk_plan.batches.order.empty());
    check_slot_lists(chunk_plan, args, groups);
    check_chunks_within_sections(chunk_plan, sections);
    CHECK(chunk_plan.batches.section_batches.size() ==
          static_cast<std::size_t>(sections.size()) + 1);
    CHECK(chunk_plan.batches.section_batches.front() == 0);
    CHECK(chunk_plan.batches.section_batches.back() + 1 ==
          static_cast<int>(chunk_plan.batches.batch_offsets.size()));
    CHECK(chunk_plan.batches.batch_offsets.front() == 0);
    CHECK(chunk_plan.batches.batch_offsets.back() ==
          static_cast<int>(num_chunks(chunk_plan)));
    CHECK(chunk_plan.batches.order.size() == num_chunks(chunk_plan));
    CHECK(chunk_plan.chunk_thread_colours.size() == num_chunks(chunk_plan));
    CHECK(chunk_plan.batches.count() + 1 == chunk_plan.batches.batch_offsets.size());
    CHECK(chunk_plan.batch_scratch_bytes.size() == chunk_plan.batches.count());

    auto targets = [&](int source) {
        std::vector<std::pair<int, int>> out;
        for (const auto& group : groups.args) {
            const op_arg& arg = args[static_cast<std::size_t>(group.arg)];
            if (arg.opt != 0)
                out.push_back({group.group,
                               arg.map_data[source * arg.map->dim + arg.idx]});
        }
        return out;
    };
    auto disjoint = [](const std::vector<std::pair<int, int>>& a,
                       const std::vector<std::pair<int, int>>& b) {
        for (const auto& x : a)
            for (const auto& y : b)
                if (x == y)
                    return false;
        return true;
    };

    std::vector<int> run(num_chunks(chunk_plan), 0);
    for (int section = 0; section < sections.size(); ++section) {
        for (int batch = 0; batch < chunk_plan.batches.batches(section); ++batch) {
            auto [begin, end] = chunk_plan.batches.positions(section, batch);
            CHECK(begin < end);
            CHECK(chunk_plan.scratch_bytes(section, batch) ==
                  max_chunk_scratch_bytes(chunk_plan, begin, end));

            std::vector<std::vector<std::pair<int, int>>> chunk_targets;
            for (int i = begin; i < end; ++i) {
                int c = chunk_plan.batches.order[static_cast<std::size_t>(i)];
                CHECK(c >= chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section)] &&
                      c < chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section) + 1]);
                ++run[static_cast<std::size_t>(c)];

                int first = chunk_plan.chunks.offsets[static_cast<std::size_t>(c)];
                int last = chunk_plan.chunks.offsets[static_cast<std::size_t>(c) + 1];
                int ncolours = chunk_plan.chunk_thread_colours[static_cast<std::size_t>(c)];
                CHECK(ncolours >= 1 && ncolours <= 255);
                CHECK(ncolours <= chunk_plan.max_thread_colours);

                std::vector<std::pair<int, int>> reached;
                for (int a = first; a < last; ++a) {
                    int colour = chunk_plan.thread_colours[static_cast<std::size_t>(a)];
                    CHECK(colour < ncolours);
                    auto a_targets = targets(a);
                    reached.insert(reached.end(), a_targets.begin(), a_targets.end());
                    for (int b = first; b < a; ++b)
                        if (chunk_plan.thread_colours[static_cast<std::size_t>(b)] == colour)
                            CHECK(disjoint(a_targets, targets(b)));
                }

                for (const auto& other : chunk_targets)
                    CHECK(disjoint(reached, other));
                chunk_targets.push_back(std::move(reached));
            }
        }
    }

    CHECK(run == std::vector<int>(num_chunks(chunk_plan), 1));
}

void test_colour_mixed_plan() {
    MixedFixture fixture;
    fixture.args[0].acc = OP_RW;
    fixture.args[1].acc = OP_READ;

    auto result = fixture.build(true);
    CHECK(result);
    const auto& chunk_plan = *result.plan;
    CHECK(chunk_plan.chunks.size == 4);
    CHECK((chunk_plan.chunks.offsets == std::vector<int>{0, 2, 6}));
    CHECK((chunk_plan.batches.order == std::vector<int>{0, 1}));
    CHECK((chunk_plan.batches.batch_offsets == std::vector<int>{0, 1, 2}));
    CHECK((chunk_plan.batches.section_batches == std::vector<int>{0, 1, 2}));
    CHECK((chunk_plan.chunk_thread_colours == std::vector<int>{2, 3}));
    CHECK((std::vector<int>(chunk_plan.thread_colours.begin(),
                            chunk_plan.thread_colours.begin() + 6) ==
           std::vector<int>{0, 1, 0, 1, 0, 2}));
    CHECK(chunk_plan.batches.count() == 2);
    CHECK(chunk_plan.max_thread_colours == 3);
    check_colouring(chunk_plan, fixture.args, fixture.groups(), fixture.sections());

    fixture.args[3].opt = 1;
    auto active = fixture.build(true);
    CHECK(active);
    check_colouring(*active.plan, fixture.args, fixture.groups(),
                    fixture.sections());
}

// Edge i of a chain reaches nodes i and i+1, so neighbouring edges and
// neighbouring chunks share a node.
struct ChainFixture {
    op_set_core edges{};
    op_set_core nodes{};
    op_dat_core dat{};
    std::vector<int> map_values;
    op_map_core map{};
    std::vector<op_arg> args;
    std::array<plan::ArgGroup, 2> group_args{{{0, 0}, {1, 0}}};

    explicit ChainFixture(int size, int core = -1, int exec = 0) {
        initialize_set(edges, size, core, exec);
        initialize_set(nodes, size + exec + 1);
        initialize_dat(dat, &nodes, 1, sizeof(int), "integer(4)");
        for (int i = 0; i < size + exec; ++i) {
            map_values.push_back(i);
            map_values.push_back(i + 1);
        }
        initialize_map(map, &edges, &nodes, map_values);
        map.dim = 2;
        for (int idx = 0; idx < 2; ++idx) {
            args.push_back(make_dat_arg(&dat, &map, 1, "integer(4)",
                                        sizeof(int), 1, OP_RW));
            args.back().idx = idx;
        }
    }

    plan::ArgGroups groups() const { return {group_args, 1}; }

    gpu::ChunkResult build(const Sections& sections, int lanes,
                        std::size_t scratch = 1 << 16) {
        return colouring(args, sections, groups(), lanes, lanes, scratch);
    }
};

void test_colour_chain_batches() {
    ChainFixture fixture(16);
    auto sections = Sections::direct(&fixture.edges);

    auto result = fixture.build(sections, 4);
    CHECK(result);
    const auto& chunk_plan = *result.plan;
    CHECK(num_chunks(chunk_plan) == 4);
    CHECK((chunk_plan.batches.order == std::vector<int>{0, 2, 1, 3}));
    CHECK((chunk_plan.batches.batch_offsets == std::vector<int>{0, 2, 4}));
    CHECK((chunk_plan.batches.section_batches == std::vector<int>{0, 2}));
    CHECK((chunk_plan.chunk_thread_colours == std::vector<int>{2, 2, 2, 2}));
    for (int edge = 0; edge < 16; ++edge)
        CHECK(chunk_plan.thread_colours[static_cast<std::size_t>(edge)] == edge % 2);
    check_colouring(chunk_plan, fixture.args, fixture.groups(), sections);

    // Colours never cross a section, and an empty section has no batches:
    // 8 core elements, no further owned ones, 8 in the exec halo.
    ChainFixture halo(8, 8, 8);
    auto split = Sections::indirect(&halo.edges, true);
    CHECK(split.size() == 3 && split[1].size() == 0);
    auto sectioned = halo.build(split, 4);
    CHECK(sectioned);
    CHECK((sectioned.plan->batches.section_batches ==
           std::vector<int>{0, 2, 2, 4}));
    check_colouring(*sectioned.plan, halo.args, halo.groups(), split);
}

// An empty section has no batches under every batching, and an empty set
// plans no chunks and no batches.
void test_empty_sections() {
    ChainFixture halo(8, 8, 8);
    auto split = Sections::indirect(&halo.edges, true);
    auto waves = colouring(halo.args, split, halo.groups(), 4, 4, 1 << 16, true);
    CHECK(waves && waves.plan->wavefront);
    CHECK((waves.plan->batches.section_batches == std::vector<int>{0, 1, 1, 2}));

    for (auto& arg : halo.args)
        arg.acc = OP_INC;
    auto staged = atomics(halo.args, split, halo.groups(), 4, 4, 1 << 16);
    CHECK(staged);
    CHECK((staged.plan->batches.section_batches == std::vector<int>{0, 1, 1, 2}));
    CHECK((staged.plan->batch_scratch_bytes.size() == 2));

    ChainFixture empty(0);
    auto none = Sections::direct(&empty.edges);
    for (bool wavefront : {false, true}) {
        auto result = colouring(empty.args, none, empty.groups(), 4, 4, 1 << 16, wavefront);
        CHECK(result && result.plan->chunks.count() == 0);
        CHECK(result.plan->batches.count() == 0 && result.plan->batches.batches(0) == 0);
    }
    for (auto& arg : empty.args)
        arg.acc = OP_INC;
    auto result = atomics(empty.args, none, empty.groups(), 4, 4, 1 << 16);
    CHECK(result && result.plan->chunks.count() == 0);
    CHECK(result.plan->batches.count() == 0 && result.plan->batches.batches(0) == 0);
}

// Every element reaches one target, so a chunk needs one colour per element.
void test_colour_rounds_and_limit() {
    for (int size : {100, 255, 256}) {
        op_set_core source{};
        op_set_core target{};
        op_dat_core dat{};
        op_map_core map{};
        std::vector<int> map_values(static_cast<std::size_t>(size), 0);
        initialize_set(source, size);
        initialize_set(target, 1);
        initialize_dat(dat, &target, 1, sizeof(double), "real(8)");
        initialize_map(map, &source, &target, map_values);
        std::array<op_arg, 1> args{
            make_dat_arg(&dat, &map, 1, "real(8)", sizeof(double))};
        std::array<plan::ArgGroup, 1> group_args{{{0, 0}}};

        auto result = colouring(args, Sections::direct(&source),
                                plan::ArgGroups{group_args, 1}, size, size, 1 << 20);
        if (size > 255) {
            expect_rejected(result, "too_many_colours");
            continue;
        }

        CHECK(result);
        CHECK(result.plan->max_thread_colours == size);
        for (int i = 0; i < size; ++i)
            CHECK(result.plan->thread_colours[
                      static_cast<std::size_t>(i)] == i);
        check_colouring(*result.plan, args, plan::ArgGroups{group_args, 1},
                        Sections::direct(&source));
    }
}

// Random maps over a small target set, with several sections and arguments.
void test_colour_random() {
    constexpr int size = 1000;
    constexpr int targets = 97;
    op_set_core source{};
    op_set_core target_set{};
    op_dat_core dat_a{};
    op_dat_core dat_b{};
    initialize_set(source, 900, 300, 100);
    initialize_set(target_set, targets);
    initialize_dat(dat_a, &target_set, 2, sizeof(double), "real(8)");
    initialize_dat(dat_b, &target_set, 1, sizeof(int), "integer(4)");

    std::uint32_t state = 12345;
    std::array<std::vector<int>, 3> values;
    std::array<op_map_core, 3> maps{};
    for (std::size_t m = 0; m < maps.size(); ++m) {
        for (int i = 0; i < size; ++i) {
            state = state * 1664525u + 1013904223u;
            values[m].push_back(static_cast<int>((state >> 8) % targets));
        }
        initialize_map(maps[m], &source, &target_set, values[m]);
    }

    std::vector<op_arg> args{
        make_dat_arg(&dat_a, &maps[0], 2, "real(8)", sizeof(double), 1, OP_RW),
        make_dat_arg(&dat_a, &maps[1], 2, "real(8)", sizeof(double), 1, OP_READ),
        make_dat_arg(&dat_b, &maps[2], 1, "integer(4)", sizeof(int), 1, OP_INC),
    };
    std::array<plan::ArgGroup, 3> group_args{{{0, 0}, {1, 0}, {2, 1}}};
    plan::ArgGroups groups{group_args, 2};
    auto sections = Sections::indirect(&source, true);

    auto result = colouring(args, sections, groups, 32, 64, 1 << 20);
    CHECK(result);
    CHECK(result.plan->batches.count() > sections.size());
    check_colouring(*result.plan, args, groups, sections);

    // A wavefront plan runs each section as one batch, each position after
    // the latest earlier position of its section sharing one of its targets.
    auto wave = colouring(args, sections, groups, 32, 64, 1 << 20, true, 4);
    CHECK(wave);
    const auto& chunk_plan = *wave.plan;
    check_slot_lists(chunk_plan, args, groups);
    CHECK(chunk_plan.wavefront);
    CHECK(chunk_plan.batches.count() == sections.size());
    CHECK(chunk_plan.wave_pred_offsets.size() == num_chunks(chunk_plan) + 1);

    const std::size_t ndats = 2;
    for (int section = 0; section < sections.size(); ++section) {
        int begin = chunk_plan.batches.batch_offsets[static_cast<std::size_t>(section)];
        int end = chunk_plan.batches.batch_offsets[static_cast<std::size_t>(section) + 1];
        CHECK(begin == chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section)]);
        CHECK(end == chunk_plan.chunks.section_offsets[static_cast<std::size_t>(section) + 1]);

        // The latest position so far reaching each (dat, target).
        std::vector<std::vector<int>> latest(ndats, std::vector<int>(targets, -1));
        for (int position = begin; position < end; ++position) {
            auto c = static_cast<std::size_t>(
                chunk_plan.batches.order[static_cast<std::size_t>(position)]);
            CHECK(static_cast<int>(c) >= begin && static_cast<int>(c) < end);
            std::vector<int> preds(
                chunk_plan.wave_preds.begin() +
                    chunk_plan.wave_pred_offsets[static_cast<std::size_t>(position)],
                chunk_plan.wave_preds.begin() +
                    chunk_plan.wave_pred_offsets[static_cast<std::size_t>(position) + 1]);
            for (int pred : preds)
                CHECK(pred >= begin && pred < position);

            for (std::size_t dat = 0; dat < ndats; ++dat) {
                const auto& staging = chunk_plan.staging;
                auto region = c * ndats + dat;
                for (int s = 0; s < staging.slot_offsets[region + 1] - staging.slot_offsets[region];
                     ++s) {
                    int target = staging.slot_targets[static_cast<std::size_t>(
                        staging.slot_offsets[region] + s)];
                    int& last = latest[dat][static_cast<std::size_t>(target)];
                    if (last >= 0)
                        CHECK(std::find(preds.begin(), preds.end(), last) != preds.end());
                    last = position;
                }
            }
        }
    }
}

void test_colour_rejections() {
    {
        ChainFixture fixture(8);
        expect_rejected(fixture.build(Sections::direct(&fixture.edges), 4, 16),
                        "insufficient_scratch");
    }
    {
        MixedFixture fixture;
        fixture.args.push_back(make_dat_arg(&fixture.dat_b, &fixture.map_b,
                                             3, "integer(4)", sizeof(int), 1,
                                             OP_READ));
        expect_rejected(fixture.build(true), "incompatible_argument");
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_rejected(fixture.build(true), "no_active_argument");
    }
    {
        // Two groups resolving to one dat would be coloured independently.
        ChainFixture fixture(8);
        std::array<plan::ArgGroup, 2> group_args{{{0, 0}, {1, 1}}};
        auto result = colouring(fixture.args, Sections::direct(&fixture.edges),
                                plan::ArgGroups{group_args, 2}, 4, 4, 1 << 16);
        expect_rejected(result, "incompatible_argument");
    }
}

} // namespace

int main() {
    try {
        test_mixed_plan();
        test_optional_and_alignment();
        test_region_walk();
        test_chunk_sizes_and_clamping();
        test_runtime_rejections();
        test_slots_in_target_order();
        test_unstaged_singles();
        test_colour_mixed_plan();
        test_colour_chain_batches();
        test_empty_sections();
        test_colour_rounds_and_limit();
        test_colour_random();
        test_colour_rejections();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "chunk chunk_plan test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("chunk chunk_plan tests passed\n");
    return EXIT_SUCCESS;
}

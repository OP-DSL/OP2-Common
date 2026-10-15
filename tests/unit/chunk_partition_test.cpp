// The chunk partition: staging slot lists, chunk sizing against the
// scratch, the coloured kind's invariants, and the rejections.

#include <loop/chunk_partition.h>

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

namespace chunk = op::chunk;
using op::loop::Configuration;
using op::loop::Sections;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

// Options with no compression floor, so small fixtures are never rejected
// for it.
chunk::Options test_options(bool unstaged_singles = false, bool wavefront = false,
                            int window = 0) {
    chunk::Options options;
    options.min_compression = 0.0;
    options.unstaged_singles = unstaged_singles;
    options.wavefront = wavefront;
    options.window = window;
    return options;
}

chunk::Constraints constraints(int lanes, int elements,
                               std::size_t scratch_bytes) {
    chunk::Constraints out;
    out.lanes = lanes;
    out.elements = elements;
    out.scratch_bytes = scratch_bytes;
    return out;
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
    std::array<chunk::ArgGroup, 4> group_args{{{0, 0}, {1, 0}, {2, 1}, {3, 2}}};

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

    chunk::ArgGroups groups() const { return {group_args, 3}; }
    Sections sections() { return Sections::indirect(&source, false); }

    chunk::Result build(chunk::Kind kind = chunk::Kind::staged) {
        return chunk::build_partition(kind, Configuration{&source, args},
                                      sections(), groups(),
                                      constraints(2, 4, 1024), test_options());
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
    std::array<chunk::ArgGroup, 1> group_args{{{0, 0}}};

    ChunkFixture() : map_values(1024) {
        initialize_set(source, 1024);
        initialize_set(target, 1024);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        for (int i = 0; i < 1024; ++i)
            map_values[static_cast<std::size_t>(i)] = i;
        initialize_map(map, &source, &target, map_values);
        args[0] = make_dat_arg(&dat, &map, 1, "double", sizeof(double));
    }

    chunk::ArgGroups groups() const { return {group_args, 1}; }

    chunk::Result build(int requested, std::size_t scratch, int lanes = 128,
                        chunk::Options options = test_options()) {
        return chunk::build_partition(
            chunk::Kind::staged, Configuration{&source, args},
            Sections::direct(&source), groups(),
            constraints(lanes, requested, scratch), options);
    }
};

void expect_rejected(const chunk::Result& result, std::string_view reason) {
    CHECK(!result);
    CHECK(result.rejected == reason);
}

chunk::StageWord stage_word(const chunk::Partition& partition,
                            std::size_t staged_arg, int source_element) {
    return partition.staging.stage_words[
        staged_arg * static_cast<std::size_t>(partition.set_stride) +
        static_cast<std::size_t>(source_element)];
}

int stage_count(const chunk::Partition& partition, std::size_t ndats,
                std::size_t chunk, std::size_t staged_dat) {
    return partition.staging.stage_counts[chunk * ndats + staged_dat];
}

// Check the staging every kernel relies on: each region lists its chunk's
// distinct targets in ascending order, a slot each, and every active staged
// reference's word names its target's slot.
void check_slot_lists(const chunk::Partition& partition,
                      std::span<const op_arg> args,
                      const chunk::ArgGroups& groups) {
    const auto& staging = partition.staging;
    const auto ndats = static_cast<std::size_t>(groups.groups);
    CHECK(staging.stage_counts.size() == partition.num_chunks() * ndats);
    CHECK(staging.slot_offsets.size() == staging.stage_counts.size());

    std::size_t slots = 0;
    for (std::size_t region = 0; region < staging.stage_counts.size();
         ++region) {
        auto first = static_cast<std::size_t>(staging.slot_offsets[region]);
        auto count = static_cast<std::size_t>(staging.stage_counts[region]);
        CHECK(first + count <= staging.slot_targets.size());
        for (std::size_t s = 1; s < count; ++s)
            CHECK(staging.slot_targets[first + s - 1] <
                  staging.slot_targets[first + s]);
        slots += count;
    }
    CHECK(slots + partition.statistics.unstaged_singles ==
          partition.statistics.distinct_targets);

    for (std::size_t chunk = 0; chunk < partition.num_chunks(); ++chunk) {
        for (int source = partition.source_offsets[chunk];
             source < partition.source_offsets[chunk + 1]; ++source) {
            for (std::size_t i = 0; i < groups.args.size(); ++i) {
                const auto& group = groups.args[i];
                const op_arg& arg = args[static_cast<std::size_t>(group.arg)];
                if (arg.opt == 0)
                    continue;

                auto region = chunk * ndats + static_cast<std::size_t>(group.group);
                int slot = stage_word(partition, i, source);
                if (slot == chunk::unstaged_word)
                    continue;
                CHECK(slot < staging.stage_counts[region]);
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
void check_chunks_within_sections(const chunk::Partition& partition,
                                  const Sections& sections) {
    CHECK(partition.section_chunk_offsets.size() ==
          static_cast<std::size_t>(sections.size()) + 1);
    CHECK(partition.section_chunk_offsets.front() == 0);
    CHECK(partition.section_chunk_offsets.back() ==
          static_cast<int>(partition.num_chunks()));

    for (int section = 0; section < sections.size(); ++section) {
        int first = partition.section_chunk_offsets[static_cast<std::size_t>(section)];
        int last = partition.section_chunk_offsets[static_cast<std::size_t>(section) + 1];
        CHECK(first <= last);

        for (int c = first; c < last; ++c) {
            int begin = partition.source_offsets[static_cast<std::size_t>(c)];
            int end = partition.source_offsets[static_cast<std::size_t>(c) + 1];
            CHECK(begin < end);
            CHECK(begin >= sections[section].begin);
            CHECK(end <= sections[section].end);
        }

        if (first < last) {
            CHECK(partition.source_offsets[static_cast<std::size_t>(first)] ==
                  sections[section].begin);
            CHECK(partition.source_offsets[static_cast<std::size_t>(last)] ==
                  sections[section].end);
        } else {
            CHECK(sections[section].size() == 0);
        }
    }
}

void test_mixed_partition() {
    MixedFixture fixture;
    auto result = fixture.build();
    CHECK(result);

    const auto& partition = *result.partition;
    CHECK(partition.chunk_size == 4);
    CHECK(partition.set_stride == 32);
    CHECK(partition.num_chunks() == 2);
    CHECK((partition.source_offsets == std::vector<int>{0, 2, 6}));
    CHECK((partition.section_chunk_offsets == std::vector<int>{0, 1, 2}));
    // One batch per section, sized by its chunk.
    CHECK((partition.section_batch_offsets == std::vector<int>{0, 1, 2}));
    CHECK(partition.batch_chunk_offsets == partition.section_chunk_offsets);
    CHECK(partition.batches(0) == 1 && partition.batches(1) == 1);
    CHECK(partition.batch_range(1, 0) == std::make_pair(1, 2));
    CHECK(partition.scratch_bytes(1, 0) ==
          partition.staging.chunk_scratch_bytes[1]);
    CHECK((partition.staging.stage_counts == std::vector<int>{2, 2, 0, 3, 3, 0}));
    CHECK(stage_count(partition, 3, 0, 0) == 2);
    CHECK(stage_count(partition, 3, 0, 1) == 2);
    CHECK((partition.staging.chunk_scratch_bytes ==
           std::vector<std::size_t>{56, 84}));
    CHECK(partition.batch_scratch_bytes == partition.staging.chunk_scratch_bytes);

    CHECK(stage_word(partition, 0, 0) == 0);
    CHECK(stage_word(partition, 1, 0) == 0);
    CHECK(stage_word(partition, 0, 1) == 0);
    CHECK(stage_word(partition, 1, 1) == 1);
    CHECK(stage_word(partition, 2, 0) == 0);
    CHECK(stage_word(partition, 2, 1) == 1);
    CHECK(stage_word(partition, 1, 5) == 2);
    CHECK(stage_word(partition, 2, 4) == 2);
    CHECK((partition.staging.slot_offsets == std::vector<int>{0, 2, 4, 4, 7, 10}));
    CHECK((partition.staging.slot_targets ==
           std::vector<int>{0, 1, 0, 1, 1, 2, 3, 0, 1, 2}));

    // The inactive argument's words stay untouched.
    for (int source = 0; source < 6; ++source)
        CHECK(stage_word(partition, 3, source) == 0);
    check_slot_lists(partition, fixture.args, fixture.groups());

    CHECK(partition.statistics.raw_references == 18);
    CHECK(partition.statistics.distinct_targets == 10);
    check_chunks_within_sections(partition, fixture.sections());

    // The default options reject a partition below 1.5x compression.
    auto rejected = chunk::build_partition(
        chunk::Kind::staged, Configuration{&fixture.source, fixture.args},
        fixture.sections(), fixture.groups(), constraints(2, 4, 1024),
        chunk::Options{});
    CHECK(rejected);
    fixture.map_a0_values = {0, 1, 0, 1, 2, 3};
    fixture.map_a1_values = {2, 3, 1, 2, 3, 0};
    fixture.map_b_values = {0, 1, 0, 1, 2, 0};
    rejected = chunk::build_partition(
        chunk::Kind::staged, Configuration{&fixture.source, fixture.args},
        fixture.sections(), fixture.groups(), constraints(2, 4, 1024),
        chunk::Options{});
    expect_rejected(rejected, "low_compression");
    CHECK(rejected.partition != nullptr);
}

void test_optional_and_alignment() {
    MixedFixture fixture;
    CHECK(fixture.build());

    fixture.args[3].opt = 1;
    auto active = fixture.build();
    CHECK(active);
    CHECK((active.partition->staging.stage_counts ==
           std::vector<int>{2, 2, 1, 3, 3, 2}));
    CHECK((active.partition->batch_scratch_bytes ==
           std::vector<std::size_t>{60, 92}));
    CHECK(active.partition->statistics.raw_references == 24);
    CHECK(active.partition->statistics.distinct_targets == 13);

    // A float region before a double one pads the double region's start.
    fixture.args[1].opt = 0;
    fixture.args[2].opt = 0;
    std::array<chunk::ArgGroup, 2> group_args{{{3, 0}, {0, 1}}};
    auto aligned = chunk::build_partition(
        chunk::Kind::staged, Configuration{&fixture.source, fixture.args},
        fixture.sections(), chunk::ArgGroups{group_args, 2},
        constraints(2, 4, 1024), test_options());
    CHECK(aligned);
    CHECK((aligned.partition->batch_scratch_bytes ==
           std::vector<std::size_t>{24, 40}));
}

void test_chunk_sizes_and_clamping() {
    ChunkFixture fixture;
    for (int chunk_size : {128, 256, 512}) {
        auto result = fixture.build(chunk_size, 1024 * sizeof(double));
        CHECK(result);
        CHECK(result.partition->chunk_size == chunk_size);
        CHECK(result.partition->num_chunks() ==
              static_cast<std::size_t>(1024 / chunk_size));
        CHECK(result.partition->source_offsets.front() == 0);
        CHECK(result.partition->source_offsets.back() == 1024);
        CHECK(result.partition->batch_scratch_bytes[0] ==
              static_cast<std::size_t>(chunk_size) * sizeof(double));
    }

    // A request beyond the scratch clamps to what fits.
    auto clamped = fixture.build(512, 2048);
    CHECK(clamped);
    CHECK(clamped.partition->chunk_size == 256);
    CHECK(clamped.partition->num_chunks() == 4);

    auto saturated = fixture.build(INT_MAX, 1024 * sizeof(double));
    CHECK(saturated);
    CHECK(saturated.partition->chunk_size == 1024);

    expect_rejected(fixture.build(512, 1023), "insufficient_scratch");

    // A region's slots must fit a 16-bit stage word, however much scratch
    // there is.
    auto bounded = fixture.build(INT_MAX, std::size_t{1} << 20);
    CHECK(bounded);
    CHECK(bounded.partition->chunk_size == chunk::max_slots / 128 * 128);
    static_assert(chunk::max_slots - 1 <=
                  std::numeric_limits<chunk::StageWord>::max());
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
        std::array<chunk::ArgGroup, 4> group_args{{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
        auto result = chunk::build_partition(
            chunk::Kind::staged, Configuration{&fixture.source, fixture.args},
            fixture.sections(), chunk::ArgGroups{group_args, 4},
            constraints(2, 4, 1024), test_options());
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
    std::array<chunk::ArgGroup, 1> group_args{{{0, 0}}};

    OrderFixture() {
        initialize_set(source, 8);
        initialize_set(target, 4);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        initialize_map(map, &source, &target, map_values);
        args[0] = make_dat_arg(&dat, &map, 1, "double", sizeof(double));
    }

    chunk::Result build(bool unstaged_singles = false) {
        return chunk::build_partition(
            chunk::Kind::staged, Configuration{&source, args},
            Sections::direct(&source), chunk::ArgGroups{group_args, 1},
            constraints(4, 4, 1024), test_options(unstaged_singles));
    }
};

std::vector<int> words_of(const chunk::Partition& partition, int elements) {
    std::vector<int> words;
    for (int element = 0; element < elements; ++element)
        words.push_back(stage_word(partition, 0, element));
    return words;
}

void test_slots_in_target_order() {
    OrderFixture fixture;
    auto result = fixture.build();
    CHECK(result);
    const auto& partition = *result.partition;
    CHECK(partition.num_chunks() == 2);
    CHECK((partition.staging.stage_counts == std::vector<int>{3, 3}));
    CHECK((partition.staging.slot_targets == std::vector<int>{0, 1, 3, 0, 1, 2}));
    CHECK((words_of(partition, 8) == std::vector<int>{2, 1, 2, 0, 2, 1, 2, 0}));
    CHECK(partition.statistics.raw_references == 8);
    CHECK(partition.statistics.distinct_targets == 6);
    check_slot_lists(partition, fixture.args, chunk::ArgGroups{fixture.group_args, 1});
}

// Targets 1 and 0 of the first chunk, and 1 and 0 of the second, are
// referenced once there, so they stay in global memory.
void test_unstaged_singles() {
    OrderFixture fixture;
    auto result = fixture.build(true);
    CHECK(result);
    const auto& partition = *result.partition;
    constexpr int u = chunk::unstaged_word;
    CHECK((partition.staging.stage_counts == std::vector<int>{1, 1}));
    CHECK((partition.staging.slot_targets == std::vector<int>{3, 2}));
    CHECK((words_of(partition, 8) == std::vector<int>{0, u, 0, u, 0, u, 0, u}));
    CHECK(partition.statistics.raw_references == 8);
    CHECK(partition.statistics.distinct_targets == 6);
    CHECK(partition.statistics.unstaged_singles == 4);
    check_slot_lists(partition, fixture.args, chunk::ArgGroups{fixture.group_args, 1});
}

// Check every colouring invariant the kernel relies on: batches partition
// each section's chunks, elements of one chunk and thread colour reach
// distinct targets, and chunks of one batch reach distinct targets.
void check_colouring(const chunk::Partition& partition,
                     std::span<const op_arg> args,
                     const chunk::ArgGroups& groups, const Sections& sections) {
    CHECK(partition.colouring.has_value());
    check_slot_lists(partition, args, groups);
    const auto& colouring = *partition.colouring;
    check_chunks_within_sections(partition, sections);
    CHECK(partition.section_batch_offsets.size() ==
          static_cast<std::size_t>(sections.size()) + 1);
    CHECK(partition.section_batch_offsets.front() == 0);
    CHECK(partition.section_batch_offsets.back() + 1 ==
          static_cast<int>(partition.batch_chunk_offsets.size()));
    CHECK(partition.batch_chunk_offsets.front() == 0);
    CHECK(partition.batch_chunk_offsets.back() ==
          static_cast<int>(partition.num_chunks()));
    CHECK(colouring.chunk_order.size() == partition.num_chunks());
    CHECK(colouring.chunk_thread_colours.size() == partition.num_chunks());
    CHECK(partition.statistics.batches + 1 == partition.batch_chunk_offsets.size());
    CHECK(partition.batch_scratch_bytes.size() == partition.statistics.batches);

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

    std::vector<int> run(partition.num_chunks(), 0);
    for (int section = 0; section < sections.size(); ++section) {
        for (int batch = 0; batch < partition.batches(section); ++batch) {
            auto [begin, end] = partition.batch_range(section, batch);
            CHECK(begin < end);
            CHECK(partition.scratch_bytes(section, batch) ==
                  partition.max_chunk_scratch_bytes(begin, end));

            std::vector<std::vector<std::pair<int, int>>> chunk_targets;
            for (int i = begin; i < end; ++i) {
                int c = colouring.chunk_order[static_cast<std::size_t>(i)];
                CHECK(c >= partition.section_chunk_offsets[static_cast<std::size_t>(section)] &&
                      c < partition.section_chunk_offsets[static_cast<std::size_t>(section) + 1]);
                ++run[static_cast<std::size_t>(c)];

                int first = partition.source_offsets[static_cast<std::size_t>(c)];
                int last = partition.source_offsets[static_cast<std::size_t>(c) + 1];
                int ncolours = colouring.chunk_thread_colours[static_cast<std::size_t>(c)];
                CHECK(ncolours >= 1 && ncolours <= 255);
                CHECK(ncolours <= partition.statistics.max_thread_colours);

                std::vector<std::pair<int, int>> reached;
                for (int a = first; a < last; ++a) {
                    int colour = colouring.thread_colours[static_cast<std::size_t>(a)];
                    CHECK(colour < ncolours);
                    auto a_targets = targets(a);
                    reached.insert(reached.end(), a_targets.begin(), a_targets.end());
                    for (int b = first; b < a; ++b)
                        if (colouring.thread_colours[static_cast<std::size_t>(b)] == colour)
                            CHECK(disjoint(a_targets, targets(b)));
                }

                for (const auto& other : chunk_targets)
                    CHECK(disjoint(reached, other));
                chunk_targets.push_back(std::move(reached));
            }
        }
    }

    CHECK(run == std::vector<int>(partition.num_chunks(), 1));
}

void test_colour_mixed_partition() {
    MixedFixture fixture;
    fixture.args[0].acc = OP_RW;
    fixture.args[1].acc = OP_READ;

    auto result = fixture.build(chunk::Kind::coloured);
    CHECK(result);
    const auto& partition = *result.partition;
    const auto& colouring = *partition.colouring;
    CHECK(partition.chunk_size == 4);
    CHECK((partition.source_offsets == std::vector<int>{0, 2, 6}));
    CHECK((colouring.chunk_order == std::vector<int>{0, 1}));
    CHECK((partition.batch_chunk_offsets == std::vector<int>{0, 1, 2}));
    CHECK((partition.section_batch_offsets == std::vector<int>{0, 1, 2}));
    CHECK((colouring.chunk_thread_colours == std::vector<int>{2, 3}));
    CHECK((std::vector<int>(colouring.thread_colours.begin(),
                            colouring.thread_colours.begin() + 6) ==
           std::vector<int>{0, 1, 0, 1, 0, 2}));
    CHECK(partition.statistics.batches == 2);
    CHECK(partition.statistics.max_thread_colours == 3);
    check_colouring(partition, fixture.args, fixture.groups(), fixture.sections());

    fixture.args[3].opt = 1;
    auto active = fixture.build(chunk::Kind::coloured);
    CHECK(active);
    check_colouring(*active.partition, fixture.args, fixture.groups(),
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
    std::array<chunk::ArgGroup, 2> group_args{{{0, 0}, {1, 0}}};

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

    chunk::ArgGroups groups() const { return {group_args, 1}; }

    chunk::Result build(const Sections& sections, int lanes,
                        std::size_t scratch = 1 << 16) {
        return chunk::build_partition(
            chunk::Kind::coloured, Configuration{&edges, args}, sections,
            groups(), constraints(lanes, lanes, scratch), test_options());
    }
};

void test_colour_chain_batches() {
    ChainFixture fixture(16);
    auto sections = Sections::direct(&fixture.edges);

    auto result = fixture.build(sections, 4);
    CHECK(result);
    const auto& partition = *result.partition;
    const auto& colouring = *partition.colouring;
    CHECK(partition.num_chunks() == 4);
    CHECK((colouring.chunk_order == std::vector<int>{0, 2, 1, 3}));
    CHECK((partition.batch_chunk_offsets == std::vector<int>{0, 2, 4}));
    CHECK((partition.section_batch_offsets == std::vector<int>{0, 2}));
    CHECK((colouring.chunk_thread_colours == std::vector<int>{2, 2, 2, 2}));
    for (int edge = 0; edge < 16; ++edge)
        CHECK(colouring.thread_colours[static_cast<std::size_t>(edge)] == edge % 2);
    check_colouring(partition, fixture.args, fixture.groups(), sections);

    // Colours never cross a section, and an empty section has no batches:
    // 8 core elements, no further owned ones, 8 in the exec halo.
    ChainFixture halo(8, 8, 8);
    auto split = Sections::indirect(&halo.edges, true);
    CHECK(split.size() == 3 && split[1].size() == 0);
    auto sectioned = halo.build(split, 4);
    CHECK(sectioned);
    CHECK((sectioned.partition->section_batch_offsets ==
           std::vector<int>{0, 2, 2, 4}));
    check_colouring(*sectioned.partition, halo.args, halo.groups(), split);
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
        std::array<chunk::ArgGroup, 1> group_args{{{0, 0}}};

        auto result = chunk::build_partition(
            chunk::Kind::coloured, Configuration{&source, args},
            Sections::direct(&source), chunk::ArgGroups{group_args, 1},
            constraints(size, size, 1 << 20), test_options());
        if (size > 255) {
            expect_rejected(result, "too_many_colours");
            continue;
        }

        CHECK(result);
        CHECK(result.partition->statistics.max_thread_colours == size);
        for (int i = 0; i < size; ++i)
            CHECK(result.partition->colouring->thread_colours[
                      static_cast<std::size_t>(i)] == i);
        check_colouring(*result.partition, args, chunk::ArgGroups{group_args, 1},
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
    std::array<chunk::ArgGroup, 3> group_args{{{0, 0}, {1, 0}, {2, 1}}};
    chunk::ArgGroups groups{group_args, 2};
    auto sections = Sections::indirect(&source, true);
    Configuration configuration{&source, args};

    auto result = chunk::build_partition(chunk::Kind::coloured, configuration,
                                         sections, groups,
                                         constraints(32, 64, 1 << 20),
                                         test_options());
    CHECK(result);
    CHECK(result.partition->statistics.batches >
          static_cast<std::size_t>(sections.size()));
    check_colouring(*result.partition, args, groups, sections);

    // A wavefront partition runs each section as one batch, each position
    // after the latest earlier position of its section sharing one of its
    // targets.
    auto wave = chunk::build_partition(
        chunk::Kind::coloured, configuration, sections, groups,
        constraints(32, 64, 1 << 20), test_options(false, true, 4));
    CHECK(wave);
    const auto& partition = *wave.partition;
    const auto& colouring = *partition.colouring;
    check_slot_lists(partition, args, groups);
    CHECK(colouring.wavefront);
    CHECK(partition.statistics.batches == static_cast<std::size_t>(sections.size()));
    CHECK(colouring.wave_pred_offsets.size() == partition.num_chunks() + 1);

    const std::size_t ndats = 2;
    for (int section = 0; section < sections.size(); ++section) {
        int begin = partition.batch_chunk_offsets[static_cast<std::size_t>(section)];
        int end = partition.batch_chunk_offsets[static_cast<std::size_t>(section) + 1];
        CHECK(begin == partition.section_chunk_offsets[static_cast<std::size_t>(section)]);
        CHECK(end == partition.section_chunk_offsets[static_cast<std::size_t>(section) + 1]);

        // The latest position so far reaching each (dat, target).
        std::vector<std::vector<int>> latest(ndats, std::vector<int>(targets, -1));
        for (int position = begin; position < end; ++position) {
            auto c = static_cast<std::size_t>(
                colouring.chunk_order[static_cast<std::size_t>(position)]);
            CHECK(static_cast<int>(c) >= begin && static_cast<int>(c) < end);
            std::vector<int> preds(
                colouring.wave_preds.begin() +
                    colouring.wave_pred_offsets[static_cast<std::size_t>(position)],
                colouring.wave_preds.begin() +
                    colouring.wave_pred_offsets[static_cast<std::size_t>(position) + 1]);
            for (int pred : preds)
                CHECK(pred >= begin && pred < position);

            for (std::size_t dat = 0; dat < ndats; ++dat) {
                const auto& staging = partition.staging;
                auto region = c * ndats + dat;
                for (int s = 0; s < staging.stage_counts[region]; ++s) {
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
        expect_rejected(fixture.build(chunk::Kind::coloured),
                        "incompatible_argument");
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_rejected(fixture.build(chunk::Kind::coloured), "no_active_argument");
    }
    {
        // Two groups resolving to one dat would be coloured independently.
        ChainFixture fixture(8);
        std::array<chunk::ArgGroup, 2> group_args{{{0, 0}, {1, 1}}};
        auto result = chunk::build_partition(
            chunk::Kind::coloured, Configuration{&fixture.edges, fixture.args},
            Sections::direct(&fixture.edges), chunk::ArgGroups{group_args, 2},
            constraints(4, 4, 1 << 16), test_options());
        expect_rejected(result, "incompatible_argument");
    }
}

} // namespace

int main() {
    try {
        test_mixed_partition();
        test_optional_and_alignment();
        test_chunk_sizes_and_clamping();
        test_runtime_rejections();
        test_slots_in_target_order();
        test_unstaged_singles();
        test_colour_mixed_partition();
        test_colour_chain_batches();
        test_colour_rounds_and_limit();
        test_colour_random();
        test_colour_rejections();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "chunk partition test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("chunk partition tests passed\n");
    return EXIT_SUCCESS;
}

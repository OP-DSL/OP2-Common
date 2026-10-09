#include <op_hier_plan.h>

#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <span>
#include <vector>

namespace f2c = op::f2c;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

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
    std::array<f2c::HierArgDescriptor, 4> arg_desc{{
        {0, 0},
        {1, 0},
        {2, 1},
        {3, 2},
    }};
    std::array<f2c::HierScalarType, 3> dat_desc{{
        f2c::HierScalarType::f64,
        f2c::HierScalarType::i32,
        f2c::HierScalarType::f32,
    }};
    std::array<f2c::ExecutionSection, 2> sections{{{0, 2}, {2, 6}}};

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

    f2c::HierArgGroups descriptor() const {
        return {arg_desc, dat_desc, -1};
    }

    f2c::HierPlanOptions options() const {
        return {2, 4, 1024};
    }

    f2c::HierPlanBuildResult build() {
        return f2c::build_hier_atomics_plan(
            &source, args, sections, descriptor(), options());
    }
};

struct ChunkFixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    std::vector<int> map_values;
    op_map_core map{};
    std::array<op_arg, 1> args{};
    std::array<f2c::HierArgDescriptor, 1> arg_desc{{{0, 0}}};
    std::array<f2c::HierScalarType, 1> dat_desc{{
        f2c::HierScalarType::f64,
    }};
    std::array<f2c::ExecutionSection, 1> sections{{{0, 1024}}};

    ChunkFixture() : map_values(1024) {
        initialize_set(source, 1024);
        initialize_set(target, 1024);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        for (int i = 0; i < 1024; ++i)
            map_values[static_cast<std::size_t>(i)] = i;
        initialize_map(map, &source, &target, map_values);
        args[0] = make_dat_arg(&dat, &map, 1, "double", sizeof(double));
    }

    f2c::HierArgGroups descriptor() const {
        return {arg_desc, dat_desc, -1};
    }

    f2c::HierPlanBuildResult build(int requested,
                                   std::size_t shared_limit,
                                   int block_size = 128) {
        f2c::HierPlanOptions options{
            block_size, requested, shared_limit};
        return f2c::build_hier_atomics_plan(
            &source, args, sections, descriptor(), options);
    }
};

void expect_reason(const f2c::HierPlanBuildResult& result,
                   f2c::FallbackReason reason) {
    CHECK(!result);
    CHECK(result.reason == reason);
    CHECK(f2c::fallback_reason_name(reason) != "unknown");
}

f2c::HierSmemStageWord stage_word(const f2c::HierPlan& plan,
                                  std::size_t staged_arg,
                                  int source_element) {
    return plan.staging->stage_words[
        staged_arg * static_cast<std::size_t>(plan.set_stride) +
        static_cast<std::size_t>(source_element)];
}

int stage_count(const f2c::HierPlan& plan, std::size_t num_stage_dats,
                std::size_t chunk, std::size_t staged_dat) {
    return plan.staging->stage_counts[chunk * num_stage_dats + staged_dat];
}

// A chunk that spanned two schedule sections would be launched by whichever
// section came first, so its later half would run before the halo wait or
// after the globals had already been processed.  The planner chunks each
// section separately to prevent that; this checks the resulting plan.
void check_chunks_within_sections(
    const f2c::HierPlan& plan,
    std::span<const f2c::ExecutionSection> sections) {
    CHECK(plan.section_chunk_offsets.size() == sections.size() + 1);
    CHECK(plan.section_chunk_offsets.front() == 0);
    CHECK(plan.section_chunk_offsets.back() ==
          static_cast<int>(plan.num_chunks()));

    for (std::size_t section = 0; section < sections.size(); ++section) {
        int first = plan.section_chunk_offsets[section];
        int last = plan.section_chunk_offsets[section + 1];
        CHECK(first <= last);

        for (int chunk = first; chunk < last; ++chunk) {
            int begin = plan.source_offsets[static_cast<std::size_t>(chunk)];
            int end = plan.source_offsets[static_cast<std::size_t>(chunk) + 1];
            CHECK(begin < end);
            CHECK(begin >= sections[section].start);
            CHECK(end <= sections[section].end);
        }

        // The section's chunks must also cover it exactly.
        if (first < last) {
            CHECK(plan.source_offsets[static_cast<std::size_t>(first)] ==
                  sections[section].start);
            CHECK(plan.source_offsets[static_cast<std::size_t>(last)] ==
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

    const auto& plan = *result.plan;
    CHECK(plan.selected_chunk_size == 4);
    CHECK(plan.set_stride == 32);
    CHECK(plan.num_chunks() == 2);
    CHECK((plan.source_offsets == std::vector<int>{0, 2, 6}));
    CHECK((plan.section_chunk_offsets == std::vector<int>{0, 1, 2}));
    CHECK((plan.staging->stage_counts == std::vector<int>{2, 2, 0, 3, 3, 0}));
    CHECK(stage_count(plan, fixture.dat_desc.size(), 0, 0) == 2);
    CHECK(stage_count(plan, fixture.dat_desc.size(), 0, 1) == 2);
    CHECK((plan.staging->section_shared_bytes ==
           std::vector<std::size_t>{56, 84}));

    CHECK(f2c::hier_smem_stage_owner(stage_word(plan, 0, 0)));
    CHECK(f2c::hier_smem_stage_slot(stage_word(plan, 0, 0)) == 0);
    CHECK(!f2c::hier_smem_stage_owner(stage_word(plan, 1, 0)));
    CHECK(f2c::hier_smem_stage_slot(stage_word(plan, 1, 0)) == 0);
    CHECK(!f2c::hier_smem_stage_owner(stage_word(plan, 0, 1)));
    CHECK(f2c::hier_smem_stage_owner(stage_word(plan, 1, 1)));
    CHECK(f2c::hier_smem_stage_slot(stage_word(plan, 1, 1)) == 1);
    CHECK(f2c::hier_smem_stage_owner(stage_word(plan, 2, 0)));
    CHECK(f2c::hier_smem_stage_slot(stage_word(plan, 2, 0)) == 0);
    CHECK(f2c::hier_smem_stage_owner(stage_word(plan, 2, 1)));
    CHECK(f2c::hier_smem_stage_slot(stage_word(plan, 2, 1)) == 1);

    for (int source = 0; source < 6; ++source) {
        CHECK(stage_word(plan, 3, source) == 0);
        // Only an owner ever flushes, so only an owner may be exclusive.
        for (std::size_t staged_arg = 0;
             staged_arg < fixture.arg_desc.size();
             ++staged_arg) {
            auto word = stage_word(plan, staged_arg, source);
            CHECK(!f2c::hier_smem_stage_exclusive(word) ||
                  f2c::hier_smem_stage_owner(word));
        }
    }

    CHECK(plan.statistics.raw_references == 18);
    CHECK(plan.statistics.distinct_targets == 10);
    check_chunks_within_sections(plan, fixture.sections);
}

void test_optional_and_alignment() {
    MixedFixture fixture;
    auto inactive = fixture.build();
    CHECK(inactive);

    fixture.args[3].opt = 1;
    auto active = fixture.build();
    CHECK(active);
    CHECK((active.plan->staging->stage_counts ==
           std::vector<int>{2, 2, 1, 3, 3, 2}));
    CHECK((active.plan->staging->section_shared_bytes ==
           std::vector<std::size_t>{60, 92}));
    CHECK(active.plan->statistics.raw_references == 24);
    CHECK(active.plan->statistics.distinct_targets == 13);

    fixture.args[1].opt = 0;
    fixture.args[2].opt = 0;
    std::array<f2c::HierArgDescriptor, 2> arg_desc{{{3, 0}, {0, 1}}};
    std::array<f2c::HierScalarType, 2> dat_desc{{
        f2c::HierScalarType::f32,
        f2c::HierScalarType::f64,
    }};
    auto descriptor = f2c::HierArgGroups{
        arg_desc, dat_desc, -1};
    auto options = fixture.options();
    auto aligned = f2c::build_hier_atomics_plan(
        &fixture.source, fixture.args, fixture.sections, descriptor, options);
    CHECK(aligned);
    CHECK((aligned.plan->staging->section_shared_bytes ==
           std::vector<std::size_t>{24, 40}));
}

void test_chunk_sizes_and_clamping() {
    ChunkFixture fixture;
    for (int chunk_size : {128, 256, 512}) {
        auto result = fixture.build(chunk_size, 1024 * sizeof(double));
        CHECK(result);
        CHECK(result.plan->selected_chunk_size == chunk_size);
        CHECK(result.plan->num_chunks() ==
              static_cast<std::size_t>(1024 / chunk_size));
        CHECK(result.plan->source_offsets.front() == 0);
        CHECK(result.plan->source_offsets.back() == 1024);
        CHECK(result.plan->staging->section_shared_bytes[0] ==
              static_cast<std::size_t>(chunk_size) * sizeof(double));
    }

    // A request beyond the shared-memory capacity clamps to what fits.
    auto clamped = fixture.build(512, 2048);
    CHECK(clamped);
    CHECK(clamped.plan->selected_chunk_size == 256);
    CHECK(clamped.plan->num_chunks() == 4);

    auto saturated = fixture.build(INT_MAX, 1024 * sizeof(double));
    CHECK(saturated);
    CHECK(saturated.plan->selected_chunk_size == 1024);

    expect_reason(fixture.build(512, 1023),
                  f2c::FallbackReason::insufficient_shared_memory);
}

void test_runtime_fallbacks() {
    {
        MixedFixture fixture;
        fixture.args.push_back(make_dat_arg(&fixture.dat_a, &fixture.map_a0,
                                             2, "double", sizeof(double), 1,
                                             OP_READ));
        expect_reason(fixture.build(),
                      f2c::FallbackReason::incompatible_argument);
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_reason(fixture.build(),
                      f2c::FallbackReason::no_active_argument);
    }
    {
        MixedFixture fixture;
        std::array<f2c::HierArgDescriptor, 4> arg_desc{{
            {0, 0},
            {1, 1},
            {2, 2},
            {3, 3},
        }};
        std::array<f2c::HierScalarType, 4> dat_desc{{
            f2c::HierScalarType::f64,
            f2c::HierScalarType::f64,
            f2c::HierScalarType::i32,
            f2c::HierScalarType::f32,
        }};
        auto descriptor = f2c::HierArgGroups{
            arg_desc, dat_desc, -1};
        auto result = f2c::build_hier_atomics_plan(
            &fixture.source, fixture.args, fixture.sections, descriptor,
            fixture.options());
        expect_reason(result,
                      f2c::FallbackReason::incompatible_argument);
    }
}

// One dat reached through one map, with a target layout chosen so that every
// owner's expected exclusivity can be checked by hand:
//
//   source  0 1 2 3 | 4 5 6 7      chunk 0 = [0,4), chunk 1 = [4,8)
//   target  0 0 1 1 | 1 2 2 3
//
// target 0 appears only in chunk 0, targets 2 and 3 only in chunk 1, and
// target 1 spans both.  So the owners at sources 0, 5 and 7 are exclusive and
// the owners at sources 2 and 4 are not.
struct ExclusiveFixture {
    op_set_core source{};
    op_set_core target{};
    op_dat_core dat{};
    std::vector<int> map_values{0, 0, 1, 1, 1, 2, 2, 3};
    op_map_core map{};
    std::vector<op_arg> args;
    std::array<f2c::HierArgDescriptor, 1> arg_desc{{{0, 0}}};
    std::array<f2c::HierScalarType, 1> dat_desc{{
        f2c::HierScalarType::f64,
    }};

    ExclusiveFixture() {
        initialize_set(source, 8);
        initialize_set(target, 4);
        initialize_dat(dat, &target, 1, sizeof(double), "double");
        initialize_map(map, &source, &target, map_values);
        args.push_back(make_dat_arg(&dat, &map, 1, "double", sizeof(double)));
    }

    f2c::HierPlanBuildResult build(
        std::span<const f2c::ExecutionSection> sections,
        bool exclusive = true) {
        f2c::HierPlanOptions options{4, 4, 1024, exclusive};
        return f2c::build_hier_atomics_plan(
            &source, args, sections, {arg_desc, dat_desc, -1}, options);
    }
};

// Collect the source elements whose word carries the given flag.
std::vector<int> sources_with(const f2c::HierPlan& plan, int stride,
                              bool (*flag)(f2c::HierSmemStageWord)) {
    std::vector<int> out;
    for (int source = 0; source < stride; ++source) {
        auto word = plan.staging->stage_words[static_cast<std::size_t>(source)];
        if (flag(word))
            out.push_back(source);
    }

    return out;
}

void test_exclusive_within_one_section() {
    ExclusiveFixture fixture;
    std::array<f2c::ExecutionSection, 1> sections{{{0, 8}}};

    auto result = fixture.build(sections);
    CHECK(result);
    CHECK(result.plan->num_chunks() == 2);
    check_chunks_within_sections(*result.plan, sections);

    auto owners = sources_with(*result.plan, result.plan->set_stride,
                               f2c::hier_smem_stage_owner);
    auto exclusive = sources_with(*result.plan, result.plan->set_stride,
                                  f2c::hier_smem_stage_exclusive);

    CHECK((owners == std::vector<int>{0, 2, 4, 5, 7}));
    CHECK((exclusive == std::vector<int>{0, 5, 7}));
    CHECK(result.plan->statistics.distinct_targets == 5);
    CHECK(result.plan->statistics.exclusive_owners == 3);

    // Every exclusive word is also an owner; the flush relies on that.
    for (int source : exclusive)
        CHECK(f2c::hier_smem_stage_owner(
            result.plan->staging->stage_words[
                static_cast<std::size_t>(source)]));
}

void test_exclusive_is_per_section() {
    ExclusiveFixture fixture;
    // The same chunk boundary as above, but now the two chunks are separate
    // schedule sections.  Their launches are ordered on the stream, so target
    // 1 appearing in both no longer forces an atomic in either.
    std::array<f2c::ExecutionSection, 2> sections{{{0, 4}, {4, 8}}};

    auto result = fixture.build(sections);
    CHECK(result);
    CHECK(result.plan->num_chunks() == 2);
    check_chunks_within_sections(*result.plan, sections);

    auto exclusive = sources_with(*result.plan, result.plan->set_stride,
                                  f2c::hier_smem_stage_exclusive);
    CHECK((exclusive == std::vector<int>{0, 2, 4, 5, 7}));
    CHECK(result.plan->statistics.exclusive_owners == 5);
}

void test_exclusive_can_be_disabled() {
    ExclusiveFixture fixture;
    std::array<f2c::ExecutionSection, 1> sections{{{0, 8}}};

    auto enabled = fixture.build(sections, true);
    auto disabled = fixture.build(sections, false);
    CHECK(enabled);
    CHECK(disabled);

    CHECK(disabled.plan->statistics.exclusive_owners == 0);
    CHECK(sources_with(*disabled.plan, disabled.plan->set_stride,
                       f2c::hier_smem_stage_exclusive).empty());

    // Disabling exclusivity must change nothing but that one bit.
    CHECK(enabled.plan->source_offsets == disabled.plan->source_offsets);
    const auto& on = *enabled.plan->staging;
    const auto& off = *disabled.plan->staging;
    CHECK(on.stage_counts == off.stage_counts);
    CHECK(enabled.plan->statistics.distinct_targets ==
          disabled.plan->statistics.distinct_targets);
    for (std::size_t i = 0; i < off.stage_words.size(); ++i)
        CHECK((on.stage_words[i] & ~f2c::hier_smem_exclusive_bit) ==
              off.stage_words[i]);
}

// Check every colouring invariant the wrapper relies on: launches partition
// each section's chunks, elements of one chunk and thread colour reach
// distinct targets, and chunks of one launch reach distinct targets.
void check_colouring(const f2c::HierPlan& plan,
                     std::span<const op_arg> args,
                     const f2c::HierArgGroups& descriptor,
                     std::span<const f2c::ExecutionSection> sections) {
    CHECK(plan.colouring.has_value() && plan.staging.has_value());
    // A launch's chunks share no target, so every owner flushes by storing.
    CHECK(plan.staging->has_exclusive == 1);
    CHECK(plan.statistics.exclusive_owners == plan.statistics.distinct_targets);
    for (auto word : plan.staging->stage_words)
        CHECK(f2c::hier_smem_stage_owner(word) ==
              f2c::hier_smem_stage_exclusive(word));
    const auto& colouring = *plan.colouring;
    check_chunks_within_sections(plan, sections);
    CHECK(colouring.section_launch_offsets.size() == sections.size() + 1);
    CHECK(colouring.section_launch_offsets.front() == 0);
    CHECK(colouring.section_launch_offsets.back() + 1 ==
          static_cast<int>(colouring.launch_chunk_offsets.size()));
    CHECK(colouring.launch_chunk_offsets.front() == 0);
    CHECK(colouring.launch_chunk_offsets.back() ==
          static_cast<int>(plan.num_chunks()));
    CHECK(colouring.chunk_order.size() == plan.num_chunks());
    CHECK(colouring.chunk_thread_colours.size() == plan.num_chunks());
    CHECK(plan.statistics.launches + 1 ==
          colouring.launch_chunk_offsets.size());

    auto targets = [&](int source) {
        std::vector<std::pair<int, int>> out;
        for (const auto& arg_desc : descriptor.args) {
            const op_arg& arg = args[static_cast<std::size_t>(arg_desc.arg_index)];
            if (arg.opt != 0)
                out.push_back({arg_desc.dat_index,
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

    std::vector<int> launched(plan.num_chunks(), 0);
    for (std::size_t section = 0; section < sections.size(); ++section) {
        for (int launch = colouring.section_launch_offsets[section];
             launch < colouring.section_launch_offsets[section + 1]; ++launch) {
            int begin = colouring.launch_chunk_offsets[static_cast<std::size_t>(launch)];
            int end = colouring.launch_chunk_offsets[static_cast<std::size_t>(launch) + 1];
            CHECK(begin < end);

            std::vector<std::vector<std::pair<int, int>>> chunk_targets;
            for (int i = begin; i < end; ++i) {
                int chunk = colouring.chunk_order[static_cast<std::size_t>(i)];
                CHECK(chunk >= plan.section_chunk_offsets[section] &&
                      chunk < plan.section_chunk_offsets[section + 1]);
                ++launched[static_cast<std::size_t>(chunk)];

                int first = plan.source_offsets[static_cast<std::size_t>(chunk)];
                int last = plan.source_offsets[static_cast<std::size_t>(chunk) + 1];
                int ncolours =
                    colouring.chunk_thread_colours[static_cast<std::size_t>(chunk)];
                CHECK(ncolours >= 1 && ncolours <= 255);
                CHECK(ncolours <= plan.statistics.max_thread_colours);

                std::vector<std::pair<int, int>> reached;
                for (int a = first; a < last; ++a) {
                    int colour = colouring.thread_colours[static_cast<std::size_t>(a)];
                    CHECK(colour < ncolours);
                    auto a_targets = targets(a);
                    reached.insert(reached.end(), a_targets.begin(),
                                   a_targets.end());
                    for (int b = first; b < a; ++b)
                        if (colouring.thread_colours[static_cast<std::size_t>(b)] ==
                            colour)
                            CHECK(disjoint(a_targets, targets(b)));
                }

                for (const auto& other : chunk_targets)
                    CHECK(disjoint(reached, other));
                chunk_targets.push_back(std::move(reached));
            }
        }
    }

    CHECK(launched == std::vector<int>(plan.num_chunks(), 1));
}

f2c::HierPlanBuildResult build_colour(MixedFixture& fixture) {
    return f2c::build_hier_colouring_plan(&fixture.source, fixture.args,
                                          fixture.sections, fixture.descriptor(),
                                       fixture.options());
}

void test_colour_mixed_plan() {
    MixedFixture fixture;
    fixture.args[0].acc = OP_RW;
    fixture.args[1].acc = OP_READ;

    auto result = build_colour(fixture);
    CHECK(result);
    const auto& plan = *result.plan;
    const auto& colouring = *plan.colouring;
    CHECK(plan.selected_chunk_size == 4);
    CHECK((plan.source_offsets == std::vector<int>{0, 2, 6}));
    CHECK((colouring.chunk_order == std::vector<int>{0, 1}));
    CHECK((colouring.launch_chunk_offsets == std::vector<int>{0, 1, 2}));
    CHECK((colouring.section_launch_offsets == std::vector<int>{0, 1, 2}));
    CHECK((colouring.chunk_thread_colours == std::vector<int>{2, 3}));
    CHECK((std::vector<int>(colouring.thread_colours.begin(),
                            colouring.thread_colours.begin() + 6) ==
           std::vector<int>{0, 1, 0, 1, 0, 2}));
    CHECK(plan.statistics.launches == 2);
    CHECK(plan.statistics.max_thread_colours == 3);
    check_colouring(plan, fixture.args, fixture.descriptor(),
                    fixture.sections);

    fixture.args[3].opt = 1;
    auto active = build_colour(fixture);
    CHECK(active);
    check_colouring(*active.plan, fixture.args, fixture.descriptor(),
                    fixture.sections);
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
    std::array<f2c::HierArgDescriptor, 2> arg_desc{{{0, 0}, {1, 0}}};
    std::array<f2c::HierScalarType, 1> dat_desc{{
        f2c::HierScalarType::i32,
    }};

    explicit ChainFixture(int size) {
        initialize_set(edges, size);
        initialize_set(nodes, size + 1);
        initialize_dat(dat, &nodes, 1, sizeof(int), "integer(4)");
        for (int i = 0; i < size; ++i) {
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

    f2c::HierArgGroups descriptor() const {
        return {arg_desc, dat_desc, -1};
    }

    f2c::HierPlanBuildResult build(
        std::span<const f2c::ExecutionSection> sections, int block_size) {
        f2c::HierPlanOptions options{block_size, block_size, 1 << 16};
        return f2c::build_hier_colouring_plan(&edges, args, sections,
                                              descriptor(), options);
    }
};

void test_colour_chain_blocks() {
    ChainFixture fixture(16);
    std::array<f2c::ExecutionSection, 1> sections{{{0, 16}}};

    auto result = fixture.build(sections, 4);
    CHECK(result);
    const auto& plan = *result.plan;
    const auto& colouring = *plan.colouring;
    CHECK(plan.num_chunks() == 4);
    CHECK((colouring.chunk_order == std::vector<int>{0, 2, 1, 3}));
    CHECK((colouring.launch_chunk_offsets == std::vector<int>{0, 2, 4}));
    CHECK((colouring.section_launch_offsets == std::vector<int>{0, 2}));
    CHECK((colouring.chunk_thread_colours == std::vector<int>{2, 2, 2, 2}));
    for (int edge = 0; edge < 16; ++edge)
        CHECK(colouring.thread_colours[static_cast<std::size_t>(edge)] == edge % 2);
    check_colouring(plan, fixture.args, fixture.descriptor(), sections);

    // Colours never cross a section, and an empty section has no launches.
    std::array<f2c::ExecutionSection, 3> split{{{0, 8}, {8, 8}, {8, 16}}};
    auto sectioned = fixture.build(split, 4);
    CHECK(sectioned);
    CHECK((sectioned.plan->colouring->section_launch_offsets ==
           std::vector<int>{0, 2, 2, 4}));
    check_colouring(*sectioned.plan, fixture.args, fixture.descriptor(),
                    split);
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
        std::array<f2c::HierArgDescriptor, 1> arg_desc{{{0, 0}}};
        std::array<f2c::HierScalarType, 1> dat_desc{{
            f2c::HierScalarType::f64,
        }};
        std::array<f2c::ExecutionSection, 1> sections{{{0, size}}};
        f2c::HierArgGroups descriptor{arg_desc, dat_desc, -1};

        auto result = f2c::build_hier_colouring_plan(
            &source, args, sections, descriptor, {size, size, 1 << 20});
        if (size > 255) {
            expect_reason(result,
                          f2c::FallbackReason::too_many_colours);
            continue;
        }

        CHECK(result);
        CHECK(result.plan->statistics.max_thread_colours == size);
        for (int i = 0; i < size; ++i)
            CHECK(result.plan->colouring->thread_colours[
                      static_cast<std::size_t>(i)] == i);
        check_colouring(*result.plan, args, descriptor, sections);
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
        make_dat_arg(&dat_a, &maps[1], 2, "real(8)", sizeof(double), 1,
                     OP_READ),
        make_dat_arg(&dat_b, &maps[2], 1, "integer(4)", sizeof(int), 1,
                     OP_INC),
    };
    std::array<f2c::HierArgDescriptor, 3> arg_desc{{{0, 0}, {1, 0}, {2, 1}}};
    std::array<f2c::HierScalarType, 2> dat_desc{{
        f2c::HierScalarType::f64,
        f2c::HierScalarType::i32,
    }};
    std::array<f2c::ExecutionSection, 3> sections{{
        {0, 300}, {300, 900}, {900, 1000}}};
    f2c::HierArgGroups descriptor{arg_desc, dat_desc, -1};

    auto result = f2c::build_hier_colouring_plan(
        &source, args, sections, descriptor, {32, 64, 1 << 20});
    CHECK(result);
    CHECK(result.plan->statistics.launches > sections.size());
    check_colouring(*result.plan, args, descriptor, sections);
}

void test_colour_fallbacks() {
    {
        ChainFixture fixture(8);
        std::array<f2c::ExecutionSection, 1> sections{{{0, 8}}};
        auto result = f2c::build_hier_colouring_plan(
            &fixture.edges, fixture.args, sections, fixture.descriptor(),
            {4, 4, 16});
        expect_reason(result,
                      f2c::FallbackReason::insufficient_shared_memory);
    }
    {
        MixedFixture fixture;
        fixture.args.push_back(make_dat_arg(&fixture.dat_b, &fixture.map_b,
                                             3, "integer(4)", sizeof(int), 1,
                                             OP_READ));
        expect_reason(build_colour(fixture),
                      f2c::FallbackReason::incompatible_argument);
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_reason(build_colour(fixture),
                      f2c::FallbackReason::no_active_argument);
    }
    {
        // Two groups resolving to one dat would be coloured independently.
        ChainFixture fixture(8);
        std::array<f2c::HierArgDescriptor, 2> arg_desc{{{0, 0}, {1, 1}}};
        std::array<f2c::HierScalarType, 2> dat_desc{{
            f2c::HierScalarType::i32,
            f2c::HierScalarType::i32,
        }};
        std::array<f2c::ExecutionSection, 1> sections{{{0, 8}}};
        auto result = f2c::build_hier_colouring_plan(
            &fixture.edges, fixture.args, sections,
            {arg_desc, dat_desc, -1}, {4, 4, 1 << 16});
        expect_reason(result,
                      f2c::FallbackReason::incompatible_argument);
    }
}

void test_packed_word_boundaries() {
    constexpr auto word = f2c::hier_smem_pack_stage_word(
        f2c::hier_smem_slot_mask, true, true);
    static_assert(f2c::hier_smem_stage_slot(word) ==
                  f2c::hier_smem_slot_mask);
    static_assert(f2c::hier_smem_stage_owner(word));
    static_assert(f2c::hier_smem_stage_exclusive(word));
    constexpr std::array fallback_reasons{
        f2c::FallbackReason::none,
        f2c::FallbackReason::no_active_argument,
        f2c::FallbackReason::incompatible_argument,
        f2c::FallbackReason::insufficient_shared_memory,
        f2c::FallbackReason::low_compression,
        f2c::FallbackReason::too_many_colours,
    };
    for (auto reason : fallback_reasons)
        CHECK(f2c::fallback_reason_name(reason) != "unknown");
}

void test_plan_owner_lifecycle() {
    int releases = 0;
    auto release = [](void *owner) {
        ++*static_cast<int *>(owner);
    };

    f2c::register_hier_plan_owner(&releases, release);
    f2c::release_hier_plan_device_storage();
    CHECK(releases == 1);

    f2c::unregister_hier_plan_owner(&releases);
    f2c::release_hier_plan_device_storage();
    CHECK(releases == 1);
}

} // namespace

int main() {
    try {
        test_mixed_plan();
        test_optional_and_alignment();
        test_chunk_sizes_and_clamping();
        test_exclusive_within_one_section();
        test_exclusive_is_per_section();
        test_exclusive_can_be_disabled();
        test_runtime_fallbacks();
        test_colour_mixed_plan();
        test_colour_chain_blocks();
        test_colour_rounds_and_limit();
        test_colour_random();
        test_colour_fallbacks();
        test_packed_word_boundaries();
        test_plan_owner_lifecycle();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hierarchical plan test failed: %s\n",
                     error.what());
        return EXIT_FAILURE;
    }

    std::printf("hierarchical plan tests passed\n");
    return EXIT_SUCCESS;
}

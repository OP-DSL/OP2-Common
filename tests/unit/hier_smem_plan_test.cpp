#include <op_hier_smem_plan.h>

#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
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
    std::array<f2c::HierSmemArgDescriptor, 4> arg_desc{{
        {0, 0},
        {1, 0},
        {2, 1},
        {3, 2},
    }};
    std::array<f2c::HierSmemDatDescriptor, 3> dat_desc{{
        {f2c::HierSmemScalarType::f64},
        {f2c::HierSmemScalarType::i32},
        {f2c::HierSmemScalarType::f32},
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

    f2c::HierSmemStagingDescriptor descriptor() const {
        return {arg_desc, dat_desc, -1};
    }

    f2c::HierSmemPlanOptions options() const {
        return {2, 4, 1024};
    }

    f2c::HierSmemPlanBuildResult build() {
        return f2c::build_hier_smem_plan(
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
    std::array<f2c::HierSmemArgDescriptor, 1> arg_desc{{{0, 0}}};
    std::array<f2c::HierSmemDatDescriptor, 1> dat_desc{{
        {f2c::HierSmemScalarType::f64},
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

    f2c::HierSmemStagingDescriptor descriptor() const {
        return {arg_desc, dat_desc, -1};
    }

    f2c::HierSmemPlanBuildResult build(int requested,
                                       std::size_t shared_limit,
                                       int block_size = 128) {
        f2c::HierSmemPlanOptions options{
            block_size, requested, shared_limit};
        return f2c::build_hier_smem_plan(
            &source, args, sections, descriptor(), options);
    }
};

void expect_reason(const f2c::HierSmemPlanBuildResult& result,
                   f2c::HierSmemFallbackReason reason) {
    CHECK(!result);
    CHECK(result.reason == reason);
    CHECK(f2c::hier_smem_fallback_reason_name(reason) != "unknown");
}

f2c::HierSmemStageWord stage_word(const f2c::HierSmemPlan& plan,
                                  std::size_t staged_arg,
                                  int source_element) {
    return plan.stage_words[
        staged_arg * static_cast<std::size_t>(plan.set_stride) +
        static_cast<std::size_t>(source_element)];
}

int stage_count(const f2c::HierSmemPlan& plan, std::size_t num_stage_dats,
                std::size_t chunk, std::size_t staged_dat) {
    return plan.stage_counts[chunk * num_stage_dats + staged_dat];
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
    CHECK((plan.stage_counts == std::vector<int>{2, 2, 0, 3, 3, 0}));
    CHECK(stage_count(plan, fixture.dat_desc.size(), 0, 0) == 2);
    CHECK(stage_count(plan, fixture.dat_desc.size(), 0, 1) == 2);
    CHECK((plan.section_shared_bytes ==
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
        for (std::size_t staged_arg = 0;
             staged_arg < fixture.arg_desc.size();
             ++staged_arg)
            CHECK(!f2c::hier_smem_stage_exclusive(
                stage_word(plan, staged_arg, source)));
    }

    CHECK(plan.statistics.raw_references == 18);
    CHECK(plan.statistics.distinct_targets == 10);
}

void test_optional_and_alignment() {
    MixedFixture fixture;
    auto inactive = fixture.build();
    CHECK(inactive);

    fixture.args[3].opt = 1;
    auto active = fixture.build();
    CHECK(active);
    CHECK((active.plan->stage_counts ==
           std::vector<int>{2, 2, 1, 3, 3, 2}));
    CHECK((active.plan->section_shared_bytes ==
           std::vector<std::size_t>{60, 92}));
    CHECK(active.plan->statistics.raw_references == 24);
    CHECK(active.plan->statistics.distinct_targets == 13);

    fixture.args[1].opt = 0;
    fixture.args[2].opt = 0;
    std::array<f2c::HierSmemArgDescriptor, 2> arg_desc{{{3, 0}, {0, 1}}};
    std::array<f2c::HierSmemDatDescriptor, 2> dat_desc{{
        {f2c::HierSmemScalarType::f32},
        {f2c::HierSmemScalarType::f64},
    }};
    auto descriptor = f2c::HierSmemStagingDescriptor{
        arg_desc, dat_desc, -1};
    auto options = fixture.options();
    auto aligned = f2c::build_hier_smem_plan(
        &fixture.source, fixture.args, fixture.sections, descriptor, options);
    CHECK(aligned);
    CHECK((aligned.plan->section_shared_bytes ==
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
        CHECK(result.plan->section_shared_bytes[0] ==
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
                  f2c::HierSmemFallbackReason::insufficient_shared_memory);
}

void test_runtime_fallbacks() {
    {
        MixedFixture fixture;
        fixture.args.push_back(make_dat_arg(&fixture.dat_a, &fixture.map_a0,
                                             2, "double", sizeof(double), 1,
                                             OP_READ));
        expect_reason(fixture.build(),
                      f2c::HierSmemFallbackReason::incompatible_argument);
    }
    {
        MixedFixture fixture;
        for (auto& arg : fixture.args)
            arg.opt = 0;
        expect_reason(fixture.build(),
                      f2c::HierSmemFallbackReason::no_active_increment);
    }
    {
        MixedFixture fixture;
        std::array<f2c::HierSmemArgDescriptor, 4> arg_desc{{
            {0, 0},
            {1, 1},
            {2, 2},
            {3, 3},
        }};
        std::array<f2c::HierSmemDatDescriptor, 4> dat_desc{{
            {f2c::HierSmemScalarType::f64},
            {f2c::HierSmemScalarType::f64},
            {f2c::HierSmemScalarType::i32},
            {f2c::HierSmemScalarType::f32},
        }};
        auto descriptor = f2c::HierSmemStagingDescriptor{
            arg_desc, dat_desc, -1};
        auto result = f2c::build_hier_smem_plan(
            &fixture.source, fixture.args, fixture.sections, descriptor,
            fixture.options());
        expect_reason(result,
                      f2c::HierSmemFallbackReason::incompatible_argument);
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
        f2c::HierSmemFallbackReason::none,
        f2c::HierSmemFallbackReason::no_active_increment,
        f2c::HierSmemFallbackReason::incompatible_argument,
        f2c::HierSmemFallbackReason::insufficient_shared_memory,
    };
    for (auto reason : fallback_reasons)
        CHECK(f2c::hier_smem_fallback_reason_name(reason) != "unknown");
}

void test_plan_owner_lifecycle() {
    int releases = 0;
    auto release = [](void *owner) {
        ++*static_cast<int *>(owner);
    };

    f2c::register_hier_smem_plan_owner(&releases, release);
    f2c::release_hier_smem_plan_device_storage();
    CHECK(releases == 1);

    f2c::unregister_hier_smem_plan_owner(&releases);
    f2c::release_hier_smem_plan_device_storage();
    CHECK(releases == 1);
}

} // namespace

int main() {
    try {
        test_mixed_plan();
        test_optional_and_alignment();
        test_chunk_sizes_and_clamping();
        test_runtime_fallbacks();
        test_packed_word_boundaries();
        test_plan_owner_lifecycle();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hierarchical smem plan test failed: %s\n",
                     error.what());
        return EXIT_FAILURE;
    }

    std::printf("hierarchical smem plan tests passed\n");
    return EXIT_SUCCESS;
}

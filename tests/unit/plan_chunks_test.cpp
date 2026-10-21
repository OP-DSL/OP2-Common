// The chunk building blocks on their own, with no device limits, and the
// OpenMP backend's plans over them: references, chunks within sections,
// each chunk's targets, coloured batches and their wavefront, and element
// colours.

#include <cpu/chunk_plan.h>
#include <plan/chunks.h>
#include <plan/staging.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// Edge i of a chain reaches nodes i and i+1 through a two-component map, so
// neighbouring edges share a node; a second dat on the nodes is read
// through the same map, and a third, on the edges, directly.
struct Chain {
    op_set_core edges{};
    op_set_core nodes{};
    op_dat_core x{};
    op_dat_core y{};
    op_dat_core w{};
    std::vector<int> map_values;
    op_map_core map{};
    std::vector<op_arg> args;

    explicit Chain(int size, int core = -1, int exec = 0) {
        edges.size = size;
        edges.core_size = core < 0 ? size : core;
        edges.exec_size = exec;
        nodes.size = size + exec + 1;
        nodes.core_size = nodes.size;
        for (auto [dat, set] : {std::pair{&x, &nodes}, {&y, &nodes}, {&w, &edges}}) {
            dat->set = set;
            dat->dim = 1;
            dat->size = sizeof(double);
            dat->type = "double";
        }
        for (int i = 0; i < size + exec; ++i) {
            map_values.push_back(i);
            map_values.push_back(i + 1);
        }
        map.from = &edges;
        map.to = &nodes;
        map.dim = 2;
        map.map = map_values.data();

        for (int idx = 0; idx < 2; ++idx)
            args.push_back(arg(&x, idx, OP_INC));
        args.push_back(arg(&y, 0, OP_READ));
        args.push_back(arg(&w, -1, OP_READ));
    }

    op_arg arg(op_dat dat, int idx, op_access access) {
        op_arg out{};
        out.opt = 1;
        out.argtype = OP_ARG_DAT;
        out.dat = dat;
        out.map = idx >= 0 ? &map : nullptr;
        out.map_data = idx >= 0 ? map_values.data() : nullptr;
        out.idx = idx;
        out.dim = 1;
        out.size = sizeof(double);
        out.type = "double";
        out.acc = access;
        return out;
    }
};

// Every element's targets as (group, target) pairs.
std::vector<std::pair<int, int>> targets_of(const plan::References& references,
                                            int element) {
    std::vector<std::pair<int, int>> out;
    for (const auto& reference : references.items)
        if (reference.arg->opt != 0)
            out.push_back({reference.group,
                           reference.arg->map_data[element * reference.arg->map->dim +
                                                   reference.arg->idx]});
    return out;
}

bool disjoint(const std::vector<std::pair<int, int>>& a,
              const std::vector<std::pair<int, int>>& b) {
    for (const auto& item : a)
        if (std::find(b.begin(), b.end(), item) != b.end())
            return false;
    return true;
}

std::vector<std::pair<int, int>> chunk_targets_of(const plan::References& references,
                                                  const plan::Chunks& chunks, int chunk) {
    std::vector<std::pair<int, int>> out;
    for (int element = chunks.begin(chunk); element < chunks.end(chunk); ++element) {
        auto targets = targets_of(references, element);
        out.insert(out.end(), targets.begin(), targets.end());
    }
    return out;
}

// Each section's batches cover its chunks once, and a batch's chunks share
// no target.
void check_batches(const plan::ChunkBatches& batches, const plan::Chunks& chunks,
                   const plan::References& references) {
    std::vector<int> runs(static_cast<std::size_t>(chunks.count()), 0);
    const int sections = static_cast<int>(chunks.section_offsets.size()) - 1;
    CHECK(static_cast<int>(batches.section_batches.size()) == sections + 1);
    for (int section = 0; section < sections; ++section)
        for (int batch = 0; batch < batches.batches(section); ++batch) {
            auto [begin, end] = batches.positions(section, batch);
            std::vector<std::vector<std::pair<int, int>>> seen;
            for (int position = begin; position < end; ++position) {
                int chunk = batches.chunk(position);
                CHECK(chunk >= chunks.section_offsets[static_cast<std::size_t>(section)] &&
                      chunk < chunks.section_offsets[static_cast<std::size_t>(section) + 1]);
                ++runs[static_cast<std::size_t>(chunk)];
                auto reached = chunk_targets_of(references, chunks, chunk);
                for (const auto& other : seen)
                    CHECK(disjoint(reached, other));
                seen.push_back(std::move(reached));
            }
        }
    CHECK(runs == std::vector<int>(static_cast<std::size_t>(chunks.count()), 1));
}

void test_references() {
    Chain chain(8);
    plan::References references;
    CHECK(plan::updated_references(chain.args, references).empty());
    // Both increments and the read of the updated dat, one group; neither
    // the other dat's read nor the direct argument.
    CHECK(references.groups.size() == 1);
    CHECK(references.groups[0].dat == &chain.x && references.groups[0].extent == 9);
    CHECK(references.items.size() == 2);

    chain.args[2].dat = &chain.x;
    CHECK(plan::updated_references(chain.args, references).empty());
    CHECK(references.items.size() == 3 && references.groups.size() == 1);

    chain.args[0].acc = OP_READ;
    chain.args[1].acc = OP_READ;
    CHECK(plan::updated_references(chain.args, references) == "no_active_argument");

    chain.args[0].acc = OP_RW;
    chain.args[1].idx = -2;
    CHECK(plan::updated_references(chain.args, references) == "incompatible_argument");
}

void test_chunks_and_targets() {
    // 6 core edges, 4 more owned, 2 in the exec halo: three sections.
    Chain chain(10, 6, 2);
    auto sections = Sections::indirect(&chain.edges, true);
    auto chunks = plan::chunk_sections(sections, 4);
    CHECK(chunks.size == 4);
    CHECK((chunks.offsets == std::vector<int>{0, 4, 6, 10, 12}));
    CHECK((chunks.section_offsets == std::vector<int>{0, 2, 3, 4}));

    plan::References references;
    CHECK(plan::updated_references(chain.args, references).empty());
    auto targets = plan::chunk_targets(chunks, references);
    CHECK(targets.groups == 1);
    CHECK((targets.offsets == std::vector<int>{0, 5, 8, 13, 16}));
    CHECK((targets.targets == std::vector<int>{0, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10,
                                               10, 11, 12}));
    CHECK((targets.references == std::vector<int>{1, 2, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2,
                                                  1, 1, 2, 1}));
    CHECK(targets.total_references == 24);
    CHECK(targets.compression() == 24.0 / 16.0);

    // One batch per section, in chunk order.
    auto batches = plan::section_batches(chunks);
    CHECK(batches.order.empty());
    CHECK((batches.batch_offsets == chunks.section_offsets));
    CHECK(batches.batches(0) == 1 && batches.positions(2, 0) == std::make_pair(3, 4));
}

void test_colouring() {
    Chain chain(64);
    plan::References references;
    CHECK(plan::updated_references(chain.args, references).empty());
    auto sections = Sections::direct(&chain.edges);
    auto chunks = plan::chunk_sections(sections, 4);
    auto targets = plan::chunk_targets(chunks, references);

    // Neighbouring chunks share a node: two colours.
    auto coloured = plan::colour_chunks(chunks, targets);
    CHECK(coloured.count() == 2);
    check_batches(coloured, chunks, references);

    // A window of one chunk colours each on its own: a batch per chunk.
    auto windowed = plan::colour_chunks(chunks, targets, 1);
    CHECK(windowed.count() == chunks.count());
    check_batches(windowed, chunks, references);

    // The wavefront runs the section as one batch, each chunk after its
    // neighbour on the left, the latest earlier position sharing its node.
    auto wave = plan::wavefront(windowed, targets);
    CHECK(wave.batches.count() == 1);
    CHECK(wave.batches.order == windowed.order);
    CHECK(wave.pred_offsets.size() == static_cast<std::size_t>(chunks.count()) + 1);
    for (int position = 0; position < chunks.count(); ++position) {
        std::vector<int> preds(wave.preds.begin() + wave.pred_offsets[static_cast<std::size_t>(position)],
                               wave.preds.begin() + wave.pred_offsets[static_cast<std::size_t>(position) + 1]);
        CHECK(preds == (position == 0 ? std::vector<int>{} : std::vector<int>{position - 1}));
    }

    // A chunk's elements alternate colours.
    auto colours = plan::colour_elements(chunks, references);
    CHECK(colours.max == 2);
    CHECK(colours.counts == std::vector<int>(static_cast<std::size_t>(chunks.count()), 2));
    for (int element = 0; element < 64; ++element)
        CHECK(colours.colours[static_cast<std::size_t>(element)] == element % 2);
}

// The staging pipeline under a backend's limits: the chunk size is a lane
// multiple, a budget no chunk fits rejects, and singles may stay unstaged.
void test_stage_chunks() {
    Chain chain(1000, 600, 100);
    chain.args.resize(2);
    auto sections = Sections::indirect(&chain.edges, false);
    std::array<plan::ArgGroup, 2> groups{{{0, 0}, {1, 0}}};
    plan::ArgGroups grouped{groups, 1};

    plan::StagedChunks staged;
    CHECK(plan::stage_chunks(chain.args, sections, grouped, false, 100, 48,
                             std::size_t{1} << 20, true, staged)
              .empty());
    CHECK(staged.chunks.size == 144);
    CHECK(staged.staging.stride == sections.set_stride());
    CHECK(staged.targets.compression() > 1.9);
    // A chunk of the chain reaches its first and last nodes once.
    CHECK(staged.staging.unstaged_singles ==
          2 * static_cast<std::size_t>(staged.chunks.count()));

    // 48 elements reach at least 49 doubles, beyond 40 doubles of scratch.
    plan::StagedChunks tight;
    CHECK(plan::stage_chunks(chain.args, sections, grouped, false, 100, 48,
                             40 * sizeof(double), false, tight) == "insufficient_scratch");

    chain.args[0].opt = chain.args[1].opt = 0;
    plan::StagedChunks inactive;
    CHECK(plan::stage_chunks(chain.args, sections, grouped, false, 100, 48,
                             std::size_t{1} << 20, false, inactive) == "no_active_argument");
}

// The OpenMP plans: chunk_colouring needs no groups, chunk_atomics stages
// the grouped increments.
void test_cpu_plans() {
    Chain chain(1000, 600, 100);
    auto sections = Sections::indirect(&chain.edges, false);

    auto coloured = op::cpu::plan_chunk_colouring(chain.args, sections, 16);
    CHECK(coloured);
    const auto& colouring = *coloured.plan;
    CHECK(colouring.chunks.size == 16);
    CHECK(colouring.staging.stage_words.empty());
    plan::References references;
    CHECK(plan::updated_references(chain.args, references).empty());
    check_batches(colouring.batches, colouring.chunks, references);

    std::array<plan::ArgGroup, 2> groups{{{0, 0}, {1, 0}}};
    chain.args.resize(2);
    auto staged = op::cpu::plan_chunk_atomics(chain.args, sections,
                                              plan::ArgGroups{groups, 1}, 100, 0.0);
    CHECK(staged);
    const auto& atomics = *staged.plan;
    CHECK(atomics.chunks.size == 100 && atomics.batches.order.empty());
    CHECK(atomics.staging.stride == sections.set_stride());
    // A chunk of 100 edges reaches 101 nodes.
    CHECK(atomics.scratch_bytes == 101 * sizeof(double));
    CHECK(atomics.compression > 1.9);

    auto rejected = op::cpu::plan_chunk_atomics(chain.args, sections,
                                                plan::ArgGroups{groups, 1}, 100, 2.5);
    CHECK(!rejected && rejected.rejected == "low_compression" && rejected.plan == nullptr);
    CHECK(rejected.compression > 1.9 && rejected.compression < 2.5);

    chain.args[0].opt = chain.args[1].opt = 0;
    CHECK(op::cpu::plan_chunk_colouring(chain.args, sections, 16).rejected ==
          "no_active_argument");

    // An empty section has no batches, and an empty set none at all.
    Chain halo(8, 8, 8);
    halo.args.resize(2);
    auto split = Sections::indirect(&halo.edges, true);
    CHECK(split.size() == 3 && split[1].size() == 0);
    auto sectioned = op::cpu::plan_chunk_atomics(halo.args, split,
                                                 plan::ArgGroups{groups, 1}, 4, 0.0);
    CHECK(sectioned);
    CHECK((sectioned.plan->batches.section_batches == std::vector<int>{0, 1, 1, 2}));
    CHECK(op::cpu::plan_chunk_colouring(halo.args, split, 4).plan->batches.batches(1) == 0);

    Chain empty(0);
    empty.args.resize(2);
    auto none = Sections::direct(&empty.edges);
    auto coloured_empty = op::cpu::plan_chunk_colouring(empty.args, none, 4);
    auto staged_empty = op::cpu::plan_chunk_atomics(empty.args, none,
                                                    plan::ArgGroups{groups, 1}, 4, 0.0);
    CHECK(coloured_empty && coloured_empty.plan->batches.count() == 0);
    CHECK(staged_empty && staged_empty.plan->batches.count() == 0);
}

} // namespace

int main() {
    try {
        test_references();
        test_chunks_and_targets();
        test_colouring();
        test_stage_chunks();
        test_cpu_plans();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "plan chunks test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("plan chunks tests passed\n");
    return EXIT_SUCCESS;
}

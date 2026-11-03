#pragma once

// One call of a par_loop: its set and arguments, its configuration's key,
// the element sections it runs in, and the reduction of its globals across
// the ranks.

#include <op2/rt/abi/descriptor.h>
#include <op_lib_core.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

namespace op::rt::loop {

// The element ranges a loop runs, in order, with where its halo wait and
// global processing fall among them.  A direct loop reads no halo: its
// exchange only brings the dats to this device, so it waits before its one
// section.  An indirect loop runs its core elements, then, once the halo has
// arrived, the rest; the owned elements form a section of their own when a
// global reduction has to be processed after them.
struct Section {
    int begin = 0;
    int end = 0;

    int size() const { return end - begin; }
};

// The phases a strategy runs a call in, sections or colours: how many, the
// one the halo wait comes before (count for after the last), and the one the
// globals are processed after, none when no phase holds owned elements.
struct Phases {
    int count = 0;
    int wait_before = 0;
    std::optional<int> globals_after;
};

class Sections {
private:
    std::array<Section, 3> m_sections{};
    int m_count = 0;
    int m_wait_before = 0;
    int m_globals_after = 0;

public:
    static Sections direct(op_set set);
    // Ending at run_size, which takes in the exec halo when the call runs it.
    static Sections indirect(op_set set, bool separate_owned, int run_size);

    int size() const { return m_count; }
    const Section& operator[](int index) const {
        return m_sections[static_cast<std::size_t>(index)];
    }
    std::span<const Section> span() const {
        return {m_sections.data(), static_cast<std::size_t>(m_count)};
    }

    // The elements they cover, up to the last section's end.
    int elements() const { return m_count > 0 ? m_sections[m_count - 1].end : 0; }

    // A section a phase.
    Phases phases() const { return {m_count, m_wait_before, m_globals_after}; }
};

// One call, as a view: the set, the arguments, which its run may write, and
// the sections they run in.  Its configuration, the shape its key
// identifies, is what plans and the tuner depend on.
class Call {
private:
    mutable std::uint64_t m_key = 0;
    mutable bool m_keyed = false;

public:
    const op_set set;
    const std::span<op_arg> args;
    // The elements the call runs, as op_mpi_halo_exchanges returned it: the
    // exec halo too when the exchange decided the call runs it.
    const int run_size;
    // Direct, or indirect with the owned elements apart when it reduces a
    // global.
    const Sections sections;

    Call(op_set set_, std::span<op_arg> args_, int run_size_, bool direct)
        : set{set_}, args{args_}, run_size{run_size_},
          sections{direct ? Sections::direct(set_)
                          : Sections::indirect(set_, global_reduction(), run_size_)} {}

    // Any active global argument reduced across the set.
    bool global_reduction() const;

    // Any active global argument the call writes: reduced, read-written or
    // written.
    bool writes_globals() const;

    // The set index, the run size and, per argument, its activity, kind, the
    // first argument on its dat, its map index, map component, access and
    // dimension: which arguments share a dat, not which dat, so a temporary
    // dat declared again keeps its plans.
    std::uint64_t key() const;

    // "edges (3.1M) via edge_nodes, reduction"; for reports only.
    std::string describe() const;
};

// What a strategy's run is given: the call, and whether to time its
// launches.
struct RunArgs {
    const Call& call;
    bool timed;
};

// A global's element type, from its C or Fortran spelling; none when OP2
// does not reduce it.
std::optional<abi::ElementType> element_type(const char *type);

// Reduce a call's INC, MIN and MAX globals across the ranks once this rank's
// elements are combined into them; an unsupported type stops the run, naming
// loop.
void reduce_mpi_globals(std::span<op_arg> args, const char *loop);

} // namespace op::rt::loop

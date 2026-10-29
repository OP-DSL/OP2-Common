#pragma once

// One call of a par_loop: its set and arguments, its configuration's key,
// the element sections it runs in, and the reduction of its globals across
// the ranks.

#include <op_lib_core.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace op::rt::loop {

// One call, as a view: the set and the arguments.  Its configuration, the
// shape its key identifies, is what plans and the tuner depend on.
class Call {
private:
    mutable std::uint64_t m_key = 0;
    mutable bool m_keyed = false;

public:
    op_set set;
    std::span<const op_arg> args;
    // The elements the call runs, as op_mpi_halo_exchanges returned it: the
    // exec halo too when the exchange decided the call runs it.
    int run_size;

    Call(op_set set_, std::span<const op_arg> args_, int run_size_)
        : set{set_}, args{args_}, run_size{run_size_} {}

    // Any active global argument reduced across the set.
    bool global_reduction() const;

    // The set index, the run size and, per argument, its activity, kind, the
    // first argument on its dat, its map index, map component, access and
    // dimension: which arguments share a dat, not which dat, so a temporary
    // dat declared again keeps its plans.
    std::uint64_t key() const;

    // "edges (3.1M) via edge_nodes, reduction"; for reports only.
    std::string describe() const;
};

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

// The stride of an indirect loop's per-element arrays: the set's owned and
// exec-halo elements, rounded up to a warp.
inline int indirect_set_stride(op_set set) {
    return (set->size + set->exec_size + 31) & ~31;
}

class Sections {
private:
    std::array<Section, 3> m_sections{};
    int m_count = 0;
    int m_wait_before = 0;
    int m_globals_after = 0;
    int m_set_stride = 0;

public:
    static Sections direct(op_set set);
    // Ending at run_size, which takes in the exec halo when the call runs it.
    static Sections indirect(op_set set, bool separate_owned, int run_size);
    // A call's sections: direct, or indirect with the owned elements apart
    // when it reduces a global.
    static Sections of(const Call& call, bool direct);

    int size() const { return m_count; }
    const Section& operator[](int index) const {
        return m_sections[static_cast<std::size_t>(index)];
    }
    std::span<const Section> span() const {
        return {m_sections.data(), static_cast<std::size_t>(m_count)};
    }

    // The section the halo wait comes before, in [0, size()], size() for
    // after the last; the section the globals are processed after.
    int wait_index() const { return m_wait_before; }
    int globals_index() const { return m_globals_after; }
    int set_stride() const { return m_set_stride; }
};

// What a strategy's run is given: the arguments, and whether to time its
// launches.
struct RunArgs {
    op_arg *args;
    int nargs;
    bool timed;
};

// A global's element type, from its C or Fortran spelling.
enum class ScalarType {
    f64,
    f32,
    i32,
    logical,
    unknown,
};

ScalarType scalar_type(const char *type);

// Reduce the call's INC, MIN and MAX globals across the ranks once this
// rank's elements are combined into them; an unsupported type stops the run,
// naming loop.
void reduce_mpi_globals(op_arg *args, int nargs, const char *loop);

} // namespace op::rt::loop

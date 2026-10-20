#pragma once

// When queued calls may share a kernel, from their arguments alone: element
// by element in queue order, or all at once.  A backend's fusion says which
// of a call's indirect increments are atomic.

#include <op_lib_core.h>

#include <algorithm>
#include <span>
#include <string>
#include <vector>

namespace op::loop::fusion {

inline bool writes(const op_arg& arg) {
    return arg.acc == OP_WRITE || arg.acc == OP_RW || arg.acc == OP_INC;
}

// How a call uses one dat.
struct Use {
    op_dat dat = nullptr;
    bool read = false;
    bool write = false;
    bool only_inc = true;
    bool all_direct = true;
    bool any_direct = false;
    bool indirect_inc = false;
};

// The active dats the arguments reach, one entry each.
inline std::vector<Use> uses(std::span<const op_arg> args) {
    std::vector<Use> out;
    for (const auto& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT)
            continue;
        auto found = std::find_if(out.begin(), out.end(),
                                  [&](const Use& use) { return use.dat == arg.dat; });
        if (found == out.end()) {
            out.push_back({arg.dat});
            found = out.end() - 1;
        }
        bool direct = arg.map == nullptr;
        found->read |= arg.acc == OP_READ || arg.acc == OP_RW;
        found->write |= writes(arg);
        found->only_inc &= arg.acc == OP_INC;
        found->all_direct &= direct;
        found->any_direct |= direct;
        found->indirect_inc |= !direct && arg.acc == OP_INC;
    }
    return out;
}

// Both only increment the dat, every increment indirect and atomic: such
// calls commute, element by element or all at once.
inline bool atomic_increments(const Use& x, bool x_atomic, const Use& y, bool y_atomic) {
    return x.only_inc && y.only_inc && !x.any_direct && !y.any_direct && x_atomic &&
           y_atomic;
}

// Whether a call using y may run after one using x for each element in
// turn, as far as their one dat goes: neither writes it, both reach it only
// directly, or both only increment it with every indirect increment atomic.
inline bool element_wise(const Use& x, bool x_atomic, const Use& y, bool y_atomic) {
    if (x.dat != y.dat || !(x.write || y.write))
        return true;
    if (x.all_direct && y.all_direct)
        return true;
    return x.only_inc && y.only_inc && (!x.indirect_inc || x_atomic) &&
           (!y.indirect_inc || y_atomic);
}

// Whether a call using y may follow one using x in a chain, whose halos are
// exchanged before either runs: element by element, and y's exchange the
// same before x runs as after.  y_exec: y runs the exec halo, where it would
// need the exchange of a dat x writes that it reads directly.
inline bool chains(const Use& x, bool x_atomic, const Use& y, bool y_atomic, bool y_exec) {
    return element_wise(x, x_atomic, y, y_atomic) &&
           !(y_exec && x.dat == y.dat && x.write && y.read && y.any_direct);
}

// Whether b may run after a for each element in turn, dat by dat.
inline bool element_wise(const std::vector<Use>& a, bool a_atomic,
                         const std::vector<Use>& b, bool b_atomic) {
    for (const auto& x : a)
        for (const auto& y : b)
            if (!element_wise(x, a_atomic, y, b_atomic))
                return false;
    return true;
}

// Whether calls using x and y may run at once, as far as their one dat
// goes: neither writes it, or both only increment it atomically.
inline bool concurrent(const Use& x, bool x_atomic, const Use& y, bool y_atomic) {
    return x.dat != y.dat || !(x.write || y.write) ||
           atomic_increments(x, x_atomic, y, y_atomic);
}

// Whether a and b may run at once, dat by dat.
inline bool independent(const std::vector<Use>& a, bool a_atomic,
                        const std::vector<Use>& b, bool b_atomic) {
    for (const auto& x : a)
        for (const auto& y : b)
            if (!concurrent(x, a_atomic, y, b_atomic))
                return false;
    return true;
}

// How a call reaches a dat, for reports: "increments through a map".
inline std::string access(const Use& use) {
    std::string verb = use.only_inc          ? "increments"
                       : use.read && use.write ? "reads and writes"
                       : use.write            ? "writes"
                                              : "reads";
    if (use.all_direct)
        return verb;
    return verb + (use.any_direct ? " directly and through a map" : " through a map");
}

// The first dat that keeps b from running after a: in a chain when
// same_range (b_exec as for chains), else at once; both null when there is
// none.
struct Conflict {
    const Use *first = nullptr;
    const Use *second = nullptr;
};

inline Conflict conflict(const std::vector<Use>& a, bool a_atomic,
                         const std::vector<Use>& b, bool b_atomic, bool same_range,
                         bool b_exec) {
    for (const auto& x : a)
        for (const auto& y : b) {
            bool fits = same_range ? chains(x, a_atomic, y, b_atomic, b_exec)
                                   : concurrent(x, a_atomic, y, b_atomic);
            if (!fits)
                return {&x, &y};
        }
    return {};
}

} // namespace op::loop::fusion

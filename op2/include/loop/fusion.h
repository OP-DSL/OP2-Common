#pragma once

// When queued calls may share a kernel, from their arguments alone: element
// by element in queue order, or all at once.  A backend's fusion says which
// of a call's indirect increments are atomic.

#include <op_lib_core.h>

#include <algorithm>
#include <span>
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

// Whether b may run after a for each element in turn: every dat either
// writes is reached only directly by both, or only incremented by both
// with every indirect increment atomic.
inline bool element_wise(const std::vector<Use>& a, bool a_atomic,
                         const std::vector<Use>& b, bool b_atomic) {
    for (const auto& x : a)
        for (const auto& y : b) {
            if (x.dat != y.dat || !(x.write || y.write))
                continue;
            if (x.all_direct && y.all_direct)
                continue;
            if (x.only_inc && y.only_inc && (!x.indirect_inc || a_atomic) &&
                (!y.indirect_inc || b_atomic))
                continue;
            return false;
        }
    return true;
}

// Whether a and b may run at once: every dat either writes only gets
// atomic indirect increments from both.
inline bool independent(const std::vector<Use>& a, bool a_atomic,
                        const std::vector<Use>& b, bool b_atomic) {
    for (const auto& x : a)
        for (const auto& y : b)
            if (x.dat == y.dat && (x.write || y.write) &&
                !atomic_increments(x, a_atomic, y, b_atomic))
                return false;
    return true;
}


} // namespace op::loop::fusion

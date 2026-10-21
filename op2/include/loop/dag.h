#pragma once

// The dependency graph of a run of queued calls, from their arguments alone,
// and what it allows: how few kernels the run could take with the calls
// reordered, and where each halo exchange could start.  Reports only, so
// far; a backend says which calls it can fuse and which increment
// atomically.

#include <loop/fusion.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace op::loop::dag {

// One call of the run.
struct Node {
    op_set set = nullptr;
    // It writes through a map, so it runs the exec halo.
    bool exec = false;
    // Its indirect increments are atomic.
    bool atomic = false;
    // The backend could fuse it with others.
    bool fusable = true;
    std::vector<fusion::Use> uses;
};

// b depends on a through dat: element by element in one chain, or only once
// all of a has run.
struct Edge {
    int from = 0;
    int to = 0;
    op_dat dat = nullptr;
    bool element = false;
};

// One version of a dat that a call needs the halo of: written by writers
// (none: before the run), first needed by reader.  Later readers of the
// version need no exchange of their own.
struct Halo {
    op_dat dat = nullptr;
    std::vector<int> writers;
    int reader = 0;

    bool before_run() const { return writers.empty(); }
};

struct Graph {
    std::vector<Edge> edges;
    std::vector<Halo> halos;
    // Each call's level: the most barriers on a path to it.  Calls on one
    // level could share a kernel, reordered.
    std::vector<int> level;
    int levels = 0;
    // With the calls the backend cannot fuse each in a launch of its own.
    int launches = 0;
};

// The writer of a halo's version that runs last: its exchange can start
// once that one is done.  -1 before the run.
inline int last_writer(const Halo& halo) {
    return halo.writers.empty() ? -1 : halo.writers.back();
}

// Whether a call reads the dat's halo: through a map, or directly where it
// runs the exec halo.
inline bool reads_halo(const fusion::Use& use, bool exec) {
    return use.read && (!use.all_direct || exec);
}

inline Graph build(std::span<const Node> nodes) {
    Graph graph;
    auto n = static_cast<int>(nodes.size());

    // Per dat: the writers of the version that atomic increments build on,
    // those increments since (they commute), the readers since, and whether
    // the current version's halo has been asked for.
    struct State {
        op_dat dat = nullptr;
        std::vector<int> base;
        std::vector<int> increments;
        std::vector<int> readers;
        bool halo_needed = false;
    };
    std::vector<State> states;
    auto state_of = [&](op_dat dat) -> State& {
        for (auto& state : states)
            if (state.dat == dat)
                return state;
        return states.emplace_back(State{dat});
    };
    auto add_edge = [&](int from, int to, op_dat dat) {
        if (from == to)
            return;
        const auto& a = nodes[static_cast<std::size_t>(from)];
        const auto& b = nodes[static_cast<std::size_t>(to)];
        bool element = a.set == b.set && a.exec == b.exec;
        if (element)
            for (const auto& x : a.uses)
                for (const auto& y : b.uses)
                    element = element && fusion::chains(x, a.atomic, y, b.atomic, b.exec);
        for (auto& edge : graph.edges)
            if (edge.from == from && edge.to == to) {
                edge.element = edge.element && element;
                return;
            }
        graph.edges.push_back({from, to, dat, element});
    };
    auto from_all = [&](const std::vector<int>& from, int to, op_dat dat) {
        for (int f : from)
            add_edge(f, to, dat);
    };

    for (int c = 0; c < n; ++c) {
        const auto& node = nodes[static_cast<std::size_t>(c)];
        for (const auto& use : node.uses) {
            auto& state = state_of(use.dat);
            bool atomic_inc = use.only_inc && !use.any_direct && node.atomic;
            if (use.read) {
                from_all(state.base, c, use.dat);
                from_all(state.increments, c, use.dat);
            }
            if (reads_halo(use, node.exec) && !state.halo_needed) {
                auto& halo = graph.halos.emplace_back(Halo{use.dat, state.base, c});
                halo.writers.insert(halo.writers.end(), state.increments.begin(),
                                    state.increments.end());
                state.halo_needed = true;
            }
            if (!use.write) {
                state.readers.push_back(c);
                continue;
            }
            if (atomic_inc) {
                // Once read, the increments so far are the version the next
                // ones build on.
                if (!state.readers.empty()) {
                    from_all(state.readers, c, use.dat);
                    state.base.insert(state.base.end(), state.increments.begin(),
                                      state.increments.end());
                    state.increments.clear();
                    state.readers.clear();
                }
                from_all(state.base, c, use.dat);
                state.increments.push_back(c);
            } else {
                from_all(state.base, c, use.dat);
                from_all(state.increments, c, use.dat);
                from_all(state.readers, c, use.dat);
                state.base.assign(1, c);
                state.increments.clear();
                state.readers.clear();
            }
            state.halo_needed = false;
        }
    }

    graph.level.assign(static_cast<std::size_t>(n), 0);
    std::sort(graph.edges.begin(), graph.edges.end(),
              [](const Edge& a, const Edge& b) { return a.to < b.to; });
    for (const auto& edge : graph.edges) {
        int level = graph.level[static_cast<std::size_t>(edge.from)] + (edge.element ? 0 : 1);
        auto& to = graph.level[static_cast<std::size_t>(edge.to)];
        to = std::max(to, level);
    }
    for (int level : graph.level)
        graph.levels = std::max(graph.levels, level + 1);
    for (auto& halo : graph.halos)
        std::sort(halo.writers.begin(), halo.writers.end(), [&](int a, int b) {
            return graph.level[static_cast<std::size_t>(a)] <
                   graph.level[static_cast<std::size_t>(b)];
        });

    // A level's fusable calls share a launch; the others take one each.
    std::vector<bool> shared(static_cast<std::size_t>(graph.levels), false);
    for (int c = 0; c < n; ++c) {
        auto level = static_cast<std::size_t>(graph.level[static_cast<std::size_t>(c)]);
        if (!nodes[static_cast<std::size_t>(c)].fusable)
            ++graph.launches;
        else if (!shared[level]) {
            shared[level] = true;
            ++graph.launches;
        }
    }
    return graph;
}

// A run's signature: each call's loop, set, and its arguments' dats, maps
// and accesses, by index.  Runs with equal signatures plan alike.
class Signature {
private:
    std::uint64_t m_hash = 1469598103934665603ull;

    void mix(std::uint64_t value) {
        m_hash ^= value;
        m_hash *= 1099511628211ull;
    }

public:
    void add(std::uint64_t loop, op_set set, std::span<const op_arg> args) {
        mix(loop);
        mix(static_cast<std::uint64_t>(set->index));
        for (const auto& arg : args) {
            mix(static_cast<std::uint64_t>(arg.opt));
            if (arg.opt == 0)
                continue;
            mix(static_cast<std::uint64_t>(arg.argtype));
            mix(static_cast<std::uint64_t>(arg.acc));
            mix(arg.dat != nullptr ? static_cast<std::uint64_t>(arg.dat->index) : ~0ull);
            mix(arg.map != nullptr ? static_cast<std::uint64_t>(arg.map->index) : ~0ull);
            mix(static_cast<std::uint64_t>(arg.idx + 2));
        }
    }

    std::uint64_t value() const { return m_hash; }
};

// Predicts the next run's signature from the runs so far: the one that
// followed the last order runs' signatures last time.
class Predictor {
private:
    std::size_t m_order;
    std::unordered_map<std::uint64_t, std::uint64_t> m_next;
    // The last order signatures, oldest first.
    std::vector<std::uint64_t> m_history;

    std::uint64_t context() const {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (auto signature : m_history)
            hash = (hash ^ signature) * 0x100000001b3ull;
        return hash;
    }

public:
    explicit Predictor(std::size_t order = 1) : m_order(order) {}

    // The prediction for the next run, if any.
    bool predict(std::uint64_t& next) const {
        if (m_history.size() < m_order)
            return false;
        auto found = m_next.find(context());
        if (found == m_next.end())
            return false;
        next = found->second;
        return true;
    }

    // Record a run; true when it was the prediction.
    bool observe(std::uint64_t signature) {
        std::uint64_t predicted = 0;
        bool hit = predict(predicted) && predicted == signature;
        if (m_history.size() == m_order)
            m_next[context()] = signature;
        m_history.push_back(signature);
        if (m_history.size() > m_order)
            m_history.erase(m_history.begin());
        return hit;
    }
};

} // namespace op::loop::dag

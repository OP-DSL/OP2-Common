#pragma once

// Run shapes, for reports under OP_DIAGS: each queue run's signature, how
// well the next run is predicted from the last, and per distinct shape the
// dependency graph and what it would allow (loop/dag.h): launches with the
// calls reordered, and how early each halo exchange could start.

#include <loop/dag.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace op::gpu {

// A queued call as the analysis sees it.
struct RunCall {
    std::uint64_t loop = 0;
    op_set set = nullptr;
    std::span<const op_arg> args;
    bool atomic = false;
    bool fusable = false;
    // Per argument: a dat dirty when the call was queued.
    std::vector<bool> dirty;
};

class RunShapes {
private:
    struct Shape {
        std::size_t runs = 0;
        std::size_t calls = 0;
        std::size_t launches = 0;
        op::loop::dag::Graph graph;
        // Of the halos needed before any write in the run, those dirty when
        // their reader was queued, last time: exchanges a prediction could
        // start early.
        std::size_t dirty_before = 0;
        std::vector<bool> halo_dirty;
        // The calls' loop names, for the halo list.
        std::vector<std::string> names;
    };

    std::unordered_map<std::uint64_t, Shape> m_shapes;
    op::loop::dag::Predictor m_predictor;
    std::size_t m_runs = 0;
    std::size_t m_hits = 0;

    static int median(std::vector<int> values) {
        if (values.empty())
            return 0;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    // Whether a halo needed before any write was dirty when its reader was
    // queued.
    static bool dirty_at_reader(const op::loop::dag::Halo& halo, std::span<const RunCall> calls) {
        const auto& reader = calls[static_cast<std::size_t>(halo.reader)];
        for (std::size_t a = 0; a < reader.args.size(); ++a)
            if (reader.args[a].opt != 0 && reader.args[a].dat == halo.dat && reader.dirty[a])
                return true;
        return false;
    }

public:
    // A run that has executed: its signature, the prediction, and its graph
    // the first time.
    void observe(std::span<const RunCall> calls, std::span<const std::string *> names,
                 std::size_t launches) {
        op::loop::dag::Signature signature;
        for (const auto& call : calls)
            signature.add(call.loop, call.set, call.args);
        auto key = signature.value();
        ++m_runs;
        m_hits += m_predictor.observe(key);

        auto& shape = m_shapes[key];
        shape.launches += launches;
        if (shape.runs++ == 0) {
            shape.calls = calls.size();
            for (const auto *name : names)
                shape.names.push_back(*name);
            std::vector<op::loop::dag::Node> nodes;
            nodes.reserve(calls.size());
            for (const auto& call : calls) {
                auto& node = nodes.emplace_back();
                node.set = call.set;
                node.atomic = call.atomic;
                node.fusable = call.fusable;
                node.uses = op::loop::fusion::uses(call.args);
                for (const auto& arg : call.args)
                    node.exec |= arg.opt != 0 && arg.argtype == OP_ARG_DAT &&
                                 arg.map != nullptr && arg.acc != OP_READ;
            }
            shape.graph = op::loop::dag::build(nodes);
        }
        shape.dirty_before = 0;
        shape.halo_dirty.clear();
        for (const auto& halo : shape.graph.halos) {
            bool dirty = halo.before_run() && dirty_at_reader(halo, calls);
            shape.halo_dirty.push_back(dirty);
            shape.dirty_before += dirty;
        }
    }

    void report() {
        if (m_runs == 0)
            return;
        op_printf("dag: %zu runs, %zu shapes, next run predicted %.1f%%\n", m_runs,
                  m_shapes.size(), 100.0 * static_cast<double>(m_hits) / static_cast<double>(m_runs));
        std::vector<std::pair<std::uint64_t, const Shape *>> shapes;
        for (const auto& [signature, shape] : m_shapes)
            shapes.emplace_back(signature, &shape);
        std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
            return a.second->runs * a.second->calls > b.second->runs * b.second->calls;
        });
        for (std::size_t s = 0; s < std::min<std::size_t>(shapes.size(), 8); ++s) {
            const auto& [signature, shape] = shapes[s];
            const auto& graph = shape->graph;
            std::vector<int> slack_calls, slack_levels, waves, readers;
            std::size_t before = 0;
            for (const auto& halo : graph.halos) {
                if (halo.before_run()) {
                    ++before;
                    continue;
                }
                int writer = op::loop::dag::last_writer(halo);
                slack_calls.push_back(halo.reader - writer - 1);
                slack_levels.push_back(graph.level[static_cast<std::size_t>(halo.reader)] -
                                       graph.level[static_cast<std::size_t>(writer)] - 1);
                waves.push_back(graph.level[static_cast<std::size_t>(writer)]);
                readers.push_back(halo.reader);
            }
            std::sort(waves.begin(), waves.end());
            auto wave_count = std::unique(waves.begin(), waves.end()) - waves.begin();
            std::sort(readers.begin(), readers.end());
            auto reader_count = std::unique(readers.begin(), readers.end()) - readers.begin();
            op_printf("dag: shape %016llx x%zu: %zu calls, %.1f launches now, %d levels, %d "
                      "launches reordered; %zu halos in the run (slack median %d calls, %d "
                      "levels; %td waves, %td exchanges eager), %zu before it (%zu dirty)\n",
                      static_cast<unsigned long long>(signature), shape->runs, shape->calls,
                      static_cast<double>(shape->launches) / static_cast<double>(shape->runs),
                      graph.levels, graph.launches, slack_calls.size(), median(slack_calls),
                      median(slack_levels), wave_count, reader_count, before,
                      shape->dirty_before);
            if (OP_diags > 2)
                for (std::size_t h = 0; h < graph.halos.size(); ++h) {
                    const auto& halo = graph.halos[h];
                    auto level = [&](int c) { return graph.level[static_cast<std::size_t>(c)]; };
                    int writer = op::loop::dag::last_writer(halo);
                    op_printf("dag:   %s%s: %s (call %d, level %d) -> %s (call %d, level %d)\n",
                              halo.dat->name, shape->halo_dirty[h] ? " dirty" : "",
                              writer < 0 ? "before the run" : shape->names[static_cast<std::size_t>(writer)].c_str(),
                              writer, writer < 0 ? -1 : level(writer),
                              shape->names[static_cast<std::size_t>(halo.reader)].c_str(), halo.reader,
                              level(halo.reader));
                }
        }
        m_shapes.clear();
        m_runs = m_hits = 0;
    }
};

} // namespace op::gpu

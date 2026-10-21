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
            std::size_t slack = 0, slack_calls_total = 0;
            for (int calls : slack_calls)
                if (calls >= 2) {
                    ++slack;
                    slack_calls_total += static_cast<std::size_t>(calls);
                }
            std::sort(waves.begin(), waves.end());
            auto wave_count = std::unique(waves.begin(), waves.end()) - waves.begin();
            std::sort(readers.begin(), readers.end());
            auto reader_count = std::unique(readers.begin(), readers.end()) - readers.begin();
            op_printf("dag: shape %016llx x%zu: %zu calls, %.1f launches now, %d levels, %d "
                      "launches reordered; %zu halos in the run (slack median %d calls, %d "
                      "levels; %zu with 2 or more calls between writer and reader, %.1f on "
                      "average; %td waves, %td exchanges eager), %zu before it (%zu dirty)\n",
                      static_cast<unsigned long long>(signature), shape->runs, shape->calls,
                      static_cast<double>(shape->launches) / static_cast<double>(shape->runs),
                      graph.levels, graph.launches, slack_calls.size(), median(slack_calls),
                      median(slack_levels), slack,
                      slack > 0 ? static_cast<double>(slack_calls_total) / slack : 0.0,
                      wave_count, reader_count, before, shape->dirty_before);
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

namespace op::gpu {

// Halo exchanges started early, from the shape of the global segment the
// queue is in: the calls between two flushes for a global, which every rank
// sees alike whatever local flushes split them.  Once a segment's shape has
// been seen, and while the calls so far match the one predicted, each
// version of a dat that a later call reads the halo of is exchanged as soon
// as its last writer has launched; at a global flush, the halos the next
// segment reads before writing them are exchanged straight away.  Every rank
// starts the same exchanges, as their segments and predictions are alike.
class EarlyExchanges {
private:
    struct Halo {
        int writer = -1;
        int reader = 0;
        op_dat dat = nullptr;
    };

    struct Segment {
        // The signature after each call, to follow a prediction.
        std::vector<std::uint64_t> prefix;
        // Halos written in the segment, by writer, and those read before.
        std::vector<Halo> halos;
        std::vector<op_dat> before;
    };

    std::unordered_map<std::uint64_t, Segment> m_segments;
    op::loop::dag::Predictor m_predictor;

    // The segment under way.
    op::loop::dag::Signature m_signature;
    std::vector<std::uint64_t> m_prefix;
    std::vector<op::loop::dag::Node> m_nodes;
    const Segment *m_predicted = nullptr;
    bool m_on_track = false;
    std::size_t m_cursor = 0;

    static Segment shape(std::vector<std::uint64_t> prefix,
                         std::span<const op::loop::dag::Node> nodes) {
        Segment segment;
        segment.prefix = std::move(prefix);
        auto graph = op::loop::dag::build(nodes);
        // Launched in program order: the exchange waits for the latest
        // writer by position, whatever the levels.
        for (const auto& halo : graph.halos) {
            if (halo.before_run())
                segment.before.push_back(halo.dat);
            else
                segment.halos.push_back(
                    {*std::max_element(halo.writers.begin(), halo.writers.end()), halo.reader,
                     halo.dat});
        }
        std::sort(segment.halos.begin(), segment.halos.end(),
                  [](const Halo& a, const Halo& b) { return a.writer < b.writer; });
        return segment;
    }

public:
    // The next call of the segment.
    void push(const RunCall& call) {
        m_signature.add(call.loop, call.set, call.args);
        m_prefix.push_back(m_signature.value());
        auto& node = m_nodes.emplace_back();
        node.set = call.set;
        node.atomic = call.atomic;
        node.fusable = call.fusable;
        node.uses = op::loop::fusion::uses(call.args);
        for (const auto& arg : call.args)
            node.exec |= arg.opt != 0 && arg.argtype == OP_ARG_DAT && arg.map != nullptr &&
                         arg.acc != OP_READ;
        auto i = m_prefix.size() - 1;
        m_on_track = m_on_track && m_predicted != nullptr && i < m_predicted->prefix.size() &&
                     m_predicted->prefix[i] == m_prefix.back();
    }

    // The segment's calls before launched have been launched: start the
    // exchanges whose last writer is among them and whose reader is not.
    template<typename Start>
    void launched(std::size_t launched, Start&& start) {
        if (!m_on_track)
            return;
        const auto& halos = m_predicted->halos;
        while (m_cursor < halos.size() &&
               static_cast<std::size_t>(halos[m_cursor].writer) < launched) {
            if (static_cast<std::size_t>(halos[m_cursor].reader) >= launched)
                start(halos[m_cursor].dat);
            ++m_cursor;
        }
    }

    std::size_t calls() const { return m_prefix.size(); }

    // A global flush ends the segment: record its shape, predict the next,
    // and start the exchanges it needs first.
    template<typename Start>
    void end(Start&& start) {
        auto key = m_signature.value();
        if (!m_prefix.empty() && m_segments.find(key) == m_segments.end())
            m_segments.emplace(key, shape(m_prefix, m_nodes));
        m_predictor.observe(key);
        std::uint64_t next = 0;
        m_predicted = nullptr;
        if (m_predictor.predict(key, next)) {
            auto found = m_segments.find(next);
            if (found != m_segments.end())
                m_predicted = &found->second;
        }
        m_signature = {};
        m_prefix.clear();
        m_nodes.clear();
        m_cursor = 0;
        m_on_track = m_predicted != nullptr;
        if (m_predicted != nullptr)
            for (op_dat dat : m_predicted->before)
                start(dat);
    }
};

} // namespace op::gpu

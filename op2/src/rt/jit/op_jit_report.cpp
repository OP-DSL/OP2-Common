#include <op2/rt/jit/report.h>

#include <op2/rt/report.h>

#include <algorithm>
#include <string>
#include <vector>

namespace op::rt::jit {

using report::appendf;
using report::counted;
using report::listed;
using report::table;

namespace {

// A loop's specialised kernels, and its configurations' modes: for each
// parameter that may move, in how many configurations it is baked in, at
// run time, and moved.
struct LoopJit {
    struct Counts {
        const Param *param;
        std::size_t baked = 0;
        std::size_t runtime = 0;
        std::size_t moved = 0;
    };

    std::string_view name;
    loop::BuildCounts builds;
    std::size_t configurations = 0;
    std::size_t settled = 0;
    std::vector<Counts> params{};

    // Something to list: a parameter at run time or moved, or a failed
    // build.
    bool notable() const {
        if (builds.failed > 0)
            return true;
        return std::any_of(params.begin(), params.end(), [](const auto& counts) {
            return counts.runtime > 0 || counts.moved > 0;
        });
    }
};

LoopJit loop_jit(const JitLoop& loop) {
    LoopJit jit{loop.name, loop.builds};
    for (const auto& configuration : loop.params->configurations()) {
        ++jit.configurations;
        jit.settled += configuration.calls >= statistics_window;
        for (const auto& outcome : configuration.params) {
            auto found = std::find_if(jit.params.begin(), jit.params.end(),
                                      [&](const auto& counts) {
                                          return counts.param == outcome.param;
                                      });
            if (found == jit.params.end())
                found = jit.params.insert(found, LoopJit::Counts{outcome.param});
            ++(outcome.mode == ParamMode::baked ? found->baked : found->runtime);
            found->moved += outcome.moved;
        }
    }
    return jit;
}

} // namespace

std::string summary(std::span<const JitLoop> loops, int ranks) {
    if (!options().enabled() || options().report_file.empty())
        return {};
    std::vector<LoopJit> jits;
    loop::BuildCounts builds;
    std::size_t configurations = 0, settled = 0, moving = 0;
    for (const auto& loop : loops) {
        auto jit = loop_jit(loop);
        auto total = jit.builds.built + jit.builds.failed + jit.builds.building +
                     jit.builds.waiting;
        if (total == 0 && jit.configurations == 0)
            continue;
        builds += jit.builds;
        configurations += jit.configurations;
        settled += jit.settled;
        moving += !jit.params.empty();
        jits.push_back(std::move(jit));
    }
    if (jits.empty())
        return {};

    std::string out = "JIT summary";
    if (ranks > 1)
        appendf(out, " (rank 0 of %d)", ranks);
    out += ":\n";
    appendf(out, "  %s built for %s", counted(builds.built, "kernel").c_str(),
            counted(jits.size(), "loop").c_str());
    if (builds.failed > 0)
        appendf(out, ", %zu failed", builds.failed);
    if (builds.building > 0)
        appendf(out, ", %zu still building", builds.building);
    if (builds.waiting > 0)
        appendf(out, ", %zu not built (asked for fewer than %d times)", builds.waiting,
                options().min_calls);
    out += "\n";
    if (moving == 0)
        return out;
    appendf(out, "  %s with parameters that may move: %s, %zu settled (observed for "
                 "%llu calls)\n",
            counted(moving, "loop").c_str(),
            counted(configurations, "configuration").c_str(), settled,
            static_cast<unsigned long long>(statistics_window));

    auto end = std::partition(jits.begin(), jits.end(),
                              [](const auto& jit) { return jit.notable(); });
    std::sort(jits.begin(), end, [](const auto& a, const auto& b) {
        return a.builds.built > b.builds.built;
    });
    std::size_t notable = static_cast<std::size_t>(end - jits.begin());
    if (notable > 0) {
        out += "  each parameter at run time or moved, in how many configurations it is "
               "baked in/at run time (moved):\n";
        std::vector<std::vector<std::string>> rows{
            {"loop", "kernels", "configurations", "settled", "parameters"}};
        for (std::size_t i = 0; i < std::min(notable, listed); ++i) {
            const auto& jit = jits[i];
            std::string kernels = std::to_string(jit.builds.built);
            if (jit.builds.failed > 0)
                appendf(kernels, " (%zu failed)", jit.builds.failed);
            std::string params;
            for (const auto& counts : jit.params) {
                if (counts.runtime == 0 && counts.moved == 0)
                    continue;
                params += params.empty() ? "" : ", ";
                appendf(params, "%s %zu/%zu", counts.param->name(), counts.baked,
                        counts.runtime);
                if (counts.moved > 0)
                    appendf(params, " (%zu)", counts.moved);
            }
            rows.push_back({std::string{jit.name}, kernels, std::to_string(jit.configurations),
                            std::to_string(jit.settled), params});
        }
        out += table(rows, {false, true, true, true, false});
        if (notable > listed)
            appendf(out, "  %s more with a parameter at run time or moved, or a failed "
                         "build\n",
                    counted(notable - listed, "loop").c_str());
    }
    std::size_t baked = 0;
    for (auto jit = end; jit != jits.end(); ++jit)
        baked += !jit->params.empty();
    if (baked > 0)
        appendf(out, "  %s kept every parameter that may move baked in\n",
                counted(baked, "loop").c_str());
    return out;
}

void rows(const JitLoop& loop, std::vector<std::string>& out) {
    for (const auto& configuration : loop.params->configurations()) {
        std::string baked, runtime, moved;
        for (const auto& outcome : configuration.params) {
            auto& list = outcome.mode == ParamMode::baked ? baked : runtime;
            list += list.empty() ? "" : ";";
            list += outcome.param->name();
            if (outcome.moved) {
                moved += moved.empty() ? "" : ";";
                moved += outcome.param->name();
            }
        }
        out.push_back(std::string{loop.name} + "," +
                      report::quoted(configuration.description) + "," +
                      std::to_string(configuration.calls) + "," +
                      (configuration.calls >= statistics_window ? "yes" : "no") + "," +
                      baked + "," + runtime + "," + moved);
    }
}

} // namespace op::rt::jit

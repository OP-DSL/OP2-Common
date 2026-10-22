#include <loop/strategy.h>

#include <loop/autotune.h>
#include <loop/jit.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <utility>

namespace op::loop {

// --- Environment variables ---------------------------------------------------

namespace env {

std::string lower(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

std::optional<bool> switch_value(std::string_view text) {
    auto lowered = lower(text);
    if (lowered == "on" || lowered == "1" || lowered == "yes" || lowered == "true")
        return true;
    if (lowered == "off" || lowered == "0" || lowered == "no" || lowered == "false")
        return false;
    return std::nullopt;
}

bool parse_bool(const char *variable, bool& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return false;

    auto parsed = switch_value(text);
    if (!parsed) {
        std::fprintf(stderr, "warning: ignoring %s='%s', expected on or off\n",
                     variable, text);
        return false;
    }
    value = *parsed;
    return true;
}

bool parse_number(const char *variable, double& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return false;

    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || parsed < 0.0) {
        std::fprintf(stderr,
                     "warning: ignoring %s='%s', expected a non-negative "
                     "number\n",
                     variable, text);
        return false;
    }
    value = parsed;
    return true;
}

bool parse_number(const char *variable, int& value) {
    double parsed = value;
    if (!parse_number(variable, parsed))
        return false;
    value = static_cast<int>(parsed);
    return true;
}

} // namespace env

// --- One call -----------------------------------------------------------------

bool Configuration::global_reduction() const {
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_GBL)
            continue;
        if (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX)
            return true;
    }
    return false;
}

std::uint64_t Configuration::key() const {
    if (m_keyed)
        return m_key;

    // FNV-1a over the words that shape the call.
    std::uint64_t hash = 14695981039346656037ull;
    auto mix = [&](std::uint64_t word) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= (word >> (8 * byte)) & 0xff;
            hash *= 1099511628211ull;
        }
    };

    mix(static_cast<std::uint64_t>(set->index));
    mix(static_cast<std::uint64_t>(run_size));
    for (const op_arg& arg : args) {
        mix(static_cast<std::uint64_t>(arg.opt));
        if (arg.opt == 0)
            continue;
        mix(static_cast<std::uint64_t>(arg.argtype));
        mix(static_cast<std::uint64_t>(arg.dat != nullptr ? arg.dat->index : -1));
        mix(static_cast<std::uint64_t>(arg.map != nullptr ? arg.map->index : -1));
        mix(static_cast<std::uint64_t>(arg.idx));
        mix(static_cast<std::uint64_t>(arg.acc));
        mix(static_cast<std::uint64_t>(arg.dim));
    }

    m_key = hash;
    m_keyed = true;
    return hash;
}

std::string Configuration::describe() const {
    std::string text = set->name != nullptr ? set->name : "?";
    text += " (" + std::to_string(set->size) + ")";

    std::string maps;
    std::string inactive;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const op_arg& arg = args[i];
        if (arg.opt == 0) {
            inactive += (inactive.empty() ? "" : " ") + std::to_string(i);
            continue;
        }
        if (arg.map == nullptr || arg.map->name == nullptr)
            continue;
        std::string name = arg.map->name;
        if (maps.find(name) == std::string::npos)
            maps += (maps.empty() ? "" : ", ") + name;
    }
    if (!maps.empty())
        text += " via " + maps;
    if (global_reduction())
        text += ", reduction";
    if (!inactive.empty())
        text += ", inactive args " + inactive;
    return text;
}

Sections Sections::direct(op_set set) {
    Sections sections;
    sections.m_sections[0] = {0, static_cast<int>(set->size)};
    sections.m_count = 1;
    sections.m_set_stride = (set->size + 31) & ~31;
    return sections;
}

Sections Sections::indirect(op_set set, bool separate_owned, int run_size) {
    assert(run_size >= set->size && run_size <= set->size + set->exec_size);
    Sections sections;
    int size = static_cast<int>(set->size);
    int end = run_size;
    if (separate_owned) {
        sections.m_sections = {{{0, set->core_size},
                                {set->core_size, size},
                                {size, end}}};
        sections.m_count = 3;
    } else {
        sections.m_sections = {{{0, set->core_size}, {set->core_size, end}}};
        sections.m_count = 2;
    }
    sections.m_wait_before = 1;
    sections.m_globals_after = 1;
    sections.m_set_stride = indirect_set_stride(set);
    return sections;
}

Sections Sections::of(const Configuration& configuration, bool direct) {
    if (direct)
        return Sections::direct(configuration.set);
    return Sections::indirect(configuration.set, configuration.global_reduction(),
                              configuration.run_size);
}

ScalarType scalar_type(const char *type) {
    auto is = [type](std::initializer_list<const char *> names) {
        return type != nullptr &&
               std::any_of(names.begin(), names.end(), [type](const char *name) {
                   return std::strcmp(type, name) == 0;
               });
    };
    if (is({"double", "r8", "real*8", "real(8)"}))
        return ScalarType::f64;
    if (is({"float", "r4", "real*4", "real(4)"}))
        return ScalarType::f32;
    if (is({"int", "i4", "integer*4", "integer(4)"}))
        return ScalarType::i32;
    if (is({"bool", "logical"}))
        return ScalarType::logical;
    return ScalarType::unknown;
}

void reduce_mpi_globals(op_arg *args, int nargs, const char *loop) {
    for (int i = 0; i < nargs; ++i) {
        auto& arg = args[i];
        if (arg.opt == 0 || arg.argtype != OP_ARG_GBL ||
            (arg.acc != OP_INC && arg.acc != OP_MIN && arg.acc != OP_MAX))
            continue;

        switch (scalar_type(arg.type)) {
        case ScalarType::f64:
            op_mpi_reduce_double(&arg, reinterpret_cast<double *>(arg.data));
            break;
        case ScalarType::f32:
            op_mpi_reduce_float(&arg, reinterpret_cast<float *>(arg.data));
            break;
        case ScalarType::i32:
            op_mpi_reduce_int(&arg, reinterpret_cast<int *>(arg.data));
            break;
        case ScalarType::logical:
            op_mpi_reduce_bool(&arg, reinterpret_cast<bool *>(arg.data));
            break;
        case ScalarType::unknown:
            std::fprintf(stderr, "error: unsupported MPI reduction type '%s' (in %s)\n",
                         arg.type == nullptr ? "<null>" : arg.type, loop);
            std::exit(1);
        }
    }
}

// --- Strategies ---------------------------------------------------------------

std::string_view Strategy::family() const {
    auto text = name();
    return text.substr(0, text.find('='));
}

namespace {

// OP_STRATEGY's items, as init read them.
std::vector<std::string> strategy_items;

// The families forced on a loop, most specific first: "<loop>:<family>"
// forces that loop, a bare family every loop that has it.
std::vector<std::string> forced_for(std::string_view loop) {
    std::vector<std::string> specific;
    std::vector<std::string> every;
    for (std::string_view item : strategy_items) {
        if (item.size() > loop.size() && item.substr(0, loop.size()) == loop &&
            item[loop.size()] == ':')
            specific.emplace_back(item.substr(loop.size() + 1));
        else if (item.find(':') == std::string_view::npos)
            every.emplace_back(item);
    }
    specific.insert(specific.end(), every.begin(), every.end());
    return specific;
}

std::string upper(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return out;
}

// Print a warning once per run, however many loops meet it.
void warn_once(const std::string& message) {
    static std::vector<std::string> warned;
    if (std::find(warned.begin(), warned.end(), message) != warned.end())
        return;
    warned.push_back(message);
    std::fprintf(stderr, "%s", message.c_str());
}

// An axis's values from one variable, in order without repeats; empty when
// the variable is unset or holds nothing usable.
std::vector<int> axis_values(const std::string& variable, const Axis& axis) {
    std::vector<int> values;
    const char *text = std::getenv(variable.c_str());
    std::string_view rest = text != nullptr ? text : "";
    while (!rest.empty()) {
        auto comma = rest.find(',');
        auto item = std::string{rest.substr(0, comma)};
        rest = comma == std::string_view::npos ? std::string_view{}
                                               : rest.substr(comma + 1);
        int value = -1;
        if (axis.binary) {
            if (auto parsed = env::switch_value(item))
                value = *parsed ? 1 : 0;
        } else {
            char *end = nullptr;
            long parsed = std::strtol(item.c_str(), &end, 10);
            if (end != item.c_str() && *end == '\0' && parsed > 0 && parsed <= INT32_MAX)
                value = static_cast<int>(parsed);
        }
        if (value < 0)
            warn_once("warning: ignoring " + variable + " item '" + item +
                      "', expected " + (axis.binary ? "on or off" : "a positive number") +
                      "\n");
        else if (std::find(values.begin(), values.end(), value) == values.end())
            values.push_back(value);
    }
    return values;
}

// Each axis's values for family: its own variable's, else the axis-wide
// variable's, else the defaults.
std::vector<std::vector<int>> family_values(std::string_view family,
                                            std::span<const Axis> axes) {
    std::vector<std::vector<int>> values;
    for (const auto& axis : axes) {
        auto variable = std::string{axis.variable};
        auto chosen = axis_values("OP_" + upper(family) + "_" + variable, axis);
        if (chosen.empty())
            chosen = axis_values("OP_" + variable, axis);
        if (chosen.empty())
            chosen = axis.defaults;
        assert(!chosen.empty());
        values.push_back(std::move(chosen));
    }
    return values;
}

// Every point of the values, the first values' first, named
// "<family>=on;<key>=<value>;...", a switch's value on or off.
std::vector<Point> points(std::string_view family, std::span<const Axis> axes,
                          const std::vector<std::vector<int>>& values) {
    std::vector<Point> out;
    std::vector<std::size_t> index(axes.size(), 0);
    while (true) {
        Point point{std::string{family} + "=on", {}};
        for (std::size_t i = 0; i < axes.size(); ++i) {
            int value = values[i][index[i]];
            point.values.push_back(value);
            point.name += ";" + std::string{axes[i].key} + "=" +
                          (axes[i].binary ? (value != 0 ? "on" : "off")
                                          : std::to_string(value));
        }
        out.push_back(std::move(point));

        std::size_t axis = 0;
        for (; axis < axes.size(); ++axis) {
            if (++index[axis] < values[axis].size())
                break;
            index[axis] = 0;
        }
        if (axis == axes.size())
            return out;
    }
}

} // namespace

void init() {
    strategy_items.clear();
    const char *text = std::getenv("OP_STRATEGY");
    std::string_view rest = text != nullptr ? text : "";
    while (!rest.empty()) {
        auto comma = rest.find(',');
        auto item = rest.substr(0, comma);
        rest = comma == std::string_view::npos ? std::string_view{}
                                               : rest.substr(comma + 1);
        if (!item.empty())
            strategy_items.emplace_back(item);
    }
    init_autotune_config();
    init_jit_policy();
}

Ladder::Ladder(std::string_view loop_name)
    : m_name{loop_name}, m_forced_names{forced_for(loop_name)} {}

void Ladder::add(std::string_view family, Default default_setting,
                 std::span<const Axis> axes, const Make& make) {
    bool enabled = default_setting == Default::enabled;
    bool off = env::parse_bool(("OP_" + upper(family)).c_str(), enabled) && !enabled;

    auto all = points(family, axes, family_values(family, axes));
    for (std::size_t i = 1; i < all.size(); ++i)
        add_entry(make(all[i]), false, false, off);
    add_entry(make(all[0]), true, enabled, off);
}

void Ladder::add_entry(std::unique_ptr<Strategy> strategy, bool runs,
                       bool enabled, bool off) {
    assert(strategy != nullptr);
    for (const auto& entry : m_entries)
        if (entry.strategy->name() == strategy->name()) {
            std::fprintf(stderr, "error: strategy %s registered twice (in %s)\n",
                         std::string{strategy->name()}.c_str(), m_name.c_str());
            std::exit(1);
        }

    Entry entry{std::move(strategy), runs, enabled, off};
    if (runs)
        for (std::size_t rank = 0;
             rank < std::min(m_forced_rank, m_forced_names.size()); ++rank)
            if (m_forced_names[rank] == entry.strategy->family()) {
                m_forced = m_entries.size();
                m_forced_rank = rank;
                break;
            }
    m_entries.push_back(std::move(entry));
    m_reasons.emplace_back();
}

Prepared Ladder::select(const Configuration& configuration) {
    assert(!m_entries.empty());
    std::fill(m_reasons.begin(), m_reasons.end(), std::string_view{});

    Prepared selection;
    std::size_t index = m_forced.value_or(0);
    for (; index < m_entries.size(); ++index) {
        auto& entry = m_entries[index];
        bool forced = index == m_forced;
        if (!entry.enabled && !forced) {
            // The tuner's variants are no part of the ladder.
            if (entry.runs)
                m_reasons[index] = "disabled";
            continue;
        }
        // The run asked for the forced strategy, so its kernel is waited for.
        selection = entry.strategy->prepare(configuration,
                                            forced ? Urgency::wait : Urgency::later);
        assert(selection.strategy == entry.strategy.get());
        if (selection.status == Status::ready) {
            report(selection, {m_reasons.data(), index});
            return selection;
        }
        m_reasons[index] = selection.reason;
        if (forced)
            warn_forced(selection.reason);
    }

    // The floor must run, so wait for its kernel; without one the loop
    // cannot run at all.
    auto& floor = m_entries.back();
    selection = floor.strategy->prepare(configuration, Urgency::wait);
    assert(selection.strategy == floor.strategy.get());
    if (selection.status != Status::ready) {
        std::fprintf(stderr, "error: %s has nothing to run: %s is %s\n",
                     m_name.c_str(), std::string{floor.strategy->name()}.c_str(),
                     std::string{selection.reason}.c_str());
        std::exit(1);
    }
    report(selection, {m_reasons.data(), m_entries.size() - 1});
    return selection;
}

void Ladder::warn_forced(std::string_view reason) {
    if (std::find(m_warned.begin(), m_warned.end(), reason) != m_warned.end())
        return;

    m_warned.emplace_back(reason);
    std::fprintf(stderr, "warning: %s passes over its forced strategy %s: %s\n",
                 m_name.c_str(), std::string{(*this)[*m_forced].name()}.c_str(),
                 m_warned.back().c_str());
}

void Ladder::report(const Prepared& selection,
                    std::span<const std::string_view> reasons) {
    if (OP_diags <= 3)
        return;

    std::string passed;
    for (std::size_t i = 0; i < reasons.size(); ++i) {
        if (reasons[i].empty())
            continue;
        passed += passed.empty() ? " (" : ", ";
        passed += m_entries[i].strategy->name();
        passed += ": ";
        passed += reasons[i];
    }
    if (!passed.empty())
        passed += ")";
    std::string outcome = std::string{selection.strategy->name()} + passed;
    if (std::find(m_reported.begin(), m_reported.end(), outcome) !=
        m_reported.end())
        return;

    m_reported.push_back(outcome);
    std::printf("strategy: %s runs %s\n", m_name.c_str(), outcome.c_str());
}

// --- Teardown -----------------------------------------------------------------

namespace {

struct Owner {
    void *owner;
    void (*release)(void *owner);
};

std::mutex owners_mutex;
std::vector<Owner> owners;

} // namespace

void register_release(void *owner, void (*release)(void *owner)) {
    assert(owner != nullptr && release != nullptr);
    std::scoped_lock lock(owners_mutex);
    assert(std::none_of(owners.begin(), owners.end(),
                        [owner](const Owner& item) { return item.owner == owner; }));
    owners.push_back({owner, release});
}

void unregister_release(void *owner) {
    std::scoped_lock lock(owners_mutex);
    std::erase_if(owners, [owner](const Owner& item) { return item.owner == owner; });
}

void release_all() {
    std::vector<Owner> current;
    {
        std::scoped_lock lock(owners_mutex);
        current = owners;
    }
    for (const auto& item : current)
        item.release(item.owner);
}

} // namespace op::loop

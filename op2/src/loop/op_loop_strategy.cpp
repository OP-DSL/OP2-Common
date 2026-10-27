#include <loop/strategy.h>

#include <extern/rapidhash.h>
#include <loop/autotune.h>
#include <loop/jit.h>
#include <loop/select.h>
#include <op_env.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <mutex>
#include <utility>

namespace op::loop {

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

    // The words that shape the call, hashed a buffer at a time: per
    // argument its activity, kind, access and map index in one word, then
    // the first argument on its dat, its map and dimension.
    constexpr std::size_t arg_words = 4;
    std::array<std::int32_t, 2 + 32 * arg_words> words;
    std::size_t count = 0;
    std::uint64_t hash = RAPID_SEED;
    words[count++] = set->index;
    words[count++] = run_size;
    auto first_on_dat = [&](std::size_t i) {
        for (std::size_t j = 0; j < i; ++j)
            if (args[j].opt != 0 && args[j].dat == args[i].dat)
                return static_cast<std::int32_t>(j);
        return static_cast<std::int32_t>(i);
    };
    for (std::size_t i = 0; i < args.size(); ++i) {
        const op_arg& arg = args[i];
        if (count + arg_words > words.size()) {
            hash = rapidhash_withSeed(words.data(), count * sizeof(words[0]), hash);
            count = 0;
        }
        if (arg.opt == 0) {
            words[count++] = 0;
            continue;
        }
        words[count++] = 1 | arg.argtype << 1 | arg.acc << 4 | (arg.idx & 0xffff) << 16;
        words[count++] = arg.dat != nullptr ? first_on_dat(i) : -1;
        words[count++] = arg.map != nullptr ? arg.map->index : -1;
        words[count++] = arg.dim;
    }

    m_key = rapidhash_withSeed(words.data(), count * sizeof(words[0]), hash);
    m_keyed = true;
    return m_key;
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

// A declared family's axes, default, OP_<FAMILY>, each axis's values, and
// its place in ladder order.
struct FamilySettings {
    std::vector<Axis> axes;
    bool enabled_by_default;
    std::optional<bool> enabled;
    std::vector<std::vector<int>> values;
    std::size_t rank;
};

std::map<std::string, FamilySettings, std::less<>> family_settings;
std::size_t declared_families = 0;

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

// An axis's values from one variable, in order without repeats; empty when
// the variable is unset or holds nothing usable.
std::vector<int> axis_values(const std::string& variable, const Axis& axis) {
    std::vector<std::string> items;
    env::parse_list(variable.c_str(), items);
    std::vector<int> values;
    for (const auto& item : items) {
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
            env::warn("warning: ignoring " + variable + " item '" + item +
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
    env::parse_list("OP_STRATEGY", strategy_items);
    family_settings.clear();
    declared_families = 0;
    init_autotune_config();
    init_jit_policy();
}

void declare_families(std::span<const Family> families) {
    for (const auto& family : families) {
        FamilySettings settings;
        settings.axes = family.axes;
        settings.enabled_by_default = family.enabled;
        env::parse_bool(("OP_" + upper(family.name)).c_str(), settings.enabled);
        settings.values = family_values(family.name, family.axes);
        settings.rank = declared_families++;
        family_settings.insert_or_assign(std::string{family.name}, std::move(settings));
    }
    for (std::string_view item : strategy_items) {
        auto family = item.substr(item.find(':') + 1);
        if (family_settings.find(family) == family_settings.end())
            env::warn("warning: ignoring OP_STRATEGY item '" + std::string{item} +
                      "', which names no strategy family\n");
    }
}

Ladder::Ladder(std::string_view loop_name)
    : m_name{loop_name}, m_forced_names{forced_for(loop_name)} {}

void Ladder::add(std::string_view family, const Make& make) {
    auto found = family_settings.find(family);
    if (found == family_settings.end()) {
        std::fprintf(stderr, "error: %s registers strategy family %s, which no backend declared\n",
                     m_name.c_str(), std::string{family}.c_str());
        std::exit(1);
    }
    const auto& settings = found->second;
    bool enabled = settings.enabled.value_or(settings.enabled_by_default);
    bool off = !enabled && settings.enabled.has_value();

    auto all = points(family, settings.axes, settings.values);
    for (std::size_t i = 1; i < all.size(); ++i)
        add_entry(make(all[i]), false, false, off, settings.rank);
    add_entry(make(all[0]), true, enabled, off, settings.rank);
}

void Ladder::add_entry(std::unique_ptr<Strategy> strategy, bool runs,
                       bool enabled, bool off, std::size_t rank) {
    assert(strategy != nullptr);
    for (const auto& entry : m_entries)
        if (entry.strategy->name() == strategy->name()) {
            std::fprintf(stderr, "error: strategy %s registered twice (in %s)\n",
                         std::string{strategy->name()}.c_str(), m_name.c_str());
            std::exit(1);
        }

    // After its family's earlier strategies and those of every family
    // declared before it.
    auto position = static_cast<std::size_t>(
        std::find_if(m_entries.begin(), m_entries.end(),
                     [&](const Entry& entry) { return entry.rank > rank; }) -
        m_entries.begin());
    if (m_forced && *m_forced >= position)
        ++*m_forced;

    Entry entry{std::move(strategy), runs, enabled, off, rank};
    if (runs)
        for (std::size_t forced = 0;
             forced < std::min(m_forced_rank, m_forced_names.size()); ++forced)
            if (m_forced_names[forced] == entry.strategy->family()) {
                m_forced = position;
                m_forced_rank = forced;
                break;
            }
    m_entries.insert(m_entries.begin() + static_cast<std::ptrdiff_t>(position),
                     std::move(entry));
    m_reasons.insert(m_reasons.begin() + static_cast<std::ptrdiff_t>(position),
                     std::string_view{});
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
                                            forced ? Urgency::wait : Urgency::later,
                                            choose_kernel(*entry.strategy, configuration));
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
    selection = floor.strategy->prepare(configuration, Urgency::wait,
                                        choose_kernel(*floor.strategy, configuration));
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
    env::warn("warning: " + m_name + " passes over its forced strategy " +
              std::string{(*this)[*m_forced].name()} + ": " + m_warned.back() + "\n");
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
    op_printf("strategy: %s runs %s\n", m_name.c_str(), outcome.c_str());
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

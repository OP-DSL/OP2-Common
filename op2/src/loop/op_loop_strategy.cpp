#include <loop/strategy.h>

#include <loop/autotune.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
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

bool parse_bool(const char *variable, bool& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return false;

    auto lowered = lower(text);
    if (lowered == "1" || lowered == "yes" || lowered == "true")
        value = true;
    else if (lowered == "0" || lowered == "no" || lowered == "false")
        value = false;
    else
        std::fprintf(stderr, "warning: ignoring %s='%s', expected 0 or 1\n",
                     variable, text);
    return true;
}

bool parse_number(const char *variable, double& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return false;

    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    if (end != text && *end == '\0' && parsed >= 0.0)
        value = parsed;
    else
        std::fprintf(stderr,
                     "warning: ignoring %s='%s', expected a non-negative "
                     "number\n",
                     variable, text);
    return true;
}

bool parse_number(const char *variable, int& value) {
    double parsed = value;
    bool set = parse_number(variable, parsed);
    value = static_cast<int>(parsed);
    return set;
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

Sections Sections::indirect(op_set set, bool separate_owned) {
    Sections sections;
    int size = static_cast<int>(set->size);
    int end = size + set->exec_size;
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

// --- Strategies ---------------------------------------------------------------

std::string_view Strategy::family() const {
    auto text = name();
    return text.substr(0, text.find(':'));
}

namespace {

// OP_STRATEGY's items, as init read them.
std::vector<std::string> strategy_items;

// The names forced on a loop, most specific first.  An item "<loop>:<name>"
// forces name on that loop; any other item is a name, which may hold a colon
// ("atomics:256", "atomics:packed"), forced on every loop that has it.
std::vector<std::string> forced_for(std::string_view loop) {
    std::vector<std::string> specific;
    std::vector<std::string> every;
    for (std::string_view item : strategy_items) {
        if (item.size() > loop.size() && item.substr(0, loop.size()) == loop &&
            item[loop.size()] == ':')
            specific.emplace_back(item.substr(loop.size() + 1));
        else
            every.emplace_back(item);
    }
    specific.insert(specific.end(), every.begin(), every.end());
    return specific;
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
}

Ladder::Ladder(std::string_view loop_name)
    : m_name{loop_name}, m_forced_names{forced_for(loop_name)} {}

void Ladder::add(std::unique_ptr<Strategy> strategy, Default default_setting) {
    assert(strategy != nullptr);
    for (const auto& entry : m_entries)
        if (entry.strategy->name() == strategy->name()) {
            std::fprintf(stderr, "error: strategy %s registered twice (in %s)\n",
                         std::string{strategy->name()}.c_str(), m_name.c_str());
            std::exit(1);
        }

    Entry entry{std::move(strategy), default_setting == Default::enabled, false};
    std::string variable = "OP_" + env::lower(entry.strategy->family());
    std::transform(variable.begin(), variable.end(), variable.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    bool enabled = entry.enabled;
    if (env::parse_bool(variable.c_str(), enabled)) {
        entry.enabled = enabled;
        entry.off = !enabled;
    }

    for (std::size_t rank = 0;
         rank < std::min(m_forced_rank, m_forced_names.size()); ++rank)
        if (m_forced_names[rank] == entry.strategy->name()) {
            m_forced = m_entries.size();
            m_forced_rank = rank;
            break;
        }
    m_entries.push_back(std::move(entry));
    m_reasons.emplace_back();
}

Selection Ladder::select(const Configuration& configuration) {
    assert(!m_entries.empty());
    std::fill(m_reasons.begin(), m_reasons.end(), std::string_view{});

    Selection selection;
    std::size_t index = m_forced.value_or(0);
    for (; index < m_entries.size(); ++index) {
        auto& entry = m_entries[index];
        if (!entry.enabled && index != m_forced) {
            m_reasons[index] = "disabled";
            continue;
        }
        selection.prepared = entry.strategy->prepare(configuration, false);
        if (selection.prepared.status == Status::ready) {
            selection.strategy = entry.strategy.get();
            report(selection, {m_reasons.data(), index});
            return selection;
        }
        m_reasons[index] = selection.prepared.reason;
    }

    // The floor, which keeps its own reason when it was passed over.
    auto& floor = m_entries.back();
    if (selection.prepared.strategy != floor.strategy.get())
        selection.prepared = floor.strategy->prepare(configuration, false);
    selection.strategy = floor.strategy.get();
    report(selection, {m_reasons.data(), m_entries.size() - 1});
    return selection;
}

void Ladder::report(const Selection& selection,
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
    if (selection.prepared.status != Status::ready)
        outcome += std::string{" ["} + std::string{selection.prepared.reason} + "]";
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

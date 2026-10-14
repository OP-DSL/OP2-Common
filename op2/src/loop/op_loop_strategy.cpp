#include <loop/strategy.h>

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

// OP_STRATEGY, read when a ladder is built: a strategy forced for every
// loop, and those forced for named loops.
struct Forced {
    std::string every;
    std::vector<std::pair<std::string, std::string>> loops;
};

Forced forced() {
        Forced result;
        const char *text = std::getenv("OP_STRATEGY");
        if (text == nullptr)
            return result;

        std::string_view rest = text;
        while (!rest.empty()) {
            auto comma = rest.find(',');
            auto item = rest.substr(0, comma);
            rest = comma == std::string_view::npos ? std::string_view{}
                                                   : rest.substr(comma + 1);
            if (item.empty())
                continue;
            // "name" or "name:size" for every loop; "loop:name" or
            // "loop:name:size" for one.  A variant's suffix is all digits.
            auto colon = item.find(':');
            auto tail = colon == std::string_view::npos
                            ? std::string_view{}
                            : item.substr(colon + 1);
            bool variant = !tail.empty() &&
                           std::all_of(tail.begin(), tail.end(), [](char c) {
                               return std::isdigit(static_cast<unsigned char>(c));
                           });
            if (colon == std::string_view::npos || variant)
                result.every = item;
            else
                result.loops.emplace_back(std::string{item.substr(0, colon)},
                                          std::string{tail});
        }
        return result;
}

std::optional<std::string> forced_for(std::string_view loop) {
    const auto parsed = forced();
    for (const auto& [name, strategy] : parsed.loops)
        if (name == loop)
            return strategy;
    if (!parsed.every.empty())
        return parsed.every;
    return std::nullopt;
}

} // namespace

Ladder::Ladder(std::string_view loop_name)
    : m_name{loop_name}, m_forced_name{forced_for(loop_name)} {}

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

    if (m_forced_name && *m_forced_name == entry.strategy->name())
        m_forced = m_entries.size();
    m_entries.push_back(std::move(entry));
}

Selection Ladder::select(const Configuration& configuration) {
    assert(!m_entries.empty());
    std::array<std::string_view, 16> reasons{};
    assert(m_entries.size() <= reasons.size());

    Selection selection;
    std::size_t index = m_forced.value_or(0);
    for (; index < m_entries.size(); ++index) {
        auto& entry = m_entries[index];
        if (!entry.enabled && index != m_forced) {
            reasons[index] = "disabled";
            continue;
        }
        selection.prepared = entry.strategy->prepare(configuration, false);
        if (selection.prepared.status == Status::ready) {
            selection.strategy = entry.strategy.get();
            report(selection, {reasons.data(), index});
            return selection;
        }
        reasons[index] = selection.prepared.reason;
    }

    // The floor, which keeps its own reason when it was passed over.
    auto& floor = m_entries.back();
    if (selection.prepared.strategy != floor.strategy.get())
        selection.prepared = floor.strategy->prepare(configuration, false);
    selection.strategy = floor.strategy.get();
    report(selection, {reasons.data(), m_entries.size() - 1});
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

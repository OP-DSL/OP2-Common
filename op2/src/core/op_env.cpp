#include <op_env.h>

#include <op_lib_core.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>

extern char **environ;

namespace op::env {

namespace {

// A recorded variable: its value as read or as the reader's default, whether
// the environment set it, and whether finish prints it.
struct Entry {
    std::string value;
    bool set = false;
    bool shown = true;
};

std::map<std::string, Entry> entries;

void record(const char *variable, std::string value, bool set, bool shown = true) {
    entries[variable] = {std::move(value), set, shown};
}

void ignore(const char *variable, const char *text, const std::string& expected) {
    warn(std::string{"warning: ignoring "} + variable + "='" + text + "', expected " +
         expected + "\n");
}

std::string lower(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

} // namespace

std::optional<bool> switch_value(std::string_view text) {
    auto lowered = lower(text);
    if (lowered == "on" || lowered == "1" || lowered == "yes" || lowered == "true")
        return true;
    if (lowered == "off" || lowered == "0" || lowered == "no" || lowered == "false")
        return false;
    return std::nullopt;
}

bool parse_bool(const char *variable, bool& value) {
    std::optional<bool> parsed;
    bool read = parse_bool(variable, parsed);
    if (read)
        value = *parsed;
    record(variable, value ? "on" : "off", read);
    return read;
}

bool parse_bool(const char *variable, std::optional<bool>& value) {
    const char *text = std::getenv(variable);
    std::optional<bool> parsed;
    if (text != nullptr) {
        parsed = switch_value(text);
        if (!parsed)
            ignore(variable, text, "on or off");
    }
    if (parsed)
        value = parsed;
    record(variable, value ? (*value ? "on" : "off") : "", parsed.has_value());
    return parsed.has_value();
}

bool parse_number(const char *variable, double& value) {
    const char *text = std::getenv(variable);
    bool read = false;
    if (text != nullptr) {
        char *end = nullptr;
        double parsed = std::strtod(text, &end);
        if (end == text || *end != '\0' || parsed < 0.0) {
            ignore(variable, text, "a non-negative number");
        } else {
            value = parsed;
            read = true;
        }
    }
    char shown[32];
    std::snprintf(shown, sizeof shown, "%g", value);
    record(variable, shown, read);
    return read;
}

bool parse_number(const char *variable, std::optional<int>& value, int min, int max) {
    const char *text = std::getenv(variable);
    bool read = false;
    if (text != nullptr) {
        char *end = nullptr;
        errno = 0;
        long parsed = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || errno == ERANGE || parsed < min || parsed > max) {
            ignore(variable, text,
                   max == std::numeric_limits<int>::max()
                       ? "an integer of at least " + std::to_string(min)
                       : "an integer from " + std::to_string(min) + " to " +
                             std::to_string(max));
        } else {
            value = static_cast<int>(parsed);
            read = true;
        }
    }
    record(variable, value ? std::to_string(*value) : "", read);
    return read;
}

bool parse_number(const char *variable, int& value, int min, int max) {
    std::optional<int> parsed = value;
    bool read = parse_number(variable, parsed, min, max);
    value = *parsed;
    return read;
}

bool parse_text(const char *variable, std::string& value) {
    const char *text = std::getenv(variable);
    if (text != nullptr)
        value = text;
    record(variable, value, text != nullptr);
    return text != nullptr;
}

bool parse_list(const char *variable, std::vector<std::string>& value) {
    const char *text = std::getenv(variable);
    if (text != nullptr) {
        value.clear();
        std::string_view rest = text;
        while (!rest.empty()) {
            auto comma = rest.find(',');
            auto item = rest.substr(0, comma);
            rest = comma == std::string_view::npos ? std::string_view{}
                                                   : rest.substr(comma + 1);
            if (!item.empty())
                value.emplace_back(item);
        }
    }
    std::string shown;
    for (const auto& item : value)
        shown += (shown.empty() ? "" : ",") + item;
    record(variable, shown, text != nullptr);
    return text != nullptr;
}

bool parse_choice(const char *variable, std::span<const Choice> choices, int& value) {
    const char *text = std::getenv(variable);
    bool read = false;
    if (text != nullptr) {
        auto lowered = lower(text);
        auto found = std::find_if(choices.begin(), choices.end(), [&](const Choice& choice) {
            return lower(choice.name) == lowered;
        });
        if (found != choices.end()) {
            value = found->value;
            read = true;
        } else {
            std::string expected;
            for (const auto& choice : choices)
                expected += (expected.empty() ? "" : ", ") + std::string{choice.name};
            ignore(variable, text, "one of " + expected);
        }
    }
    auto named = std::find_if(choices.begin(), choices.end(),
                              [&](const Choice& choice) { return choice.value == value; });
    record(variable, named != choices.end() ? std::string{named->name} : "", read);
    return read;
}

void known(const char *variable) { record(variable, "", false, false); }

void retired(const char *variable, const char *instead) {
    if (std::getenv(variable) != nullptr)
        warn(std::string{"warning: "} + variable + " is no longer read; " + instead + "\n");
    record(variable, "", false, false);
}

void warn(const std::string& message) {
    static std::vector<std::string> warned;
    if (std::find(warned.begin(), warned.end(), message) != warned.end())
        return;
    warned.push_back(message);
    if (op_is_root())
        std::fprintf(stderr, "%s", message.c_str());
}

std::vector<std::string> unread() {
    std::vector<std::string> names;
    for (char **entry = environ; entry != nullptr && *entry != nullptr; ++entry) {
        std::string_view text = *entry;
        if (text.substr(0, 3) != "OP_")
            continue;
        std::string name{text.substr(0, text.find('='))};
        if (entries.find(name) == entries.end())
            names.push_back(std::move(name));
    }
    std::sort(names.begin(), names.end());
    return names;
}

void finish() {
    bool display = false;
    parse_bool("OP_DISPLAY_ENV", display);
    for (const auto& name : unread())
        warn("warning: " + name + " is set, but nothing in this build of OP2 reads it\n");
    if (!display || !op_is_root())
        return;

    std::printf("OP2 DISPLAY ENVIRONMENT BEGIN\n");
    for (const auto& [name, entry] : entries) {
        if (!entry.shown)
            continue;
        if (entry.value.empty() && !entry.set)
            std::printf("  %s not set\n", name.c_str());
        else
            std::printf("  %s = '%s'%s\n", name.c_str(), entry.value.c_str(),
                        entry.set ? " (set)" : "");
    }
    std::printf("OP2 DISPLAY ENVIRONMENT END\n");
}

} // namespace op::env

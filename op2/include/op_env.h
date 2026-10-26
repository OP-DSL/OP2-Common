#pragma once

// OP2's environment variables, read through one parser that records each
// variable it is asked for.  op_init has every layer read its own, then
// calls finish, which warns about any OP_ variable nothing read and, under
// OP_DISPLAY_ENV, prints them all.  Command-line arguments stay with
// op_set_args.

#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::env {

std::string lower(std::string_view text);

// A switch's text, on or off (also 1/0, yes/no, true/false); none for
// anything else.
std::optional<bool> switch_value(std::string_view text);

// A name a choice variable takes, matched ignoring case, and its value.
struct Choice {
    std::string_view name;
    int value;
};

// Each reads variable into value when it is set and valid, warns about and
// ignores anything else, and records the variable with the value it leaves;
// whether value was read from it.  In order: a switch, a switch with no
// default, a non-negative number, an integer in [min, max], one with no
// default, any text, comma-separated items (empty ones dropped), and one of
// choices.
bool parse_bool(const char *variable, bool& value);
bool parse_bool(const char *variable, std::optional<bool>& value);
bool parse_number(const char *variable, double& value);
bool parse_number(const char *variable, int& value, int min = 0,
                  int max = std::numeric_limits<int>::max());
bool parse_number(const char *variable, std::optional<int>& value, int min = 0,
                  int max = std::numeric_limits<int>::max());
bool parse_text(const char *variable, std::string& value);
bool parse_list(const char *variable, std::vector<std::string>& value);
bool parse_choice(const char *variable, std::span<const Choice> choices, int& value);

// Record a variable a translator reads, so finish does not warn about it.
void known(const char *variable);

// Record a variable no longer read, warning with what to use instead when it
// is set.
void retired(const char *variable, const char *instead);

// Print a warning once per run, on the root rank.
void warn(const std::string& message);

// The OP_ variables in the environment that nothing recorded.
std::vector<std::string> unread();

// Once op_init has read everything: warn about each unread OP_ variable
// and, when OP_DISPLAY_ENV is on, print every variable recorded.
void finish();

} // namespace op::env

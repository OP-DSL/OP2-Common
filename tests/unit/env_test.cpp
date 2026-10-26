// The environment parser: each kind of variable read, ignored when unusable
// and recorded, and the OP_ variables nothing read.

#include <op_env.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test_support.h"

namespace env = op::env;

namespace {

bool unread(const char *variable) {
    auto names = env::unread();
    return std::find(names.begin(), names.end(), variable) != names.end();
}

// Each parse reads a usable value, keeps the default otherwise, and says
// whether it read.
void test_parse() {
    bool flag = false;
    unsetenv("OP_TEST_FLAG");
    CHECK(!env::parse_bool("OP_TEST_FLAG", flag) && !flag);
    setenv("OP_TEST_FLAG", "Yes", 1);
    CHECK(env::parse_bool("OP_TEST_FLAG", flag) && flag);
    setenv("OP_TEST_FLAG", "maybe", 1);
    CHECK(!env::parse_bool("OP_TEST_FLAG", flag) && flag);

    std::optional<bool> no_default;
    unsetenv("OP_TEST_SWITCH");
    CHECK(!env::parse_bool("OP_TEST_SWITCH", no_default) && !no_default);
    setenv("OP_TEST_SWITCH", "off", 1);
    CHECK(env::parse_bool("OP_TEST_SWITCH", no_default) && no_default == false);

    int count = 7;
    setenv("OP_TEST_COUNT", "2.5", 1);
    CHECK(!env::parse_number("OP_TEST_COUNT", count) && count == 7);
    setenv("OP_TEST_COUNT", "2", 1);
    CHECK(env::parse_number("OP_TEST_COUNT", count) && count == 2);
    for (const char *text : {"-1", "3", "99999999999"}) {
        setenv("OP_TEST_COUNT", text, 1);
        CHECK(!env::parse_number("OP_TEST_COUNT", count, 0, 2) && count == 2);
    }
    std::optional<int> limit;
    unsetenv("OP_TEST_LIMIT");
    CHECK(!env::parse_number("OP_TEST_LIMIT", limit) && !limit);
    setenv("OP_TEST_LIMIT", "4", 1);
    CHECK(env::parse_number("OP_TEST_LIMIT", limit) && limit == 4);
    double fraction = 0.5;
    setenv("OP_TEST_FRACTION", "0.25x", 1);
    CHECK(!env::parse_number("OP_TEST_FRACTION", fraction) && fraction == 0.5);

    std::string text = "default";
    setenv("OP_TEST_TEXT", "a,b", 1);
    CHECK(env::parse_text("OP_TEST_TEXT", text) && text == "a,b");

    std::vector<std::string> items{"kept"};
    unsetenv("OP_TEST_LIST");
    CHECK(!env::parse_list("OP_TEST_LIST", items) && items.size() == 1);
    setenv("OP_TEST_LIST", "a,,b,", 1);
    CHECK(env::parse_list("OP_TEST_LIST", items) &&
          (items == std::vector<std::string>{"a", "b"}));

    constexpr env::Choice modes[] = {{"silent", 1}, {"warn", 2}, {"warning", 2}};
    int mode = 0;
    setenv("OP_TEST_MODE", "WARNING", 1);
    CHECK(env::parse_choice("OP_TEST_MODE", modes, mode) && mode == 2);
    setenv("OP_TEST_MODE", "loud", 1);
    CHECK(!env::parse_choice("OP_TEST_MODE", modes, mode) && mode == 2);
}

// An OP_ variable is unread until something records it: a parse, even of an
// unusable value, or a translator's or a retired variable's name.
void test_unread() {
    setenv("OP_TEST_TYPO", "1", 1);
    setenv("OP_TEST_TRANSLATOR", "1", 1);
    setenv("OP_TEST_RETIRED", "1", 1);
    setenv("NOT_OP_TEST", "1", 1);
    CHECK(unread("OP_TEST_TYPO") && unread("OP_TEST_TRANSLATOR") &&
          unread("OP_TEST_RETIRED") && !unread("NOT_OP_TEST"));
    CHECK(!unread("OP_TEST_FLAG") && !unread("OP_TEST_MODE"));

    env::known("OP_TEST_TRANSLATOR");
    env::retired("OP_TEST_RETIRED", "nothing replaces it");
    CHECK(unread("OP_TEST_TYPO") && !unread("OP_TEST_TRANSLATOR") &&
          !unread("OP_TEST_RETIRED"));

    // finish warns about the typo and, under OP_DISPLAY_ENV, prints the rest.
    setenv("OP_DISPLAY_ENV", "on", 1);
    env::finish();
    CHECK(unread("OP_TEST_TYPO") && !unread("OP_DISPLAY_ENV"));
}

} // namespace

int main() {
    try {
        test_parse();
        test_unread();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "env test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("env tests passed\n");
    return EXIT_SUCCESS;
}

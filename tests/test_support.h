#pragma once

// The unit and runtime tests' check: report the failed condition and its
// line, then exit.

#include <cstdio>
#include <cstdlib>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "check failed: %s at %s:%d\n", #condition,  \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (false)

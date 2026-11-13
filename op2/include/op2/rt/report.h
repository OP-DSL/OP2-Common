#pragma once

// The text the loop runtime's reports share: formatting, counts, CSV
// quoting, tallies and aligned tables.  Defined in src/rt/op_rt_report.cpp.

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::rt::report {

// Append printf's formatting of format.
void appendf(std::string& out, const char *format, ...);

// "1 loop", "2 loops".
std::string counted(std::size_t count, const char *noun);

// A CSV field, quoted.
std::string quoted(std::string_view text);

// Labels and how many configurations each was found for: the label alone
// when there is one, else "2 colouring, 1 atomics jit", most first.
class Tally {
private:
    std::vector<std::pair<std::string, std::size_t>> m_counts;

public:
    void add(const std::string& label);
    std::string text() const;
};

// Rows of cells in columns two apart, indented by two; a column is as wide as
// its widest cell, and those in numeric are right-aligned.
std::string table(const std::vector<std::vector<std::string>>& rows,
                  const std::vector<bool>& numeric);

// Loops listed by name past which the rest are collapsed into one line.
constexpr std::size_t listed = 20;

// Write rows under header to path, ".<rank>" appended when there is more
// than one rank; nothing when path or rows are empty.
void write_csv(const std::string& path, int rank, int ranks, const char *header,
               const std::vector<std::string>& rows);

} // namespace op::rt::report

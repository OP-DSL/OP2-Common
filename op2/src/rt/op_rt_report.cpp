#include <op2/rt/report.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>

namespace op::rt::report {

void appendf(std::string& out, const char *format, ...) {
    char text[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    out += text;
}

std::string counted(std::size_t count, const char *noun) {
    return std::to_string(count) + " " + noun + (count == 1 ? "" : "s");
}

std::string quoted(std::string_view text) {
    std::string out = "\"";
    for (char c : text)
        out += c == '"' ? "\"\"" : std::string{c};
    return out + "\"";
}

void Tally::add(const std::string& label) {
    for (auto& [known, count] : m_counts)
        if (known == label) {
            ++count;
            return;
        }
    m_counts.emplace_back(label, 1);
}

std::string Tally::text() const {
    if (m_counts.size() == 1)
        return m_counts[0].first;
    auto sorted = m_counts;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string out;
    for (const auto& [label, count] : sorted)
        out += (out.empty() ? "" : ", ") + std::to_string(count) + " " + label;
    return out;
}

std::string table(const std::vector<std::vector<std::string>>& rows,
                  const std::vector<bool>& numeric) {
    std::vector<std::size_t> widths(numeric.size(), 0);
    for (const auto& row : rows)
        for (std::size_t i = 0; i < row.size(); ++i)
            widths[i] = std::max(widths[i], row[i].size());
    std::string out;
    for (const auto& row : rows) {
        std::string line;
        for (std::size_t i = 0; i < row.size(); ++i) {
            std::string pad(widths[i] - row[i].size(), ' ');
            line += "  " + (numeric[i] ? pad + row[i] : row[i] + pad);
        }
        line.erase(line.find_last_not_of(' ') + 1);
        out += line + "\n";
    }
    return out;
}

void write_csv(const std::string& path, int rank, int ranks, const char *header,
               const std::vector<std::string>& rows) {
    if (path.empty() || rows.empty())
        return;
    std::string name = path;
    if (ranks > 1)
        name += "." + std::to_string(rank);
    FILE *file = std::fopen(name.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "warning: cannot write report %s\n", name.c_str());
        return;
    }
    std::fprintf(file, "%s\n", header);
    for (const auto& row : rows)
        std::fprintf(file, "%s\n", row.c_str());
    std::fclose(file);
}

} // namespace op::rt::report

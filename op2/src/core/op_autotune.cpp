#include <op_autotune.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace op::f2c {
namespace {

std::mutex report_mutex;
std::vector<std::pair<void *, AutotuneSource>> sources;
std::string header_row;
std::vector<std::string> rows;

} // namespace

void autotune_register_source(void *owner, AutotuneSource add_rows) {
    std::scoped_lock lock(report_mutex);
    sources.emplace_back(owner, add_rows);
}

void autotune_unregister_source(void *owner) {
    std::scoped_lock lock(report_mutex);
    std::erase_if(sources,
                  [owner](const auto& source) { return source.first == owner; });
}

void autotune_record(std::string_view header, std::string row) {
    std::scoped_lock lock(report_mutex);
    if (header_row.empty())
        header_row = header;
    rows.push_back(std::move(row));
}

void autotune_write_report(int rank, int ranks) {
    const char *path = std::getenv("OP_AUTOTUNE_REPORT");
    if (path == nullptr || path[0] == '\0')
        return;

    // The sources call autotune_record, so call them unlocked.
    std::vector<std::pair<void *, AutotuneSource>> current;
    {
        std::scoped_lock lock(report_mutex);
        current = sources;
    }
    for (const auto& [owner, add_rows] : current)
        add_rows(owner);

    std::scoped_lock lock(report_mutex);
    if (rows.empty())
        return;

    std::string name = path;
    if (ranks > 1)
        name += "." + std::to_string(rank);

    FILE *file = std::fopen(name.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "warning: cannot write autotune report %s\n",
                     name.c_str());
        rows.clear();
        return;
    }

    std::fprintf(file, "%s\n", header_row.c_str());
    for (const auto& row : rows)
        std::fprintf(file, "%s\n", row.c_str());
    std::fclose(file);
    rows.clear();
}

} // namespace op::f2c

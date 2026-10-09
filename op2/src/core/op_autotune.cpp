#include <op_autotune.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace op::f2c {
namespace {

std::mutex rows_mutex;
std::string header_row;
std::vector<std::string> rows;

} // namespace

void autotune_record(std::string_view header, std::string row) {
    std::scoped_lock lock(rows_mutex);
    if (header_row.empty())
        header_row = header;
    rows.push_back(std::move(row));
}

void autotune_write_report(int rank, int ranks) {
    std::scoped_lock lock(rows_mutex);
    const char *path = std::getenv("OP_AUTOTUNE_REPORT");
    if (path == nullptr || path[0] == '\0' || rows.empty())
        return;

    std::string name = path;
    if (ranks > 1)
        name += "." + std::to_string(rank);

    FILE *file = std::fopen(name.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "warning: cannot write autotune report %s\n",
                     name.c_str());
        return;
    }

    std::fprintf(file, "%s\n", header_row.c_str());
    for (const auto& row : rows)
        std::fprintf(file, "%s\n", row.c_str());
    std::fclose(file);
    rows.clear();
}

} // namespace op::f2c

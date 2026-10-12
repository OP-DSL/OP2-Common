#pragma once

// Collects strategy autotuning results from every loop and writes them out
// at op_profile_output and op_exit.

#include <string>
#include <string_view>

namespace op::f2c {

// A loop's tuning state, which adds a row per explored configuration with
// autotune_record whenever the report is written.
using AutotuneSource = void (*)(void *owner);
void autotune_register_source(void *owner, AutotuneSource add_rows);
void autotune_unregister_source(void *owner);

// Record one configuration as a CSV row; header names the row's columns and
// is written once, ahead of the first row.
void autotune_record(std::string_view header, std::string row);

// Write every source's rows, as they stand, to OP_AUTOTUNE_REPORT, suffixed
// with the rank when there is more than one.  A later call overwrites the
// file, so applications that never call op_exit still get a report.
void autotune_write_report(int rank, int ranks);

} // namespace op::f2c

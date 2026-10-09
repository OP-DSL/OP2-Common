#pragma once

// Collects strategy autotuning results from every loop and writes them out
// when the backend shuts down.

#include <string>
#include <string_view>

namespace op::f2c {

// Record one finished or abandoned tuning key as a CSV row; header names the
// row's columns and is written once, ahead of the first row.
void autotune_record(std::string_view header, std::string row);

// Write the recorded rows to OP_AUTOTUNE_REPORT, suffixed with the rank when
// there is more than one.  Called at op_exit, after the loops have flushed.
void autotune_write_report(int rank, int ranks);

} // namespace op::f2c

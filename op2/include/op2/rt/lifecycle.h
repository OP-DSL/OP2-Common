#pragma once

// What the core calls of the loop runtime, and all it includes from op2/rt:
// op_init starts the runtime, op_profile_output has it report, and op_exit
// releases it.  From src/rt/op_rt_lifecycle.cpp, init_cpu from
// src/rt/cpu/op_cpu_init.cpp, init_gpu from src/rt/gpu/op_gpu_options.cpp.

namespace op::rt {

// Read the run-wide variables: the ladder's, the tuner's, the JIT's and the
// chunked strategies'; op_init_core calls it before the backend declares its
// families.
void init();

// Declare the CPU families; the CPU libraries' op_init calls it.
void init_cpu();

// Read the GPU options and the device's facts, and declare the GPU families;
// the GPU libraries' op_init calls it once the device is up.
void init_gpu();

// Write the tuner's and the JIT's reports; op_profile_output and op_exit.
void write_reports(int rank, int ranks);

// Write the reports, then release every loop; op_exit, while the device is
// still up.
void finalise(int rank, int ranks);

} // namespace op::rt

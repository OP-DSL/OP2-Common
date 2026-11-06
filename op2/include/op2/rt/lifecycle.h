#pragma once

// What the core calls of the loop runtime, and all it includes from op2/rt:
// op_init starts the runtime, op_profile_output has it report, and op_exit
// releases it.  From src/rt/op_rt_lifecycle.cpp, but for the backends'
// inits.

namespace op::rt {

// Make the run's context (op2/rt/context.h), reading the run-wide
// variables: OP_STRATEGY, the JIT's and the tuner's; op_init_core calls it
// before the backend declares its families.  A later call replaces the
// context.
void init();

// Add the CPU backend's part to the run's context and declare its families;
// the CPU libraries' op_init calls it.  From src/rt/cpu/op_cpu_init.cpp.
void init_cpu();

// Add the GPU backend's part to the run's context, its options and the
// device's facts, and declare its families; the GPU libraries' op_init calls
// it once the device is up.  From src/rt/gpu/op_gpu_init.cpp.
void init_gpu();

// Write the tuner's and the JIT's reports; op_profile_output.
void write_reports(int rank, int ranks);

// Write the reports, then destroy the context, its loops first; op_exit,
// while the device is still up.
void finalise(int rank, int ranks);

} // namespace op::rt

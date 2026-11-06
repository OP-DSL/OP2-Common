# plan: planning building blocks

Hardware-free algorithms each backend composes into its own plans: set
colouring, chunks and their batches, staging slots, wavefronts, plan keys and
sharing.

- Every policy value (thresholds, sizes, limits) arrives as an argument.
  Nothing here reads a variable, holds an option, or knows a device limit or
  encoding: those belong to the backend (`gpu/chunk_plan.h`,
  `cpu/chunk_plan.h`).
- The shared plan registries live in each backend's part of the context.

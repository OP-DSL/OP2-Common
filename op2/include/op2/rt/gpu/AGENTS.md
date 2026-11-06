# gpu: the CUDA and HIP backend

One source for both, through the `gpu*` shims (`op2/gpu/shims.h`) and
`OP2_CUDA`/`OP2_HIP`.  Owns the device side of every strategy: executors,
strategies, chunk plans under the device's limits and encodings, offline and
JIT kernels, symbol uploads, and its part of the context (`gpu::Backend`:
options, device facts, heuristic options, shared plans).

- Every model of what pays on the device lives in `heuristics.h`, the
  defaults the families' axes start from included; the core sees only
  `loop::Estimate`.
- State is reached through the executor (`backend()`, `context()`).
- Host sources compile against ROCm 6.x and 7.x headers (the older HIP API is
  less const-correct); hipRTC serialises builds within a process.

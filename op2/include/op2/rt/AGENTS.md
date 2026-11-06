# The loop runtime (op::rt)

Runs a translated `op_par_loop`: the generated code describes its loop to the
library, which selects a strategy, plans the call and launches the
backend's kernel.  `op2/include/README.md` maps every header.

## Layers

| Layer | What | Includes from op2/rt |
|---|---|---|
| `abi/` | the contract with the generated code: descriptors, parameter rows, launch records; no behaviour | abi |
| `kernel/` | skeletons and hooks compiled with the generated code into op2_kernels, never into a library | abi, kernel |
| `loop/`, `jit/`, `tune/`, `plan/`, the top-level headers | the neutral core: a call, the ladder, families and registry; the JIT's policy, parameters, cache and build pool; the tuner; planning building blocks; the run's context | neutral abi, each other |
| `gpu/`, `cpu/` | the backends: executors, strategies, chunk plans, kernels, options, heuristics, device facts | the neutral layers, their own abi part |

`abi/gpu`, `abi/cpu`, `kernel/gpu` and `kernel/cpu` belong to their backend
alone, and the top-level headers of `abi/` and `kernel/` include neither.
`scripts/check_rt_layering.sh` (ctest `rt_layering`) checks these includes,
the core's (only `lifecycle.h`), device headers, and which files read
variables.

## Rules
- Neutral code holds only what every backend shares and none decides.  A
  backend owns its options, defaults, limits, encodings, heuristics and device
  facts, and gives the core neutral values (`loop::Estimate`, `Prepared`).
- Building blocks take every policy value as an argument.
- Run state lives in `rt::Context` (`context.h`): `op::rt::init` makes it
  (a later call replaces it), `init_cpu`/`init_gpu` add the backend's part,
  `finalise` destroys it.  Loops and their parts take what they read from
  it; no globals.  Each variable is read once, by its owner.
- A run that skips op_exit must still exit cleanly: nothing destroys the
  context at exit.
- A backend extends a neutral type by inheriting it (`abi::gpu::Param :
  abi::Param`), and composes the neutral loop's parts rather than deriving
  from a loop that owns them.
- `op_profile` scopes are not an interface: restructure them freely.

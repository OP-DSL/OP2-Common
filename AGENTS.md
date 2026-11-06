# OP2

An unstructured-mesh DSL: an application's `op_par_loop`s over sets, maps and
dats are translated per target (`translator-v2/`) and run by the library
(`op2/`).

## Layout
- `op2/`: the core library and the runtime (MPI, the CUDA and HIP data
  layer, Fortran bindings).
- `translator-v2/`: the Python translator and its Jinja templates.
- `apps/`, `tests/`, `docs/`.

## Building and testing
- Configure with CMake (`OP2_ENABLE_*`; the tests need `OP2_ENABLE_TESTS`, the
  applications' `OP2_ENABLE_APP_TESTS`).  Build past a failing target
  (`-- -k` under Make): otherwise later test binaries stay stale.
- `ctest -L <label>` or `-R <regex>`; labels unit, functional and apps, plus
  the language and serial or mpi.  GPU tests need a device.
- Check each change by what it can break: the affected targets build without
  warnings, and the tests or runs that reach the change pass.  Full validation
  only at checkpoints and before a merge.

## Conventions
- C++20 throughout: place code where it belongs (an overload beside the API
  it extends), not by worries about older standards.  Lines up to 96 columns.
- Every library header lives under `op2/include`, internal ones included;
  sources are named `op_<dir>_<topic>.cpp`.
- Comments are short: what code is for, or the trap it guards.  Reasoning
  belongs in the commit or the review.
- Tests use the real interfaces: nothing exists only for a test.
- Legacy code keeps its style (manual memory, `op_profile` enter/exit); new
  runtime code owns its state through RAII.
- Small fixes stay at their site: no helper, macro or file just to share a
  few lines, and no parameter only for an error message.  A substantial piece
  written several times does get a helper.
- Environment variables are read through `op::env` (`op2/env.h`) by their
  owner at op_init, never with `getenv` where they are used; op_init warns
  about any `OP_` variable nothing read.
- State that lives for a run is set up and torn down in the op_init and
  op_exit chains, not in function-local statics or reset hooks.
- Per-par_loop host code (dispatch, halo exchanges) is strong-scaling
  critical: no per-call parsing, string work or allocation; resolve facts
  once and fix a cache's lifetime rather than drop the cache.
- Real runs are many MPI ranks: no single-rank specialisations.
- Device code: never raise a kernel's registers for tidiness; compare
  per-kernel registers (sm_90, gfx942) for every kernel a change reaches.
- Commit messages are one concise subject line, at most 72 columns: what changed,
  not why.

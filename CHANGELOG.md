# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

No changes yet.

---

## [v2.0.0] - TBD

Major changes since `v1.1.0` (high-level).

### New translator (`translator-v2`) — primary change

- A new OP2 code-generation engine (`translator-v2/op2-translator`) replaces the legacy translator as the default.
- Based on Python / libclang (C/C++ parsing) / fparser2 (Fortran parsing) / Jinja2 (code templates) — produces improved C/C++ and Fortran backends.
- Supports all current targets: `seq`, `genseq`, `openmp`, `cuda`, `hip`, `c_cuda`, `c_hip` (and `mpi_` variants of each).
- JIT GPU targets (`c_cuda`, `c_hip`) compile device kernels at application start-up via NVRTC / HIP RTC, removing the dependency on `nvcc`/`hipcc` at build time and enabling runtime GPU architecture selection.
- Bundled Python environment and `fpp` preprocessor redistributed inside `translator-v2/` — no separate installation required when using the OP2 Makefiles.
- Legacy translator retained in `translator/c/` and `translator/fortran/` for compatibility; not recommended for new projects.

### New hardware and backend support

- **AMD HIP** support added: HIP shim layer, `hipcc`-compiled AOT kernels, and JIT HIP kernels via HIP RTC (`c_hip` target).
- **NVIDIA Hopper** GPU architecture support (`NV_ARCH=Hopper`).
- **KaHIP** graph partitioner support (`op_partition("KAHIP", "KWAY", ...)`); partitioner can be selected at runtime between ParMETIS and KaHIP.
- **OP_FALLBACK_MODE**: translator emits a fallback sequential kernel for loops it cannot fully translate; a runtime warning is issued when the fallback executes.

### API additions and changes

- **Large-mesh global index support (PR #251)**: two typedef aliases introduced for index types:
  - `idx_l_t` (`int`) — local (per-process) set sizes and mapping-table entries.
  - `idx_g_t` (`long long`) — global element counts and cross-rank offsets.
- **`op_decl_set` signature change**: `size` parameter is now `idx_l_t` (was `idx_g_t`); set sizes are always local.
- **`op_decl_map` signature change**: `imap` parameter is now `idx_l_t *` (was `int *`).
- **New `op_decl_map_long`**: variant of `op_decl_map` accepting `idx_g_t *` for meshes with global element counts exceeding `INT_MAX`; available in C and Fortran.
- `op_get_global_set_offset` return type changed from `int` to `idx_g_t`.
- `SafeLong` debug wrapper (`op2/include/SafeLong.h`, `op2/src/core/SafeLong.cpp`): optional arithmetic overflow/underflow checker for `idx_g_t`, enabled via `-DUSE_SAFELONG`.
- `op_arg_idx` / `op_arg_info` support in C (previously Fortran only); 2-dim map variant added for Fortran.
- `op_profile`: improved timing and instrumentation API (`op2/include/op_profile.h`).
- `op_mpi_probe_halo_index`, `op_force_part`: new MPI utility routines.
- `op_reset_data_ptr` `real(4)` variants added (Fortran).
- `op_get_global_set_offset` and Fortran bindings for `op_mpi_get_data`.
- `OP_WORK` access type: per-thread scratch space, initialised per-element; supported in C and Fortran CUDA backends.
- `OP_JIT_MAX_THREADS`: runtime flag to cap OpenMP thread count for JIT builds.
- `OP_FALLBACK_MODE` runtime flag: controls fallback-to-sequential behaviour when a kernel cannot be executed on the target backend.
- `OP_CUDA_REDUCTIONS_MIB` default raised to 10 MiB.
- Option to disable MPI global reductions for debugging.
- `op_decl_dat` now accepts a `nullptr` data pointer for MPI backends (deferred initialisation).

### Fortran improvements

- Fortran/CUDA JIT backend (`c_cuda`, `c_hip`) via Fortran-to-C conversion layer.
- Fortran `real(4)` dat/arg support throughout.
- Extended Fortran global arg variants (dim-0 globals, runtime-dimension arrays).
- SIMD vectorisation support for Fortran OpenMP indirect loops.
- Fortran CUDA optional global `atomicInc` support.
- Fortran CUDA unknown-dimension reductions (including `op_arg_info`).
- Improved Fortran `op_decl_*` and `op_arg_*` bindings.
- Fortran MPI fixes: grouped halo exchanges, tag wrapping for large MPI universes.
- Timer support added to generated Fortran loop hosts (`c_cuda` and `cuda` Jinja templates): `op_timers` and `op_timing_output` called automatically, enabling per-kernel timing in Fortran applications.
- Functional tests for Fortran strides added alongside existing `const`, `gbl`, and `dat_reductions` functional test suites.
- `op_decl_map_long` Fortran binding added; Fortran reduction example updated to use revised map/set API.

### Example applications

- New `apps/c/jac1/longint/`: Jacobi iteration demonstrating large-mesh support with `idx_g_t`/`idx_l_t` types and `op_decl_map_long` under MPI.
- New `apps/fortran/jac1_long/`: Fortran equivalent of the large-mesh Jacobi demo.

### Build system

- **Persistent configuration**: `make config` detects compilers and libraries once and saves the result; subsequent `make` invocations do not re-scan.
- `make` prints buildable library and app variants at the start of every invocation.
- Intel C/C++ compilers switched from `icc`/`icpc` to `icx`/`icpx`.
- CUDA version auto-detected from `nvcc`/`nvfortran`; `--minimal` flag added automatically for nvcc ≥ 12.4.
- `makefiles/c_app.mk` and `makefiles/f_app.mk` simplified; external app builds now supported (applications outside the OP2-Common tree).
- `VARIANT_FILTER` / `VARIANT_FILTER_OUT` variables allow selective per-app target building.
- `APP_EXTRA_TRANSLATOR_FLAGS` variable added to `c_app.mk` for per-application translator flag overrides.
- HDF5 parallel and sequential builds can coexist; app variants needing HDF5 are gated on the appropriate build.
- JIT HIP Jinja templates converted to symlinks to eliminate duplication.

### Bug fixes

- Fixed `binary_search` bug causing sporadic MPI failures.
- Fixed CUDA MPI Gather/Scatter non-sync race condition affecting GPUDirect builds.
- Fixed GPU Direct gather kernel sync (`Fixed syncing gather kernels when using gpudirect`).
- Fixed import buffer sizes for temporary and overlay dats with partial halo exchanges.
- Fixed `op_arg_idx` parsing edge cases in the translator and direct/indirect index handling in C/C++ backends.
- Fixed C indexing with `OP_AUTO_SOA=1`.
- Fixed `op_par_loop` opt gbl reduction zero-initialisation.
- Fixed vector arg handling in C++ seq and OpenMP backends.
- Fixed temp dat integer overflow for large meshes.
- Fixed seg fault when `nullptr` passed to `op_decl_dat` with SoA enabled.
- MPI tag wrapping to avoid `MPI_TAG_UB` overflow on large runs.
- Fallback to MPI spec maximum tag if `MPI_TAG_UB` attribute is absent on communicator.
- Fortran OpenMP codegen: pass variables for `dim` of `op_arg_dat` to fix runtime-dimension array handling.
- Fortran CUDA runtime: fixed `c_cuda`/`c_hip` compilation issue in the Fortran runtime support layer.
- `numawrap`: fix undefined `PERHOST` variable; generalise NUMA node detection to dynamically query available nodes at runtime.

### Documentation

- Full documentation rewrite on Read the Docs (this release):
  - New Getting Started guide covering manual build and Spack.
  - New Code Generation page explaining the v2 translator and automatic Makefile integration.
  - New Developer Guide covering OP2 internals: execution plans, GPU atomics/color2, MPI halo construction, partitioning, heterogeneous backends.
  - API reference updated with Fortran 90 API section, HDF5 routines, `op_arg_idx`, `op_arg_info`, `OP_WORK`, and all runtime flags.
  - Developer Application tutorial (airfoil walkthrough, Steps 1–7) completed.
  - CUDA 11.8 minimum version and C++17 compiler requirement documented.

### Notes

- The legacy translator (`translator/c/`, `translator/fortran/`) is still present and unchanged.  It is not invoked by the default Makefiles.
- Add short one-line entries under the relevant category for each PR or commit.
- Keep `Unreleased` updated during development; on release, copy `Unreleased` to a new versioned section with the release date, then clear `Unreleased`.
- Link PRs or commits where helpful (e.g. "Fix memory leak (PR #123)").

---

## [v1.1.0] - 2022-01-18

Build system re-write and initial Read the Docs documentation.

### Added

- Unified Makefile system (`makefiles/common.mk`, `makefiles/c_app.mk`, `makefiles/f_app.mk`) replacing per-app ad-hoc Makefiles.
- Compiler profiles under `makefiles/compilers/` and `makefiles/profiles/`.
- Dependency Makefiles for CUDA, HIP, HDF5, ParMETIS, PT-Scotch, KaHIP under `makefiles/dependencies/`.
- Skeleton Read the Docs documentation with Getting Started, API reference, and example application pages.
- Automatic app update pipeline for OP2-APPS repository.
- Unified CUDA architecture selection via `NV_ARCH` (replaces per-compiler flags).

### Changed

- All example applications (`apps/c/`, `apps/fortran/`) migrated to the unified Makefile system.
- README rewritten in Markdown.

---

## [v1.0.0] - 2021-11-10

Initial public release.

### Included

- OP2 C/C++ and Fortran 90 runtime libraries (sequential, OpenMP, CUDA, MPI variants).
- Legacy code translator (`translator/c/`, `translator/fortran/`) targeting seq, OpenMP, CUDA, OpenACC, OpenMP 4.0, and MPI combinations.
- Example applications: airfoil (C and Fortran), aero, jac1, jac2, reduction, bin (C); airfoil, jac1, reduction (Fortran).
- PT-Scotch, ParMETIS, and inertial-bisection mesh partitioners.
- Parallel HDF5 I/O routines.
- Gibbs–Poole–Stockmeyer mesh renumbering via PT-Scotch.

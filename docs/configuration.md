# OP2 CMake configuration reference

Every option, cache variable, and environment hint the OP2 CMake build understands, in one place.

## Build requirements

- **CMake ≥ 3.26.** This is a hard floor, not just a `cmake_minimum_required` formality: CMake's `NVIDIA-CUDA` compiler module only gained a C++20 flag table (`CMAKE_CUDA20_STANDARD_COMPILE_OPTION`) in 3.26 - on 3.25 and earlier, configuring with CUDA enabled fails outright (`requires the language dialect "CUDA20"... CMake does not know the flags to enable it`). If your system package manager is stuck on an older CMake, `pip install cmake` (optionally in a venv) gets you a current one without touching the system install.
- **C++20** throughout. The OP2 library targets declare `cxx_std_20` (and `cuda_std_20` / `hip_std_20`) as `PUBLIC` compile features, so anything linking them is compiled as C++20 too, whatever your project's own default is. For CUDA this additionally needs **nvcc ≥ 12.0** (CMake's C++20 support for CUDA is itself gated on that compiler version).
- **`CMAKE_CUDA_ARCHITECTURES`** defaults to `80;90` (Ampere, Hopper) when neither `-DCMAKE_CUDA_ARCHITECTURES=…` nor the `CUDAARCHS` env var is set. Override for other/older hardware, e.g. `-DCMAKE_CUDA_ARCHITECTURES=70` for Volta, or `-DCMAKE_CUDA_ARCHITECTURES=native` on a machine with the target GPU present. Must be set before `enable_language(CUDA)` is reached (i.e. passed at the initial `cmake -B build` invocation), since CMP0104 otherwise defers to nvcc's own default.

## Feature toggles

Cache options controlling which library variants and pieces of the tree get built. All are `ON` by default unless noted; pass `-D<NAME>=OFF` at configure to disable.

| Option | Default | Effect when `OFF` |
|---|---|---|
| `OP2_ENABLE_MPI` | `ON` | Skip MPI library variants (`op2_mpi`, `op2_mpi_*`, `op2_for_mpi*`). |
| `OP2_ENABLE_CUDA` | `ON` | Skip CUDA library variants; don't `enable_language(CUDA)`. |
| `OP2_ENABLE_HIP` | `ON` | Skip HIP library variants; don't `enable_language(HIP)`. |
| `OP2_ENABLE_OPENMP` | `ON` | Skip the OpenMP variants. The C++ and Fortran sides are independent: OpenMP is probed per language, and having it for only one of them builds only that language's OpenMP variants. |
| `OP2_ENABLE_HDF5` | `ON` | Skip HDF5 discovery; disables `op2_hdf5`, `op2_for_hdf5`, and every MPI variant (which needs parallel HDF5). |
| `OP2_ENABLE_FORTRAN` | `ON` | Skip Fortran library variants and every Fortran-linked app. |
| `OP2_ENABLE_APPS` | `OFF` | Skip building the example apps under `apps/`. |
| `OP2_ENABLE_TESTS` | `OFF` | Skip the functional tests under `tests/` and their `ctest` registration. |
| `OP2_ENABLE_TRANSLATOR` | `ON` | Skip the Python probe. `OP2AppSupport.cmake` becomes unusable (no translator command available); library variants still build. |
| `OP2_ENABLE_INSTALL` | top-level: `ON`<br>subproject: `OFF` | Skip **all** `install()` rules. Defaults to `OFF` automatically when OP2 is consumed via `add_subdirectory()`, on the assumption that the parent project owns installation; set it explicitly either way to override. |

Every feature is *soft-fail*: if the required compiler or dependency is missing, the affected variants silently drop with a status message. Toggling `OFF` is only needed when you want the feature explicitly hidden.

## Dependency hints

OP2 finds its dependencies via CMake's standard mechanisms. You can point the build at your deps in any of the ways below - pick whichever fits your environment. All variables accept both cache-form (`-DFOO=…`) and env-var-form (`export FOO=…`) unless noted.

### Convenience shortcut: `OP2_DEPS_ROOT`

If you built the deps via `scripts/setup_deps.sh` (or `cmake -B deps/build -S cmake/deps`), the standard output layout is:

```
<repo>/deps/
    hdf5/       parmetis/     metis/      ptscotch/     kahip/
```

Pass one variable to pick them all up:

```bash
cmake -B build -DOP2_DEPS_ROOT=/path/to/deps
```

This populates every `*_ROOT` below that hasn't been set explicitly.

Alternatively, `scripts/setup_deps.sh` writes `deps/op2-deps.cmake` - a CMake cache-init file that sets all `*_ROOT` variables. Load it with `-C`:

```bash
cmake -B build -C /path/to/deps/op2-deps.cmake
```

### Per-dependency variables

These steer the **OP2 build**. Each is checked (in order of preference) against: `<Package>_ROOT` (matches the exact `find_package(<Package>)` spelling), Spack-convention `<PKG>_DIR`. All also accept environment-variable form.

They do **not** apply to a project consuming an installed OP2 - see [Downstream usage](#downstream-usage) for why. To change the partitioner an application links, rebuild OP2 against it.

| Dependency | Preferred hint | Also accepted | Discovery mechanism |
|---|---|---|---|
| **HDF5** | `HDF5_ROOT` | (CMake standard) | `find_package(HDF5 CONFIG)` - needs an HDF5 install with `hdf5-config.cmake` |
| **PT-Scotch** | `PTScotch_ROOT` | `SCOTCH_DIR`, `PTSCOTCH_DIR` | `cmake/FindPTScotch.cmake` (bundled) |
| **ParMETIS** | `ParMETIS_ROOT` | `ParMETIS_DIR`, `PARMETIS_DIR` | `cmake/FindParMETIS.cmake` (bundled) |
| **METIS** (required by ParMETIS) | `METIS_ROOT` | `METIS_DIR` | probed by `FindParMETIS` |
| **KaHIP** | `KaHIP_ROOT` | `KAHIP_DIR` | `cmake/FindKaHIP.cmake` (bundled) |

Any of these can be replaced with an entry on `CMAKE_PREFIX_PATH`:

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/opt/hdf5;/opt/scotch;/opt/parmetis"
```

### Module-system users

If your HPC site provides packages via Environment Modules / Lmod, `module load hdf5-parallel` (etc.) typically exports `HDF5_ROOT` or `HDF5_DIR` and adds prefixes to `CMAKE_PREFIX_PATH`. OP2's Find modules honour those, so `cmake -B build` after loading the right modules "just works" without any explicit `-D` on the command line.

## Translator dependency hints

The translator needs Python 3.8+ with `jinja2`, `fparser`, `pcpp`, `sympy`, and `libclang` (importable as `clang.cindex`) available at import time. Python is treated like any other OP2 dependency (MPI, HDF5, ...): CMake finds it, it doesn't provision it. There's no venv, no pip install, and no network access during configure or build - the interpreter `Python3_EXECUTABLE` points at must already have these packages installed, e.g.:

```bash
python3 -m pip install -r translator-v2/requirements.txt
```

If the picked interpreter doesn't satisfy that import check, `OP2_HAS_TRANSLATOR` is set to `FALSE` and a `STATUS` message explains what's missing and how to fix it. This is never fatal. Library variants still build, and so do apps and tests - `seq` variants need no translation at all, so `op2_add_app_variants()` simply produces a narrower set. A missing translator shows up as a smaller variant count in the configuration summary, not a configure error.

| Variable | Default | Effect |
|---|---|---|
| `Python3_EXECUTABLE` | auto | Bypass CMake's Python search; point at a specific interpreter that already has the packages installed. |
| `OP2_PYTHON` (env, legacy make only) | *(unset)* | Overrides the Python that `translator-v2/op2-translator.sh` picks. Used by the legacy `apps/**/Makefile` path. |

## Discovery outputs

Read-only variables OP2's CMake sets, useful downstream via `find_package(OP2 CONFIG)`:

| Variable | Meaning |
|---|---|
| `OP2_HAS_MPI` / `OP2_HAS_CUDA` / `OP2_HAS_HIP` / `OP2_HAS_HDF5` | `TRUE` when the feature is available in this build |
| `OP2_HAS_OPENMP_CXX` / `OP2_HAS_OPENMP_FORTRAN` | `TRUE` when OpenMP was found for that language. There is no unsuffixed `OP2_HAS_OPENMP`: the two are independent, and OP2 has no use for a combined answer |
| `OP2_HAS_HDF5_PARALLEL` | `TRUE` when the linked HDF5 is MPI-parallel |
| `OP2_HAS_PTSCOTCH` / `OP2_HAS_PARMETIS` / `OP2_HAS_KAHIP` | `TRUE` when the partitioner was found |
| `OP2_HAS_TRANSLATOR` | `TRUE` when a suitable Python 3 was found (needed for `op2_add_app_variants`) |
| `OP2_TRANSLATOR_COMMAND` | Two-element list `[Python3_EXECUTABLE; pkg-dir]` - pass unquoted to `add_custom_command(COMMAND …)` |
| `OP2_INCLUDE_DIR` | Plain path to OP2's C/C++ headers (in-tree source dir vs installed `<prefix>/include/op2`) |

## Downstream usage

```cmake
project(my_app LANGUAGES CXX)          # see "Required languages" below
find_package(OP2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE op2::op2_mpi)
# Or, to build variants driven by the translator:
op2_add_app_variants(NAME my_app LANGUAGE cpp SOURCES my_app.cpp)
```

Point CMake at the OP2 install prefix with `-DCMAKE_PREFIX_PATH=/opt/op2`. That is all you need - you do not have to locate OP2's dependencies yourself.

The mesh partitioners (PT-Scotch, ParMETIS/METIS, KaHIP) are **pinned**: OP2 links them by library path, so `install(EXPORT)` records the exact artefacts it was built against in `OP2Targets.cmake`, and none of them is searched for again. This is deliberate. `libop2_mpi.a` is compiled against those specific headers, and the partitioner APIs are not ABI-stable across builds - `scotch.h` sizes its opaque handles per build (`double dummy[N]`) and OP2 stack-allocates a `SCOTCH_Strat` - so binding a different build would corrupt memory rather than fail to link. **To change partitioner, rebuild OP2**; the libraries take seconds to build. If a pinned library has since been deleted or moved, the build stops with that path named, rather than silently binding whatever else happens to be installed.

Everything else - MPI, OpenMP, CUDA, HIP, HDF5 - provides CMake imported targets, so it is **re-found** rather than pinned, and the usual hints apply. But because OP2's libraries are static, your application relinks those same libraries into its own binaries: finding a *different* build of one is what breaks, not finding none. So each re-found dependency **defaults to the one OP2 was built against**:

| Dependency | Default recorded | Overridden by | Suppressed when |
|---|---|---|---|
| MPI | `MPI_CXX_COMPILER`, `MPI_C_COMPILER` (the wrappers) | `MPI_<lang>_COMPILER`, `MPI_HOME` | `MPI_HOME` is set, or the wrapper no longer exists |
| CUDA | `CUDAToolkit_ROOT` | `CUDAToolkit_ROOT`, an enabled `CUDA` language, `CMAKE_CUDA_COMPILER` | the directory no longer exists |
| HIP | `hip_DIR` | `hip_DIR`, `CMAKE_PREFIX_PATH` | the directory no longer exists |
| HDF5 | `HDF5_DIR` | `HDF5_ROOT`, `HDF5_DIR`, `CMAKE_PREFIX_PATH` | the directory no longer exists |
| OpenMP | *(nothing)* | - | always: OpenMP is compiler flags, and your compiler's are the right ones |

These are defaults, not pins - anything you set explicitly wins, and a recorded path that has since moved is ignored so the install stays usable. On a cluster where the same modules are loaded for OP2 and for your application, the defaults simply agree with what discovery would have found anyway; they matter when they would not have. Baking the MPI wrapper is the significant one: it determines every include and library path `FindMPI` derives, and MPI has no cross-implementation ABI, so an OP2 built against Open MPI cannot link against MPICH.

HDF5 is additionally **checked**, because it is the one whose mismatch can be quiet: a wrong MPI leaves undefined references at link time, and a toolkit too old for OP2's CUDA/HIP dialect is rejected at generate time, but OP2 links HDF5 statically and calls version-specific symbols, so a mismatched build can link cleanly and then misbehave. The HDF5 you supply must be **compatible with the version OP2 was built against** (HDF5's own policy - same major.minor series) and **parallel if OP2's is**. Either mismatch fails `find_package(OP2)` with a message saying which.

If you call `find_package(OP2)` somewhere other than your top-level `CMakeLists.txt` - say in one subdirectory, and `op2_add_app_variants()` in a sibling - add the `GLOBAL` keyword:

```cmake
find_package(OP2 CONFIG REQUIRED GLOBAL)
```

Imported targets are directory scoped by default, so without it the `op2::*` targets do not exist in the sibling directory and `op2_add_app_variants()` reports `no buildable variants`. The `OP2_HAS_*` and `OP2_INCLUDE_DIR` variables are cached and need no such treatment. Calling `find_package(OP2)` at the top level, the usual arrangement, is unaffected either way.

### Required languages

Every `op2::*` target is a C++ library - the Fortran ones are a Fortran bridge over the same C++ core - so your project must enable `CXX`. If OP2 was built against **parallel HDF5** you must also enable `C`, because HDF5's own imported targets reference `MPI::MPI_C`. Enabling `Fortran` is only needed if you use the `op2::op2_for_*` targets.

If a required language is missing, `find_package(OP2)` fails with a message naming it rather than something obscure from inside `FindMPI`:

```
OP2 requires the C language, which the consuming project has not enabled.
This OP2 build needs: CXX C.  Add them to your project() call, e.g.
project(myapp LANGUAGES CXX C).
```

### Public helpers

`OP2Config.cmake` brings two functions with it:

| Function | Purpose |
|---|---|
| `op2_add_app_variants(NAME … LANGUAGE … SOURCES …)` | Build `<name>_<variant>` for every variant this environment supports. `VARIANTS` narrows the set, `EXCLUDE_VARIANTS` subtracts from it; asking for a variant the toolchain can't build is not an error, it just doesn't appear. Pass `COMPILE_DEFINITIONS` / `INCLUDE_DIRECTORIES` here rather than setting them on the resulting targets - the translator runs against them at configure time. `TARGETS_VAR <var>` returns the executables created, so you can apply your own requirements to them (see below). |
| `op2_translate(OUT_DIR … LANGUAGE … SOURCES … VARIANTS …)` | The low-level single translator invocation, if you want to drive code generation yourself. |

#### Your app's own dependencies

`op2_add_app_variants()` attaches only what OP2 itself needs: its libraries, and whatever the code the translator generates requires. Anything your own sources pull in is yours to find and link, via the targets `TARGETS_VAR` hands back:

```cmake
find_package(MPI COMPONENTS Fortran REQUIRED)   # our Fortran does `use mpi`

op2_add_app_variants(NAME my_app LANGUAGE fortran SOURCES my_app.F90
                     TARGETS_VAR my_app_targets)

foreach(t IN LISTS my_app_targets)
    target_link_libraries(${t} PRIVATE MPI::MPI_Fortran)
endforeach()
```

The list is empty rather than undefined when nothing was buildable, so the loop is always safe. To keep a variant out entirely when a dependency of yours is missing, pass its name to `EXCLUDE_VARIANTS`.

Note in particular that **OP2 does not find the MPI Fortran bindings** - no OP2 library or generated kernel uses them - so a Fortran app calling MPI directly must do the above.

### The translator on `PATH`

An OP2 install also ships `<prefix>/bin/op2-translator`, a self-locating wrapper that runs the translator with whatever `python3` is on your `PATH`:

```bash
mkdir -p generated                      # -o must already exist
op2-translator -t seq -o generated -I /opt/op2/include/op2 my_app.cpp
```

The install bundles no Python of its own, so that interpreter needs the packages from `requirements.txt` (shipped alongside at `<libexecdir>/op2/translator/requirements.txt`) just as the build-time one does.

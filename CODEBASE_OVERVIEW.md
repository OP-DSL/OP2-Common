# OP2-Common Codebase Overview

> This file is a comprehensive reference for contributors and documentation authors.
> It covers project purpose, directory layout, API, build system, and all major subsystems.

---

## Table of Contents

1. [What is OP2?](#1-what-is-op2)
2. [Repository Layout](#2-repository-layout)
3. [Core Runtime Library](#3-core-runtime-library)
   - [Public Headers](#31-public-headers)
   - [Backend Sources](#32-backend-sources)
4. [Public API Reference](#4-public-api-reference)
5. [Build System](#5-build-system)
   - [Environment Variables](#51-environment-variables)
   - [Building the Libraries](#52-building-the-libraries)
   - [Building an Application](#53-building-an-application)
6. [Parallelization Backends](#6-parallelization-backends)
7. [Translator Tooling](#7-translator-tooling)
   - [Legacy Translator (v1)](#71-legacy-translator-v1)
   - [Next-Generation Translator (v2)](#72-next-generation-translator-v2)
8. [Example Applications](#8-example-applications)
9. [Dependencies](#9-dependencies)
10. [Documentation Sources](#10-documentation-sources)
11. [License and Authors](#11-license-and-authors)

---

## 1. What is OP2?

OP2 is a high-level **embedded domain-specific language (eDSL)** for writing algorithms on **unstructured meshes**, embedded in both C/C++ and Fortran. A user writes a single sequential description of their mesh operations using the OP2 API. The OP2 **code translator** then generates parallelized source-code variants targeting multiple hardware platforms automatically, without requiring the user to write any parallel code directly.

**Supported parallelization targets:**

| Target | Description |
|---|---|
| Sequential | Single-thread reference implementation |
| OpenMP | Shared-memory multi-core CPU |
| CUDA | NVIDIA GPU (CUDA 11.8+) |
| HIP | AMD GPU (ROCm) |
| OpenMP 4.0 | Device offload using OpenMP target directives |
| MPI + any above | Distributed-memory via halo exchange |

The OP2 runtime handles data movement, halo exchanges, mesh partitioning, parallel I/O (HDF5), execution planning (colouring), and performance instrumentation automatically.

OP2 targets production-grade CFD/scientific computing applications such as [Rolls-Royce Hydra](https://www.rolls-royce.com).

**Key publication:** Mudalige et al., *OP2: An Active Library Framework for Solving Unstructured Mesh-based Applications on Multi-Core and Many-Core Architectures*, IEEE InPar 2012. doi: `10.1109/InPar.2012.6339594`

---

## 2. Repository Layout

```
OP2-Common/
├── op2/                    # Core runtime library
│   ├── include/            # Public headers (API)
│   └── src/                # Runtime source code (all backends)
│       ├── core/           # Global state, set/map/dat management, execution plans
│       ├── sequential/     # Sequential backend
│       ├── openmp/         # OpenMP backend
│       ├── openmp4/        # OpenMP 4.0 device-offload backend
│       ├── cuda/           # CUDA backend
│       ├── mpi/            # MPI backend (+ MPI+CUDA, MPI+HIP, HDF5 I/O, partitioning)
│       ├── externlib/      # Bundled external libraries
│       └── fortran/        # Fortran C-interop wrappers
├── apps/                   # Example and benchmark applications
│   ├── c/                  # C/C++ apps: airfoil, aero, jac1, jac2, min, reduction
│   ├── fortran/            # Fortran apps: airfoil, jac1, reduction
│   └── mesh_generators/    # MATLAB mesh generation scripts
├── translator/             # Legacy Python code generators (v1)
│   ├── c/                  # C/C++ generators (seq, openmp, cuda, openacc, mpi_vec, openmp4)
│   └── fortran/            # Fortran generators (mpiseq, openmp, cuda, openacc, openmp4)
├── translator-v2/          # Next-generation translator (Jinja2 + libclang)
│   └── op2-translator/     # Main Python package (CLI, parsers, Jinja templates)
├── makefiles/              # Shared GNU Make infrastructure
│   ├── common.mk           # Main include; orchestrates everything
│   ├── c_app.mk            # Rules for building C/C++ application variants
│   ├── f_app.mk            # Rules for building Fortran application variants
│   ├── compilers.mk        # Compiler selection and flag definitions
│   ├── compilers/          # Per-compiler .mk files (gnu, intel, cray, nvhpc, xl, clang, hip)
│   ├── dependencies/       # Dependency detection .mk files
│   └── profiles/           # Cluster-specific environment profiles
├── docs/                   # Sphinx RST documentation source
├── scripts/                # Pre-commit hooks, compiler environment scripts
├── AUTHORS                 # List of contributors
├── LICENSE                 # BSD 3-Clause
└── README.md               # Top-level overview and quick-start
```

---

## 3. Core Runtime Library

### 3.1 Public Headers (`op2/include/`)

| Header | Purpose |
|---|---|
| `op_lib_core.h` | Core types: `op_set`, `op_map`, `op_dat`, `op_arg`, `op_plan`, `op_access`, … |
| `op_lib_c.h` | C-level API functions — used by all backends and Fortran bindings |
| `op_lib_cpp.h` | C++ template API (`op_decl_dat<T>`, `op_par_loop` with typed overloads) |
| `op_seq.h` | Sequential/MPI-sequential `op_par_loop` inline implementation |
| `op_rt_support.h` | Runtime plan data structures and low-level plan API |
| `op_openmp_rt_support.h` | OpenMP-specific execution plan functions (`op_plan_get`) |
| `op_cuda_rt_support.h` | CUDA runtime functions (device init, memory allocation/transfer) |
| `op_cuda_reduction.h` | GPU reduction templates using shared memory (INC, MAX, MIN, WRITE) |
| `op_gpu_shims.h` | Unified GPU shim layer mapping `gpu*` macros to CUDA or HIP |
| `op_mpi_core.h` | MPI halo data structures: `halo_list`, import/export lists, MPI comms |
| `op_lib_mpi.h` | MPI runtime state: exec/non-exec halo lists, partition tables |
| `op_hdf5.h` | Parallel HDF5 I/O API |
| `op_timing2.h` | Tree-based timing instrumentation (JSON output, 4 detail levels) |
| `op_util.h` | Utility functions |
| `SafeLong.h` | Debug wrapper type `SafeLong` for `idx_g_t` — detects integer overflow/underflow at runtime (enabled via `-DUSE_SAFELONG`) |
| `fortran/` | Fortran C-interop headers |
| `extern/` | Bundled external headers (`json.hpp`, `uthash.h`) |

### 3.2 Backend Sources (`op2/src/`)

| Source Directory | Description |
|---|---|
| `core/op_lib_core.cpp` | Global OP2 state: set/map/dat registration, global declarations |
| `core/op_rt_support.cpp` | Execution plan construction (colouring for indirect loop safety) |
| `sequential/` | Sequential reference backend |
| `openmp/op_openmp_decl.cpp` | OpenMP multi-threaded backend |
| `openmp4/op_openmp4_decl.cpp` | OpenMP 4.0 device-offload backend |
| `cuda/op_cuda_decl.cpp` | CUDA backend declarations |
| `cuda/op_cuda_rt_support.cpp` | CUDA runtime support (memory, device initialization) |
| `mpi/op_mpi_core.cpp` | MPI halo creation and exchange |
| `mpi/op_mpi_decl.cpp` | MPI backend declarations (wraps sequential backend) |
| `mpi/op_mpi_hdf5.cpp` | Parallel HDF5 I/O |
| `mpi/op_mpi_part_core.cpp` | Mesh partitioning (PTScotch, ParMETIS, KaHIP, Inertial) |
| `mpi/op_mpi_cuda_decl.cpp` | MPI+CUDA combined backend |
| `mpi/op_mpi_cuda_kernels.cu` | CUDA kernels for GPU-side halo packing/unpacking |
| `externlib/` | Bundled external libraries |
| `fortran/` | Fortran interop wrapper layers |

---

## 4. Public API Reference

### Initialization and Shutdown

```c
void op_init(int argc, char **argv, int diag_level);
void op_exit(void);
```

- `op_init` initializes OP2 and (under MPI backends) calls `MPI_Init`.
- `diag_level` controls verbosity: `0` = none, `1` = basic setup info, `2` = more detail, `3+` = verbose diagnostic.
- `op_exit` finalizes OP2 and (under MPI backends) calls `MPI_Finalize`.

### Index Types

OP2 defines two typedef aliases to support meshes with more than 2 billion elements globally:

| Type | Underlying type | Usage |
|---|---|---|
| `idx_l_t` | `int` | Local (per-process) set sizes and mapping-table entries |
| `idx_g_t` | `long long` | Global element counts and cross-rank offsets |

A debug-only `SafeLong` wrapper (`op2/include/SafeLong.h`) can be substituted for `idx_g_t` by compiling with `-DUSE_SAFELONG`; it instruments arithmetic operations to detect overflow/underflow at runtime.

### Declaring Mesh Data

```c
op_set  op_decl_set(idx_l_t size, char const *name);
op_map  op_decl_map(op_set from, op_set to, int dim, idx_l_t *imap, char const *name);
op_map  op_decl_map_long(op_set from, op_set to, int dim, idx_g_t *imap, char const *name);
op_dat  op_decl_dat(op_set set, int dim, char const *type, void *data, char const *name);
void    op_decl_const(int dim, char const *type, void *dat);
```

- **sets** represent mesh entities: cells, nodes, edges, boundary faces, etc.
- **maps** describe connectivity between two sets (e.g. cell→node). Use `op_decl_map_long` when global element counts exceed `INT_MAX`.
- **dats** hold data associated with a set (e.g. coordinates on nodes, solution on cells).
- **consts** expose values globally to all kernel functions.

C++ template variant:
```cpp
template <typename T>
op_dat op_decl_dat(op_set set, int dim, char const *type, T *data, char const *name);
```

### Parallel Loop Execution

```c
void op_par_loop(void (*kernel)(...), char const *name, op_set set, op_arg arg0, ...);
```

A parallel loop iterates over all elements of `set`, invoking `kernel` for each element with the provided arguments. The translator rewrites this call into parallelized backend-specific code.

**Loop argument constructors:**

```c
// Access data through a mapping (indirect access) or directly (idx = -1, map = OP_ID)
op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char const *type, op_access acc);

// Global (reduction) argument — same value broadcast to or reduced from all iterations
op_arg op_arg_gbl(void *data, int dim, char const *type, op_access acc);
```

**Access modes:**

| Constant | Meaning |
|---|---|
| `OP_READ` | Read-only access |
| `OP_WRITE` | Write-only (result independent per element) |
| `OP_RW` | Read-write access |
| `OP_INC` | Increment (use for indirect or global accumulation) |

**Special map values:**

| Constant | Meaning |
|---|---|
| `OP_ID` | Identity map — direct access to the iterated set |
| `OP_GBL` | Global (scalar or array reduction) |

### MPI and Partitioning

```c
void op_partition(char const *lib, char const *method,
                  op_set prime_set, op_map prime_map, op_dat coords);
```

`lib` values: `"PTScotch"`, `"ParMETIS"`, `"KaHIP"`, `"INERTIAL"`, `"RANDOM"`, `"EXTERNAL"`

Partitioning methods (ParMETIS): `"KWAY"`, `"GEOM"`, `"GEOMKWAY"`

Call `op_partition` before the first `op_par_loop` when using MPI.

### HDF5 I/O

```c
op_set op_decl_set_hdf5(char const *file, char const *name);
op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file, char const *name);
void   op_dump_to_hdf5(char const *filename);
```

---

## 5. Build System

OP2 uses **GNU Make ≥ 4.2** exclusively. No CMake.

### 5.1 Environment Variables

#### Compiler Selection

| Variable | Description |
|---|---|
| `OP2_COMPILER` | Selects a compiler suite: `gnu`, `cray`, `intel`, `xl`, `nvhpc` |
| `OP2_C_COMPILER` | Override C compiler selection independently |
| `OP2_F_COMPILER` | Override Fortran compiler selection independently |
| `OP2_C_CUDA_COMPILER` | Override CUDA C compiler (e.g. for `nvcc` vs `nvhpc`) |
| `OP2_PROFILE` | Load a cluster profile (e.g. `cirrus-intel-cuda`) |

#### GPU Targets

| Variable | Example | Description |
|---|---|---|
| `CUDA_INSTALL_PATH` | `/usr/local/cuda` | Path to CUDA toolkit |
| `NV_ARCH` | `Pascal,Volta,Ampere` | Comma-separated NVIDIA GPU architectures to target |
| `HIP_INSTALL_PATH` | `/opt/rocm` | Path to ROCm/HIP |
| `HIP_ARCH` | `gfx90a` | AMD GPU architectures to target |

#### Optional Libraries

| Variable | Purpose |
|---|---|
| `PTSCOTCH_INSTALL_PATH` | PT-Scotch mesh partitioner |
| `PARMETIS_INSTALL_PATH` | ParMETIS mesh partitioner |
| `KAHIP_INSTALL_PATH` | KaHIP mesh partitioner |
| `HDF5_SEQ_INSTALL_PATH` | Sequential HDF5 |
| `HDF5_PAR_INSTALL_PATH` | Parallel (MPI) HDF5 |

### 5.2 Building the Libraries

```sh
# 1. Set compiler
export OP2_COMPILER=gnu

# 2. (Optional) Verify compiler and library detection
make -C op2 config

# 3. Build all detected runtime libraries
make -C op2 -j$(nproc)
```

**Library variants built** (subject to available compilers and libraries):

| Library | Required |
|---|---|
| `seq` | C compiler |
| `openmp` | C compiler + OpenMP support |
| `hdf5` | C compiler + sequential HDF5 |
| `openmp4` | C compiler + OpenMP offload support |
| `cuda` | NVCC or NVHPC |
| `hip` | HIP compiler |
| `mpi` | MPI wrappers + parallel HDF5 |
| `mpi_cuda` | MPI + CUDA |
| `mpi_hip` | MPI + HIP |
| `f_*` (all above) | Fortran compiler |

### 5.3 Building an Application

Application build variants are defined in [`makefiles/c_app.mk`](makefiles/c_app.mk) (C/C++) and [`makefiles/f_app.mk`](makefiles/f_app.mk) (Fortran):

```sh
# Build sequential variant of the Airfoil benchmark
make -C apps/c/airfoil/airfoil_plain/dp -j$(nproc)

# Common targets within an app directory
make seq        # Sequential
make openmp     # OpenMP
make cuda       # CUDA
make mpi_seq    # MPI + sequential
make mpi_openmp # MPI + OpenMP
make mpi_cuda   # MPI + CUDA
```

---

## 6. Parallelization Backends

### Sequential
Reference single-threaded implementation. Used for correctness validation. Source: `op2/src/sequential/`.

### OpenMP
Shared-memory CPU parallelization using OpenMP pragmas. Execution plans (coloring of mesh elements) ensure correctness of indirect increments. Source: `op2/src/openmp/`.

### CUDA (NVIDIA GPU)
Requires CUDA 11.8+. Uses `nvcc` or `nvc++` (NVHPC).
- Device initialization and memory management: `op2/src/cuda/op_cuda_rt_support.cpp`
- Reductions via shared-memory templates in `op_cuda_reduction.h`
- Architecture targets: Pascal, Volta, Turing, Ampere, Hopper (set via `NV_ARCH`)
- GPU-unified shim layer: `op_gpu_shims.h` (maps `gpu*` → CUDA or HIP symbols)

### HIP (AMD GPU)
Parallel to CUDA, targeting AMD ROCm. Enable by setting `HIP_INSTALL_PATH`.
- Define `OP2_HIP` to activate HIP shims in `op_gpu_shims.h`.
- Target architectures set via `HIP_ARCH` (e.g. `gfx90a`).

### OpenMP 4.0 Device Offload
For compilers with OpenMP target directive support (NVIDIA, AMD, Intel). Source: `op2/src/openmp4/`.

### MPI (Distributed Memory)
All backends have an MPI variant (e.g. `mpi_seq`, `mpi_openmp`, `mpi_cuda`).

**Halo exchange model:**
- Exec-halo elements (EEH/IEH): mesh elements on the boundary between MPI ranks that are accessed but owned by a remote rank.
- Non-exec-halo elements (INH/ENH): ghost-cell data required for correctness of indirect accesses.
- Full exchange implemented in `op2/src/mpi/op_mpi_core.cpp`.

**Mesh partitioning methods:**
- `PTScotch`: graph-based (K-way) — recommended
- `ParMETIS`: graph and geometry-based (`KWAY`, `GEOM`, `GEOMKWAY`)
- `KaHIP`: graph-based
- `INERTIAL`: geometric bisection (no external library required)
- `RANDOM`: random partitioning (testing only)
- `EXTERNAL`: read partition from HDF5 file

---

## 7. Translator Tooling

The translator reads a user's OP2 source file and generates parallelized variants. Generated files are placed alongside the original source.

### 7.1 Legacy Translator (v1)

Located in `translator/`. Written in Python. Each `op2_gen_*.py` script handles one target.

**C/C++ usage:**
```sh
cd translator/c
# Edit op2.py: uncomment the desired generator (op2_gen_openmp, op2_gen_cuda, etc.)
python3 op2.py path/to/myapp.cpp
```

**Available C/C++ generators:**

| Script | Target |
|---|---|
| `op2_gen_seq.py` | Sequential |
| `op2_gen_openmp.py` / `op2_gen_openmp_simple.py` | OpenMP variants |
| `op2_gen_omp_vec.py` | OpenMP + SIMD vectorization |
| `op2_gen_cuda.py` | CUDA (Fermi) |
| `op2_gen_cuda_simple.py` | CUDA (Kepler+, optimized) |
| `op2_gen_cuda_simple_hyb.py` | Hybrid OpenMP + CUDA |
| `op2_gen_mpi_vec.py` | MPI + vectorization |
| `op2_gen_openacc.py` | OpenACC |
| `op2_gen_openmp4.py` | OpenMP 4.0 offload |

**Fortran usage:**
```sh
cd translator/fortran
# Edit op2_fortran.py: uncomment the desired generator
python3 op2_fortran.py path/to/myapp.F90
```

### 7.2 Next-Generation Translator (v2)

Located in `translator-v2/`. Uses **Jinja2 templating** and `libclang` for robust source parsing. Supports multi-file projects.

**Setup:**
```sh
cd translator-v2
pip install -r requirements.txt
```

**Usage:**
```sh
python3 op2-translator [options] <source files ...>
# e.g.
python3 op2-translator --target cuda myapp.cpp kernels.h
```

**Key modules:**

| Module | Purpose |
|---|---|
| `__main__.py` | CLI entry point — parses arguments, orchestrates pipeline |
| `op.py` | Python data model: `OpSet`, `OpMap`, `OpDat`, `OpArg`, `OpLoop` |
| `scheme.py` | `Scheme` registry: language × optimization target pairs |
| `target.py` | `Target` class: seq, openmp, cuda, etc. |
| `language.py` | `Lang` class: C/C++ or Fortran |
| `store.py` | `Application` container built from parsed sources |
| `cpp/parser.py` | C/C++ parser using `libclang` |
| `fortran/parser.py` | Fortran source parser |
| `resources/templates/` | Jinja2 templates for each language/target combination |

---

## 8. Example Applications

### Airfoil (`apps/c/airfoil/`, `apps/fortran/airfoil/`)

The canonical OP2 benchmark. Solves the non-linear 2D inviscid flow over an airfoil (Euler equations, finite-volume discretization on an unstructured mesh, ~720k cells).

**Five parallel loops:**

| Kernel | Loop type | Description |
|---|---|---|
| `save_soln` | Direct | Save current solution |
| `adt_calc` | Indirect | Compute local time step |
| `res_calc` | Indirect | Residual calculation |
| `bres_calc` | Indirect | Boundary residual |
| `update` | Direct | Advance the solution |

**Variants:**

| Directory | Description |
|---|---|
| `airfoil_plain` | Uses ASCII mesh files |
| `airfoil_hdf5` | Uses HDF5 mesh files |
| `airfoil_tempdats` | Demonstrates temporary dat patterns |
| `airfoil_tutorial` | Step-by-step tutorial version |

**Running:**
```sh
# Sequential
./airfoil_plain_seq grid.dat

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./airfoil_plain_openmp grid.dat

# MPI (4 ranks)
mpirun -np 4 ./airfoil_plain_mpi_seq grid.dat
```

Mesh files (HDF5 format): available at the links in `docs/examples.rst`.

### Other C/C++ Apps

| App directory | Description |
|---|---|
| `apps/c/aero/` | 3D aerodynamics application (`aero_plain`, `aero_hdf5`) |
| `apps/c/jac1/`, `apps/c/jac2/` | Jacobi iteration on unstructured mesh (dp and sp) |
| `apps/c/jac1/longint/` | Jacobi iteration demonstrating large-mesh support via `idx_g_t`/`idx_l_t` and `op_decl_map_long` |
| `apps/c/min/` | Minimal OP2 example |
| `apps/c/reduction/` | Reduction operations demonstration |

### Fortran Apps

| App directory | Description |
|---|---|
| `apps/fortran/airfoil/` | Fortran Airfoil benchmark |
| `apps/fortran/jac1/` | Fortran Jacobi iteration |
| `apps/fortran/jac1_long/` | Fortran Jacobi iteration with large-mesh global index support (`op_decl_map_long`) |
| `apps/fortran/reduction/` | Fortran reduction demo |

### Mesh Generators

MATLAB scripts in `apps/mesh_generators/`:
- `naca0012.m` — generates the NACA0012 airfoil mesh
- `naca_fem.m` — FEM mesh variant

---

## 9. Dependencies

### Required

| Dependency | Minimum Version | Notes |
|---|---|---|
| GNU Make | ≥ 4.2 | The build system |
| C/C++ compiler | C++17 support | GCC, Clang, Intel oneAPI, Cray, IBM XL, NVHPC |

### Optional

| Dependency | Purpose | How to enable |
|---|---|---|
| Fortran compiler | Fortran bindings + Fortran apps | Set `OP2_F_COMPILER` |
| MPI (mpicc/mpicxx/mpif90) | Distributed-memory parallelism | Auto-detected via MPI wrappers |
| NVIDIA CUDA ≥ 11.8 | GPU parallelism on NVIDIA hardware | Set `CUDA_INSTALL_PATH` and `NV_ARCH` |
| AMD HIP (ROCm) | GPU parallelism on AMD hardware | Set `HIP_INSTALL_PATH` and `HIP_ARCH` |
| HDF5 (sequential) | Serial file I/O | Set `HDF5_SEQ_INSTALL_PATH` |
| HDF5 (parallel, MPI) | Parallel checkpoint/restart I/O | Set `HDF5_PAR_INSTALL_PATH` (also requires MPI) |
| PT-Scotch | MPI mesh partitioning | Set `PTSCOTCH_INSTALL_PATH` (build with 32-bit index, no pthreads) |
| ParMETIS | MPI mesh partitioning | Set `PARMETIS_INSTALL_PATH` (build with `-DIDXSIZE32`) |
| KaHIP | MPI mesh partitioning | Set `KAHIP_INSTALL_PATH` |

> **Note:** MPI and parallel HDF5 are jointly required for the MPI library variant.

### Python (Translator)

| Package | Purpose |
|---|---|
| Python ≥ 3.8 | Runtime |
| `libclang` | C/C++ source parsing in v2 translator |
| See `translator-v2/requirements.txt` | All v2 dependencies |

---

## 10. Documentation Sources

All documentation is written in **reStructuredText (RST)** and built with **Sphinx**:

```sh
cd docs
make html    # Build HTML documentation
make latexpdf
```

| File | Content |
|---|---|
| `docs/getting_started.rst` | Full installation guide: build steps, all env vars, compiler + dependency setup |
| `docs/api.rst` | Complete C/C++ API reference |
| `docs/devapp.rst` | Developer tutorial: building an OP2 application using Airfoil |
| `docs/examples.rst` | Example applications, mesh download links, MGCFD and Volna references |
| `docs/developer_guide.rst` | Links to PDF guides for OP2 internals and MPI implementation |
| `docs/perf.rst` | Performance documentation |
| `docs/pubs.rst` | Publications list |

Additional README files:

| File | Content |
|---|---|
| `README.md` | Top-level quick-start |
| `makefiles/README.md` | Detailed Make infrastructure documentation |
| `translator/c/README.md` | Legacy C/C++ translator usage |
| `translator/fortran/README.md` | Legacy Fortran translator usage |
| `translator-v2/README.md` | v2 translator quick-start |
| `apps/c/airfoil/README.md` | Airfoil application: build, run, validate |
| `op2/include/README.md` | Header file overview |
| `scripts/README.md` | Pre-commit hooks, compiler environment scripts |

---

## 11. License and Authors

**License:** BSD 3-Clause — see [LICENSE](LICENSE)

Copyright © 2011–present Gihan Mudalige, Istvan Reguly, Mike Giles, and contributors.

**Key contributors** (see [AUTHORS](AUTHORS)):
Mike Giles, Gihan Mudalige, Istvan Reguly, Carlo Bertolli, Adam Betts, Paul Kelly, Lawrence Mitchell, David Ham, Graham Markall, Florian Rathgeber, Attila Sulyok, Daniel Balogh, Endre László, David Radford, Andrew Owenson.

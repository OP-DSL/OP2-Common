Building with CMake
===================

The reference for OP2's CMake build: every option, cache variable, and
environment hint it understands, how it finds its dependencies, and what a
downstream project gets from ``find_package(OP2)``.

For a short walkthrough of configuring and building, see the
:ref:`getting_started:CMake Build` section of Getting Started. The GNU Make
build is documented there too, and is unaffected by any of this.

Build requirements
------------------

- **CMake >= 3.26.** This is a hard floor, not just a
  ``cmake_minimum_required`` formality: CMake's ``NVIDIA-CUDA`` compiler module
  only gained a C++20 flag table (``CMAKE_CUDA20_STANDARD_COMPILE_OPTION``) in
  3.26 - on 3.25 and earlier, configuring with CUDA enabled fails outright
  (``requires the language dialect "CUDA20"... CMake does not know the flags to
  enable it``). If your system package manager is stuck on an older CMake,
  ``pip install cmake`` (optionally in a venv) gets you a current one without
  touching the system install.
- **C++20** throughout. The OP2 library targets declare ``cxx_std_20`` (and
  ``cuda_std_20`` / ``hip_std_20``) as ``PUBLIC`` compile features, so anything
  linking them is compiled as C++20 too, whatever your project's own default
  is. For CUDA this additionally needs **nvcc >= 12.0** (CMake's C++20 support
  for CUDA is itself gated on that compiler version).
- **``CMAKE_CUDA_ARCHITECTURES``** defaults to ``80;90`` (Ampere, Hopper) when
  neither ``-DCMAKE_CUDA_ARCHITECTURES=...`` nor the ``CUDAARCHS`` env var is
  set. Override for other/older hardware, e.g.
  ``-DCMAKE_CUDA_ARCHITECTURES=70`` for Volta, or
  ``-DCMAKE_CUDA_ARCHITECTURES=native`` on a machine with the target GPU
  present. Must be set before ``enable_language(CUDA)`` is reached (i.e. passed
  at the initial ``cmake -B build`` invocation), since CMP0104 otherwise defers
  to nvcc's own default.

.. _building-op2:

Building OP2
------------

The whole build is four commands. Each step is expanded below.

.. code-block:: shell

   scripts/setup_deps.sh                          # 1. bootstrap dependencies
   cmake -B build -C deps/op2-deps.cmake          # 2. configure
   cmake --build build -j$(nproc)                 # 3. build
   cmake --install build --prefix /opt/op2        # 4. install (optional)

1. Bootstrap the dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``scripts/setup_deps.sh`` downloads and builds the four external libraries OP2
can use - HDF5, ParMETIS (with its bundled METIS), PT-Scotch, and KaHIP - into
``deps/`` under the repository root. It is a thin wrapper around
``cmake/deps/CMakeLists.txt``, so anything it does can also be driven directly
with ``cmake -B deps/build -S cmake/deps``.

This step is entirely optional. It exists so that you do not have to hand-build
four libraries with mutually compatible integer widths and MPI settings; if
your site already provides them, skip it and use the hints in
:ref:`cmake:Dependency hints` instead.

.. code-block:: shell

   # Everything, with compilers auto-detected from CC / CXX / FC
   scripts/setup_deps.sh

   # Name the compilers explicitly, and cap the build parallelism
   scripts/setup_deps.sh -j8 \
       -DCMAKE_C_COMPILER=gcc \
       -DCMAKE_CXX_COMPILER=g++ \
       -DCMAKE_Fortran_COMPILER=gfortran

   # Skip what you do not need
   scripts/setup_deps.sh -DOP2_DEPS_KAHIP=OFF -DOP2_DEPS_HDF5=OFF

   # Start over
   rm -rf deps && scripts/setup_deps.sh

Two things are decided for you here, because getting them wrong produces
silent breakage rather than a build error. **MPI is required**: every library
built by this step uses it, HDF5 included, so the HDF5 it produces is always
parallel. And **all three partitioners are built with 64-bit integer indices**,
to match OP2's own ``idx_g_t``, which is ``long long``.

The result is one prefix per library plus a cache-init file:

.. code-block:: text

   deps/
       hdf5/          parallel HDF5
       parmetis/      ParMETIS
       metis/         the METIS bundled with it
       ptscotch/      PT-Scotch
       kahip/         KaHIP, including ParHIP
       op2-deps.cmake  CMake cache initializer naming all of the above

2. Configure
^^^^^^^^^^^^

``deps/op2-deps.cmake`` sets every ``*_ROOT`` variable and the compilers the
dependencies were built with, so loading it with ``-C`` is all a full-featured
configure needs:

.. code-block:: shell

   cmake -B build -C deps/op2-deps.cmake

Other starting points:

.. code-block:: shell

   # Minimal: sequential and OpenMP only, no MPI, no GPU, no HDF5
   cmake -B build -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

   # Point at the deps tree without the cache file
   cmake -B build -DOP2_DEPS_ROOT=deps

   # Turn a feature off explicitly rather than relying on auto-detection
   cmake -B build -C deps/op2-deps.cmake -DOP2_ENABLE_CUDA=OFF -DOP2_ENABLE_HIP=OFF

   # Shared libraries, apps, and the functional tests
   cmake -B build -C deps/op2-deps.cmake \
       -DBUILD_SHARED_LIBS=ON -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_TESTS=ON

Nothing here is mandatory. Every feature auto-detects, and every one of them
is soft-failing: a missing compiler or library removes the affected library
variants and prints a status message, rather than stopping the configure. That
makes the summary block at the end of the configure the thing to read, because
it is the only place that says what you actually got:

.. code-block:: text

   =============== OP2 configuration ===============
     Version:        1.2.0
     Install prefix: /opt/op2
     Build type:     (not set - defaults to unoptimised)
     Library type:   static

     C compiler:     /usr/bin/cc (GNU 14.2.0)
     C++ compiler:   /usr/bin/c++ (GNU 14.2.0)
     Fortran:        /usr/bin/f95 (GNU 14.2.0)
     CUDA:           (no compiler found)
     HIP:            (no compiler found)

     MPI:            Open MPI v5.0.7 ...
     OpenMP:         C++ v4.5, Fortran v4.5
     HDF5:           v1.14.4 parallel - target hdf5-static
     Partitioners:   PT-Scotch, ParMETIS, KaHIP
     Python:         /usr/bin/python3 (v3.13.5)
     Translator:     /usr/bin/python3

     C/C++ libs:     op2_seq, op2_openmp, op2_hdf5, op2_mpi
     Fortran libs:   op2_for_seq, op2_for_openmp, op2_for_hdf5, op2_for_mpi
     Apps:           31 apps, 38 buildable variants
     Tests:          10 directories, 40 registered tests
   ==================================================

If a library variant you expected is absent from the ``C/C++ libs`` or
``Fortran libs`` lines, the reason is above it: no CUDA compiler, no parallel
HDF5, no MPI. A ``Translator:`` line reporting missing packages means the
translated variants (everything except ``seq``) will not be built; see
:ref:`cmake:Translator dependency hints`.

3. Build
^^^^^^^^

.. code-block:: shell

   cmake --build build -j$(nproc)

CMake has no default optimisation level, so an unset ``CMAKE_BUILD_TYPE``
builds unoptimised - which is rarely what you want for a performance library.
Set it at configure time:

.. code-block:: shell

   cmake -B build -C deps/op2-deps.cmake -DCMAKE_BUILD_TYPE=Release

4. Test
^^^^^^^

With ``-DOP2_ENABLE_TESTS=ON``, the functional tests under ``tests/functional``
are registered with CTest, one test per built variant:

.. code-block:: shell

   ctest --test-dir build -j4              # everything
   ctest --test-dir build -L mpi           # only the MPI variants
   ctest --test-dir build -L fortran       # only the Fortran tests
   ctest --test-dir build -R idx           # one directory
   ctest --test-dir build -N               # list without running

Each test carries labels for its language, its category, and whether it is
``serial`` or ``mpi``, and the MPI ones declare their rank count via the
``PROCESSORS`` property so ``ctest -j`` schedules them without oversubscribing.

5. Install
^^^^^^^^^^

.. code-block:: shell

   cmake --install build --prefix /opt/op2

which produces:

.. code-block:: text

   /opt/op2/
       bin/op2-translator          standalone translator wrapper
       include/op2/                headers, with fortran/<variant>/ module files
       lib/                        libop2_*.a or .so
       lib/cmake/op2/              OP2Config.cmake and friends
       libexec/op2/translator/     translator payload and requirements.txt

Installing is optional: a downstream project can also consume OP2 straight
from the build tree, or via ``add_subdirectory()``. See :ref:`downstream-usage`.


Feature toggles
---------------

Cache options controlling which library variants and pieces of the tree get
built. All are ``ON`` by default unless noted; pass ``-D<NAME>=OFF`` at
configure to disable.

.. list-table::
   :header-rows: 1
   :widths: 22 13 65

   * - Option
     - Default
     - Effect when ``OFF``
   * - ``OP2_ENABLE_MPI``
     - ``ON``
     - Skip MPI library variants (``op2_mpi``, ``op2_mpi_*``, ``op2_for_mpi*``).
   * - ``OP2_ENABLE_CUDA``
     - ``ON``
     - Skip CUDA library variants; don't ``enable_language(CUDA)``.
   * - ``OP2_ENABLE_HIP``
     - ``ON``
     - Skip HIP library variants; don't ``enable_language(HIP)``.
   * - ``OP2_ENABLE_OPENMP``
     - ``ON``
     - Skip the OpenMP variants. The C++ and Fortran sides are independent:
       OpenMP is probed per language, and having it for only one of them builds
       only that language's OpenMP variants.
   * - ``OP2_ENABLE_HDF5``
     - ``ON``
     - Skip HDF5 discovery; disables ``op2_hdf5``, ``op2_for_hdf5``, and every
       MPI variant (which needs parallel HDF5).
   * - ``OP2_ENABLE_FORTRAN``
     - ``ON``
     - Skip Fortran library variants and every Fortran-linked app.
   * - ``OP2_ENABLE_APPS``
     - ``OFF``
     - Skip building the example apps under ``apps/``.
   * - ``OP2_ENABLE_TESTS``
     - ``OFF``
     - Skip the functional tests under ``tests/`` and their ``ctest``
       registration.
   * - ``OP2_ENABLE_TRANSLATOR``
     - ``ON``
     - Skip the Python probe. ``OP2AppSupport.cmake`` becomes unusable (no
       translator command available); library variants still build.
   * - ``OP2_ENABLE_INSTALL``
     - top-level ``ON``, subproject ``OFF``
     - Skip **all** ``install()`` rules. Defaults to ``OFF`` automatically when
       OP2 is consumed via ``add_subdirectory()``, on the assumption that the
       parent project owns installation; set it explicitly either way to
       override.

Every feature is *soft-fail*: if the required compiler or dependency is
missing, the affected variants silently drop with a status message. Toggling
``OFF`` is only needed when you want the feature explicitly hidden.

.. _shared-vs-static:

Shared vs static libraries
--------------------------

OP2 honours the standard CMake ``BUILD_SHARED_LIBS`` - not an OP2-specific
option. ``OFF`` (CMake's own default) builds every ``op2_*`` target ``STATIC``,
exactly as before this was supported. ``ON`` switches every one of them to
``SHARED``; there is no mode that builds both at once, so pick whichever your
project needs at configure time.

A few things specific to going shared:

- **The partitioner and HDF5 static archives get absorbed into each ``.so``.**
  Building a ``.so`` is a real link step (unlike a static archive, which only
  records the requirement), so PT-Scotch/ParMETIS/METIS/KaHIP/HDF5's object
  code becomes part of ``libop2_mpi.so`` etc. ``-Wl,--exclude-libs=ALL`` hides
  symbols pulled in from those archives from the ``.so``'s exported table -
  OP2's own object code isn't in an archive at this link step, so this only
  touches what got absorbed, not OP2's own public API (which mixes several
  naming conventions - ``op_*``/``OP_*``, gfortran-mangled Fortran module
  procedures, and a handful of plain-named C-Fortran bridge helpers - so a
  name-based export list isn't a viable way to draw this line). This is what
  prevents a symbol collision if your own project also links
  PT-Scotch/ParMETIS/HDF5 directly.
- **PIC is required for every static dependency being linked into a shared
  OP2.** The ``scripts/setup_deps.sh`` bootstrap already builds HDF5,
  PT-Scotch, ParMETIS and KaHIP with position-independent code, so
  ``BUILD_SHARED_LIBS=ON`` against those "just works". A **system- or
  module-provided** partitioner or HDF5 has no such guarantee - if a shared OP2
  build fails at link time with "recompile with -fPIC", that's the site's
  library, not something OP2's CMake can fix for you.
- **Installed executables get an RPATH back to OP2's own install prefix
  automatically** (see :ref:`downstream-usage` for what a consumer needs to do
  for its own executables).

Dependency hints
----------------

OP2 finds its dependencies via CMake's standard mechanisms. You can point the
build at your deps in any of the ways below - pick whichever fits your
environment. All variables accept both cache-form (``-DFOO=...``) and
env-var-form (``export FOO=...``) unless noted.

Convenience shortcut: ``OP2_DEPS_ROOT``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you built the deps via ``scripts/setup_deps.sh`` (or
``cmake -B deps/build -S cmake/deps``), the standard output layout is:

.. code-block:: text

   <repo>/deps/
       hdf5/       parmetis/     metis/      ptscotch/     kahip/

Pass one variable to pick them all up:

.. code-block:: shell

   cmake -B build -DOP2_DEPS_ROOT=/path/to/deps

This populates every ``*_ROOT`` below that hasn't been set explicitly.

Alternatively, ``scripts/setup_deps.sh`` writes ``deps/op2-deps.cmake`` - a
CMake cache-init file that sets all ``*_ROOT`` variables. Load it with ``-C``:

.. code-block:: shell

   cmake -B build -C /path/to/deps/op2-deps.cmake

Per-dependency variables
^^^^^^^^^^^^^^^^^^^^^^^^

These steer the **OP2 build**. Each is checked (in order of preference)
against: ``<Package>_ROOT`` (matches the exact ``find_package(<Package>)``
spelling), Spack-convention ``<PKG>_DIR``. All also accept
environment-variable form.

They do **not** apply to a project consuming an installed OP2 - see
:ref:`downstream-usage` for why. To change the partitioner an application
links, rebuild OP2 against it.

.. list-table::
   :header-rows: 1
   :widths: 22 18 22 38

   * - Dependency
     - Preferred hint
     - Also accepted
     - Discovery mechanism
   * - **HDF5**
     - ``HDF5_ROOT``
     - (CMake standard)
     - ``find_package(HDF5 CONFIG)`` - needs an HDF5 install with
       ``hdf5-config.cmake``
   * - **PT-Scotch**
     - ``PTScotch_ROOT``
     - ``SCOTCH_DIR``, ``PTSCOTCH_DIR``
     - ``cmake/FindPTScotch.cmake`` (bundled)
   * - **ParMETIS**
     - ``ParMETIS_ROOT``
     - ``ParMETIS_DIR``, ``PARMETIS_DIR``
     - ``cmake/FindParMETIS.cmake`` (bundled)
   * - **METIS** (required by ParMETIS)
     - ``METIS_ROOT``
     - ``METIS_DIR``
     - probed by ``FindParMETIS``
   * - **KaHIP**
     - ``KaHIP_ROOT``
     - ``KAHIP_DIR``
     - ``cmake/FindKaHIP.cmake`` (bundled)

Any of these can be replaced with an entry on ``CMAKE_PREFIX_PATH``:

.. code-block:: shell

   cmake -B build -DCMAKE_PREFIX_PATH="/opt/hdf5;/opt/scotch;/opt/parmetis"

Module-system users
^^^^^^^^^^^^^^^^^^^

If your HPC site provides packages via Environment Modules / Lmod,
``module load hdf5-parallel`` (etc.) typically exports ``HDF5_ROOT`` or
``HDF5_DIR`` and adds prefixes to ``CMAKE_PREFIX_PATH``. OP2's Find modules
honour those, so ``cmake -B build`` after loading the right modules "just
works" without any explicit ``-D`` on the command line.

Translator dependency hints
---------------------------

The translator needs Python 3.8+ with ``jinja2``, ``fparser``, ``pcpp``,
``sympy``, and ``libclang`` (importable as ``clang.cindex``) available at
import time. Python is treated like any other OP2 dependency (MPI, HDF5, ...):
CMake finds it, it doesn't provision it. There's no venv, no pip install, and
no network access during configure or build - the interpreter
``Python3_EXECUTABLE`` points at must already have these packages installed,
e.g.:

.. code-block:: shell

   python3 -m pip install -r translator-v2/requirements.txt

If the picked interpreter doesn't satisfy that import check,
``OP2_HAS_TRANSLATOR`` is set to ``FALSE`` and a ``STATUS`` message explains
what's missing and how to fix it. This is never fatal. Library variants still
build, and so do apps and tests - ``seq`` variants need no translation at all,
so ``op2_add_app_variants()`` simply produces a narrower set. A missing
translator shows up as a smaller variant count in the configuration summary,
not a configure error.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Effect
   * - ``Python3_EXECUTABLE``
     - auto
     - Bypass CMake's Python search; point at a specific interpreter that
       already has the packages installed.
   * - ``OP2_PYTHON`` (env, legacy make only)
     - *(unset)*
     - Overrides the Python that ``translator-v2/op2-translator.sh`` picks.
       Used by the legacy ``apps/**/Makefile`` path.

Discovery outputs
-----------------

Read-only variables OP2's CMake sets, useful downstream via
``find_package(OP2 CONFIG)``:

.. list-table::
   :header-rows: 1
   :widths: 33 67

   * - Variable
     - Meaning
   * - ``OP2_HAS_MPI`` / ``OP2_HAS_CUDA`` / ``OP2_HAS_HIP`` / ``OP2_HAS_HDF5``
     - ``TRUE`` when the feature is available in this build
   * - ``OP2_HAS_OPENMP_CXX`` / ``OP2_HAS_OPENMP_FORTRAN``
     - ``TRUE`` when OpenMP was found for that language. There is no
       unsuffixed ``OP2_HAS_OPENMP``: the two are independent, and OP2 has no
       use for a combined answer
   * - ``OP2_HAS_HDF5_PARALLEL``
     - ``TRUE`` when the linked HDF5 is MPI-parallel
   * - ``OP2_HAS_PTSCOTCH`` / ``OP2_HAS_PARMETIS`` / ``OP2_HAS_KAHIP``
     - ``TRUE`` when the partitioner was found
   * - ``OP2_HAS_TRANSLATOR``
     - ``TRUE`` when a suitable Python 3 was found (needed for
       ``op2_add_app_variants``)
   * - ``OP2_TRANSLATOR_COMMAND``
     - Two-element list ``[Python3_EXECUTABLE; pkg-dir]`` - pass unquoted to
       ``add_custom_command(COMMAND ...)``
   * - ``OP2_INCLUDE_DIR``
     - Plain path to OP2's C/C++ headers (in-tree source dir vs installed
       ``<prefix>/include/op2``)
   * - ``OP2_HDF5_LIBRARIES``
     - The HDF5 targets OP2 linked, for an app that calls the HDF5 API itself -
       OP2 links HDF5 ``PRIVATE``, so it passes on no HDF5 include directory
       (see :ref:`downstream-usage`)

.. _downstream-usage:

Downstream usage
----------------

An installed OP2 is found with config-mode discovery and exports namespaced
targets, one per library variant:

.. code-block:: shell

   cmake -B build -DCMAKE_PREFIX_PATH=/opt/op2

That is all the discovery you need. ``OP2Config.cmake`` records where OP2 found
each of its own dependencies and falls back to those locations, so you do not
have to locate HDF5 or a partitioner yourself.

There are two ways to build against OP2, and which one you want depends on
whether your code contains ``op_par_loop`` calls:

- **An OP2 application** - source with ``op_par_loop`` in it - must be put
  through the translator, once per parallelisation variant, before it can be
  compiled. ``op2_add_app_variants()`` does that and the compiling and linking
  with it. This is the canonical path, and the one the apps in this repository
  use.
- **A program that merely links OP2** - a mesh converter, a test harness, a
  tool calling ``op_fetch_data`` - needs no translation. Link the library
  variant you want and build it as an ordinary target.

The canonical application build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.26)
   project(my_app LANGUAGES CXX C)    # C is needed against a parallel HDF5

   find_package(OP2 CONFIG REQUIRED)

   op2_add_app_variants(
       NAME     my_app
       LANGUAGE cpp
       SOURCES  my_app.cpp)

Enabling ``C`` alongside ``CXX`` is what an OP2 built against **parallel**
HDF5 requires, which is the usual case and what ``scripts/setup_deps.sh``
produces; a serial-HDF5 or HDF5-less OP2 needs only ``CXX``. Get it wrong and
``find_package(OP2)`` says so by name rather than failing somewhere inside
``FindMPI`` - see :ref:`cmake:Required languages`.

One call produces one executable per variant this environment can build, named
``<NAME>_<variant>``: ``my_app_seq``, ``my_app_genseq``, ``my_app_openmp``,
``my_app_cuda``, ``my_app_mpi_seq``, ``my_app_mpi_cuda``, and so on. The
variant names and what each one means are listed under
:ref:`getting_started:Application Build Variants`; they are the same set the
GNU Make build offers.

For each variant, the call:

1. runs the translator over ``SOURCES`` into
   ``${CMAKE_CURRENT_BINARY_DIR}/generated/<NAME>/`` (override with
   ``OUTPUT_DIR``), producing the rewritten user program and, in a
   subdirectory named for the translator target, the master kernel file. The
   subdirectory is keyed by translator target rather than by variant, so
   variants needing identical generated code share one run: ``genseq`` and
   ``mpi_genseq`` both use ``seq/``, and the ``seq`` variant needs no
   translation at all;
2. creates the executable from your sources plus that generated code, enabling
   whichever of ``CUDA``, ``HIP`` or ``Fortran`` the variant needs;
3. links the matching OP2 library - ``op2::op2_seq`` for ``seq``,
   ``op2::op2_openmp`` for ``openmp``, ``op2::op2_mpi_cuda`` for ``mpi_cuda``,
   and so on - along with MPI, OpenMP or the CUDA runtime as required;
4. defines ``USE_MPI`` on every ``mpi_*`` variant.

The translation is a normal build step with a depfile, so editing a kernel
header retranslates and rebuilds only what depends on it. You do not run the
translator yourself, and the generated sources are build artefacts - do not
check them in.

Nothing about the call fails when a toolchain is absent. With no CUDA
compiler, the ``cuda`` variants simply are not created; with no MPI, none of
the ``mpi_*`` ones are. Narrow the set deliberately with ``VARIANTS`` and
``EXCLUDE_VARIANTS``, which take shell-style globs - ``mpi_*`` being the one
you will actually reach for, and the equivalent of the Make build's
``VARIANT_FILTER := mpi_%``.

That matters because a single source file is often not valid for both. An
application that sets up its mesh by reading a file on one rank looks different
from one that partitions it across ranks, so most of the apps in this
repository keep two programs and make two calls, one per family:

.. code-block:: cmake

   op2_add_app_variants(
       NAME     my_app
       LANGUAGE cpp
       SOURCES  my_app.cpp
       EXCLUDE_VARIANTS mpi_*)          # my_app_seq, my_app_openmp, ...

   op2_add_app_variants(
       NAME     my_app_par
       LANGUAGE cpp
       SOURCES  my_app_mpi.cpp
       VARIANTS mpi_*)                  # my_app_par_mpi_seq, ...

Each call owns a distinct ``NAME``, so the two sets of executables do not
collide. If your program handles both cases from one source - guarded by
``#ifdef USE_MPI``, which every ``mpi_*`` variant defines for you - then one
call covering every variant is enough.

Two keywords are easy to miss. Compile definitions and include directories
must be passed **to the call**, not applied to the resulting targets with
``target_compile_definitions()``, because the translator parses your sources at
configure time and has to see the same ones:

.. code-block:: cmake

   op2_add_app_variants(
       NAME     my_app
       LANGUAGE cpp
       SOURCES  my_app.cpp
       COMPILE_DEFINITIONS  DOUBLE_PRECISION
       INCLUDE_DIRECTORIES  "${CMAKE_CURRENT_SOURCE_DIR}/include"
       WITH_HDF5                     # for op_decl_dat_hdf5 et al.
       INSTALL)                      # install the executables to bin/

``WITH_HDF5`` links the standalone HDF5 backend (``op2::op2_hdf5``, or
``op2::op2_for_hdf5``) into the non-MPI variants; the MPI backends already
carry the HDF5 entry points, so it is a no-op for those. ``INSTALL`` adds every
executable created to the install set.

A Fortran application is the same call with ``LANGUAGE fortran``, plus
``--consts-module`` so the translator can resolve the module holding your
``op_decl_const`` declarations:

.. code-block:: cmake

   project(my_app LANGUAGES CXX Fortran)

   op2_add_app_variants(
       NAME     my_app
       LANGUAGE fortran
       SOURCES  my_consts.F90 my_kernels.F90 my_app.F90
       TRANSLATOR_ONLY_ARGS
           --consts-module "${CMAKE_CURRENT_SOURCE_DIR}/my_consts.F90")

``TRANSLATOR_ONLY_ARGS`` reaches the translator without affecting compilation,
which is what you want for flags like that one and ``--force_soa``.

Building and running is then ordinary CMake:

.. code-block:: shell

   cmake -B build -DCMAKE_PREFIX_PATH=/opt/op2 -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ./build/my_app_genseq
   mpirun -np 8 ./build/my_app_mpi_genseq

.. note::

   If you call ``find_package(OP2)`` somewhere other than your top-level
   ``CMakeLists.txt`` - say in one subdirectory, with
   ``op2_add_app_variants()`` in a sibling - add the ``GLOBAL`` keyword:
   ``find_package(OP2 CONFIG REQUIRED GLOBAL)``. Imported targets are directory
   scoped by default, so without it the ``op2::*`` targets do not exist in the
   sibling directory and the call reports ``no buildable variants``. The
   ``OP2_HAS_*`` and ``OP2_INCLUDE_DIR`` variables are cached and need no such
   treatment.

Linking a library variant directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Code with no ``op_par_loop`` in it needs no translation, so it is an ordinary
target linking one ``op2::*`` variant:

.. code-block:: cmake

   project(convert_mesh LANGUAGES CXX C)   # C for parallel HDF5
   find_package(OP2 CONFIG REQUIRED)

   add_executable(convert_mesh convert_mesh.cpp)
   target_link_libraries(convert_mesh PRIVATE op2::op2_mpi)

The available targets follow the library variants OP2 was built with:
``op2::op2_seq``, ``op2::op2_openmp``, ``op2::op2_cuda``, ``op2::op2_hip``,
``op2::op2_hdf5``, ``op2::op2_mpi``, ``op2::op2_mpi_cuda``,
``op2::op2_mpi_hip``, and an ``op2::op2_for_*`` for each on the Fortran side.
Which of them exist depends on how OP2 was configured, so branch on the
``OP2_HAS_*`` variables rather than assuming:

.. code-block:: cmake

   if(OP2_HAS_MPI AND OP2_HAS_HDF5_PARALLEL)
       add_executable(convert_mesh_mpi convert_mesh_mpi.cpp)
       target_link_libraries(convert_mesh_mpi PRIVATE op2::op2_mpi MPI::MPI_CXX)
   endif()

Note that this only works for code the translator would have nothing to do
with. Compiling a source containing ``op_par_loop`` without translating it
first will not link.

How OP2's own dependencies are resolved
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mesh partitioners (PT-Scotch, ParMETIS/METIS, KaHIP) are **pinned**: OP2
links them by library path, so ``install(EXPORT)`` records the exact artefacts
it was built against in ``OP2Targets.cmake``, and none of them is searched for
again. This is deliberate. ``libop2_mpi.a`` is compiled against those specific
headers, and the partitioner APIs are not ABI-stable across builds -
``scotch.h`` sizes its opaque handles per build (``double dummy[N]``) and OP2
stack-allocates a ``SCOTCH_Strat`` - so binding a different build would corrupt
memory rather than fail to link. **To change partitioner, rebuild OP2**; the
libraries take seconds to build. If a pinned library has since been deleted or
moved, the build stops with that path named, rather than silently binding
whatever else happens to be installed.

Everything else - MPI, OpenMP, CUDA, HIP, HDF5 - provides CMake imported
targets, so it is **re-found** rather than pinned, and the usual hints apply.
If OP2 was built ``STATIC`` (the default), your application relinks those same
libraries into its own binary: finding a *different* build of one is what
breaks, not finding none. If OP2 was built ``SHARED``, MPI/OpenMP/CUDA/HIP are
still your own binary's direct runtime dependencies exactly as before (a
``.so``-to-``.so`` link, nothing new there), but the partitioners and HDF5 are
no longer separate runtime dependencies at all - they were absorbed into
``libop2_*.so`` itself at OP2's own build time (see
:ref:`shared-vs-static`). Either way, each re-found dependency **defaults to
the one OP2 was built against**:

.. list-table::
   :header-rows: 1
   :widths: 14 30 28 28

   * - Dependency
     - Default recorded
     - Overridden by
     - Suppressed when
   * - MPI
     - ``MPI_CXX_COMPILER``, ``MPI_C_COMPILER`` (the wrappers)
     - ``MPI_<lang>_COMPILER``, ``MPI_HOME``
     - ``MPI_HOME`` is set, or the wrapper no longer exists
   * - CUDA
     - ``CUDAToolkit_ROOT``
     - ``CUDAToolkit_ROOT``, an enabled ``CUDA`` language,
       ``CMAKE_CUDA_COMPILER``
     - the directory no longer exists
   * - HIP
     - ``hip_DIR``
     - ``hip_DIR``, ``CMAKE_PREFIX_PATH``
     - the directory no longer exists
   * - HDF5
     - ``HDF5_DIR``
     - ``HDF5_ROOT``, ``HDF5_DIR``, ``CMAKE_PREFIX_PATH``
     - the directory no longer exists
   * - OpenMP
     - *(nothing)*
     - \-
     - always: OpenMP is compiler flags, and your compiler's are the right ones

These are defaults, not pins - anything you set explicitly wins, and a recorded
path that has since moved is ignored so the install stays usable. On a cluster
where the same modules are loaded for OP2 and for your application, the
defaults simply agree with what discovery would have found anyway; they matter
when they would not have. Baking the MPI wrapper is the significant one: it
determines every include and library path ``FindMPI`` derives, and MPI has no
cross-implementation ABI, so an OP2 built against Open MPI cannot link against
MPICH.

HDF5 is additionally **checked**, because it is the one whose mismatch can be
quiet: a wrong MPI leaves undefined references at link time, and a toolkit too
old for OP2's CUDA/HIP dialect is rejected at generate time, but OP2 links HDF5
statically and calls version-specific symbols, so a mismatched build can link
cleanly and then misbehave. The HDF5 you supply must be **compatible with the
version OP2 was built against** (HDF5's own policy - same major.minor series)
and **parallel if OP2's is**. Either mismatch fails ``find_package(OP2)`` with
a message saying which.

HDF5 and the partitioners are linked ``PRIVATE``, in both library types, so
neither appears in your compile line: no public OP2 header exposes either API.
A static OP2 still hands the libraries themselves to your link line (CMake
exports a static target's private dependencies as ``$<LINK_ONLY:>``), so this
changes what you *compile* against, not what you link.

If your own code calls the raw HDF5 API alongside OP2, name HDF5 explicitly.
``find_package(OP2)`` has already found it - the same install OP2 was built
against, unless you steered it elsewhere - and publishes the targets it chose
as ``OP2_HDF5_LIBRARIES``:

.. code-block:: cmake

   find_package(OP2 CONFIG REQUIRED)
   target_link_libraries(my_app PRIVATE op2::op2_mpi ${OP2_HDF5_LIBRARIES})

Runtime library paths
^^^^^^^^^^^^^^^^^^^^^

If you consume a ``SHARED`` OP2 build, your own executables need to be able to
find ``libop2_*.so`` at runtime. OP2 sets ``RPATH`` on the executables **it**
installs (its own tests and apps), but ``op2_add_app_variants(... INSTALL)``
called from *your* project builds *your* executables - OP2 has no way to inject
``RPATH`` policy into a build it doesn't control. Set the usual CMake knobs in
your own project, the same as you would for any other shared-library
dependency:

.. code-block:: cmake

   set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

Required languages
^^^^^^^^^^^^^^^^^^

Every ``op2::*`` target is a C++ library - the Fortran ones are a Fortran
bridge over the same C++ core - so your project must enable ``CXX``. If OP2 was
built against **parallel HDF5** you must also enable ``C``, because HDF5's own
imported targets reference ``MPI::MPI_C``. Enabling ``Fortran`` is only needed
if you use the ``op2::op2_for_*`` targets.

If a required language is missing, ``find_package(OP2)`` fails with a message
naming it rather than something obscure from inside ``FindMPI``:

.. code-block:: text

   OP2 requires the C language, which the consuming project has not enabled.
   This OP2 build needs: CXX C. Add them to your project() call, e.g.
   project(myapp LANGUAGES CXX C).

Public helpers
^^^^^^^^^^^^^^

``OP2Config.cmake`` brings two functions with it:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Purpose
   * - ``op2_add_app_variants(NAME ... LANGUAGE ... SOURCES ...)``
     - Build ``<name>_<variant>`` for every variant this environment supports.
       ``VARIANTS`` narrows the set, ``EXCLUDE_VARIANTS`` subtracts from it;
       asking for a variant the toolchain can't build is not an error, it just
       doesn't appear. Pass ``COMPILE_DEFINITIONS`` / ``INCLUDE_DIRECTORIES``
       here rather than setting them on the resulting targets - the translator
       runs against them at configure time. ``TARGETS_VAR <var>`` returns the
       executables created, so you can apply your own requirements to them (see
       below). ``OUTPUT_DIR`` relocates the generated sources,
       ``TRANSLATOR_ONLY_ARGS`` passes flags to the translator without
       affecting compilation, ``WITH_HDF5`` links the standalone HDF5 API
       alongside the non-MPI variants, and ``INSTALL`` adds the executables to
       the install set.
   * - ``op2_translate(OUT_DIR ... LANGUAGE ... SOURCES ... VARIANTS ...)``
     - The low-level single translator invocation, if you want to drive code
       generation yourself.

Your app's own dependencies
"""""""""""""""""""""""""""

``op2_add_app_variants()`` attaches only what OP2 itself needs: its libraries,
and whatever the code the translator generates requires. Anything your own
sources pull in is yours to find and link, via the targets ``TARGETS_VAR``
hands back:

.. code-block:: cmake

   find_package(MPI COMPONENTS Fortran REQUIRED)   # our Fortran does `use mpi`

   op2_add_app_variants(NAME my_app LANGUAGE fortran SOURCES my_app.F90
                        TARGETS_VAR my_app_targets)

   foreach(t IN LISTS my_app_targets)
       target_link_libraries(${t} PRIVATE MPI::MPI_Fortran)
   endforeach()

The list is empty rather than undefined when nothing was buildable, so the loop
is always safe. To keep a variant out entirely when a dependency of yours is
missing, pass its name to ``EXCLUDE_VARIANTS``.

Note in particular that **OP2 does not find the MPI Fortran bindings** - no OP2
library or generated kernel uses them - so a Fortran app calling MPI directly
must do the above.

The translator on ``PATH``
^^^^^^^^^^^^^^^^^^^^^^^^^^

An OP2 install also ships ``<prefix>/bin/op2-translator``, a self-locating
wrapper that runs the translator with whatever ``python3`` is on your
``PATH``:

.. code-block:: shell

   mkdir -p generated                      # -o must already exist
   op2-translator -t seq -o generated -I /opt/op2/include/op2 my_app.cpp

The install bundles no Python of its own, so that interpreter needs the
packages from ``requirements.txt`` (shipped alongside at
``<libexecdir>/op2/translator/requirements.txt``) just as the build-time one
does.

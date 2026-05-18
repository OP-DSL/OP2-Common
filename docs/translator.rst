Code Generation
===============

OP2 uses a code translator to transform a user's sequential OP2 source files into parallelised variants targeting specific hardware backends. The current OP2 translator uses:

- **libclang** to parse C/C++ source files.
- **fparser2** (the ``fparser`` PyPI package) to parse Fortran source files.
- **Jinja2** to render backend-specific kernel code from templates.

It is the recommended tool for all projects. A **legacy translator** is also retained for compatibility, consisting of a collection of standalone Python scripts.

.. note::
   **For most users the translator runs automatically.**  Any application that uses the standard OP2 Makefiles (``makefiles/common.mk`` + ``makefiles/c_app.mk`` or ``f_app.mk``) will invoke the translator transparently when you run ``make <app_name>_<variant>``.  You only need to be aware of the translator's options if you are setting up a custom build system, running the translator in isolation for debugging, or need to override its behaviour.

----

OP2 Translator
--------------

Requirements
^^^^^^^^^^^^

The translator and its dependencies are bundled inside ``translator-v2/`` and are set up automatically by the OP2 Makefiles.  If you need to run the translator outside the Makefile (e.g., in a custom CI pipeline), install the dependencies manually:

- Python >= 3.8
- Python packages: ``jinja2``, ``fparser`` (fparser2 API), ``libclang``, ``pcpp``, ``sympy``

.. code-block:: shell

   cd translator-v2
   pip install -r requirements.txt

.. note::
   No system Clang installation is required.  The ``libclang`` PyPI wheel (pinned to 18.1.1 in ``requirements.txt``) is a self-contained ``manylinux`` wheel that bundles its own ``libclang.so`` — no ``apt install libclang-dev`` or equivalent is needed.  The ``fparser`` package provides the ``fparser.two`` (fparser2) API used to parse Fortran source files.

Manual Usage
^^^^^^^^^^^^

When invoked manually, the translator is called as a Python module:

.. code-block:: shell

   python3 op2-translator [options] <source files ...>

Key options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-t <target>``, ``--target <target>``
     - Code-generation target (see `Targets`_ below). Can be specified multiple times to generate several targets in a single invocation.
   * - ``-o <dir>``, ``--out <dir>``
     - Output directory for generated files. Defaults to the directory of the first input source file.
   * - ``-soa``, ``--force_soa``
     - Force Struct of Arrays data layout for all datasets. Equivalent to setting ``OP_AUTO_SOA=1``.
   * - ``-I <dir>``
     - Add a directory to the include search path.
   * - ``-D <define>``
     - Add a preprocessor define.
   * - ``-c <json>``, ``--config <json>``
     - Pass a JSON object of target-specific configuration options (can be repeated).
   * - ``-v``, ``--verbose``
     - Enable verbose output.

Example — generate OpenMP and JIT CUDA variants:

.. code-block:: shell

   python3 op2-translator -t openmp -t c_cuda airfoil.cpp

The translator produces a ``generated/`` subdirectory in the output directory containing one subdirectory per target (e.g. ``generated/airfoil/openmp/``, ``generated/airfoil/c_cuda/``). Each subdirectory contains the generated ``op2_kernels.*`` file(s) ready to be compiled as part of the application build.

C/C++ Targets
^^^^^^^^^^^^^

The following code-generation targets are available for C/C++ applications (``.cpp`` source files):

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Target name
     - Build variant
     - Description
   * - ``seq``
     - ``genseq``
     - Code-generated sequential CPU implementation. Produces a single ``op2_kernels.cpp``. Preferable to the development sequential build for accurate benchmarking.
   * - ``openmp``
     - ``openmp``
     - Multi-threaded CPU implementation using OpenMP. Produces ``op2_kernels.cpp`` with OpenMP pragmas. Set ``OMP_NUM_THREADS`` at runtime to control thread count.
   * - ``cuda``
     - ``cuda``
     - Ahead-of-time compiled NVIDIA GPU implementation using CUDA. Produces ``op2_kernels.cu`` compiled offline by ``nvcc`` during the application build. Requires ``CUDA_INSTALL_PATH`` and ``NV_ARCH`` to be set.
   * - ``hip``
     - ``hip``
     - Ahead-of-time compiled AMD GPU implementation using HIP. Produces ``op2_kernels.cpp`` compiled by ``hipcc`` during the application build. Requires ``HIP_INSTALL_PATH`` and ``HIP_ARCH`` to be set.
   * - ``c_cuda``
     - ``c_cuda``
     - JIT-compiled NVIDIA GPU implementation. Produces ``op2_kernels.cu`` compiled by ``nvcc`` during the application build. Device kernel source strings are embedded in the binary via INCBIN and compiled at application start-up by NVRTC, so the GPU architecture is selected at runtime. Requires ``CUDA_INSTALL_PATH`` and ``nvcc``.
   * - ``c_hip``
     - ``c_hip``
     - JIT-compiled AMD GPU implementation. Produces ``op2_kernels.cpp`` compiled by ``hipcc`` during the application build. Device kernel source strings are embedded in the binary and compiled at application start-up by the HIP RTC library, so the GPU architecture is selected at runtime. Requires ``HIP_INSTALL_PATH`` and ``hipcc``.

All targets have a corresponding distributed-memory MPI variant built automatically when an MPI-enabled OP2 library is available (e.g. the ``cuda`` target produces both ``<app>_cuda`` and ``<app>_mpi_cuda``).

.. note::
   The ``c_cuda`` and ``c_hip`` JIT targets use the same CUDA/HIP runtime libraries as the ``cuda`` and ``hip`` AOT targets, and both also require ``nvcc``/``hipcc`` at build time. The key difference is that device kernel source is embedded in the binary and compiled at application launch via NVRTC/HIP RTC, allowing the GPU architecture to be selected at runtime rather than fixed at build time.

Fortran Targets
^^^^^^^^^^^^^^^

The language is detected automatically from the file extension (``.F90`` or ``.f90``). The same target name strings are used as for C/C++, and the translator selects the appropriate Fortran code-generation scheme.

Example — generate Fortran OpenMP and C_CUDA variants:

.. code-block:: shell

   python3 op2-translator -t openmp -t c_cuda myapp.F90

The following targets are available for Fortran applications:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Target name
     - Description
   * - ``seq``
     - Code-generated sequential Fortran implementation. Produces a ``master_kernel.F90`` containing all loop host subroutines.
   * - ``openmp``
     - Multi-threaded Fortran implementation using OpenMP pragmas. Includes optional SIMD vectorisation of indirect loops.
   * - ``cuda``
     - Native Fortran CUDA implementation using CUDA Fortran (``.CUF``). Device kernels are written in Fortran with ``attributes(device)`` annotations. Requires a CUDA Fortran-capable compiler (NVHPC).
   * - ``c_seq``
     - Fortran interop with C sequential kernels. Generates both a Fortran host file (``.F90``) and a C kernel file (``.cpp``). Useful for transitioning Fortran applications to portable C kernel implementations.
   * - ``c_cuda``
     - Fortran interop with CUDA kernels using JIT compilation via NVRTC. Generates Fortran host code (``.F90``) and CUDA device kernel source (``.cu``). Device kernels are compiled at application start-up. This is the primary recommended GPU target for Fortran applications.
   * - ``c_hip``
     - Fortran interop with HIP kernels using JIT compilation via the HIP RTC library. Generates Fortran host code (``.F90``) and HIP device kernel source (``.hip.cpp``). Device kernels are compiled at application start-up.

.. note::
   For Fortran applications, the ``c_cuda`` and ``c_hip`` JIT targets are the primary recommended GPU backends. The native Fortran ``cuda`` target (CUDA Fortran) is also available but requires the NVHPC compiler.

Choosing Between AOT and JIT GPU Targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This applies to both C/C++ and Fortran applications:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Consideration
     - AOT (``cuda`` / ``hip``)
     - JIT (``c_cuda`` / ``c_hip``)
   * - Build-time GPU compiler required
     - Yes (``nvcc`` / ``hipcc``)
     - Yes (``nvcc`` / ``hipcc``)
   * - Target architecture fixed at build time
     - Yes (via ``NV_ARCH`` for NVIDIA; ``HIP_ARCH`` for AMD)
     - No — resolved at application start-up
   * - Application start-up overhead
     - None
     - Small (kernel compilation on first run and when constant values change at runtime)
   * - Fortran native GPU Fortran available
     - Yes (``cuda`` target with NVHPC)
     - N/A — uses C interop layer
   * - Recommended for
     - Deployments with a known GPU
     - When a large number of constant values are determined at runtime and used in a kernel

SoA Data Layout
^^^^^^^^^^^^^^^

By default, OP2 stores datasets in Array of Structs (AoS) layout. Struct of Arrays (SoA) layout can improve GPU memory access patterns and is often beneficial for CUDA and HIP targets.

To enable SoA layout for all datasets, choose one of:

- Append ``:soa`` to individual ``type`` strings in :c:func:`op_decl_dat` calls in your source for per-dataset control. For example, to store a ``double`` dataset in SoA layout:

  .. code-block:: c

     op_dat p_K = op_decl_dat(cells, 16, "double:soa", K, "p_K");

  Without the suffix the dataset uses the default AoS layout. Note that the data supplied by the user should remain in AoS layout regardless — OP2 performs the conversion internally.
- Pass the ``-soa`` flag to the translator.  When using the OP2 Makefiles, append it to the ``TRANSLATOR`` variable in the application Makefile before including ``c_app.mk``:

  .. code-block:: make

     TRANSLATOR += --force_soa
     include ../../../../../makefiles/common.mk
     include ../../../../../makefiles/c_app.mk

  Or when invoking the translator manually:

  .. code-block:: shell

     python3 op2-translator -soa -t cuda myapp.cpp

Makefile Integration Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the OP2 Makefiles (``makefiles/c_app.mk`` / ``makefiles/f_app.mk``), the following Make variables can be set in your application Makefile *before* the ``include`` of the OP2 Makefile fragment to customise the build:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Description
   * - ``APP_EXTRA_TRANSLATOR_FLAGS``
     - Extra command-line flags appended to every translator invocation for the application. Useful for passing additional ``-I`` include paths or ``-D`` defines that the translator needs to parse your source correctly, without altering the shared ``TRANSLATOR`` variable.
   * - ``VARIANT_FILTER``
     - A Make pattern (default ``%``, matches everything) used to *keep* only the matching build variants. For example, set ``VARIANT_FILTER := %cuda%`` to build only CUDA-related variants.
   * - ``VARIANT_FILTER_OUT``
     - A Make pattern used to *exclude* matching build variants from the set of targets printed and built. For example, ``VARIANT_FILTER_OUT := %hip%`` suppresses all HIP variants.

Example — restrict an application to CUDA variants only and pass an extra define:

.. code-block:: make

   VARIANT_FILTER := %cuda%
   APP_EXTRA_TRANSLATOR_FLAGS := -DUSE_FEATURE_X
   include path/to/makefiles/common.mk
   include path/to/makefiles/c_app.mk

----

Legacy Translator
-----------------

The v1 translator is a collection of standalone Python scripts located in ``translator/c/`` (C/C++) and ``translator/fortran/`` (Fortran). Each script targets a single parallelisation strategy.

.. note::
   The legacy translator is retained for compatibility. For new projects, use the :ref:`OP2 Translator <OP2 Translator>`.

C/C++ Targets
^^^^^^^^^^^^^

Scripts are located in ``translator/c/``. To use, uncomment the desired generator inside ``op2.py`` and invoke:

.. code-block:: shell

   cd translator/c
   python3 op2.py path/to/myapp.cpp

Available generators:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Target
   * - ``op2_gen_seq.py``
     - Sequential (reference)
   * - ``op2_gen_openmp.py`` / ``op2_gen_openmp_simple.py``
     - OpenMP multi-threaded CPU
   * - ``op2_gen_omp_vec.py``
     - OpenMP with SIMD vectorisation
   * - ``op2_gen_cuda.py``
     - CUDA (Fermi)
   * - ``op2_gen_cuda_simple.py``
     - CUDA (Kepler and later, optimised)
   * - ``op2_gen_cuda_simple_hyb.py``
     - Hybrid OpenMP + CUDA
   * - ``op2_gen_mpi_vec.py``
     - MPI + SIMD vectorisation
   * - ``op2_gen_openacc.py``
     - OpenACC
   * - ``op2_gen_openmp4.py``
     - OpenMP 4.0 device offload

Fortran Targets
^^^^^^^^^^^^^^^

Scripts are located in ``translator/fortran/``. To use, uncomment the desired generator inside ``op2_fortran.py`` and invoke:

.. code-block:: shell

   cd translator/fortran
   python3 op2_fortran.py path/to/myapp.F90

Available generators:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Target
   * - ``op2_gen_mpiseq.py`` / ``op2_gen_mpiseq2.py`` / ``op2_gen_mpiseq3.py``
     - MPI + sequential host stubs
   * - ``op2_gen_mpivec.py``
     - MPI + sequential with Intel vectorisation
   * - ``op2_gen_openmp.py`` / ``op2_gen_openmp2.py`` / ``op2_gen_openmp3.py``
     - OpenMP variants
   * - ``op2_gen_openmpINC.py``
     - OpenMP with INC staging
   * - ``op2_gen_cuda.py`` / ``op2_gen_cudaINC.py`` / ``op2_gen_cuda_color2.py`` / ``op2_gen_cuda_permute.py``
     - CUDA variants
   * - ``op2_gen_openacc.py``
     - OpenACC
   * - ``op2_gen_openmp4.py``
     - OpenMP 4.0 device offload

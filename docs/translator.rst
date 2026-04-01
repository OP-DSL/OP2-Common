Code Generation
===============

OP2 uses a code translator to transform a user's sequential OP2 source files into parallelised variants targeting specific hardware backends. The current OP2 translator is based on Jinja2 templating and ``libclang`` parsing and is the recommended tool for all projects. A **legacy translator** is also retained for compatibility, consisting of a collection of standalone Python scripts.

----

OP2 Translator
--------------

Requirements
^^^^^^^^^^^^

- Python >= 3.8
- ``libclang`` (for Debian-based systems: ``sudo apt-get install libclang-dev``)

.. code-block:: shell

   cd translator-v2
   pip install -r requirements.txt

Usage
^^^^^

The translator is invoked as a Python module from the command line:

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
     - JIT-compiled NVIDIA GPU implementation. Produces ``op2_kernels.cu`` containing device kernel source strings that are compiled at application start-up using NVRTC. Avoids a separate offline CUDA compilation step and enables runtime kernel specialisation. Requires the CUDA runtime library at link time.
   * - ``c_hip``
     - ``c_hip``
     - JIT-compiled AMD GPU implementation. Produces ``op2_kernels.cpp`` containing device kernel source strings compiled at application start-up using the HIP RTC library. Requires the HIP RTC library at link time.

All targets have a corresponding distributed-memory MPI variant built automatically when an MPI-enabled OP2 library is available (e.g. the ``cuda`` target produces both ``<app>_cuda`` and ``<app>_mpi_cuda``).

.. note::
   The ``c_cuda`` and ``c_hip`` JIT targets use the same CUDA/HIP runtime libraries as the ``cuda`` and ``hip`` AOT targets. The key difference is that device kernels are compiled at application launch rather than at build time. This removes the dependency on ``nvcc``/``hipcc`` during the build and allows the GPU architecture to be selected at runtime.

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
     - No
   * - Target architecture fixed at build time
     - Yes (via ``NV_ARCH`` / ``HIP_ARCH``)
     - No — resolved at application start-up
   * - Application start-up overhead
     - None
     - Small (kernel compilation on first run)
   * - Fortran native GPU Fortran available
     - Yes (``cuda`` target with NVHPC)
     - N/A — uses C interop layer
   * - Recommended for
     - Production deployments with a known GPU
     - Portable deployments or rapid development

SoA Data Layout
^^^^^^^^^^^^^^^

By default, OP2 stores datasets in Array of Structs (AoS) layout. Struct of Arrays (SoA) layout can improve GPU memory access patterns and is often beneficial for CUDA and HIP targets.

To enable SoA layout, either:

- Set the environment variable before invoking the translator:

  .. code-block:: shell

     OP_AUTO_SOA=1 python3 op2-translator -t cuda myapp.cpp

- Or pass the ``-soa`` flag directly:

  .. code-block:: shell

     python3 op2-translator -soa -t cuda myapp.cpp

- Or append ``:soa`` to the ``type`` string in individual :c:func:`op_decl_dat` calls in your source.

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
   * - ``op2_gen_mpiseq.py`` / ``op2_gen_mpiseq3.py``
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

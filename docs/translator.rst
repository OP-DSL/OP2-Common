Code Generation
===============

OP2 uses a code translator to transform a user's sequential OP2 source files into parallelised variants targeting specific hardware backends. Two generations of translator are provided: a **legacy translator** (v1) written as a collection of standalone Python scripts, and a **next-generation translator** (v2) based on Jinja2 templating and ``libclang`` parsing.

The next-generation translator (v2) is recommended for all new projects.

.. contents:: On this page
   :local:
   :depth: 2

----

Next-Generation Translator (v2)
--------------------------------

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

Targets
^^^^^^^

The following code-generation targets are available for C/C++ applications:

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

Choosing Between AOT and JIT GPU Targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Legacy Translator (v1)
-----------------------

The v1 translator is a collection of standalone Python scripts located in ``translator/c/`` (C/C++) and ``translator/fortran/`` (Fortran). Each script targets a single parallelisation strategy.

.. note::
   The v1 translator is retained for compatibility. For new projects, use the :ref:`next-generation translator <Next-Generation Translator (v2)>`.

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

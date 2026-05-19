Getting Started
===============

Manual Build
------------

Toolchain and Build Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **GNU Make** > 4.2
- **C/C++17 compiler** (GCC, Clang, Cray, Intel, IBM XL, NVHPC).
- Optional: **Fortran compiler** (GFortran, Cray, Intel, IBM XL, NVHPC).
- Optional: **MPI implementation** supporting ``mpicc``, ``mpicxx``, and ``mpif90`` compiler wrappers.
- Optional: **NVIDIA CUDA** >= 11.8
- Optional: **AMD HIP** (ROCm)

These are likely provided in some form by either your distribution's package manager or pre-installed and loaded via commands such as with `Environment Modules <http://modules.sourceforge.net/>`_.

Library Dependencies
^^^^^^^^^^^^^^^^^^^^

These dependencies can also come from package managers or modules, but they must be built with a specific configuration and the same compiler toolchain that you will use to build OP2.

- Optional: `(PT-)Scotch <https://www.labri.fr/perso/pelegrin/scotch/>`_: Used for mesh partitioning. You must build both the sequential Scotch and parallel PT-Scotch with 32-bit indicies (``-DIDXSIZE=32``) and without threading support (remove ``-DSCOTCH_PTHREAD``).
- Optional: `ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`_: Used for mesh partitioning.
- Optional: `KaHIP <https://kahip.github.io/>`_: Used for mesh partitioning.
- Optional: `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_: Used for HDF5 I/O. You may build with and without ``--enable-parallel`` depending on whether MPI support is needed, and then specify both builds using the environment variables listed below.

.. note::
   Building the MPI-enabled OP2 libraries require a parallel HDF5 build. A sequential HDF5 build is needed only for HDF5 support in the sequential OP2 libraries.

Building
^^^^^^^^

(1) Clone the repository:

.. code-block:: shell

   git clone https://github.com/OP-DSL/OP2-Common.git
   cd OP2-Common

(2) Select your compiler:

.. code-block:: shell

   export OP2_COMPILER={gnu, cray, intel, xl, nvhpc}

Alternatively, for a greater level of control:

.. code-block:: shell

   export OP2_C_COMPILER={gnu, clang, cray, intel, xl, nvhpc}
   export OP2_C_CUDA_COMPILER={nvhpc}  # optional
   export OP2_C_HIP_COMPILER={hip}  # optional
   export OP2_F_COMPILER={gnu, cray, intel, xl, nvhpc}   # optional

.. note::
   In some scenarios you may be able to use a profile rather than specifying an ``OP2_COMPILER``. See `Makefile-README <https://github.com/OP-DSL/OP2-Common/blob/master/makefiles/README.md>`_ for more information.

(3) Set library paths (if needed):

.. code-block:: shell

   export PTSCOTCH_INSTALL_PATH=<path/to/ptscotch>
   export PARMETIS_INSTALL_PATH=<path/to/parmetis>
   export KAHIP_INSTALL_PATH=<path/to/kahip>
   export HDF5_{SEQ, PAR}_INSTALL_PATH=<path/to/hdf5>

   export CUDA_INSTALL_PATH=<path/to/cuda/toolkit>
   export HIP_INSTALL_PATH=<path/to/hip/rocm>

.. note::
   You may not need to specify the ``X_INSTALL_PATH`` varaibles if the include paths and library search paths are automatically injected by your package manager or module system.

If you are using CUDA or HIP, you may also specify a comma separated list of target architectures for which to generate code for:

.. code-block:: shell

   export NV_ARCH={Pascal, Volta, ..., Hopper}[,{Pascal, ...}]
   export HIP_ARCH={gfx803, gfx90a, ..., gfx908}[,{gfx803, ...}]

(4) Configure the build: 

.. code-block:: shell

    make -C op2 config

.. note::
   Check the terminal log to ensure the compilers, libraries, and flags are as expected.

(5) Build OP2 library and an example app:

.. code-block:: shell

   make -C op2 -j$(nproc)
   make -C apps/c/airfoil/airfoil_plain/dp -j$(nproc)

.. note::
   A new folder ``generated`` will be created inside the example app folder containing the generated source files. The compiled executable will be in the example app folder.

.. warning::
   MPI builds require an MPI wrapper (``mpicxx``) pointing to the compiler defined by ``OP2_COMPILER``. You can manually set the MPI executable path using ``MPI_INSTALL_PATH``.

Application Build Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^

When building an application, the following parallelisation variants are available as Make targets:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Target
     - Description
   * - ``seq``
     - Single-threaded sequential build using the developer sequential library.
   * - ``genseq``
     - Code-generated sequential build (translator-v2 ``seq`` target). Recommended over ``seq`` for performance measurement.
   * - ``openmp``
     - Multi-threaded CPU build using OpenMP.
   * - ``cuda``
     - NVIDIA GPU build using CUDA (translator-v2 ``cuda`` target, ahead-of-time compiled).
   * - ``hip``
     - AMD GPU build using HIP (translator-v2 ``hip`` target, ahead-of-time compiled).
   * - ``c_cuda``
     - NVIDIA GPU build using CUDA with JIT compilation (translator-v2 ``c_cuda`` target). Device kernels are compiled at application start-up using NVRTC, enabling runtime specialisation.
   * - ``c_hip``
     - AMD GPU build using HIP with JIT compilation (translator-v2 ``c_hip`` target). Device kernels are compiled at application start-up using the HIP RTC library.
   * - ``mpi_<variant>``
     - Distributed-memory MPI variant of any of the above (e.g. ``mpi_cuda``, ``mpi_c_hip``). Requires an MPI-enabled OP2 library build.

For example, to build the JIT CUDA variant of the Airfoil benchmark:

.. code-block:: shell

   make -C apps/c/airfoil/airfoil_plain/dp c_cuda

See :doc:`translator` for details on how to generate the required source files for each variant.

Fortran Application Build Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fortran application variants are prefixed with ``f_``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Target
     - Description
   * - ``f_seq``
     - Sequential Fortran build.
   * - ``f_openmp``
     - OpenMP multi-threaded Fortran build.
   * - ``f_cuda``
     - Native CUDA Fortran build. Requires a CUDA Fortran-capable compiler (NVHPC).
   * - ``f_c_cuda``
     - Fortran interop with JIT CUDA kernels (recommended GPU target for Fortran).
   * - ``f_c_hip``
     - Fortran interop with JIT HIP kernels.
   * - ``f_mpi_<variant>``
     - Distributed-memory MPI variant of any of the above.

For example, to build the Fortran Airfoil benchmark with JIT CUDA:

.. code-block:: shell

   make -C apps/fortran/airfoil f_c_cuda

See :ref:`op2-fortran-api` for the Fortran API reference and :doc:`translator` for Fortran code generation targets.
   
Spack
-----

A Spack package for OP2 is not yet available. Building from source using the manual steps above is currently the recommended installation method.

If you are using a Spack-managed environment, the required compilers and libraries (MPI, CUDA, HDF5) will generally be available through the Spack-generated environment or compiler wrappers. Once the appropriate modules or environment is activated, follow the manual build steps. You do not need to set ``X_INSTALL_PATH`` variables if the include and library paths are already injected by the module system.

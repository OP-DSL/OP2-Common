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
   
Spack
-----

Coming soon.

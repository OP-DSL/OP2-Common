Getting Started
===============

Spack
-----

Coming soon.

Manual Build
------------

Toolchain and Build Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are likely provided in some form by either your distribution's package manager or pre-installed and loaded via commands such as with `Environment Modules <http://modules.sourceforge.net/>`_:

- GNU Make > 4.2
- A C/C++ compiler: Currently supported compilers are GCC, Clang, Cray, Intel, IBM XL and NVHPC.
- (Optional) A Fortran compiler: Currently supported compilers are GFortran, Cray, Intel, IBM XL and NVHPC.
- (Optional) An MPI implementation: Any implementation with the ``mpicc``, ``mpicxx``, and ``mpif90`` wrappers is supported.
- (Optional) NVIDIA CUDA > 9.2

Library Dependencies
^^^^^^^^^^^^^^^^^^^^

These may also be provided from various package managers and modules, however they must be built with a specific configuration and with the same compiler toolchain that you plan on using to build OP2:

- `(PT-)Scotch <https://www.labri.fr/perso/pelegrin/scotch/>`_: Used for mesh partitioning. You must build both the sequential Scotch and parallel PT-Scotch with 32-bit indicies (``-DIDXSIZE=32``) and without threading support (remove ``-DSCOTCH_PTHREAD``).
- `ParMETIS <http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview>`_: Used for mesh partitioning.
- (Optional) `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_: Used for HDF5 I/O. You may build with and without ``--enable-parallel`` (depending on if you need MPI), and then specify both builds via the environment variables listed below.

.. note::
   To build the MPI enabled OP2 libraries you will need a parallel HDF5 build, however you only need a sequential HDF5 build if you need HDF5 support for the sequential OP2 libraries.

Building
^^^^^^^^

First, clone the repository:

.. code-block:: shell

   git clone https://github.com/OP-DSL/OP2-Common.git
   cd OP2-Common

Then, setup toolchain configuration:

.. code-block:: shell

   export OP2_COMPILER={gnu, cray, intel, xl, nvhpc}

Alternatively for a greater level of control:

.. code-block:: shell

   export OP2_C_COMPILER={gnu, clang, cray, intel, xl, nvhpc}
   export OP2_C_CUDA_COMPILER={nvhpc}
   export OP2_F_COMPILER={gnu, cray, intel, xl, nvhpc}

.. note::
   In some scenarios you may be able to use a profile rather than specifying an ``OP2_COMPILER``. See :gh-blob:`makefiles/README.md` for more information.

Then, specify the paths to the library dependency installation directories:

.. code-block:: shell

   export PT_SCOTCH_INSTALL_PATH=<path/to/ptscotch>
   export PARMETIS_INSTALL_PATH=<path/to/parmetis>
   export HDF5_{SEQ, PAR}_INSTALL_PATH=<path/to/hdf5>

   export CUDA_INSTALL_PATH=<path/to/cuda/toolkit>

.. note::
   You may not need to specify the ``X_INSTALL_PATH`` varaibles if the include paths and library search paths are automatically injected by your package manager or module system. For some cases where the dependency libraries are built with non-standard names and provided via injection into the compiler executable you may need to set ``NONSTANDARD_IMPLICIT_LIBS=true``.

Finally, build OP2 and an example app:

.. code-block:: shell

   cd op2
   make -j$(nproc)

   cd ../apps/c/airfoil/airfoil_plain/dp
   make -j$(nproc)

.. warning::
   The MPI variants of the libraries and apps will only be built if an ``mpicxx`` executable is found. It is up to you to ensure that the MPI wrapper wraps the compiler you specify via ``OP2_COMPILER``. To manually set the path to the MPI executables you may use ``MPI_INSTALL_PATH``.

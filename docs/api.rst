OP2 C/C++ Manual
================

The key concept behind OP2 is that unstructured grids can be described by a number of sets. Depending on the application, these sets might be of nodes, edges, faces, cells of a variety of types, far-field boundary nodes, wall boundary faces, etc. Associated with these are data (e.g. coordinate data at nodes) and mappings to other sets (e.g. edge mapping to the two nodes at each end of the edge). All of the numerically-intensive operations can then be described as a loop over all members of a set, carrying out some operations on data associated directly with the set or with another set through a mapping.

OP2 makes the important restriction that the order in which the function is applied to the members of the set must not affect the final result to within the limits of finite precision floating-point arithmetic. This allows the parallel implementation to choose its own ordering to achieve maximum parallel efficiency. Two other restrictions are that the sets and maps are static (i.e. they do not change) and the operands in the set operations are not referenced through a double level of mapping indirection (i.e. through a mapping to another set which in turn uses another mapping to data in a third set).

OP2 currently enables users to write a single program which can be built into three different executables for different single-node platforms:

- Single-threaded on a CPU.
- Multi-threaded using OpenMP for multicore CPU systems.
- Parallelised using CUDA for NVIDIA GPUs.

Further to these there are also in-development versions that can emit SYCL and AMD HIP for parallelisation on a wider range of GPUs. In addition to this, there is support for distributed-memory MPI parallelisation in combination with any of the above. The user can either use OP2’s parallel file I/O capabilities for HDF5 files with a specified structure, or perform their own parallel file I/O using custom MPI code.

.. note::
   This documentation describes the C++ API, but FORTRAN 90 is also supported with a very similar API.

Overview
--------

A computational project can be viewed as involving three steps:

- Writing the program.
- Debugging the program, often using a small testcase.
- Running the program on increasingly large applications.

With OP2 we want to simplify the first two tasks, while providing as much performance as possible for the third.

To achieve the high performance for large applications, a preprocessor is needed to generate the CUDA code for GPUs or OpenMP code for multicore x86 systems. However, to keep the initial development simple, a development single-threaded executable can be created without any special tools; the user’s main code is simply linked to a set of library routines, most of which do little more than error-checking to assist the debugging process by checking the correctness of the user’s program. Note that this single-threaded version will not execute efficiently. The preprocessor is needed to generate efficient single-threaded and OpenMP code for CPU systems.

Figure 1 shows the build process for a single thread CPU executable. The user’s main program (in this case ``jac.cpp``) uses the OP2 header file ``op_seq.h`` and is linked to the appropriate OP2 libraries using ``g++``, perhaps controlled by a Makefile.

Figure 2 shows the build process for the corresponding CUDA executable. The preprocessor parses the user’s main program and produces a modified main program and a CUDA file which includes a separate file for each of the kernel functions. These are then compiled and linked to the OP libraries using g++ and the NVIDIA CUDA compiler nvcc, again perhaps controlled by a Makefile. Figure 3 shows the OpenMP build process which is very similar to the CUDA process except that it uses ``*.cpp`` files produced by the preprocessor instead of ``*.cu`` files.

In looking at the API specification, users may think it is a little verbose in places. For example, users have to re-supply information about the datatype of the datasets being used in a parallel loop. This is a deliberate choice to simplify the task of the preprocessor, and therefore hopefully reduce the chance for errors. It is also motivated by the thought that "programming is easy; it’s debugging which is difficult": writing code isn’t time-consuming, it’s correcting it which takes the time. Therefore, it’s not unreasonable to ask the programmer to supply redundant information, but be assured that the preprocessor or library will check that all redundant information is self-consistent. If you declare a dataset as being of type :c:type:`OP_DOUBLE` and later say that it is of type :c:type:`OP_FLOAT` this will be flagged up as an error at run-time.

.. todo::
   Bring across LaTeX figures.

Initialisation and Termination
------------------------------

.. c:function:: void op_init(int argc, char **argv, int diags_level)

   This routine must be called before all other OP routines. Under MPI back-ends, this routine also calls :c:func:`MPI_Init()` unless its already called previously.

   :param argc: The number of command line arguments.
   :param argv: The command line arguments, as passed to :c:func:`main()`.
   :param diags_level: Determines the level of debugging diagnostics and reporting to be performed.

   The values for **diags_level** are as follows:

   - 0: None.
   - 1: Error-checking.
   - 2: Info on plan construction.
   - 3: Report execution of parallel loops.
   - 4: Report use of old plans.
   - 7: Report positive checks in :c:func:`op_plan_check()`

.. c:function:: void op_exit()

   This routine must be called last to cleanly terminate the OP2 runtime. Under MPI back-ends, this routine also calls :c:func:`MPI_Finalize()` unless its has been called previously. A runtime error will occur if :c:func:`MPI_Finalize()` is called after :c:func:`op_exit()`.

.. c:function:: op_set op_decl_set(int size, char *name)

   This routine declares a set.

   :param size: Number of set elements.
   :param name: A name to be used for output diagnostics.
   :returns: A set ID.

.. c:function:: op_map op_decl_map(op_set from, op_set to, int dim, int *imap, char *name)

   This routine defines a mapping between sets.

   :param from: Source set.
   :param to: Destination set.
   :param dim: Number of mappings per source element.
   :param imap: Mapping table.
   :param name: A name to be used for output diagnostics.

.. c:function:: void op_decl_const(int dim, char *type, T *dat)

   This routine defines constant data with global scope that can be used in kernel functions.

   :param dim: Number of data elements. For maximum efficiency this should be an integer literal.
   :param type: The type of the data as a string. This can be either intrinsic (`"float"`, `"double"`, `"int"`, `"uint"`, `"ll"`, `"ull"`, or "`bool`") or user-defined.
   :param dat: A pointer to the data, checked for type consistency at run-time.

.. note::
   If **dim** is 1 then the variable is available in the kernel functions with type :c:type:`T`, otherwise it will be available with type :c:type:`T*`.

.. warning::
   If the executable is not preprocessed, as is the case with the development sequential build, then you must define an equivalent global scope variable to use the data within the kernels.

.. c:function:: op_dat op_decl_dat(op_set set, int dim, char *type, T *data, char *name)

   This routine defines a dataset.

   :param set: The set the data is associated with.
   :param dim: Number of data elements per set element.
   :param type: The datatype as a string, as with :c:func:`op_decl_const()`. A qualifier may be added to control data layout - see `Dataset Layout`_.
   :param data: Input data of type :c:type:`T` (checked for consistency with **type** at run-time). The data must be provided in AoS form with each of the **dim** elements per set element contiguous in memory.
   :param name: A name to be used for output diagnostics.

.. note::
   At present **dim** must be an integer literal. This restriction will be removed in the future but an integer literal will remain more efficient.

.. c:function:: op_dat op_decl_dat_temp(op_set set, int dim, char *type, T *data, char *name)

    Equivalent to :c:func:`op_decl_dat()` but the dataset may be released early with :c:func:`op_free_dat_temp()`.

.. c:function:: void op_free_dat_temp(op_dat dat)

   This routine releases a temporary dataset defined with :c:func:`op_decl_dat_temp()`

   :param dat: The dataset to free.


Parallel Loops
--------------

.. c:function:: void op_par_loop(void (*kernel)(...), char *name, op_set set, op_arg arg1, op_arg arg2, ..., op_arg argN)

   This routine executes a parallelised loop over the given **set**, with arguments provided by the :c:func:`op_arg_gbl()`, :c:func:`op_arg_dat()`, and :c:func:`op_opt_arg_dat()` routines.

   :param kernel: The kernel function to execute. The number of arguments to the kernel should match the number of :c:type:`op_arg` arguments provided to this routine.
   :param name: A name to be used for output diagnostics.
   :param set: The set to loop over.
   :param arg1..N: The arguments passed to each invokation of the kernel.

.. c:function:: op_arg op_arg_gbl(T* data, int dim, char *type, op_access acc)

   This routine defines an :c:type:`op_arg` that may be used either to pass non-constant read-only data or to compute a global sum, maximum or minimum.

   :param data: Source or destination data array.
   :param dim: Number of data elements.
   :param type: The datatype as a string. This is checked for consistency with **data** at run-time.
   :param acc: The access type.

   Valid access types for this routine are:

   - :c:data:`OP_READ`: Read-only.
   - :c:data:`OP_INC`: Global reduction to compute a sum.
   - :c:data:`OP_MAX`: Global reduction to compute a maximum.
   - :c:data:`OP_MIN`: Global reduction to compute a minimum.

.. c:function:: op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char *type, op_access acc)

   This routine defines an :c:type:`op_arg` that can be used to pass a dataset either directly attached to the target :c:type:`op_set` or attached to an :c:type:`op_set` reachable through a mapping.

   :param dat: The dataset.
   :param idx: The per-set-element index into the map to use. You may pass a negative value here to use a range of indicies - see `Vector Arguments`_. This argument is ignored if the identity mapping is used.
   :param map: The mapping to use. Pass :c:data:`OP_ID` for the identity mapping if no mapping indirection is required.
   :param dim: The dimension of the dataset, checked for consistency at run-time.
   :param type: The datatype of the dataset as a string, checked for consistency at run-time.
   :param acc: The access type.

   Valid access types for this routine are:

   - :c:data:`OP_READ`: Read-only.
   - :c:data:`OP_WRITE`: Write-only.
   - :c:data:`OP_RW`: Read and write.
   - :c:data:`OP_INC`: Increment or global reduction to compute a sum.

.. warning::
   :c:data:`OP_WRITE` and :c:data:`OP_RW` accesses *must not* have any potential data conflicts. This means that two different elements of the set cannot, through a map, reference the same elements of the dataset.

   Furthermore with :c:data:`OP_WRITE` the kernel function *must* set the value of all **dim** components of the dataset. If this is not possible then :c:data:`OP_RW` access should be specified.

.. note::
   At present **dim** must be an integer literal. This restriction will be removed in the future but an integer literal will remain more efficient.

.. c:function:: op_arg op_opt_arg_dat(op_dat dat, int idx, op_map map, int dim, char *type, op_access acc, int flag)

   This routine is equivalent to :c:func:`op_arg_dat()` except for an extra **flag** parameter that governs whether the argument will be used (non-zero) or not (zero). This is intended to ease development of large application codes where many features may be enabled or disabled based on flags.

   The argument must not be dereferenced in the user kernel if **flag** is set to zero. If the value of the flag needs to be passed to the kernel then use an additional :c:func:`op_arg_gbl()` argument.

Advanced Features
-----------------

Dataset Layout
^^^^^^^^^^^^^^



Vector Arguments
^^^^^^^^^^^^^^^^

HDF5 I/O
--------

Other I/O and Utilities
-----------------------

.. c:function:: void op_diagnostic_output()

   This routine prints diagnostics relating to sets, mappings and datasets.

Developer Guide
===============

This guide covers the internal algorithms and implementation details of OP2. It is intended for those who are developing or extending OP2; those who only need to use OP2 in an application should instead read the :ref:`op2-c-api` or :ref:`op2-fortran-api`.

.. note::
   This page incorporates material originally published as two PDF technical reports by Mike Giles, Gihan R. Mudalige, and Istvan Reguly (December 2013).  The original PDFs remain available at `dev.pdf <https://op-dsl.github.io/docs/OP2/dev.pdf>`__ and `mpi-dev.pdf <https://op-dsl.github.io/docs/OP2/mpi-dev.pdf>`__. **It is strongly recommended to read these reports in conjunction with this guide, as they contain the core concepts in OP2 that underpin the DSL.**


Parallelisation Architecture
-----------------------------

OP2 uses a **hierarchical parallelism** model with two principal levels:

- **Distributed-memory level:** OP2 is parallelised across distributed-memory clusters using MPI message-passing.  The domain is partitioned among the compute nodes and import/export halos are constructed for communication.  Data conflicts when incrementing indirectly-referenced datasets are avoided by using an "owner-compute" model, in which each process performs the computations required to update data owned by its partition.  This may involve some redundant computation — for example, if an edge computation updates nodes belonging to two different partitions, both partitions must execute that computation — but this stays minimal when each compute node has GBs of memory.

- **Shared-memory/thread level:** within a single node OP2 supports:

  - NVIDIA GPUs via CUDA (and optionally HIP for AMD GPUs), with multiple thread blocks and multiple threads per block.
  - CPUs via OpenMP multi-threading, with optional vectorisation via SSE/AVX/AVX-512.

A key design constraint is that **memory bandwidth is a major bottleneck** in both the GPU (graphics memory ↔ GPU) and CPU (main memory ↔ CPU) cases, and that available fast (shared/L1) memory is very limited.  These assumptions drive the execution-plan and data-layout choices described below.


GPU Parallelisation and Execution Plans
----------------------------------------

Execution Plans
~~~~~~~~~~~~~~~

Because GPU execution is driven by data residency in graphics memory, the blocking for each parallel loop can be chosen independently for each loop, unlike in MPI where repartitioning is expensive.  Inspired by FFTW, OP2 constructs for each parallel loop an execution **plan** — a customised blocking for the GPU that makes optimum use of the shared memory on each streaming multiprocessor (SM), given the precise memory requirements of that loop.

For *indirect loops* (loops that access datasets indirectly through a mapping), data conflicts on ``OP_INC`` arguments are resolved using **hardware atomics** (``atomicAdd``) by default.  Each thread computes its increment into a thread-local accumulator and then atomically adds it to global memory — eliminating the need for any synchronisation between threads.  This is the default strategy for all GPU targets (``cuda``, ``hip``, ``c_cuda``, ``c_hip``) in the v2 translator.

An alternative **two-level colouring** strategy (``color2`` mode) is also available for cases where atomic operations are undesirable:

1. **Block colouring:** blocks of the same colour have no data conflict, so each colour is dispatched as a separate kernel call with a synchronisation barrier between colours.
2. **Thread colouring:** within each block, threads are coloured and increments applied one colour at a time with ``__syncthreads`` between thread colours.  This avoids atomics but incurs some warp divergence overhead.

The strategy is selected at code-generation time; the default is ``atomics=True, color2=False``.

Renumbering
~~~~~~~~~~~

Breaking the execution set into contiguous blocks is good for direct datasets (they will be in contiguous memory, giving coalesced transfers), but for indirect datasets each mini-partition should ideally reference elements held close together in graphics memory.  OP2 therefore provides an optional renumbering step — for example, recursive coordinate bisection — to improve data locality so that neighbouring elements have proximate memory locations.

Plan Construction (``op_plan_core``)
--------------------------------------

The plan-construction algorithm lives in ``op_plan_core`` inside ``op2/src/core/op_rt_support.cpp``.  It is invoked whenever a parallel loop has at least one indirect dataset.  The following concepts are central to it.

**Sets, Datasets, and Indirect Datasets**

A *set* is a collection of abstract elements over which parallel loops execute (e.g. nodes, edges, cells).  A *dataset* (``op_dat``) is data associated with a set, such as flow variables or edge weights.  An *indirect dataset* is one that is referenced through a mapping from another set.  More than one argument of a loop can address the same indirect dataset.

**Indirect Dataset Local Renumbering**

For each plan block, and each indirect dataset used in it, the algorithm:

1. Builds a list of all references to the dataset.
2. Sorts the list and removes duplicates, defining the local-to-global index mapping.
3. Inverts that mapping (via a large work array) to obtain global-to-local indices for the elements in the block.
4. Creates a new copy of the mapping table using the new local indices.

.. note::
   In cases where two arguments share the same original mapping table (e.g. flow variables and flux residuals both indexed via the edge-to-node mapping), the current implementation duplicates the renumbered mapping table.  Deduplication is a known future improvement.

**Element and Block Colouring** *(used in ``color2`` mode)*

When the ``color2`` strategy is selected, colouring is used at two levels to avoid data conflicts.

*Element colouring* assigns a colour to each element such that no two elements of the same colour reference the same indirect dataset element.  The efficient implementation uses a 32-bit integer as a bitmask per indirect dataset element, with bit ``i`` set when the element has been referenced by a colour-``i`` element.  The ``ffs`` instruction finds the first zero bit.  A single pass handles up to 32 colours; in the rare case that more are needed the loop is repeated with the bitmasks re-initialised for the new set of colours.

*Block colouring* follows the same algorithm but considers all indirect dataset elements referenced by all elements in a block (not just by a single element).

When ``atomics=True`` (the default), colouring is still computed to determine ``ncolors_core`` and ``ncolors_owned`` (used for MPI communication overlap), but the actual element ordering used at execution time is the flat permutation in ``col_reord`` — threads within a kernel call process contiguous elements regardless of colour and use ``atomicAdd`` for all indirect increments.

**Block Mapping**

The ``blkmap`` / ``col_reord`` arrays map from the colour-grouped or flat element ordering to the stored block order, along with ``col_offsets`` (per-block colour start/end) and ``ncolblk`` (per-colour block counts).

**``op_plan`` Struct**

The ``op_plan`` struct holds:

- ``nthrcol`` / ``thrcol``: number of thread colours per block, and colour assignment per element.
- ``col_reord``: permutation of elements ordered by block colour (flat reordering).
- ``col_offsets``: per-block offsets to the start of each thread-colour group.
- ``color2_offsets``: offsets for the flat (cross-block) colouring used by some backends.
- ``offset`` / ``offset_d``: primary-set offset for the start of each block (host and Fortran-GPU device copies).
- ``ind_map`` / ``ind_maps``: backing array and 2D pointer array for local ↔ global renumbering of indirect datasets.
- ``ind_offs`` / ``ind_sizes``: block offsets and block sizes for each indirect dataset.
- ``nindirect``: total size (across all blocks) for each indirect dataset.
- ``loc_map`` / ``loc_maps``: backing array and 2D pointer array (``short`` type) mapping arguments to local shared-memory indices.
- ``nelems`` / ``nelems_d``: number of primary-set elements per block (host and Fortran-GPU device copies).
- ``ncolors_core``: number of block colours that contain only core elements (used in MPI+GPU to separate locally-computable work from halo-dependent work).
- ``ncolors_owned``: number of block colours for blocks whose elements are all owned (no indirect-reduction contributions from imported elements).
- ``ncolors`` / ``ncolblk`` / ``blkmap`` / ``blkmap_d``: total block colour count, per-colour block counts, block mapping (host and Fortran-GPU device copies).
- ``nsharedCol``: array of per-block-colour shared memory requirements (bytes).
- ``nshared``: maximum shared memory required over all blocks.

Device arrays (``*_d`` variants) are Fortran-backend device copies.  All other execution-plan arrays are generated on the host; backends copy them to the device and retain only the device pointers.

The ``op_plan_check`` routine validates the plan's self-consistency; it is run automatically based on the diagnostics level ``OP_DIAGS``.


Data Layout
-----------

A key implementation choice is how to store datasets with multiple items per set element (e.g. four flow variables per cell in the airfoil test case).

- **Structure of Arrays (SoA):** for each component, store data for all set elements as a contiguous block.  Natural for vector hardware (CRAY-style); achieves memory coalescence for direct loops.
- **Array of Structures (AoS):** for each set element, store all components together.  Preferred for cache hierarchies: once a cache line is loaded to access one component, the remaining components are immediately available.

OP2 defaults to **AoS** because with indirect addressing the worst-case SoA cache line efficiency is 1/K of AoS (where K is the number of items per element), since a random access to one item loads an entire cache line of the same component for other elements rather than loading all K components of the target element.  For K=4 (airfoil), AoS can be up to 4× more efficient in data transfer.

However, OP2 also supports **automatic AoS→SoA conversion** at runtime, enabling SoA layout on the device for backends where it is more appropriate (e.g. when access patterns are more regular).  This is enabled by passing ``OP_AUTO_SOA`` as a runtime argument to ``op_init``, or on a per-dataset basis by appending ``:soa`` to the type string in ``op_decl_dat`` (e.g. ``"double:soa"``).  When enabled, OP2 prints ``Enabling Automatic AoS->SoA Conversion`` at startup.  ``op_set_core`` also exposes a ``stride`` field used by GPU backends for SoA indexing.

Memory coalescence for direct loops is still achieved in AoS by staging through shared memory:

.. code-block:: cuda

   // Coalesced load into shared memory (direct loop)
   for (int m = 0; m < 4; m++)
       ((float *)arg_s)[tid + m*nelems] = arg0[tid + m*nelems + offset*4];

   // Copy from shared to local registers
   for (int m = 0; m < 4; m++)
       arg0_l[m] = ((float *)arg_s)[m + tid*4];

Each warp gets its own shared memory scratchpad to eliminate ``__syncthreads`` between warps:

.. code-block:: cuda

   char *arg_s = shared + offset_s * (threadIdx.x / OP_WARPSIZE);

For indirect loops, the mapping indices ``ind_arg_map[]`` are stored in sorted ascending order so that the resulting accesses into the indirect array are also ascending, maximising coalesced accesses.


Code Generation
---------------

The translator (see :doc:`translator`) generates backend-specific code for each ``op_par_loop`` call.

**CUDA stub routine**

The stub is the host function called from user code.  It:

1. Transfers any local constants or global-reduction arrays to the GPU.
2. Calls ``op_plan_get`` to retrieve or construct a plan.
3. With the default **atomics** strategy: iterates in two rounds — first over core elements (``[0, set->core_size)``), then over exec-halo elements (``[set->core_size, set->size + set->exec_size)``) after the MPI wait.  Each kernel call dispatches a contiguous range of elements; indirect increments are applied with ``atomicAdd``.
   With the legacy **color2** strategy: loops over block colours dispatching one kernel per colour; a synchronisation barrier between colours prevents data conflicts.
4. After the kernel, for global reductions, fetches partial results back to the CPU and combines them.

**CUDA kernel routine**

The generated kernel:

- Declares all arguments and local working arrays (likely held in registers by nvcc).
- Retrieves block ID via the ``blkmap`` mapping.
- Sets shared-memory pointers for indirect datasets.
- Zeroes incremented indirect data; copies read-only indirect data into shared memory or relies on L2 cache for reuse.
- Loops over set elements; with atomics, accumulates increments in thread-local arrays and applies them via ``atomicAdd``; with ``color2``, applies thread colouring with ``__syncthreads`` between colours.
- Writes back indirect data; completes global reductions.

.. note::
   ``op_par_loop`` array dimensions should always be specified as literal constants so that nvcc can map them to registers.

**OpenMP files**

The OpenMP execution pattern is similar to CUDA but without element colouring within a mini-partition: the implementation loops over elements, calling user kernels with pointers to global memory with no staging.  Global reductions use per-thread local variables combined in a final sequential step.

**Generated sequential files (``genseq``)**

While OP2 can run sequentially via the generic ``op_seq.h`` header (used for debugging), the translator generates loop-specific sequential stub files (``genseq``) that reduce overhead by spelling out mapping pointers explicitly and reading indirect mappings only once per iteration.


Global Reductions
-----------------

**Summation**

Each thread block has a separate entry in a device array initialised to zero.  Thread-level partial sums are combined within the block via a binary-tree reduction using shared memory (based on the CUDA SDK reduction example).  The per-block sums are transferred back to the CPU and added to the starting value after the kernel call.

**Min/Max**

Each thread block entry is initialised to the current CPU input value.  Thread-level min/max values are combined via a binary-tree approach and used to update the global value.  The final per-block values are combined on the CPU.

**User-defined types**

Summation works directly.  Min/max works provided the user has correctly overloaded the inequality operators to form a total order.

In MPI mode, after the per-process loop reduction, ``op_mpi_reduce()`` calls ``MPI_Allreduce`` with the appropriate operation and data type.


MPI Implementation
------------------

Introduction
~~~~~~~~~~~~

The MPI implementation in OP2 follows the same "owner-compute" parallelisation strategy as the original OPlus library.  The domain is partitioned among compute nodes; import/export halos are constructed for message passing; and each process computes only the updates that affect data it owns.

Parallel Startup
~~~~~~~~~~~~~~~~

An OP2 application running under MPI has multiple copies of the same program running as separate MPI processes.  Data initialisation supports two approaches:

1. **User-managed I/O:** the application developer handles file I/O and calls ``op_decl_set``, ``op_decl_map``, and ``op_decl_dat`` with the partition of the data local to each MPI rank.
2. **HDF5 parallel I/O:** OP2 provides ``op_decl_set_hdf5``, ``op_decl_map_hdf5``, and ``op_decl_dat_hdf5`` routines that read sets, mappings, and data from a prescribed HDF5 file format.

In either case, the initial distribution serves only as an input convention; OP2 repartitions the data, migrates all datasets and mappings to the correct MPI processes, and renumbers mapping tables.

.. note::
   If the application allocates arrays passed to ``op_decl_map`` or ``op_decl_dat``, the MPI backend makes an internal copy (halo creation may reallocate memory and invalidate the original pointer).  The application is responsible for freeing its copy; OP2 frees its internal copy on ``op_exit()``.  When using HDF5 I/O, OP2 manages cleanup entirely.

Runtime arguments
~~~~~~~~~~~~~~~~~

Several runtime behaviours are controlled by arguments passed to ``op_init`` (parsed by ``op_set_args``), or by the equivalent environment variables:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Effect
   * - ``OP_BLOCK_SIZE=<n>``
     - CUDA/HIP thread-block size override.
   * - ``OP_PART_SIZE=<n>``
     - Partition size (mini-partition element count) override.
   * - ``OP_CACHE_LINE_SIZE=<n>``
     - Cache line size assumption used in plan construction.
   * - ``OP_NO_REALLOC``
     - Disables internal data copy on ``op_decl_dat``/``op_decl_map``; the user's pointer is used directly.
   * - ``OP_TEST_FREQ=<n>``
     - Frequency of ``MPI_Test`` polling during computation-communication overlap.
   * - ``-gpudirect``
     - Enable NVIDIA GPUDirect for direct GPU-to-GPU MPI transfers.
   * - ``OP_AUTO_SOA``
     - Automatically convert all dataset storage to SoA layout on the device.
   * - ``OP_PARTIAL_EXCHANGE``
     - Enable partial MPI halo exchange (per-mapping halos instead of full set halos).
   * - ``OP_HYBRID_BALANCE=<f>``
     - GPU-to-CPU partition size ratio for hybrid CPU/GPU runs.  Can also be set via the ``OP_HYBRID_BALANCE`` environment variable.
   * - ``OP_MAPS_BASE_INDEX=<0|1>``
     - Mapping table base index: 0 (C/C++ default) or 1 (Fortran-style).
   * - ``OP_CUDA_REDUCTIONS_MIB=<n>``
     - Size (MiB) of the GPU reduction buffer.

Constructing Halo Lists
~~~~~~~~~~~~~~~~~~~~~~~

After partitioning, OP2 calls ``op_halo_create()`` (defined in ``op2/src/mpi/op_mpi_core.cpp``) to classify set elements into five categories:

``core``
    An element is *core* to the MPI process that owns it if every element referenced through any mapping from that element is also core on that process.

``export execute halo (eeh)``
    An element belongs to the *export execute halo* if at least one element it references via any mapping is NOT core on the local process.  The element must be sent to foreign processes that own those referenced elements so that those processes can execute the loop and update their data.

``import execute halo (ieh)``
    An element belongs to the *import execute halo* if it is referenced by an element on a foreign process.  The foreign element must be imported to allow the local process to compute all required contributions.

``import non-execute halo (inh)``
    Any element referenced (via any mapping, including mappings from the ``ieh``) by a local element, that is NOT already in the ``ieh``, must also be imported.  These elements do not require a loop execution but their data values are needed.

``export non-execute halo (enh)``
    Elements in ``core`` that are referenced by elements on foreign processes must be exported (if not already in ``eeh``).  ``enh`` is a subset of ``core``.

The halo list structure (``halo_list_core``) stores, for each set:

.. code-block:: c

   typedef struct {
       op_set set;      // set to which this list belongs
       int size;        // number of elements in the list
       int *ranks;      // MPI ranks to export to / import from
       int  ranks_size; // number of neighbouring MPI ranks
       int *disps;      // displacement for each rank's element list
       int *sizes;      // number of elements per rank
       int *list;       // the full element list
   } halo_list_core;

Four global arrays — ``OP_export_exec_list``, ``OP_import_exec_list``, ``OP_import_nonexec_list``, ``OP_export_nonexec_list`` — are indexed by ``set->index``.

Halo creation in ``op_halo_create()`` proceeds in 12 steps:

1. Build ``eeh`` export lists (elements whose referenced data spans a partition boundary).
2. Exchange ``eeh`` lists between neighbours to construct ``ieh`` import lists.
3. Exchange mapping table entries using the ``eeh``/``ieh`` lists; imported entries are appended to each ``op_map->map`` array.
4. Build ``inh`` import lists using the now-extended mapping tables.
5. Exchange ``inh`` lists to construct ``enh`` export lists.
6. Exchange execute-halo data; append to each ``op_dat->data`` array.
7. Exchange non-execute halo data; append to each ``op_dat->data`` array.
8. Renumber all mapping tables to use local indices.
9. Create MPI send buffers (``op_mpi_buffer`` struct) for each ``op_dat``.
10. Separate core elements into a contiguous block (index range ``[0, set->core_size)``); elements in ``[set->core_size, set->size)`` are export-execute-halo (``eeh``) elements.  The full iteration range, including imported halo elements, extends to ``set->size + set->exec_size + set->nonexec_size``.  The ``op_set_core`` struct exposes ``core_size``, ``exec_size``, and ``nonexec_size`` for this purpose.
11. Save the original set-element ordering (stored in the ``part`` struct) for ``op_fetch_data()`` and output routines.
12. Free temporaries; compute a rough estimate of the average worst-case halo size.

After halo creation, the element ordering within each set is: ``core | eeh | ieh | inh``, with sizes ``set->core_size``, ``set->size - set->core_size``, ``set->exec_size``, ``set->nonexec_size``.

Halo Exchanges
~~~~~~~~~~~~~~

When ``op_par_loop`` executes under MPI:

- **Direct loops** (no indirect arguments) loop only over the local set size; no halo exchange is needed.
- **Indirect loops** use the algorithm:

  1. For each indirect ``op_arg`` with access ``OP_READ`` or ``OP_RW`` and a dirty bit, trigger a halo exchange and clear the dirty bit.
  2. If all indirect arguments are ``OP_READ``, loop over ``set->size``; otherwise loop over ``set->size + ieh->size``.
  3. After the loop, set the dirty bit for each ``op_arg`` with access ``OP_INC``, ``OP_WRITE``, or ``OP_RW``.

``op_exchange_halo()`` (``op2/src/mpi/op_mpi_rt_support.cpp``) checks the conditions, packs halo data into pre-defined send buffers, issues non-blocking ``MPI_Isend`` / ``MPI_Irecv`` operations, and returns immediately.  The ``op_par_loop`` is structured so that:

1. All ``op_exchange_halo()`` calls are issued first.
2. Core elements are computed (no halo data required).
3. ``op_wait_all()`` is called.
4. The remaining elements (``ieh``) are computed.

This maximises overlap of computation with communication.

Partial Halo Exchange
~~~~~~~~~~~~~~~~~~~~~

For loops over a *boundary set* with sparse connectivity to an internal set, exchanging the full internal halo is wasteful.  OP2 implements a partial halo exchange via ``op_halo_permap_create()``: a per-mapping halo is computed and used if the number of mapping table entries crossing partition boundaries (for that map) is less than 30% of the full halo size for the exchanged set.

Global Operations
~~~~~~~~~~~~~~~~~

For ``op_arg`` of type ``OP_ARG_GBL`` (global reduction), contributions from the ``ieh`` are excluded (a dummy value is passed for those elements).  After the loop, ``op_mpi_reduce()`` calls ``MPI_Allreduce`` with the appropriate type and operation (``OP_INC``, ``OP_MAX``, or ``OP_MIN``).

Fetching Data
~~~~~~~~~~~~~

``op_fetch_data()`` returns the current values of an ``op_dat``'s data array in the original element order that was supplied to OP2 (before repartitioning).  The implementation copies the current data and reorders it using the original global indices saved in step 11 of halo creation.

Performance Instrumentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

OP2 provides two timer routines:

- ``op_timers_core()`` (``op2/src/core/op_lib_core.cpp``) — elapsed time on a single MPI process.
- ``op_timers()`` (``op2/src/mpi/op_mpi_decl.cpp``) — includes an implicit ``MPI_Barrier`` to measure wall time across the whole MPI universe.

Per-loop statistics are accumulated in an ``op_mpi_kernel`` struct:

.. code-block:: c

   typedef struct {
       UT_hash_handle hh;              // uthash intrusive handle
       char name[NAMESIZE];            // kernel name
       double time;                    // total compute + comm-overlap time
       int count;                      // number of kernel invocations
       int num_indices;                // number of op_dat communication entries
       op_dat_mpi_comm_info *comm_info;// per-dat MPI communication info
       int cap;                        // capacity of comm_info array
   } op_mpi_kernel;

MPI message monitoring can be enabled at compile time with ``-DCOMM_PERF``.  On ``op_exit()``, all halo lists, MPI send buffers, and performance-measurement tables are freed.


HDF5 File I/O
-------------

OP2's HDF5 file format follows the layout of the NACA 0012 airfoil mesh generator (``apps/mesh_generators/naca0012.m``).  The structure and contents of a generated HDF5 file can be inspected with the ``h5dump`` utility.

Parallel HDF5 I/O routines are documented in :ref:`op2-c-api`; an example use is in the ``apps/c/airfoil/airfoil_hdf5`` application.  OP2 assumes that each HDF5 file uses a flat keyword-based hierarchy where set names, map names, and dat names serve as dataset keys.


Partitioning
------------

Distributing an unstructured mesh across the MPI universe requires a mesh partitioner to minimise halo sizes.  Once all ``op_decl_*`` calls have been made, the application calls ``op_partition()`` with arguments that select the partitioner and identify the primary set (and its coordinates or connectivity), triggering automatic redistribution.

Supported partitioners:

- **KaHIP k-way:** ``op_partition("KAHIP", "KWAY", ...)`` — k-way graph partitioning using KaHIP.
- **ParMetis geometric:** ``op_partition("PARMETIS", "GEOM", ...)`` — uses node coordinates, suitable for geometrically regular meshes.
- **ParMetis k-way:** ``op_partition("PARMETIS", "KWAY", ...)`` — graph-based k-way partitioning.
- **ParMetis geometric k-way:** ``op_partition("PARMETIS", "GEOMKWAY", ...)`` — combined geometric and k-way.
- **PT-Scotch k-way:** ``op_partition("PTSCOTCH", "KWAY", ...)`` — alternative graph-based partitioner.
- **Inertial coordinate bisection:** ``op_partition("INERTIAL", ...)`` — built-in, for 3D meshes.
- **User-defined:** ``op_partition_external()`` — partitioning array supplied externally via an ``op_dat``.
- **Random:** ``op_partition("RANDOM", ...)`` — for debugging only.

The **primary set** (e.g. nodes, given its XY coordinates) is partitioned first.  All secondary sets (e.g. cells, edges) inherit the partitioning from the primary set: for each mapping table, the set element that maximises overlap with an already-partitioned set is assigned to that partition.  After assignment, ``migrate_all()`` migrates data and mappings to new MPI ranks and ``renumber_maps()`` renumbers mapping table entries.

Mesh Renumbering
~~~~~~~~~~~~~~~~

OP2 implements a mesh renumbering routine using the Gibbs–Poole–Stockmeyer algorithm from PT-Scotch (``op2/src/externlib/op_renumber.cpp``) to improve cache locality: elements that are executed consecutively should reference data stored at adjacent memory locations.

.. note::
   This renumbering currently runs only on a single node (no MPI support).  The recommended workflow is: read an unoptimised mesh into an HDF5 file, apply renumbering to produce an optimised mesh HDF5 file, and use that optimised file for both single-node and distributed-memory runs.


Heterogeneous and Hybrid Backends
-----------------------------------

GPU Cluster (MPI + CUDA)
~~~~~~~~~~~~~~~~~~~~~~~~~

On a GPU cluster, OP2 assigns **one MPI process per GPU**.  Nodes with multiple GPUs run multiple MPI processes; each process selects an available GPU device at runtime.

To overlap computation with communication on the GPU, the execution is split into rounds separated by an MPI wait.  With the default **atomics** strategy, the pseudo-code for one ``op_par_loop`` on a GPU cluster is:

.. code-block:: text

   trigger non-blocking MPI halo exchanges for all dirty op_dats

   round 0: execute GPU kernel over core elements [0, core_size)
            (no halo data needed — overlaps with MPI communication)
   round 1: wait for all MPI communications to complete
            copy import halo data from host to GPU
            execute GPU kernel over exec-halo elements [core_size, size + exec_size)

With the legacy **color2** strategy, the execution uses block colours to achieve the same overlap:

.. code-block:: text

   for each op_dat requiring a halo exchange:
       execute CUDA kernel to gather export halo data
       copy export halo data from GPU to host
       start non-blocking MPI communication

   for each colour i:
       if colour == ncolors_core:
           wait for all MPI communications to complete
           for each op_dat requiring a halo exchange:
               copy import halo data from host to GPU
       execute CUDA kernel for colour-i mini-partitions

In both variants, the key property is that ``ncolors_core`` (atomics) or the core-element range marks the boundary between locally-computable work and halo-dependent work.

.. note::
   The above uses PCIe-bridged GPU ↔ host copies for halo data.  When built with GPUDirect support, the intermediate host copy is eliminated and MPI send/receive operations transfer data directly between GPUs over the network fabric.

CPU Cluster (MPI + OpenMP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-threaded CPU cluster implementation (MPI + OpenMP) follows the same design as the GPU cluster, except that all data resides in CPU main memory and there is no GPU data transfer step.

Hybrid CPU/GPU Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

OP2 supports fully hybrid execution where some MPI processes run on GPUs and others on CPUs within the same run.  On a node with *N* GPUs, the first *N* MPI processes assigned to that node acquire a GPU; the remainder become CPU processes.

Generated kernel files include code for both MPI+CUDA and MPI+OpenMP so each MPI process can call the appropriate kernel at runtime based on its hardware assignment.

The key challenge for hybrid execution is **load balancing**: the relative performance of CPU and GPU varies across different loops.  The recommended approach (currently the only supported one) is heterogeneous load balancing via ParMetis.  For example:

.. code-block:: bash

   export OP_HYBRID_BALANCE=2.5
   mpirun -np 3 ./my_op2_app

This configuration on a two-CPU + one-GPU node assigns the GPU a partition 2.5× larger than each CPU process (i.e. 1.25× the combined CPU partition).


Contributing
------------

To contribute to OP2, use the following workflow:

1. Fork or clone the OP2-Common repository.
2. Create a new branch from ``master``.
3. Make your changes in the new branch.
4. Submit a Pull Request targeting the ``master`` branch of the OP2-Common repository.

Accumulated contributions in ``master`` are included in new releases.
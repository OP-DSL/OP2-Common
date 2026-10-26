.. _perf:

Performance Tuning
==================

Executing with GPUDirect
------------------------

OP2 supports execution with GPU direct MPI when using the MPI + CUDA builds.

This is detected automatically at ``op_init`` and needs no flag: OP2 asks the MPI
library whether it is CUDA or ROCm aware (``MPIX_Query_cuda_support`` and
``MPIX_Query_rocm_support``), and for the MPI implementations that offer no such
query it falls back to the vendor variables ``MPICH_GPU_SUPPORT_ENABLED``,
``I_MPI_OFFLOAD`` and ``MV2_USE_CUDA``. The decision is printed once at startup as
``OP2: GPU-direct MPI = ...`` along with the mechanism that decided it.

Detection only ever turns the feature *on* when it can confirm support, because
handing a device pointer to an MPI that cannot take one fails inside the transport,
whereas staging through host memory is always correct and merely slower.

Set ``OP_GPU_DIRECT`` to override the detection in either direction: ``1`` forces
it on for an MPI that supports it but does not advertise it, ``0`` forces the
staged path for comparison or to work around a broken transport.

Note that some MPI implementations need their own environment variables set before
they will accept device pointers at all, so check your cluster's user guide.

OpenMP and OpenMP+MPI
---------------------
It is recommended that you assign one MPI rank per NUMA region when executing MPI+OpenMP parallel code. 

Usually for a multi-CPU system a single CPU socket is a single NUMA region. Thus, for a 4 socket system, OP2's MPI+OpenMP code should be executed with 4 MPI processes with each MPI process having multiple OpenMP threads (typically specified by the ``OMP_NUM_THREADS`` flag). 

Additionally on some systems using ``numactl`` to bind threads to cores could give performance improvements.

numawrap
--------

The ``scripts/numawrap`` script automates NUMA binding and GPU assignment for MPI + GPU runs. It detects the MPI local rank from common launchers (Open MPI, MVAPICH2, Hydra, MPISPAWN) and then:

- Sets ``CUDA_VISIBLE_DEVICES`` to the local rank, ensuring each MPI rank uses a distinct GPU.
- Calls ``numactl --cpunodebind`` to bind the process to the NUMA node corresponding to the local rank (round-robined across the available NUMA nodes).

Usage: pass it as the process wrapper to your MPI launcher:

.. code-block:: shell

   mpirun -np 4 scripts/numawrap ./my_op2_application

This is equivalent to manually calling ``numactl`` per rank but works portably across the MPI implementations above without per-rank launch scripts.


.. _strategies:

Execution Strategies
--------------------

A loop generated for the ``cuda``, ``hip``, ``cuda_f2c``, ``hip_f2c`` or ``openmp_f2c`` target carries a kernel for each way it can be run, and OP2 picks one for every call.  The strategies:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - How the loop runs
   * - ``plain``
     - One element at a time, nothing protected: a loop that updates no indirect dat.
   * - ``atomics``
     - Each indirect increment as an atomic add.
   * - ``colouring``
     - The elements coloured so that no two of a colour share a target, one colour after another.
   * - ``chunk_atomics``
     - The elements in chunks whose increment targets are staged (GPU shared memory, or a host thread's scratch), accumulated there, then added to their targets once each.
   * - ``chunk_colouring``
     - Chunks coloured so that no two running together share a target; each chunk's targets are staged, updated by its elements one thread colour at a time, and stored back.

A loop with an indirect ``OP_RW`` or ``OP_WRITE`` argument never runs the atomic strategies.  On the GPU the chunked strategies exist only as JIT kernels, so a call waits for (or, until it is built, passes over) their kernel.

Each loop keeps its strategies in a ladder, ``chunk_atomics``, ``atomics``, ``chunk_colouring``, ``colouring`` (or ``plain`` alone), and a call runs the first that is enabled and ready for it; a chunked strategy may decline a call, for instance when its chunks share too few targets to be worth staging (``low_compression``).  The last strategy runs when none is.  Enabled by default:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Backend
     - Enabled strategies
   * - CUDA (``cuda``, ``cuda_f2c``)
     - ``plain``, ``atomics``, ``colouring``
   * - HIP (``hip``, ``hip_f2c``)
     - ``plain``, ``atomics``, ``colouring``, ``chunk_atomics``
   * - OpenMP (``openmp_f2c``)
     - ``plain``, ``chunk_colouring``, ``colouring``

Environment variables (read by ``op_init``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Effect
   * - ``OP_<STRATEGY>``, e.g. ``OP_CHUNK_ATOMICS``
     - ``on`` or ``off``: enables or disables the strategy; ``off`` also keeps the autotuner from trying it.
   * - ``OP_STRATEGY``
     - ``<strategy>`` or ``<loop>:<strategy>``, a comma-separated list: forces the strategy (a loop's own item wins).  A loop that cannot run it warns ``<loop> passes over its forced strategy <strategy>: <reason>``.
   * - ``OP_BLOCK_SIZES``
     - GPU thread-block size, default 128.
   * - ``OP_CHUNK_SIZES``
     - Elements per chunk of the chunked strategies, default 128 on the GPU and 256 on the host.
   * - ``OP_CHUNK_ATOMICS_UNSTAGED_SINGLES``
     - ``on`` or ``off`` (GPU): ``chunk_atomics`` leaves a target only one of a chunk's elements references in global memory, adding to it directly.  On under HIP, off under CUDA.
   * - ``OP_CHUNK_COLOURING_WAVEFRONT``
     - ``on`` or ``off`` (GPU, default off): ``chunk_colouring`` runs a section's chunks in one launch, each once the earlier chunks sharing one of its targets are done, instead of a launch per colour.
   * - ``OP_CHUNK_COLOURING_WINDOW``
     - Integer, default 4096 (GPU, wavefront only): chunks coloured at a time; 0 colours a section at once.
   * - ``OP_CHUNK_ATOMICS_MIN_COMPRESSION``
     - Default 1.5: ``chunk_atomics`` declines a call whose chunks average fewer references per staged target; 0 accepts every call.  Ignored with unstaged singles.

``OP_BLOCK_SIZES``, ``OP_CHUNK_SIZES`` and the two switches above are each a strategy's *axis*: ``OP_<STRATEGY>_<AXIS>`` (e.g. ``OP_CHUNK_ATOMICS_BLOCK_SIZES``) sets it for one strategy, and every axis takes one value or a comma-separated list, of which the first runs and the rest are only tried by the autotuner.

Autotuning
----------

``OP_AUTOTUNE=report`` times every candidate (each strategy at each combination of its axes' values) for each configuration of a loop (its set, run size and arguments) once that configuration has been seen ``OP_AUTOTUNE_MIN_CALLS`` times (default 2), with ``OP_AUTOTUNE_WARMUP`` untimed runs (default 1) and ``OP_AUTOTUNE_SAMPLES`` timed ones (default 3), and still runs the ladder's choice.  ``OP_AUTOTUNE=select`` also runs a configuration's fastest candidate once it is explored, when that beats the ladder's choice by more than ``OP_AUTOTUNE_MARGIN`` (default 0.03, a fraction of the ladder's time).  Either mode writes a CSV report to ``OP_AUTOTUNE_REPORT`` (``.<rank>`` appended under MPI) with the columns ``loop``, ``configuration``, ``calls``, ``strategy``, ``status``, ``median_us``, ``samples``, ``ladder``, ``fastest``, ``speedup``, ``selected``, ``state`` and ``statistics``.

JIT Kernels
-----------

The GPU targets also compile each kernel at run time, specialised on the values it sees: see ``OP_JIT_LEVEL`` and the other ``OP_JIT_*`` variables in :doc:`api`.  A specialised kernel is built once the same values have been seen ``OP_JIT_MIN_CALLS`` times (default 9); until then, and for a strategy with no JIT kernel needed, the offline kernel runs.

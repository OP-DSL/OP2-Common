Developing an OP2 Application
=============================

This page provides a tutorial in the basics of using OP2 for unstructured-mesh application development.

Example Application
-------------------
Original - Load mesh and initialization
---------------------------------------
Original - Main iteration
-------------------------
Original - Loops over mesh
--------------------------
5 loops, Direct and indirect loops - details of loops


Build OP2
---------
Build OP2 using instructions in the `Getting Started <fhttps://op2-dsl.readthedocs.io/en/latest/getting_started.html>`__. page.

Step 1 - Preparing to use OP2
-----------------------------
* Header files
* Initialize and finalize OP2
* Link with sequential back-end (need Step 1 Makefile)
* The airfoil mesh can be downloaded from `here <https://op-dsl.github.io/docs/OP2/new_grid.dat>`__.


Step 2 - OP2 Declaration
------------------------
* Declare sets
* Declare maps
* Declare dats
* Declare constants

Step 3 - First parallel loop : direct loop
------------------------------------------
* Outline kernel
* Declare parallel loop with ``op_par_loop``

Step 4 - Indirect loops
-----------------------
* Details of ``op_par_loop`` for indirect loops

Step 5 - Global reductions
--------------------------
* Details of ``op_par_loop`` for specifying global reductions


Step 6 - Handing it all to OP2
------------------------------
* Now can run a sequential version and validate results
* Partitioning call for MPI
* Parallel file I/O
* details on ``op_fetch_data`` call

Step 7 - Code generation
------------------------
* Code-gen command
* Link and execute parallel versions with Makefiles

Code generated versions
-----------------------

Optimizations
-------------
* Brief notes on runtime and optimization flags
* Provide link to Performance tuning page in the docs

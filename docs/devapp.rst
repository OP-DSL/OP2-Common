Developing an OP2 Application
=============================

This page provides a tutorial in the basics of using OP2 for unstructured-mesh application development.

Example Application
-------------------

The tutorial will use the Airfoil application, a simple non-linear  2D  inviscid  airfoil code that uses an unstructured mesh. It is a finite volume application that solves the 2D Euler equations using a scalar numerical dissipation. The algorithm iterates towards the steady state solution, in each iteration using a control volume approach - for example the rate at which the mass changes within a control volume is equal to the net flux of mass into the control volume across the four faces around the cell.

Airfoil consists of five loops, ``save_soln`` , ``adt_calc`` , ``res_calc`` , ``bres_calc`` and ``update``, within a time-marching iterative loop. Out of these, ``save_soln`` and ``update`` are what we classify as direct loops where all the data accessed in the loop is defined on the mesh element over which the loop iterates over. Thus for example in a direct loop a loop over edges will only access data defined on edges. The other three loops are indirect loops. In this case when looping over a given type of elements, data on other types of elements will be accessed indirectly, using mapping tables. Thus for example ``res_calc`` iterates over edges and increments data on cells, accessing them indirectly via a mapping table that gives the explicit connectivity information between edges and cells.

The `standard mesh <https://op-dsl.github.io/docs/OP2/new_grid.dat>`__  size solved with Airfoil consists of 1.5M edges.  Here the most compute intensive loop is ``res_calc``, which is called 2000 times during the total execution of the application and performs about 100 floating-point operations per mesh edge.

* Go to the ``OP2/apps/c/airfoil/airfoil/airfoil_tutorial/original`` directory and open the ``airfoil_orig.cpp`` file to view the original application.
* Use the Makefile in the same directory to compile and then run the application. The ``new_grid.dat`` needs to be present in the same directory as the executable.


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

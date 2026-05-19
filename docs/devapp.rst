Developing an OP2 Application
=============================

Building an OP2 application involves four key steps:

(1) **Declare sets:** OP2 organizes elements, such as cells, nodes, or edges, into sets.
(2) **Declare connectivity:** Use maps to specify the unstructured mesh topology (e.g., cell-to-nodes mapping).
(3) **Declare data:** Assign data fields associated with sets, such as node-pressure.
(4) **Operations over sets:** Implement computations over sets, accessing data either directly or indirectly through mappings.

Below is a tutorial on using OP2 for unstructured mesh applications.

Example Application
-------------------

This tutorial will use the Airfoil application, a simple non-linear 2D inviscid airfoil code with an unstructured mesh. 
It solves the 2D Euler equations using finite volumes and iterates toward a steady-state solution. 
Each iteration uses a control volume approach, where the mass change in a volume equals the net flux across its faces.

Airfoil has five loops inside a time-marching iteration:

* **Direct loops:** ``save_soln`` and ``update``
    * Access data defined on the mesh element being iterated. For example, a loop over edges only accessing data defined on edges. 

* **Indirect loops:** ``adt_calc`` , ``res_calc`` , ``bres_calc``
    * Access data on other sets via mapping tables. For example, ``res_calc`` loops over edges and updates cell data using an edge-to-cell map.

The standard Airfoil mesh has 1.5M edges. The most compute-intensive loop is ``res_calc``, called 2,000 times during the total execution, performing ~100 floating-point operations per edge.

Try the Original Code Yourself
------------------------------

(1) Navigate to:
    ``OP2/apps/c/airfoil/airfoil_tutorial/original`` and open ``airfoil_orig.cpp`` file to view the original application.
(2) Compile and run using the Makefile in the same directory.
(3) Ensure ``new_grid.dat`` (downloadable from `here <https://op-dsl.github.io/docs/OP2/new_grid.dat>`__) is in the same folder as the executable.
(4) The program reports RMS pressure values on cells every 100 iterations. At 1,000 iterations, it compares the result to a reference solution. If the RMS value matches within machine precision, the run passes validation. The same criterion used for parallelized OP2 versions.

What is happening in the code?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Load mesh and initialization
""""""""""""""""""""""""""""

The original code begins with allocating memory to hold the mesh data and then initializing them by reading in the mesh data, form the ``new_grid.dat`` text file:

.. raw:: html

   <details class="sphinx-code-toggle"><summary>Show C code</summary>

.. code-block:: c

   FILE *fp;
   if ((fp = fopen(FILE_NAME_PATH, "r")) == NULL) {
     printf("can't open file FILE_NAME_PATH\n");
     exit(-1);
   }
   if (fscanf(fp, "%d %d %d %d \n", &nnode, &ncell, &nedge, &nbedge) != 4) {
     printf("error reading from FILE_NAME_PATH\n");
     exit(-1);
   }

   cell   = (int *)malloc(4 * ncell  * sizeof(int));
   edge   = (int *)malloc(2 * nedge  * sizeof(int));
   ecell  = (int *)malloc(2 * nedge  * sizeof(int));
   bedge  = (int *)malloc(2 * nbedge * sizeof(int));
   becell = (int *)malloc(1 * nbedge * sizeof(int));
   bound  = (int *)malloc(1 * nbedge * sizeof(int));
   x      = (double *)malloc(2 * nnode * sizeof(double));
   q      = (double *)malloc(4 * ncell * sizeof(double));
   qold   = (double *)malloc(4 * ncell * sizeof(double));
   res    = (double *)malloc(4 * ncell * sizeof(double));
   adt    = (double *)malloc(1 * ncell * sizeof(double));

   for (int n = 0; n < nnode; n++) {
     if (fscanf(fp, "%lf %lf \n", &x[2 * n], &x[2 * n + 1]) != 2) {
         printf("error reading from FILE_NAME_PATH\n");
         exit(-1);
     }
   }

   for (int n = 0; n < ncell; n++) {
     if (fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
         &cell[4 * n + 2], &cell[4 * n + 3]) != 4) {
       printf("error reading from FILE_NAME_PATH\n");
       exit(-1);
     }
   }
   for (int n = 0; n < nedge; n++) {
     if (fscanf(fp, "%d %d %d %d \n", &edge[2 * n], &edge[2 * n + 1],
           &ecell[2 * n], &ecell[2 * n + 1]) != 4) {
       printf("error reading from FILE_NAME_PATH\n");
       exit(-1);
     }
   }
   for (int n = 0; n < nbedge; n++) {
     if (fscanf(fp, "%d %d %d %d \n", &bedge[2 * n], &bedge[2 * n + 1],
         &becell[n], &bound[n]) != 4) {
       printf("error reading from FILE_NAME_PATH\n");
       exit(-1);
     }
   }
   fclose(fp);

.. raw:: html

   </details>


The code then initialize ``q`` and ``res`` data arrays to 0.

Main iteration and loops over mesh
""""""""""""""""""""""""""""""""""

The main iterative loop is a for loop that iterates for some ``NUM_ITERATIONS`` (set to 1,000 iterations).  
Within this main iterative loops there are 5 loops over various mesh elements (as noted above).


Convert original application to use OP2 DSL
-------------------------------------------
Build OP2 using instructions in the :doc:`getting_started` page.

Step 1 - Preparing to use OP2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, include the following header files, then initialize OP2 and finalize it as follows:

.. code-block:: C

  #include "op_seq.h"
  ...
  ...
  int main(int argc, char **argv) {
    //Initialise the OP2 library, passing runtime args, and setting diagnostics level to low (1)
    op_init(argc, argv, 1);
    ...
    ...
    ...
    free(adt);
    free(res);

    //Finalising the OP2 library
    op_exit();
  }

By now, you should have OP2 set up. 
The Makefile in `step 1 <https://github.com/OP-DSL/OP2-Common/tree/master/apps/c/airfoil/airfoil_tutorial/step1>`_ shows the include and library paths and links against the ``op2_seq`` back-end. 

Including ``op_seq.h`` and linking with the sequential OP2 library gives a **developer sequential version** of the application. 
Use this version to build the rest of the app with the OP2 API and validate numerical results.

Step 2 - OP2 Declaration
^^^^^^^^^^^^^^^^^^^^^^^^

Declare sets
""""""""""""

Airfoil uses four mesh element types, called sets: nodes, edges, cells, and boundary edges. Declare each set with the ``op_set`` API and its element count.

.. code-block:: C

  op_set nodes  = op_decl_set(nnode,  "nodes" );
  op_set edges  = op_decl_set(nedge,  "edges" );
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells  = op_decl_set(ncell,  "cells" );

Later, you can read the number of mesh elements directly from an HDF5 file using ``op_set_hdf5`` (explained below).

When creating or converting an application to OP2, you must decide which mesh element types (sets) to declare to define the full mesh. 
A good starting point is to check which mesh elements are used in the loops over the mesh.

Declare maps
""""""""""""

Examining Airfoil’s loops shows that we need mappings between:

* edges → nodes
* edges → cells
* boundary edges → nodes
* boundary edges → cells
* cells → nodes

These mappings are required due to the indirect data accesses in the main iteration loops. They are declared using the ``op_decl_map`` API.

.. code-block:: C

  op_map pedge   = op_decl_map(edges,  nodes, 2, edge,   "pedge"  );
  op_map pecell  = op_decl_map(edges,  cells, 2, ecell,  "pecell" );
  op_map pbedge  = op_decl_map(bedges, nodes, 2, bedge,  "pbedge" );
  op_map pbecell = op_decl_map(bedges, cells, 1, becell, "pbecell");
  op_map pcell   = op_decl_map(cells,  nodes, 4, cell,   "pcell"  );

The ``op_decl_map`` requires the names of the two sets being mapped, its arity, mapping data (stored in integer memory blocks in this case) and a string name.

Declare data
""""""""""""

All data associated with sets should be declared using the ``op_decl_dat`` API call. 

For Airfoil this includes:

* the mesh coordinates data ``x`` (declared on nodes set)
* new and old solution ``q`` and ``q_old`` (declared on cells set)
* area time step ``adt`` (declared on cells set)
* flux residual ``res`` (declared on cells set)
* boundary flag ``bound`` indicating whether an edge is a boundary edge (declared on bedges set)

.. code-block:: C

  op_dat p_bound = op_decl_dat(bedges, 1, "int",    bound, "p_bound");
  op_dat p_x     = op_decl_dat(nodes,  2, "double", x,     "p_x"    );
  op_dat p_q     = op_decl_dat(cells,  4, "double", q,     "p_q"    );
  op_dat p_qold  = op_decl_dat(cells,  4, "double", qold,  "p_qold" );
  op_dat p_adt   = op_decl_dat(cells,  1, "double", adt,   "p_adt"  );
  op_dat p_res   = op_decl_dat(cells,  4, "double", res,   "p_res"  );

Declare constants
""""""""""""""""""

Finally, global constants used in any of the computations in the loops must be declared. 
This is necessary because global constants must be available on the GPU before they can be used in a kernel, which is handled automatically during code generation for parallelization (e.g., with CUDA).

Declaring them with the ``op_decl_const`` API informs the OP2 code generator to handle these constants specially, generating the necessary code to copy them to the GPU for the relevant back-ends.

.. code-block:: C

  op_decl_const(1, "double", &gam  );
  op_decl_const(1, "double", &gm1  );
  op_decl_const(1, "double", &cfl  );
  op_decl_const(1, "double", &eps  );
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf  );


Diagnose and test
"""""""""""""""""

At this point, we have declared the mesh sets, connectivity, and data.

Now, information about the declared mesh can be viewed by setting a diagnostics level of ``2`` in ``op_init`` and calling ``op_diagnostic_output()`` API:

.. code-block:: C

  int main(int argc, char **argv) {
    
    op_init(argc, argv, 2); //Initialise the OP2 library by setting diagnostics level to (2)
    ...
    ... op_decl_set ...
    ... op_decl_map ...
    ... op_decl_dat ...
    ... op_decl_const  ...
    ...
    op_diagnostic_output(); //output mesh information

Finally compile the ``step2`` application and execute using the following runtime flag:

.. code-block:: bash

  ./airfoil_step2 OP_NO_REALLOC

If everything is correct, you should see the mesh information printed to the terminal.

The ``OP_NO_REALLOC`` runtime flag instructs the OP2 back-end to use the already allocated memory for sets, maps, and data without internally deallocating it. 
This allows the developer to gradually convert an application to the OP2 API while validating each step, as we do here.

However, this behavior applies **only to the developer sequential version** that we are using. 
Parallel versions generated by the code generator, as well as the generated sequential version (``genseq``), will not work with this flag, since they deallocate the initial memory and move the mesh to achieve optimal parallel performance.

Step 3 - First parallel loop : direct loop
------------------------------------------

We can now convert the first loop to use the OP2 API. In this case its a direct loop called ``save_soln`` that iterates over cells and saves the previous time-iteration's solution, ``q`` to ``q_old``:

.. code-block:: C

  //save_soln : iterates over cells
  for (int iteration = 0; iteration < (ncell * 4); ++iteration) {
    qold[iteration] = q[iteration];
  }

This is a **direct loop**, since all data accessed in the computation are defined on the same set over which the loop iterates. In this case, the iteration set is the cells set.

To convert this loop to the OP2 API, we first extract the loop body (the elemental kernel) into a separate subroutine.

.. code-block:: C

  //outlined elemental kernel
  inline void save_soln(const double *q, double *qold) {
    for (int n = 0; n < 4; ++n)
      qold[n] = q[n];
  }

  //save_soln : iterates over cells
  for (int iteration = 0; iteration < ncell; ++iteration) {
    save_soln(&q[iteration*4], &qold[iteration*4]);
  }

The loop can now be declared directly using the ``op_par_loop`` API as follows:

.. code-block:: C

  op_par_loop(save_soln, "save_soln", cells,
              op_arg_dat(p_q,    -1, OP_ID, 4, "double", OP_READ ),
              op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_WRITE));

Note that we have now:

- Specified the elemental kernel ``save_soln`` as the first argument to ``op_par_loop``
- Used the ``op_dat`` names ``p_q`` and ``p_qold`` in the API call
- Identified the iteration set as ``cells`` (the third argument)
- Indicated direct access to ``q`` and ``q_old`` using ``OP_ID``
- Specified that ``p_q`` is read-only (``OP_READ``) and ``q_old`` is write-only (``OP_WRITE``) by inspecting how they are accessed within the elemental kernel
- Marked ``p_q`` as read-only in the elemental kernel by declaring it with the ``const`` keyword in ``save-soln``
- Specified the data dimension in the fourth argument of ``op_arg_dat``; for ``p_q`` and ``q_old``, this is four doubles per mesh point

Compile and execute the modified application (again using ``OP_NO_REALLOC``; see code in ``../step3``) and verify that the solution validates.

Step 4 - Indirect loops
-----------------------

The next loop in the application, ``adt_calc``, calculates the area/timstep while iterating over the ``cells`` set. 
In this case, the **loop is indirect**, since the data ``x`` on the four nodes connected to each cell are accessed indirectly via a ``cell-to-nodes`` mapping. 
In addition, the data ``adt`` are accessed directly, as ``adt`` is defined on the ``cells`` set.

.. code-block:: C

  //adt_calc - calculate area/timstep : iterates over cells
  for (int iteration = 0; iteration < ncell; ++iteration) {
    int map1idx = cell[iteration * 4 + 0];
    int map2idx = cell[iteration * 4 + 1];
    int map3idx = cell[iteration * 4 + 2];
    int map4idx = cell[iteration * 4 + 3];

    double dx, dy, ri, u, v, c;

    ri = 1.0f / q[4 * iteration + 0];
    u = ri * q[4 * iteration + 1];
    v = ri * q[4 * iteration + 2];
    c = sqrt(gam * gm1 * (ri * q[4 * iteration + 3] - 0.5f * (u * u + v * v)));

    dx = x[2 * map2idx + 0] - x[2 * map1idx + 0];
    dy = x[2 * map2idx + 1] - x[2 * map1idx + 1];
    adt[iteration] = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x[2 * map3idx + 0] - x[2 * map2idx + 0];
    dy = x[2 * map3idx + 1] - x[2 * map2idx + 1];
    adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x[2 * map4idx + 0] - x[2 * map3idx + 0];
    dy = x[2 * map4idx + 1] - x[2 * map3idx + 1];
    adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x[2 * map1idx + 0] - x[2 * map4idx + 0];
    dy = x[2 * map1idx + 1] - x[2 * map4idx + 1];
    adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    adt[iteration] = adt[iteration] / cfl;
  }

Similar to the direct loop, we first extract the loop body into a separate subroutine and then invoke it within the loop as follows:

.. code-block:: C

  //outlined elemental kernel - adt_calc
  inline void adt_calc(double *x1, double *x2, double *x3,
                       double *x4, double *q, double *adt) {
    double dx, dy, ri, u, v, c;

    ri = 1.0f / q[0];
    u = ri * q[1];
    v = ri * q[2];
    c = sqrt(gam * gm1 * (ri * q[3] - 0.5f * (u * u + v * v)));

    dx = x2[0] - x1[0];
    dy = x2[1] - x1[1];
    *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x3[0] - x2[0];
    dy = x3[1] - x2[1];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x4[0] - x3[0];
    dy = x4[1] - x3[1];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    dx = x1[0] - x4[0];
    dy = x1[1] - x4[1];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

    *adt = (*adt) / cfl;
  }

  //adt_calc - calculate area/timstep : iterates over cells
  for (int iteration = 0; iteration < ncell; ++iteration) {
    int map1idx = cell[iteration * 4 + 0];
    int map2idx = cell[iteration * 4 + 1];
    int map3idx = cell[iteration * 4 + 2];
    int map4idx = cell[iteration * 4 + 3];

    adt_calc(&x[2 * map1idx], &x[2 * map2idx], &x[2 * map3idx],
             &x[2 * map4idx], &q[4 * iteration], &adt[iteration]);
  }

We now convert the loop to use the ``op_par_loop`` API:

.. code-block:: C

  //adt_calc - calculate area/timstep : iterates over cells
  op_par_loop(adt_calc, "adt_calc", cells,
              op_arg_dat(p_x,   0, pcell, 2, "double", OP_READ ),
              op_arg_dat(p_x,   1, pcell, 2, "double", OP_READ ),
              op_arg_dat(p_x,   2, pcell, 2, "double", OP_READ ),
              op_arg_dat(p_x,   3, pcell, 2, "double", OP_READ ),
              op_arg_dat(p_q,  -1, OP_ID, 4, "double", OP_READ ),
              op_arg_dat(p_adt,-1, OP_ID, 1, "double", OP_WRITE));

Note that, in this case, the indirections are specified using the mapping declared as the OP2 map ``pcell``, by indicating the to-set index (second argument) and the access mode ``OP_READ``.

Similarly, the ``res_calc`` and ``bres_calc`` indirect loops can be converted to use the OP2 API.

Step 5 - Global reductions
--------------------------

At this stage, the only remaining loop is ``update`` which updates the flow field, and requires special handling due to its **global reduction**.

.. code-block:: C

  rms = 0.0f;
  for (int iteration = 0; iteration < ncell; ++iteration) {
    double del, adti;

    adti = 1.0f / (adt[iteration]);

    for (int n = 0; n < 4; ++n) {
      del = adti * res[iteration * 4 + n];
      q[iteration * 4 + n] = qold[iteration * 4 + n] - del;
      res[iteration * 4 + n] = 0.0f;
      rms += del * del;
    }
  }

Here, the global variable ``rms`` is used as a reduction variable to compute the root-mean-square (RMS) value of the residual. The kernel can be outlined as follows:

.. code-block:: C

  //outlined elemental kernel - update
  inline void update(const double *qold, double *q, double *res,
                     const double *adt, double *rms) {
    double del, adti;

    adti = 1.0f / (*adt);

    for (int n = 0; n < 4; ++n) {
      del = adti * res[n];
      q[n] = qold[n] - del;
      res[n] = 0.0f;
      *rms += del * del;
    }
  }

It is then called in the application:

.. code-block:: C

  //update = update flow field - iterates over cells
  rms = 0.0f;
  for (int iteration = 0; iteration < ncell; ++iteration) {
    update(&qold[iteration * 4], &q[iteration * 4], &res[iteration * 4],
           &adt[iteration], &rms);
  }

The global reduction requires the ``op_arg_gbl`` API call, using the ``OP_INC`` access mode to indicate that the variable participates in a global reduction.

.. code-block:: C

  //update = update flow field - iterates over cells
  rms = 0.0f;
  op_par_loop(update, "update", cells,
              op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_READ ),
              op_arg_dat(p_q,    -1, OP_ID, 4, "double", OP_WRITE),
              op_arg_dat(p_res,  -1, OP_ID, 4, "double", OP_RW   ),
              op_arg_dat(p_adt,  -1, OP_ID, 1, "double", OP_READ ),
              op_arg_gbl(&rms,    1,           "double", OP_INC  ));

At this point, all loops have been converted to use the ``op_par_loop`` API, and the application should validate correctly when executed as a sequential, single-threaded CPU program. 
You should now also be able to run it without the ``OP_NO_REALLOC`` runtime flag and still obtain a valid result.

However, in this case, OP2 will create internal copies of the data declared for ``op_map`` and ``op_dat``. 
When developing applications for performance, it is advisable to free the initial memory allocated immediately after the relevant ``op_decl_map`` and ``op_decl_dat`` calls.

In the next step, we avoid freeing this "application-developer-allocated" memory by using **HDF5 file I/O**, allowing the mesh data to be read directly from file into OP2-allocated internal memory.


Step 6 - Handing it all to OP2
--------------------------------------

Once the developer sequential version has been created and the numerical output validates the application can be prepared to obtain a developer distributed memory parallel version. This step can be completed to obtain a parallel executable, without code-generation if the following steps are implemented.

(1) File I/O needs to be extended to allow distributed memory execution with MPI. The current Airfoil application simply reads the mesh data from a text file and such a simple setup will not be workable on a distributed memory system, such as a cluster and more importantly will not be scalable with MPI. The simplest solution is to use OP2's HDF5 API for declaring the mesh by replacing ``op_decl_set, op_decl_map, op_decl_dat`` and ``op_decl_const`` by its HDF5 counterparts as follows:

.. code-block:: C

  // declare sets
  op_set nodes  = op_decl_set_hdf5(file,  "nodes" );
  op_set edges  = op_decl_set_hdf5(file,  "edges" );
  op_set bedges = op_decl_set_hdf5(file, "bedges");
  op_set cells  = op_decl_set_hdf5(file,  "cells" );

  //declare maps
  op_map pedge   = op_decl_map_hdf5(edges,  nodes, 2, file, "pedge"  );
  op_map pecell  = op_decl_map_hdf5(edges,  cells, 2, file, "pecell" );
  op_map pbedge  = op_decl_map_hdf5(bedges, nodes, 2, file, "pbedge" );
  op_map pbecell = op_decl_map_hdf5(bedges, cells, 1, file, "pbecell");
  op_map pcell   = op_decl_map_hdf5(cells,  nodes, 4, file, "pcell"  );

  //declare data on sets
  op_dat p_bound = op_decl_dat_hdf5(bedges, 1, "int",    file, "p_bound");
  op_dat p_x     = op_decl_dat_hdf5(nodes,  2, "double", file, "p_x"    );
  op_dat p_q     = op_decl_dat_hdf5(cells,  4, "double", file, "p_q"    );
  op_dat p_qold  = op_decl_dat_hdf5(cells,  4, "double", file, "p_qold" );
  op_dat p_adt   = op_decl_dat_hdf5(cells,  1, "double", file, "p_adt"  );
  op_dat p_res   = op_decl_dat_hdf5(cells,  4, "double", file, "p_res"  );

  //read and declare global constants
  op_get_const_hdf5("gam",   1, "double", (char *)&gam,  file);
  op_get_const_hdf5("gm1",   1, "double", (char *)&gm1,  file);
  op_get_const_hdf5("cfl",   1, "double", (char *)&cfl,  file);
  op_get_const_hdf5("eps",   1, "double", (char *)&eps,  file);
  op_get_const_hdf5("alpha", 1, "double", (char *)&alpha,file);
  op_get_const_hdf5("qinf",  4, "double", (char *)&qinf, file);

  op_decl_const(1, "double", &gam  );
  op_decl_const(1, "double", &gm1  );
  op_decl_const(1, "double", &cfl  );
  op_decl_const(1, "double", &eps  );
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf  );

Note here that we assume that the mesh is already available as an HDF5 file named ``new_grid.h5``. (See the ``convert_mesh.cpp`` utility application in ``OP2-Common/apps/c/airfoil/airfoil_hdf5/dp`` to understand how we can create an HDF5 file to be compatible with the OP2 API for Airfoil starting from mesh data defined in a text file.)

When the application has been switched to use the HDF5 API calls, manually allocated memory for the mesh elements can be removed. Additionally all ``printf`` statements should use ``op_printf`` so that output to terminal will only be done by the ROOT mpi process. We can also replace the timer routines with OP2's ``op_timers`` which times the execution of the code the ROOT.

Given that the mesh was read via HDF5, to obtain the global sizes of the mesh, OP2's ``op_get_size()`` API call need to be used. This is required for the Airfoil application to obtain the number of cells to compute the rms value for every 100 iterations to validate the application:

.. code-block:: C

  //get global number of cells
  ncell = op_get_size(cells);


(2) Add the OP2 partitioner call ``op_partition`` to the code in order to signal to the MPI back-end, the point in the program that all mesh data have been defined and mesh can be partitioned and MPI halos can be created:

.. code-block:: C

  ...
  ...
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf  );

  //output mesh information
  op_diagnostic_output();

  //partition mesh and create mpi halos
  op_partition("BLOCK", "ANY", edges, pecell, p_x);

  ...
  ...

See the API documentation for practitioner options. In this case no special partitioner is used leaving the initial block partitioning of data at the time of file I/O through HDF5.

Take a look at the code in the ``/step6`` for the full code changes done to the Airfoil application. The application can  now be compiled to obtain a developer distributed-memory (MPI) parallel executable using the Makefile in the same directory. Note how the executable is created by linking with the OP2 MPI back-end, ``libop2_mpi`` together with the HDF5 library ``libhdf5``. You will need to have had HDF5 library installed on your system to carry out this step.

The resulting executable is called a ``developer MPI`` version of the application, which should again be used to verify validity of the application by running with ``mpirun`` in the usual way of executing an MPI application.

.. * details on ``op_fetch_data`` call


Step 7 - Code generation
------------------------

Now that both the sequential and MPI developer versions work and validate, its time to generate other parallel versions. However, first we should move the elemental kernels to header files so that after the code generation the modified main application will not have the same elemental kernel definitions. This is currently a limitation of the code-generator, which will be remedied in future versions.

We move the elemental kernels to header files, each with the name of the kernel and include them in the ``airfoil_step7.cpp`` main file:

.. code-block:: C

  ...
  ...
  /* Global Constants */
  double gam, gm1, cfl, eps, mach, alpha, qinf[4];

  //
  // kernel routines for parallel loops
  //
  #include "adt_calc.h"
  #include "bres_calc.h"
  #include "res_calc.h"
  #include "save_soln.h"
  #include "update.h"
  ...

Application Makefile
^^^^^^^^^^^^^^^^^^^^

The application needs a Makefile that tells the OP2 build system the name and source files of the application.  The Makefile in ``/step7`` is minimal — you only need to supply four things, then include the two OP2 makefiles:

.. code-block:: make

   APP_NAME           := airfoil_step7  # (1) base name for all generated executables
   APP_SRC            := airfoil_step7.cpp  # (2) main source file(s)
   OP2_LIBS_WITH_HDF5 := true           # (3) link against HDF5 (omit if not using HDF5 I/O)

   include ../../../../../makefiles/common.mk  # (4) OP2 configuration
   include ../../../../../makefiles/c_app.mk   # (5) all build targets

The ``makefiles/c_app.mk`` file provides ready-made targets for every parallelisation variant.  When you request a target, it automatically:

1. Runs the OP2 v2 translator on ``$(APP_SRC)`` to produce backend-specific kernel files under ``generated/airfoil_step7/<variant>/``.
2. Compiles the generated kernel file for that variant.
3. Links the compiled application with the appropriate OP2 library.

You do not need to invoke the translator manually or know its command-line options — the Makefile does it for you.

Available build targets
"""""""""""""""""""""""

Running ``make`` with a build target from the ``/step7`` directory (or the application directory for any OP2 app that uses these makefiles):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Target
     - What is built
   * - ``seq``
     - Developer sequential build (no code generation; links directly).
   * - ``genseq``
     - Code-generated sequential build.  Preferred over ``seq`` for benchmarking.
   * - ``openmp``
     - Code-generated multi-threaded CPU build using OpenMP.
   * - ``cuda``
     - Code-generated NVIDIA GPU build (AOT compiled by ``nvcc``).
   * - ``hip``
     - Code-generated AMD GPU build (AOT compiled by ``hipcc``).
   * - ``c_cuda``
     - Code-generated NVIDIA GPU build with JIT compilation (via NVRTC).
   * - ``c_hip``
     - Code-generated AMD GPU build with JIT compilation (via HIP RTC).
   * - ``mpi_<variant>``
     - Distributed-memory MPI version of any of the above (e.g. ``mpi_cuda``).
   * - ``all``
     - Builds every variant for which the required libraries are available.
   * - ``clean``
     - Removes all built executables and the ``generated/`` directory.

For example, to build just the JIT CUDA variant and its MPI counterpart:

.. code-block:: shell

   make c_cuda mpi_c_cuda

Targets that require hardware or libraries that were not found at configuration time are silently skipped.  The first thing printed by any ``make`` invocation is the list of variants that will actually be built, e.g.:

.. code-block:: text

   Buildable app variants for airfoil_step7: seq genseq openmp c_cuda mpi_seq mpi_genseq mpi_openmp mpi_c_cuda

This will build all the currently supported parallel versions for your system, provided that the relevant OP2 back-end libraries have been built.


Final - Code generated versions and execution
---------------------------------------------

The following parallel versions will be generated from the code generator. These and the previously developed *developer* versions and can be executed as follows (see the `\/final <https://github.com/OP-DSL/OP2-APPS/tree/main/apps/c/airfoil/airfoil_tutorial/final>`__ directory in the `OP2-APPS <https://github.com/OP-DSL/OP2-APPS>`__ repository for all the generated code) :

(1) Developer sequential and developer mpi - no code-generation required.

.. code-block:: bash

  #developer sequential
  ./airfoil_seq

  #developer distributed memory with mpi, on 4 mpi procs
  $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_seq

(2) Code-gen sequential and MPI + Code-gen sequential

.. code-block:: bash

  # code-gen sequential
  ./airfoil_genseq
  #On 4 mpi procs
  $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_genseq


(3) Code-gen OpenMP, on 4 OpenMP threads, with mini-partition size of 256 and MPI + Code-gen OpenMP, on 4 MPI x 8 OpenMP with mini-partition size of 256

.. code-block:: bash

  # on 4 OMP threads
  export OMP_NUM_THREADS=4; ./airfoil_openmp OP_PART_SIZE=256
  #On 4 mpi procs with each proc running 8 OpenMP threads
  export OMP_NUM_THREADS=8; $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_openmp OP_PART_SIZE=256

(4) Code-gen SIMD vectorized and MPI + Code-gen SIMD vectorized, on 4 MPI

.. code-block:: bash

  #SIMD vec
  ./airfoil_vec
  #On 4 mpi procs with each proc running SIMD vec
  $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_vec

(5) Code-gen CUDA with mini-partition size of 128 and CUDA thread-block size of 192 and MPI + Code-gen CUDA

.. code-block:: bash

  #On a single GPU
  ./airfoil_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192
  #On 4 mpi procs, each proc having a GPU
  $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_cuda OP_PART_SIZE=128 OP_BLOCK_SIZE=192

(6) MPI + Code-gen hybrid CUDA with mini-partition size of 128 and CUDA thread-block size of 192

  The hybrid version can be run on both CPUs and GPUs at the same time. If there is only 4 GPUs are available the following execution will allocated 4 MPI procs to be run on 4 GPUs and 8 MPI procs allocated to the remaining CPU cores.

.. code-block:: bash

  #On 12 mpi procs
  $MPI_INSTALL_PATH/bin/mpirun -np 12 ./airfoil_mpi_cuda_hyb OP_PART_SIZE=128 OP_BLOCK_SIZE=192

(7) Code-gen OpenACC with mini-partition size of 128 and thread-block size of 192 and MPI + Code-gen OpenACC

.. code-block:: bash

  #On a single GPU
  ./airfoil_openacc OP_PART_SIZE=128 OP_BLOCK_SIZE=192
  #On 4 mpi procs, each proc having a GPU
  $MPI_INSTALL_PATH/bin/mpirun -np 4 ./airfoil_mpi_openacc OP_PART_SIZE=128 OP_BLOCK_SIZE=192


Optimizations
-------------

See the :ref:`perf` section for a number of specific compile-time and runtime flags to obtain better performance.


Airfoil is a nonlinear 2D inviscid airfoil code that uses an unstructured grid. It is a finite volume application that
solves the 2D Euler equations using a scalar numerical dissipation. The algorithm iterates towards the steady state
solution, in each iteration using a control volume approach – for example the rate at which the mass changes within a
control volume is equal to the net flux of mass into the control volume across the four faces around the cell. This is
representative of the 3D viscous flow calculations OP2 aims to eventually support for production-grade CFD applications
(such as the Hydra CFD code at Rolls-Royce plc.). The application implements a predictor/corrector time-marching
loop, by (1) looping over the cells of the mesh and computing the time step for each cell, (2) computing the flux over
internal edges (3) computing the flux over boundary edges (4) updating the solution and (5) saving the old solution
before repeating. These main stages of the application is solved in Airfoil within five parallel loops: adt_calc,
res_calc, bres_calc, update and save_soln. Out of these, save_soln and update are direct loops while the other three are
indirect loops.

Please see airfoil-doc under the ../../doc directory for further OP2 application development details

## Airfoil Application Directory Structure

Airfoil has been the main development, testing and benchmarking application in OP2. As such this directory contains
several versions of Airfoil that demonstrate the use of various features of OP2.






program airfoil
#ifdef HDF5
    use op2_fortran_hdf5_declarations
#else
    use op2_fortran_declarations
#endif

    use op2_fortran_reference
    use op2_fortran_rt_support

    use airfoil_constants
    use airfoil_kernels

    use, intrinsic :: iso_c_binding

    implicit none

    integer(4), parameter :: file_id = 1

    character(*), parameter :: file_name = "new_grid.dat"
    character(*), parameter :: file_name_h5 = "new_grid.h5"

    integer(4), parameter :: niter = 1000
    integer(4) :: iter, k

    integer(4) :: nnode, ncell, nbedge, nedge

#ifdef PLAIN
    integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
    real(8), dimension(:), allocatable, target :: x, q, qold, adt, res
#endif

#ifdef ARG_PTRS
    integer(4), dimension(*) :: ecell, bound, edge, bedge, becell, cell
    real(8), dimension(*) :: x, q, qold, adt, res

    pointer (ptr_ecell, ecell), (ptr_bound, bound), (ptr_edge, edge), (ptr_bedge, bedge), &
        (ptr_becell, becell), (ptr_cell, cell)

    pointer (ptr_x, x), (ptr_q, q), (ptr_qold, qold), (ptr_adt, adt), (ptr_res, res)
#endif

#ifdef HDF5
    integer(4) :: stat
#endif

    real(8), dimension(2) :: rms

    type(op_set) :: nodes, edges, bedges, cells
    type(op_map) :: pedge, pecell, pcell, pbedge, pbecell
    type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res

    real(kind=c_double) :: start_time, end_time

    real(8) :: diff

#ifndef HDF5
    print *, "Reading input file"
    call read_input()

    print *, ncell
#endif

    call op_init_base(0, 0)

#ifdef HDF5
    print *, "Declaring OP2 sets (HDF5)"
    call op_decl_set_hdf5(nnode, nodes, file_name_h5, "nodes")
    call op_decl_set_hdf5(nedge, edges, file_name_h5, "edges")
    call op_decl_set_hdf5(nbedge, bedges, file_name_h5, "bedges")
    call op_decl_set_hdf5(ncell, cells, file_name_h5, "cells")

    print *, "Declaring OP2 maps (HDF5)"
    call op_decl_map_hdf5(edges, nodes, 2, pedge, file_name_h5, "pedge", stat)
    call op_decl_map_hdf5(edges, cells, 2, pecell, file_name_h5, "pecell", stat)
    call op_decl_map_hdf5(bedges, nodes, 2, pbedge, file_name_h5, "pbedge", stat)
    call op_decl_map_hdf5(bedges, cells, 1, pbecell, file_name_h5, "pbecell", stat)
    call op_decl_map_hdf5(cells, nodes, 4, pcell, file_name_h5, "pcell", stat)

    print *, "Declaring OP2 data (HDF5)"
    call op_decl_dat_hdf5(bedges, 1, p_bound, "integer(4)", file_name_h5, "p_bound", stat)
    call op_decl_dat_hdf5(nodes, 2, p_x, "real(8)", file_name_h5, "p_x", stat)
    call op_decl_dat_hdf5(cells, 4, p_q, "real(8)", file_name_h5, "p_q", stat)
    call op_decl_dat_hdf5(cells, 4, p_qold, "real(8)", file_name_h5, "p_qold", stat)
    call op_decl_dat_hdf5(cells, 1, p_adt, "real(8)", file_name_h5, "p_adt", stat)
    call op_decl_dat_hdf5(cells, 4, p_res, "real(8)", file_name_h5, "p_res", stat)
#else
    print *, "Declaring OP2 sets"
    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nedge, edges, "edges")
    call op_decl_set(nbedge, bedges, "bedges")
    call op_decl_set(ncell, cells, "cells")

    print *, "Declaring OP2 maps"
    call op_decl_map(edges, nodes, 2, edge, pedge, "pedge")
    call op_decl_map(edges, cells, 2, ecell, pecell, "pecell")
    call op_decl_map(bedges, nodes, 2, bedge, pbedge, "pbedge")
    call op_decl_map(bedges, cells, 1, becell, pbecell, "pbecell")
    call op_decl_map(cells, nodes, 4, cell, pcell, "pcell")

    print *, "Declaring OP2 data"
    call op_decl_dat(bedges, 1, "integer(4)", bound, p_bound, "p_bound")
    call op_decl_dat(nodes, 2, "real(8)", x, p_x, "p_x")
    call op_decl_dat(cells, 4, "real(8)", q, p_q, "p_q")
    call op_decl_dat(cells, 4, "real(8)", qold, p_qold, "p_qold")
    call op_decl_dat(cells, 1, "real(8)", adt, p_adt, "p_adt")
    call op_decl_dat(cells, 4, "real(8)", res, p_res, "p_res")

    call release_buffers()
#endif

    print *, "Declaring OP2 constants"
    call op_decl_const(gam, 1, "real(8)")
    call op_decl_const(gm1, 1, "real(8)")
    call op_decl_const(cfl, 1, "real(8)")
    call op_decl_const(eps, 1, "real(8)")
    call op_decl_const(mach, 1, "real(8)")
    call op_decl_const(alpha, 1, "real(8)")
    call op_decl_const(qinf, 4, "real(8)")

    call op_partition("PTSCOTCH", "KWAY", edges, pecell, p_x)
    call op_timers(start_time)

    do iter = 1, niter
        call op_par_loop_2(save_soln, cells, &
            op_arg_dat(p_q,    -1, OP_ID, 4, "real(8)", OP_READ), &
            op_arg_dat(p_qold, -1, OP_ID, 4, "real(8)", OP_WRITE))

        do k = 1, 2
            call op_par_loop_6(adt_calc, cells, &
                op_arg_dat(p_x,    1, pcell, 2, "real(8)", OP_READ), &
                op_arg_dat(p_x,    2, pcell, 2, "real(8)", OP_READ), &
                op_arg_dat(p_x,    3, pcell, 2, "real(8)", OP_READ), &
                op_arg_dat(p_x,    4, pcell, 2, "real(8)", OP_READ), &
                op_arg_dat(p_q,   -1, OP_ID, 4, "real(8)", OP_READ), &
                op_arg_dat(p_adt, -1, OP_ID, 1, "real(8)", OP_WRITE))

            call op_par_loop_8(res_calc, edges, &
                op_arg_dat(p_x,    1, pedge,  2, "real(8)", OP_READ), &
                op_arg_dat(p_x,    2, pedge,  2, "real(8)", OP_READ), &
                op_arg_dat(p_q,    1, pecell, 4, "real(8)", OP_READ), &
                op_arg_dat(p_q,    2, pecell, 4, "real(8)", OP_READ), &
                op_arg_dat(p_adt,  1, pecell, 1, "real(8)", OP_READ), &
                op_arg_dat(p_adt,  2, pecell, 1, "real(8)", OP_READ), &
                op_arg_dat(p_res,  1, pecell, 4, "real(8)", OP_INC),  &
                op_arg_dat(p_res,  2, pecell, 4, "real(8)", OP_INC))

            call op_par_loop_6(bres_calc, bedges, &
                op_arg_dat(p_x,      1, pbedge,  2, "real(8)",    OP_READ), &
                op_arg_dat(p_x,      2, pbedge,  2, "real(8)",    OP_READ), &
                op_arg_dat(p_q,      1, pbecell, 4, "real(8)",    OP_READ), &
                op_arg_dat(p_adt,    1, pbecell, 1, "real(8)",    OP_READ), &
                op_arg_dat(p_res,    1, pbecell, 4, "real(8)",    OP_INC),  &
                op_arg_dat(p_bound, -1, OP_ID,   1, "integer(4)", OP_READ))

            rms = 0.0
            call op_par_loop_5(update, cells, &
                op_arg_dat(p_qold, -1, OP_ID, 4, "real(8)", OP_READ),  &
                op_arg_dat(p_q,    -1, OP_ID, 4, "real(8)", OP_WRITE), &
                op_arg_dat(p_res,  -1, OP_ID, 4, "real(8)", OP_RW),    &
                op_arg_dat(p_adt,  -1, OP_ID, 1, "real(8)", OP_READ),  &
                op_arg_gbl(rms,     2,           "real(8)", OP_INC))
        end do

        rms(2) = sqrt(rms(2) / real(ncell))

        if (op_is_root() .eq. 1 .and. mod(iter, 100) == 0) then
            print *, iter, rms(2)
        end if
    end do

    call op_timers(end_time)
    call op_timing_output()

    print *
    print *, "Time =", end_time - start_time, "seconds"

    if (niter == 1000 .and. ncell == 720000) then
        diff = abs((100.0_8 * (rms(2) / 0.0001060114637578_8)) - 100.0_8)

        print *
        write (*, "(A, I0, A, E16.7, A)") " Test problem with ", ncell , &
            " cells is within ", diff, "% of the expected solution"

        if(diff < 0.00001_8) THEN
            print *, "Test PASSED"
        else
            print *, "Test FAILED"
        end if
    end if

    call op_exit()

contains

#ifndef HDF5
    subroutine read_input()
        implicit none

        integer(4) :: i

        open(file_id, file=file_name)

        read(file_id, *) nnode, ncell, nedge, nbedge

#ifdef PLAIN
        allocate(x(2 * nnode))

        allocate(cell(4 * ncell))
        allocate(edge(2 * nedge))
        allocate(ecell(2 * nedge))
        allocate(bedge(2 * nbedge))
        allocate(becell(nbedge))
        allocate(bound(nbedge))
#endif

#ifdef ARG_PTRS
        call op_memalloc(ptr_x, 2 * nnode * int(sizeof(x(1))))

        call op_memalloc(ptr_cell, 4 * ncell * int(sizeof(cell(1))))
        call op_memalloc(ptr_edge, 2 * nedge * int(sizeof(edge(1))))
        call op_memalloc(ptr_ecell, 2 * nedge * int(sizeof(ecell(1))))
        call op_memalloc(ptr_bedge, 2 * nbedge * int(sizeof(bedge(1))))
        call op_memalloc(ptr_becell, nbedge * int(sizeof(becell(1))))
        call op_memalloc(ptr_bound, nbedge * int(sizeof(bound(1))))
#endif

        do i = 1, nnode
            read(file_id, *) x(2 * (i - 1) + 1), x(2 * (i - 1) + 2)
        end do

        do i = 1, ncell
            read(file_id, *) cell(4 * (i - 1) + 1), cell(4 * (i - 1) + 2), cell(4 * (i - 1) + 3), cell(4 * (i - 1) + 4)
        end do

        do i = 1, nedge
            read(file_id, *) edge(2 * (i - 1) + 1), edge(2 * (i - 1) + 2), ecell(2 * (i - 1) + 1), ecell(2 * (i - 1) + 2)
        end do

        do i = 1, nbedge
            read(file_id, *) bedge(2 * (i - 1) + 1), bedge(2 * (i - 1) + 2), becell(i), bound(i)
        end do

        close(file_id)

#ifdef PLAIN
        allocate(q(4 * ncell))
        allocate(qold(4 * ncell))
        allocate(res(4 * ncell))
        allocate(adt(ncell))
#endif

#ifdef USE_ARG_PTRS
        call op_memalloc(ptr_q, 4 * ncell * int(sizeof(q(1))))
        call op_memalloc(ptr_qold, 4 * ncell * int(sizeof(qold(1))))
        call op_memalloc(ptr_res, 4 * ncell * int(sizeof(res(1))))
        call op_memalloc(ptr_adt, ncell * int(sizeof(adt(1))))
#endif

        do i = 1, ncell
            q(4 * (i - 1) + 1:4 * (i - 1) + 4) = qinf
        end do

        qold(:4 * ncell) = 0.0_8
        res(:4 * ncell) = 0.0_8
        adt(:ncell) = 0.0_8
    end subroutine
#endif

#ifndef HDF5
    subroutine release_buffers()
#ifdef PLAIN
        deallocate(edge)
        deallocate(ecell)
        deallocate(bedge)
        deallocate(becell)
        deallocate(cell)
        deallocate(bound)

        deallocate(x)
        deallocate(q)
        deallocate(qold)
        deallocate(adt)
        deallocate(res)
#endif

#ifdef USE_ARG_PTRS
        call free(ptr_edge)
        call free(ptr_ecell)
        call free(ptr_bedge)
        call free(ptr_becell)
        call free(ptr_cell)
        call free(ptr_bound)

        call free(ptr_x)
        call free(ptr_q)
        call free(ptr_qold)
        call free(ptr_adt)
        call free(ptr_res)
#endif
    end subroutine
#endif

end program

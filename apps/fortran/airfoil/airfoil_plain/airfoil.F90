program airfoil
    use op2_fortran_declarations
    use op2_fortran_reference

    use airfoil_constants
    use airfoil_input
    use airfoil_kernels

    use, intrinsic :: iso_c_binding

    implicit none

    integer(4), parameter :: niter = 1000
    integer(4) :: iter, i, j, k

    integer(4) :: nnode, ncell, nbedge, nedge

    integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
    real(8), dimension(:), allocatable, target :: x, q, qold, adt, res
    real(8), dimension(2) :: rms

    type(op_set) :: nodes, edges, bedges, cells
    type(op_map) :: pedge, pecell, pcell, pbedge, pbecell
    type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res

    real(kind=c_double) :: start_time, end_time

    real(8) :: diff

    print *, "Reading input file"
    call read_input(nnode, ncell, nedge, nbedge, x, cell, edge, ecell, bedge, becell, bound)

    print *, ncell

    allocate(q(4 * ncell))
    allocate(qold(4 * ncell))
    allocate(res(4 * ncell))
    allocate(adt(ncell))

    do i = 1, ncell
        q(4 * (i - 1) + 1:) = qinf
    end do

    qold = 0.0_8
    res = 0.0_8
    adt = 0.0_8

    call op_init_base(0, 0)

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
                op_arg_gbl(rms, 2, "real(8)", OP_INC))
        end do

        rms(2) = sqrt(rms(2) / real(ncell))

        if (op_is_root() .eq. 1 .and. mod(iter, 100) == 0) then
            print *, iter, rms(2)
        end if
    end do

    if (niter == 1000 .and. ncell == 720000) then
        diff = abs((100.0_8 * (rms(2) / 0.0001060114637578_8)) - 100.0_8)

        write (*, "(A, I0, A, E16.7, A)") "Test problem with ", ncell , &
            " cells is within ", diff, "% of the expected solution"

        if(diff < 0.00001) THEN
            print *, "Test Passed"
        else
            print *, "Test Failed"
        end if
    end if

    call op_timers(end_time)
    call op_timing_output()

    print *, "Time =", end_time - start_time, "seconds"

    call op_exit()
end program airfoil

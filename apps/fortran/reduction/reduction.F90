program reduction
#ifdef HDF5
    use op2_fortran_hdf5_declarations
#else
    use op2_fortran_declarations
#endif

    use op2_fortran_reference
    use op2_fortran_rt_support

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), parameter :: file_id = 1

    character(*), parameter :: file_name = "new_grid.dat"
    character(*), parameter :: file_name_h5 = "new_grid.h5"

    integer(4) :: nnode, ncell, nedge
    integer(4) :: ncell_total, nedge_total

#ifndef HDF5
    integer(4), dimension(:), allocatable, target :: ecell
    real(8), dimension(:), allocatable, target :: res
#endif

    type(op_set) :: edges, cells
    type(op_map) :: pecell
    type(op_dat) :: p_res, p_dummy

    real(kind = c_double) :: start_time, end_time

    integer(4) :: i, cell_count_result, edge_count_result

#ifndef HDF5
    integer(4) :: dummy_int
    real(8) :: dummy_real
#else
    integer(4) :: stat
#endif

    call op_init_base(0, 0)

#ifndef HDF5
    open(file_id, file = file_name)

    read(file_id, *) nnode, ncell, nedge, dummy_int

    allocate(ecell(2 * nedge))

    do i = 1, nnode
        read(file_id, *) dummy_real, dummy_real
    end do

    do i = 1, ncell
        read(file_id, *) dummy_int, dummy_int, dummy_int, dummy_int
    end do

    do i = 1, nedge
        read(file_id, *) dummy_int, dummy_int, ecell(2 * (i - 1) + 1), ecell(2 * (i - 1) + 2)
    end do

    close(file_id)

    allocate(res(4 * ncell))
    res = 0.0
#endif

#ifdef HDF5
    call op_decl_set_hdf5(nedge, edges, file_name_h5, "edges")
    call op_decl_set_hdf5(ncell, cells, file_name_h5, "cells")

    call op_decl_map_hdf5(edges, cells, 2, pecell, file_name_h5, "pecell", stat)

    call op_decl_dat_hdf5(cells, 4, p_res, "real(8)", file_name_h5, "p_res", stat)
#else
    call op_decl_set(nedge, edges, "edges")
    call op_decl_set(ncell, cells, "cells")

    call op_decl_map(edges, cells, 2, ecell, pecell, "pecell")

    call op_decl_dat(cells, 4, "real(8)", res, p_res, "p_res")
#endif

#ifndef HDF5
    deallocate(ecell)
    deallocate(res)
#endif

    call op_partition("PTSCOTCH", "KWAY", edges, pecell, p_dummy)
    call op_timers(start_time)

    ncell_total = op_get_size(cells)
    nedge_total = op_get_size(edges)

    cell_count_result = 0
    edge_count_result = 0

    call op_par_loop_2(cell_count, cells, &
        op_arg_dat(p_res, -1, OP_ID, 4, "real(8)", OP_RW), &
        op_arg_gbl(cell_count_result, 1, "integer(4)", OP_INC))

    call op_par_loop_2(edge_count, edges, &
        op_arg_dat(p_res, 1, pecell, 4, "real(8)", OP_RW), &
        op_arg_gbl(edge_count_result, 1, "integer(4)", OP_INC))

    call op_timers(end_time)
    call op_timing_output()

    if (op_is_root() == 1) then
        print *
        print *, "Direct reduction: cell count = ", cell_count_result, ", target = ", ncell_total
        print *, "Indirect reduction: edge count = ", edge_count_result, ", target = ", nedge_total
        print *

        if (cell_count_result == ncell_total .and. edge_count_result == nedge_total) then
            print *, "Test PASSED"
        else
            print *, "Test FAILED"
        end if

        print *
        print *, 'Time = ', end_time - start_time, 'seconds'
    end if

    call op_exit()

contains

    subroutine cell_count(res, cell_count_result)

        implicit none

        real(8), dimension(4) :: res
        integer(4), dimension(1) :: cell_count_result

        res = 0.0
        cell_count_result = cell_count_result + 1

    end subroutine

    subroutine edge_count(res, edge_count_result)

        implicit none

        real(8), dimension(4) :: res
        integer(4), dimension(1) :: edge_count_result

        res = 0.0
        edge_count_result = edge_count_result + 1

    end subroutine

end program

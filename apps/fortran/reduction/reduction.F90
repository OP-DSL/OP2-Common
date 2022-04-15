program reduction

    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer(4), parameter :: file_id = 1
    character(len = *), parameter :: file_name = "new_grid.dat"

    integer(4) :: nnode, ncell, nedge

    integer(4), dimension(:), allocatable, target :: ecell
    real(8), dimension(:), allocatable, target :: res

    type(op_set) :: edges, cells
    type(op_map) :: pecell
    type(op_dat) :: p_res

    real(kind = c_double) :: start_time, end_time

    integer(4) :: i, cell_count_result, edge_count_result

    integer(4) :: dummy_int
    real(8) :: dummy_real

    call op_init_base(0, 0)

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

    call op_decl_set(nedge, edges, 'edges')
    call op_decl_set(ncell, cells, 'cells')

    call op_decl_map(edges, cells, 2, ecell, pecell, 'pecell')

    call op_decl_dat(cells, 4, 'real(8)', res, p_res, 'p_res')

    deallocate(ecell)
    deallocate(res)

    call op_timers(start_time)

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

    print *
    print *, "Direct reduction: cell count = ", cell_count_result, ", target = ", ncell
    print *, "Indirect reduction: edge count = ", edge_count_result, ", target = ", nedge
    print *

    if (cell_count_result == ncell .and. edge_count_result == nedge) then
        print *, "Test PASSED"
    else
        print *, "Test FAILED"
    end if

    print *
    print *, 'Time = ', end_time - start_time, 'seconds'

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

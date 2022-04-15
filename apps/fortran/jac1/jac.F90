program jac

    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    use, intrinsic :: ISO_C_BINDING

    implicit none

    real(8), parameter :: tolerance = 1e-12
    integer(4), parameter :: nn = 6
    integer(4), parameter :: niter = 2

    real(kind = c_double) :: start_time, end_time
    logical :: valid

    integer(4) :: i, j, p

    real(8) :: u_sum, u_max, alpha, beta

    integer(4) :: nnode, nedge
    integer(4), dimension(:), allocatable :: pp

    real(8), dimension(:), allocatable :: A, r, u, du

    type(op_set) :: nodes, edges
    type(op_map) :: ppedge
    type(op_dat) :: p_A, p_r, p_u, p_du

    call op_init(0)

    nnode = (nn - 1) * (nn - 1)
    nedge = nnode + 4 * (nn - 1) * (nn - 2)

    allocate(pp(nedge * 2))
    allocate(A(nedge))
    allocate(r(nnode))
    allocate(u(nnode))
    allocate(du(nnode))

    call init_data()

    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nedge, edges, "edges")

    call op_decl_map(edges, nodes, 2, pp, ppedge, "ppedge")

    call op_decl_dat(edges, 1, "real(8)", A, p_A, "p_A")
    call op_decl_dat(nodes, 1, "real(8)", r, p_r, "p_r")
    call op_decl_dat(nodes, 1, "real(8)", u, p_u, "p_u")
    call op_decl_dat(nodes, 1, "real(8)", du, p_du, "p_du")

    deallocate(pp)
    deallocate(A)
    deallocate(r)
    deallocate(u)
    deallocate(du)

    alpha = 1.0

    call op_decl_const(alpha, 1, "real(8)")

    call op_timers(start_time)

    beta = 1.0

    do i = 1, niter
        call op_par_loop_4(res, edges, &
            op_arg_dat(p_A, -1, OP_ID,  1, "real(8)", OP_READ), &
            op_arg_dat(p_u,  2, ppedge, 1, "real(8)", OP_READ), &
            op_arg_dat(p_du, 1, ppedge, 1, "real(8)", OP_INC),  &
            op_arg_gbl(beta, 1, "real(8)", OP_READ))

        u_sum = 0.0
        u_max = 0.0

        call op_par_loop_5(update, nodes, &
            op_arg_dat(p_r,  -1, OP_ID, 1, "real(8)", OP_READ), &
            op_arg_dat(p_du, -1, OP_ID, 1, "real(8)", OP_RW),   &
            op_arg_dat(p_u,  -1, OP_ID, 1, "real(8)", OP_INC),  &
            op_arg_gbl(u_sum, 1, "real(8)", OP_INC), &
            op_arg_gbl(u_max, 1, "real(8)", OP_MAX))

        write (*, "(1X, A, F7.4, A, F10.8)") "u max = ", u_max, "; u rms = ", sqrt(u_sum / nnode)
    end do

    call op_timers(end_time)
    call op_timing_output()

    allocate(u(nnode))
    call op_fetch_data(p_u, u)

    print *
    write (*, "(1X, A, I0, A)") "Results after ", niter, " iterations:"

    call output_data()

    valid = check_data()
    if (valid) then
        print *, "Test PASSED"
    else
        print *, "Test FAILED"
    endif

    print *
    write (*, "(1X, A, F7.5, A)") "Time = ", end_time - start_time, " seconds"

    deallocate(u)
    call op_exit()

contains

    subroutine res(A, u, du, beta)
        implicit none
        real(8), dimension(1) :: A, u, du, beta

        du(1) = du(1) + beta(1) * A(1) * u(1)
    end subroutine

    subroutine update(r, du, u, u_sum, u_max)
        implicit none
        real(8), dimension(1) :: r, du, u, u_sum, u_max

        u(1) = u(1) + du(1) + alpha * r(1)
        du(1) = 0.0

        u_sum(1) = u_sum(1) + u(1) ** 2
        u_max(1) = max(u_max(1), u(1))
    end subroutine

    subroutine init_data()
        implicit none

        integer(4) :: n, e, i2, j2
        integer(4), dimension(4) :: i_p, j_p

        i_p = (/-1, 1, 0, 0/)
        j_p = (/0, 0, -1, 1/)

        e = 1

        do i = 1, nn - 1
            do j = 1, nn - 1
                n = i + (j - 1) * (nn - 1)

                r(n) = 0.0
                u(n) = 0.0
                du(n) = 0.0

                pp(2 * (e - 1) + 1) = n
                pp(2 * (e - 1) + 2) = n

                A(e) = -1.0

                e = e + 1

                do p = 1, 4
                    i2 = i + i_p(p)
                    j2 = j + j_p(p)

                    if (i2 == 0 .or. i2 == nn .or. j2 == 0 .or. j2 == nn) then
                        r(n) = r(n) + 0.25
                    else
                        pp(2 * (e - 1) + 1) = n
                        pp(2 * (e - 1) + 2) = i2 + (j2 - 1) * (nn - 1)

                        A(e) = 0.25
                        e = e + 1
                    end if
                end do
            end do
        end do
    end subroutine

    subroutine output_data()
        implicit none

        do j = nn - 1, 1, -1
            do i = 1, nn - 1
                write(*, "(1X, F7.4)", advance="no") u(i + (j - 1) * (nn - 1))
            end do

            write(*, *)
        end do

        write(*, *)
    end subroutine

    function check_data() result(valid)
        implicit none

        integer(4) :: n
        logical :: valid

        valid = .true.

        do i = 1, nn - 1
            do j = 1, nn - 1
                n = i + (j - 1) * (nn - 1)

                if ((i == 1 .or. i == nn - 1) .and. (j == 1 .or. j == nn - 1)) then
                    ! Corners
                    valid = check_value(u(n), 0.6250_8) .and. valid
                else if ((i == 1 .or. i == nn - 1) .and. (j == 2 .or. j == nn - 2)) then
                    ! Horizontally adjacent to a corner
                    valid = check_value(u(n), 0.4375_8) .and. valid
                else if ((j == 1 .or. j == nn - 1) .and. (i == 2 .or. i == nn - 2)) then
                    ! Vertically adjacent to a corner
                    valid = check_value(u(n), 0.4375_8) .and. valid
                else if ((i == 2 .or. i == nn - 2) .and. (j == 2 .or. j == nn - 2)) then
                    ! Diagonally adjacent to a corner
                    valid = check_value(u(n), 0.1250_8) .and. valid
                else if (i == 1 .or. i == nn - 1 .or. j == 1 .or. j == nn - 1) then
                    ! Other edge nodes
                    valid = check_value(u(n), 0.3750_8) .and. valid
                else if (i == 2 .or. i == nn - 2 .or. j == 2 .or. j == nn - 2) then
                    ! Other nodes 1 node from edge
                    valid = check_value(u(n), 0.0625_8) .and. valid
                else
                    ! Other nodes
                    valid = check_value(u(n), 0.0000_8) .and. valid
                endif
            end do
        end do
    end function

    function check_value(x, ref) result(valid)
        real(8) :: x, ref
        logical :: valid

        valid = abs(x - ref) < tolerance

        if (.not. valid) then
            write(*, "(1X, A, F7.4, A, F7.4, A, I0, A, I0)") &
                "Node check failed: expected = ", ref, "; actual = ", x, "; i = ", i, "; j = ", j
        endif
    end function

end program

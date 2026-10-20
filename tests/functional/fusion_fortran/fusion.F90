program fusion
!
! Functional test for lazy execution and runtime kernel fusion.  Each sweep
! queues nine calls with no fetch between them, so the queue runs them
! together: a direct chain that reads its own writes, a call reading through
! a map from a set to itself, which must not fuse with the call before it,
! indirect increments over two sets, the same loop twice in a row, and an
! indirect read followed by a reduction.  Every result is checked against a
! host reference.
!
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    implicit none

    real(8), parameter :: tol = 1.0e-10_8
    integer(4), parameter :: nnode = 1001
    integer(4), parameter :: nedge = 1001
    integer(4), parameter :: ncell = 333
    integer(4), parameter :: nsweeps = 30

    integer(4) :: i, e, sweep, failures, opt_flag
    integer(4), dimension(:), allocatable :: e2n, n2n, c2n
    real(8), dimension(:), allocatable :: a, b, c, d, w, f
    real(8), dimension(:), allocatable :: ref_a, ref_b, ref_c, ref_d, ref_f
    real(8) :: k, scale, total, ref_total
    logical :: use_opt

    type(op_set) :: nodes, edges, cells
    type(op_map) :: p_e2n, p_n2n, p_c2n
    type(op_dat) :: p_a, p_b, p_c, p_d, p_w, p_f

    call op_init(0)

    allocate(e2n(2 * nedge), n2n(nnode), c2n(ncell))
    do e = 1, nedge
        e2n(2 * e - 1) = e
        e2n(2 * e) = mod(e, nnode) + 1
    end do
    do i = 1, nnode
        n2n(i) = mod(i, nnode) + 1
    end do
    do i = 1, ncell
        c2n(i) = mod(3 * i, nnode) + 1
    end do

    allocate(a(2 * nnode), b(nnode), c(nnode), d(nnode), w(nedge), f(nedge))
    do i = 1, nnode
        a(2 * i - 1) = 0.01_8 * dble(i)
        a(2 * i) = 1.0_8 - 0.001_8 * dble(i)
        b(i) = 0.0_8
        c(i) = 0.5_8 + 0.002_8 * dble(i)
        d(i) = 0.0_8
    end do
    do e = 1, nedge
        w(e) = 0.25_8 + 0.0001_8 * dble(e)
        f(e) = 0.0_8
    end do
    ref_a = a
    ref_b = b
    ref_c = c
    ref_d = d
    ref_f = f

    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nedge, edges, "edges")
    call op_decl_set(ncell, cells, "cells")
    call op_decl_map(edges, nodes, 2, e2n, p_e2n, "e2n")
    call op_decl_map(nodes, nodes, 1, n2n, p_n2n, "n2n")
    call op_decl_map(cells, nodes, 1, c2n, p_c2n, "c2n")

    call op_decl_dat(nodes, 2, "real(8)", a, p_a, "a")
    call op_decl_dat(nodes, 1, "real(8)", b, p_b, "b")
    call op_decl_dat(nodes, 1, "real(8)", c, p_c, "c")
    call op_decl_dat(nodes, 1, "real(8)", d, p_d, "d")
    call op_decl_dat(edges, 1, "real(8)", w, p_w, "w")
    call op_decl_dat(edges, 1, "real(8)", f, p_f, "f")

    scale = 0.25_8
    failures = 0

    do sweep = 1, nsweeps
        ! Two values, so each fused kernel compiles twice at most.
        k = dble(mod(sweep, 2) + 1)
        use_opt = mod(sweep, 3) == 0
        opt_flag = merge(1, 0, use_opt)

        ! A direct chain: the second call reads what the first writes.
        call op_par_loop_4(halve_add, nodes, &
            op_arg_dat(p_a, -1, OP_ID, 2, "real(8)", OP_RW), &
            op_opt_arg_dat(use_opt, p_d, -1, OP_ID, 1, "real(8)", OP_RW), &
            op_arg_gbl(k, 1, "real(8)", OP_READ), &
            op_arg_gbl(opt_flag, 1, "integer(4)", OP_READ))
        call op_par_loop_2(sum_pair, nodes, &
            op_arg_dat(p_a, -1, OP_ID, 2, "real(8)", OP_READ), &
            op_arg_dat(p_b, -1, OP_ID, 1, "real(8)", OP_WRITE))

        ! The next node's b: other elements of the call before.
        call op_par_loop_2(add_next, nodes, &
            op_arg_dat(p_b, 1, p_n2n, 1, "real(8)", OP_READ), &
            op_arg_dat(p_d, -1, OP_ID, 1, "real(8)", OP_INC))

        ! Increments of c over edges and over cells, independent of each
        ! other.
        call op_par_loop_4(spread, edges, &
            op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ), &
            op_arg_dat(p_b, 1, p_e2n, 1, "real(8)", OP_READ), &
            op_arg_dat(p_c, 1, p_e2n, 1, "real(8)", OP_INC), &
            op_arg_dat(p_c, 2, p_e2n, 1, "real(8)", OP_INC))
        call op_par_loop_1(bump, cells, &
            op_arg_dat(p_c, 1, p_c2n, 1, "real(8)", OP_INC))

        ! The same loop twice.
        call op_par_loop_2(damp, nodes, &
            op_arg_dat(p_c, -1, OP_ID, 1, "real(8)", OP_RW), &
            op_arg_gbl(scale, 1, "real(8)", OP_READ))
        call op_par_loop_2(damp, nodes, &
            op_arg_dat(p_c, -1, OP_ID, 1, "real(8)", OP_RW), &
            op_arg_gbl(scale, 1, "real(8)", OP_READ))

        ! An indirect read of c, then a reduction over it.
        call op_par_loop_3(difference, edges, &
            op_arg_dat(p_c, 1, p_e2n, 1, "real(8)", OP_READ), &
            op_arg_dat(p_c, 2, p_e2n, 1, "real(8)", OP_READ), &
            op_arg_dat(p_f, -1, OP_ID, 1, "real(8)", OP_WRITE))
        total = 0.0_8
        call op_par_loop_2(accumulate, nodes, &
            op_arg_dat(p_c, -1, OP_ID, 1, "real(8)", OP_READ), &
            op_arg_gbl(total, 1, "real(8)", OP_INC))

        call reference_sweep()
        if (.not. close_to(total, ref_total)) then
            failures = failures + 1
            if (failures < 20) print *, "MISMATCH total sweep ", sweep, " expected ", &
                ref_total, " got ", total
        end if
    end do

    call op_fetch_data(p_a, a)
    call op_fetch_data(p_b, b)
    call op_fetch_data(p_c, c)
    call op_fetch_data(p_d, d)
    call op_fetch_data(p_f, f)

    call check("a", a, ref_a)
    call check("b", b, ref_b)
    call check("c", c, ref_c)
    call check("d", d, ref_d)
    call check("f", f, ref_f)

    if (failures == 0) then
        print *, "Test PASSED"
        call op_exit()
    else
        print *, "Test FAILED with", failures, "mismatches"
        call op_exit()
        error stop 1
    end if

contains

    subroutine halve_add(x, y, kk, flag)
        real(8), dimension(2) :: x
        real(8) :: y, kk
        integer(4) :: flag
        x(1) = 0.5_8 * x(1) + kk
        x(2) = 0.5_8 * x(2) - kk
        if (flag /= 0) y = y + 1.0_8
    end subroutine halve_add

    subroutine sum_pair(x, y)
        real(8), dimension(2) :: x
        real(8) :: y
        y = x(1) + x(2) * x(2)
    end subroutine sum_pair

    subroutine add_next(next, y)
        real(8) :: next, y
        y = y + 0.001_8 * next
    end subroutine add_next

    subroutine spread(weight, b1, c1, c2)
        real(8) :: weight, b1, c1, c2
        c1 = c1 + weight * b1
        c2 = c2 - 0.5_8 * weight * b1
    end subroutine spread

    subroutine bump(c1)
        real(8) :: c1
        c1 = c1 + 1.0_8
    end subroutine bump

    subroutine damp(x, s)
        real(8) :: x, s
        x = x * s + 0.125_8
    end subroutine damp

    subroutine difference(c1, c2, out)
        real(8) :: c1, c2, out
        out = c1 - c2
    end subroutine difference

    subroutine accumulate(x, sum)
        real(8) :: x, sum
        sum = sum + x
    end subroutine accumulate

    subroutine reference_sweep()
        integer(4) :: n, m, j
        real(8) :: weight

        do n = 1, nnode
            ref_a(2 * n - 1) = 0.5_8 * ref_a(2 * n - 1) + k
            ref_a(2 * n) = 0.5_8 * ref_a(2 * n) - k
            if (use_opt) ref_d(n) = ref_d(n) + 1.0_8
        end do
        do n = 1, nnode
            ref_b(n) = ref_a(2 * n - 1) + ref_a(2 * n) * ref_a(2 * n)
        end do
        do n = 1, nnode
            ref_d(n) = ref_d(n) + 0.001_8 * ref_b(n2n(n))
        end do
        do j = 1, nedge
            weight = w(j)
            n = e2n(2 * j - 1)
            m = e2n(2 * j)
            ref_c(n) = ref_c(n) + weight * ref_b(n)
            ref_c(m) = ref_c(m) - 0.5_8 * weight * ref_b(n)
        end do
        do j = 1, ncell
            ref_c(c2n(j)) = ref_c(c2n(j)) + 1.0_8
        end do
        do n = 1, nnode
            ref_c(n) = ref_c(n) * scale + 0.125_8
            ref_c(n) = ref_c(n) * scale + 0.125_8
        end do
        do j = 1, nedge
            ref_f(j) = ref_c(e2n(2 * j - 1)) - ref_c(e2n(2 * j))
        end do
        ref_total = 0.0_8
        do n = 1, nnode
            ref_total = ref_total + ref_c(n)
        end do
    end subroutine reference_sweep

    logical function close_to(actual, expected)
        real(8), intent(in) :: actual, expected
        close_to = abs(actual - expected) <= tol * max(1.0_8, abs(expected))
    end function close_to

    subroutine check(name, actual, expected)
        character(*), intent(in) :: name
        real(8), dimension(:), intent(in) :: actual, expected
        integer(4) :: j
        do j = 1, size(expected)
            if (.not. close_to(actual(j), expected(j))) then
                failures = failures + 1
                if (failures < 20) print *, "MISMATCH ", name, " index ", j, &
                    " expected ", expected(j), " got ", actual(j)
            end if
        end do
    end subroutine check

end program fusion

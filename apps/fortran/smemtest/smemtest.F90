program smemtest
!
! Functional test for indirect OP_INC handling.  It deliberately combines
! mixed scalar types and dimensions, repeated/aliased arguments, two target
! sets, runtime dimensions, optional arguments, and repeated loop calls.
!
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    implicit none

    real(8), parameter :: tol8 = 1.0e-10_8
    real(4), parameter :: tol4 = 5.0e-5_4
    integer(4), parameter :: nn = 17
    integer(4), parameter :: nnode = (nn - 1) * (nn - 1)
    integer(4), parameter :: nother = nnode + 7
    integer(4), parameter :: nsweeps = 20

    logical :: enable_optional, valid
    integer(4) :: i, j, e, p, n, d, i2, j2, sweep
    integer(4) :: nedge, runtime_dim, optional_scale
    integer(4) :: failures
    integer(4), dimension(4) :: i_p, j_p

    integer(4), dimension(:), allocatable :: pp, pp_rev, to_other
    real(8), dimension(:), allocatable :: edge_weight
    real(8), dimension(:), allocatable :: u8
    real(4), dimension(:), allocatable :: u4

    real(8), dimension(:), allocatable :: du_a
    real(4), dimension(:), allocatable :: du_b
    integer(4), dimension(:), allocatable :: du_c
    real(8), dimension(:), allocatable :: du_d
    real(4), dimension(:), allocatable :: du_e
    real(8), dimension(:), allocatable :: du_runtime
    real(4), dimension(:), allocatable :: du_optional

    real(8), dimension(:), allocatable :: ref_a
    real(4), dimension(:), allocatable :: ref_b
    integer(4), dimension(:), allocatable :: ref_c
    real(8), dimension(:), allocatable :: ref_d
    real(4), dimension(:), allocatable :: ref_e
    real(8), dimension(:), allocatable :: ref_runtime
    real(4), dimension(:), allocatable :: ref_optional

    type(op_set) :: nodes, other_nodes, edges
    type(op_map) :: ppedge, ppedge_rev, edge_to_other
    type(op_dat) :: p_edge_weight, p_u8, p_u4
    type(op_dat) :: p_du_a, p_du_b, p_du_c, p_du_d, p_du_e
    type(op_dat) :: p_du_runtime, p_du_optional

    real(8) :: beta

    call op_init(0)

    runtime_dim = 4
    nedge = nnode + 4 * (nn - 1) * (nn - 2)
    if (mod(nedge, 128) == 0) error stop "smemtest source size must not be divisible by 128"

    allocate(pp(2 * nedge), pp_rev(2 * nedge), to_other(nedge))
    allocate(edge_weight(nedge), u8(nnode), u4(2 * nnode))
    allocate(du_a(2 * nnode), du_b(nnode), du_c(3 * nnode))
    allocate(du_d(nnode), du_e(2 * nnode))
    allocate(du_runtime(runtime_dim * nother), du_optional(nnode))

    allocate(ref_a(2 * nnode), ref_b(nnode), ref_c(3 * nnode))
    allocate(ref_d(nnode), ref_e(2 * nnode))
    allocate(ref_runtime(runtime_dim * nother), ref_optional(nnode))

    i_p = (/-1, 1, 0, 0/)
    j_p = (/0, 0, -1, 1/)

    ! Build a five-point-stencil edge list.  Fortran maps are one-based.
    e = 1
    do i = 1, nn - 1
        do j = 1, nn - 1
            n = i + (j - 1) * (nn - 1)

            pp(2 * e - 1) = n
            pp(2 * e) = n
            edge_weight(e) = -1.0_8
            e = e + 1

            do p = 1, 4
                i2 = i + i_p(p)
                j2 = j + j_p(p)
                if (i2 == 0 .or. i2 == nn .or. j2 == 0 .or. j2 == nn) cycle

                pp(2 * e - 1) = n
                pp(2 * e) = i2 + (j2 - 1) * (nn - 1)
                edge_weight(e) = 0.25_8
                e = e + 1
            end do
        end do
    end do

    if (e /= nedge + 1) error stop "smemtest edge count is inconsistent"

    do e = 1, nedge
        pp_rev(2 * e - 1) = pp(2 * e)
        pp_rev(2 * e) = pp(2 * e - 1)
        to_other(e) = mod(5 * (e - 1) + pp(2 * e - 1) + 2 * pp(2 * e), nother) + 1
    end do

    ! Use component- and element-dependent nonzero initial values.  Buffers
    ! supplied to op_decl_dat are AoS even when automatic SoA is enabled.
    do n = 1, nnode
        u8(n) = 0.5_8 * real(n, 8) - 0.125_8
        do d = 1, 2
            u4(2 * (n - 1) + d) = 0.25_4 * real(n, 4) + 0.03125_4 * real(d, 4)
            du_a(2 * (n - 1) + d) = 100.0_8 + 0.01_8 * real(n, 8) + real(d, 8)
            du_e(2 * (n - 1) + d) = 800.0_4 + 0.02_4 * real(n, 4) + real(d, 4)
        end do

        du_b(n) = 300.0_4 + 0.03_4 * real(n, 4)
        do d = 1, 3
            du_c(3 * (n - 1) + d) = 400_4 + 10_4 * d + n
        end do
        du_d(n) = 700.0_8 + 0.04_8 * real(n, 8)
        du_optional(n) = 1000.0_4 + 0.05_4 * real(n, 4)
    end do

    do n = 1, nother
        do d = 1, runtime_dim
            du_runtime(runtime_dim * (n - 1) + d) = &
                900.0_8 + 0.01_8 * real(n, 8) + 0.1_8 * real(d, 8)
        end do
    end do

    ! Copy the initialized payloads into the host references only after all
    ! initialization is complete.
    ref_a = du_a
    ref_b = du_b
    ref_c = du_c
    ref_d = du_d
    ref_e = du_e
    ref_runtime = du_runtime
    ref_optional = du_optional

    call op_decl_set(nnode, nodes, "nodes")
    call op_decl_set(nother, other_nodes, "other_nodes")
    call op_decl_set(nedge, edges, "edges")
    call op_decl_map(edges, nodes, 2, pp, ppedge, "ppedge")
    call op_decl_map(edges, nodes, 2, pp_rev, ppedge_rev, "ppedge_rev")
    call op_decl_map(edges, other_nodes, 1, to_other, edge_to_other, "edge_to_other")

    call op_decl_dat(edges, 1, "real(8)", edge_weight, p_edge_weight, "edge_weight")
    call op_decl_dat(nodes, 1, "real(8)", u8, p_u8, "u8")
    call op_decl_dat(nodes, 2, "real(4)", u4, p_u4, "u4")
    call op_decl_dat(nodes, 2, "real(8)", du_a, p_du_a, "du_a")
    call op_decl_dat(nodes, 1, "real(4)", du_b, p_du_b, "du_b")
    call op_decl_dat(nodes, 3, "integer(4)", du_c, p_du_c, "du_c")
    call op_decl_dat(nodes, 1, "real(8)", du_d, p_du_d, "du_d")
    call op_decl_dat(nodes, 2, "real(4)", du_e, p_du_e, "du_e")
    call op_decl_dat(other_nodes, runtime_dim, "real(8)", du_runtime, &
                     p_du_runtime, "du_runtime")
    call op_decl_dat(nodes, 1, "real(4)", du_optional, p_du_optional, "du_optional")

    beta = 2.0_8
    failures = 0

    call op_profile_start("SMEMTEST")

    do sweep = 1, nsweeps
        enable_optional = mod(sweep, 2) == 1
        if (enable_optional) then
            optional_scale = 1
        else
            optional_scale = 0
        end if

        do e = 1, nedge
            call accum_reference(e, pp(2 * e - 1), pp(2 * e), to_other(e), optional_scale)
        end do

        call op_par_loop_14(resk, edges, &
            op_arg_dat(p_edge_weight, -1, OP_ID,          1, "real(8)",    OP_READ), &
            op_arg_dat(p_u8,           2, ppedge,         1, "real(8)",    OP_READ), &
            op_arg_dat(p_u4,           1, ppedge,         2, "real(4)",    OP_READ), &
            op_arg_dat(p_du_a,         1, ppedge,         2, "real(8)",    OP_INC),  &
            op_arg_dat(p_du_b,         2, ppedge,         1, "real(4)",    OP_INC),  &
            op_arg_dat(p_du_c,         1, ppedge,         3, "integer(4)", OP_INC),  &
            op_arg_dat(p_du_d,         2, ppedge,         1, "real(8)",    OP_INC),  &
            op_arg_dat(p_du_e,         1, ppedge,         2, "real(4)",    OP_INC),  &
            op_arg_dat(p_du_d,         1, ppedge_rev,     1, "real(8)",    OP_INC),  &
            op_arg_dat(p_du_a,         1, ppedge,         2, "real(8)",    OP_INC),  &
            op_arg_dat(p_du_runtime,   1, edge_to_other, runtime_dim, "real(8)", OP_INC), &
            op_opt_arg_dat(enable_optional, p_du_optional, 1, ppedge, &
                           1, "real(4)", OP_INC), &
            op_arg_gbl(optional_scale, 1, "integer(4)", OP_READ), &
            op_arg_gbl(beta,           1, "real(8)",    OP_READ))
    end do

    call op_fetch_data(p_du_a, du_a)
    call op_fetch_data(p_du_b, du_b)
    call op_fetch_data(p_du_c, du_c)
    call op_fetch_data(p_du_d, du_d)
    call op_fetch_data(p_du_e, du_e)
    call op_fetch_data(p_du_runtime, du_runtime)
    call op_fetch_data(p_du_optional, du_optional)

    call validate_results()
    valid = failures == 0

    call op_profile_end()

    if (valid) then
        print *, "Test PASSED"
        call op_exit()
    else
        print *, "Test FAILED with", failures, "mismatches"
        call op_exit()
        error stop 1
    end if

contains

    subroutine resk(a1, u1, u2, b1, b2, c1, d1, e1, f1, g1, r1, o1, oscale, beta_value)
        implicit none

        real(8) :: a1, u1, d1, f1, beta_value
        real(4) :: b2, o1
        integer(4) :: oscale
        real(4), dimension(2) :: u2, e1
        real(8), dimension(2) :: b1, g1
        integer(4), dimension(3) :: c1
        real(8), dimension(4) :: r1

        real(8) :: c
        real(4) :: cf

        c = beta_value * a1
        cf = c

        b1(1) = b1(1) + c * (u1 + dble(u2(1)))
        b1(2) = b1(2) - c * (u1 - dble(u2(2)))
        b2 = b2 + cf * u2(1)
        c1(1) = c1(1) + int(mod(abs(c * 100.0_8), 7.0_8)) + 1
        c1(2) = c1(2) + int(mod(abs(c * 10.0_8), 3.0_8)) + 1
        c1(3) = c1(3) + 1
        d1 = d1 + c
        e1(1) = e1(1) + cf
        e1(2) = e1(2) - cf

        ! f1 aliases d1 through the reversed map.  g1 exactly aliases b1.
        f1 = f1 + c * u1
        g1(1) = g1(1) + c * (u1 + dble(u2(1)))
        g1(2) = g1(2) - c * (u1 - dble(u2(2)))

        r1(1) = r1(1) + c
        r1(2) = r1(2) + c * u1
        r1(3) = r1(3) + c * dble(u2(1))
        r1(4) = r1(4) + c * dble(u2(2))

        if (oscale .ne. 0) o1 = o1 + cf * (u2(1) + u2(2))
    end subroutine resk

    subroutine accum_reference(edge, n1, n2, other, oscale)
        implicit none

        integer(4), intent(in) :: edge, n1, n2, other, oscale
        real(8) :: c, read_u8
        real(4) :: cf, read_u4_1, read_u4_2

        c = beta * edge_weight(edge)
        cf = real(c, 4)
        read_u8 = u8(n2)
        read_u4_1 = u4(2 * (n1 - 1) + 1)
        read_u4_2 = u4(2 * (n1 - 1) + 2)

        ! p_du_a appears twice with the same map and map index.
        ref_a(2 * (n1 - 1) + 1) = ref_a(2 * (n1 - 1) + 1) + &
            2.0_8 * c * (read_u8 + real(read_u4_1, 8))
        ref_a(2 * (n1 - 1) + 2) = ref_a(2 * (n1 - 1) + 2) - &
            2.0_8 * c * (read_u8 - real(read_u4_2, 8))

        ref_b(n2) = ref_b(n2) + cf * read_u4_1
        ref_c(3 * (n1 - 1) + 1) = ref_c(3 * (n1 - 1) + 1) + &
            int(mod(abs(c * 100.0_8), 7.0_8), 4) + 1
        ref_c(3 * (n1 - 1) + 2) = ref_c(3 * (n1 - 1) + 2) + &
            int(mod(abs(c * 10.0_8), 3.0_8), 4) + 1
        ref_c(3 * (n1 - 1) + 3) = ref_c(3 * (n1 - 1) + 3) + 1

        ! Both p_du_d arguments resolve to n2, through different maps.
        ref_d(n2) = ref_d(n2) + c + c * read_u8
        ref_e(2 * (n1 - 1) + 1) = ref_e(2 * (n1 - 1) + 1) + cf
        ref_e(2 * (n1 - 1) + 2) = ref_e(2 * (n1 - 1) + 2) - cf

        ref_runtime(runtime_dim * (other - 1) + 1) = &
            ref_runtime(runtime_dim * (other - 1) + 1) + c
        ref_runtime(runtime_dim * (other - 1) + 2) = &
            ref_runtime(runtime_dim * (other - 1) + 2) + c * read_u8
        ref_runtime(runtime_dim * (other - 1) + 3) = &
            ref_runtime(runtime_dim * (other - 1) + 3) + c * real(read_u4_1, 8)
        ref_runtime(runtime_dim * (other - 1) + 4) = &
            ref_runtime(runtime_dim * (other - 1) + 4) + c * real(read_u4_2, 8)

        if (oscale /= 0) then
            ref_optional(n1) = ref_optional(n1) + cf * (read_u4_1 + read_u4_2)
        end if
    end subroutine accum_reference

    subroutine validate_results()
        implicit none

        integer(4) :: elem, component, idx

        do elem = 1, nnode
            do component = 1, 2
                idx = 2 * (elem - 1) + component
                if (.not. chk8("du_a", elem, component, ref_a(idx), du_a(idx))) failures = failures + 1
                if (.not. chk4("du_e", elem, component, ref_e(idx), du_e(idx))) failures = failures + 1
            end do

            if (.not. chk4("du_b", elem, 1, ref_b(elem), du_b(elem))) failures = failures + 1
            do component = 1, 3
                idx = 3 * (elem - 1) + component
                if (.not. chki("du_c", elem, component, ref_c(idx), du_c(idx))) failures = failures + 1
            end do
            if (.not. chk8("du_d", elem, 1, ref_d(elem), du_d(elem))) failures = failures + 1
            if (.not. chk4("du_optional", elem, 1, ref_optional(elem), &
                           du_optional(elem))) failures = failures + 1
        end do

        do elem = 1, nother
            do component = 1, runtime_dim
                idx = runtime_dim * (elem - 1) + component
                if (.not. chk8("du_runtime", elem, component, &
                               ref_runtime(idx), du_runtime(idx))) failures = failures + 1
            end do
        end do
    end subroutine validate_results

    logical function chk8(name, elem, component, expected, actual)
        implicit none

        character(*), intent(in) :: name
        integer(4), intent(in) :: elem, component
        real(8), intent(in) :: expected, actual

        chk8 = abs(actual - expected) <= tol8 * max(1.0_8, abs(expected))
        if (.not. chk8 .and. failures < 20) then
            print *, "MISMATCH ", name, " element ", elem, " component ", component, &
                     " expected ", expected, " got ", actual
        end if
    end function chk8

    logical function chk4(name, elem, component, expected, actual)
        implicit none

        character(*), intent(in) :: name
        integer(4), intent(in) :: elem, component
        real(4), intent(in) :: expected, actual

        chk4 = abs(actual - expected) <= tol4 * max(1.0_4, abs(expected))
        if (.not. chk4 .and. failures < 20) then
            print *, "MISMATCH ", name, " element ", elem, " component ", component, &
                     " expected ", expected, " got ", actual
        end if
    end function chk4

    logical function chki(name, elem, component, expected, actual)
        implicit none

        character(*), intent(in) :: name
        integer(4), intent(in) :: elem, component, expected, actual

        chki = actual == expected
        if (.not. chki .and. failures < 20) then
            print *, "MISMATCH ", name, " element ", elem, " component ", component, &
                     " expected ", expected, " got ", actual
        end if
    end function chki

end program smemtest

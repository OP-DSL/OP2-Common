! A loop the c_cuda target warns about and runs through its seq fallback:
! an OP_INC argument assigned rather than incremented; with INFO_DIM, also an
! info argument narrower than its maximum, which the translator rejects, and
! with VECTOR_MAP a vector map argument, which it rejects too; with IMPLIED_SAVE
! an initialised local, which c_cuda leaves to seq; with SLICE slices of a
! dim-4 argument, which only the legacy targets leave to seq; with
! CONFLICTING_DAT one dat passed as two types, which it rejects.
program rejects
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    implicit none

    integer(4) :: edge_nodes(4) = (/ 0, 1, 2, 3 /)
    real(8) :: x(4) = 0.0_8, w(4) = 1.0_8
    type(op_set) :: nodes, edges
    type(op_map) :: p_edge_nodes
    type(op_dat) :: p_x, p_w
#ifdef SLICE
    real(8) :: q(16) = 0.0_8
    type(op_dat) :: p_q
#endif
#ifdef INFO_DIM
    real(8) :: mx(2) = 0.0_8
    integer(4) :: loc(1) = 0
#endif

    call op_init(0)
    call op_decl_set(4, nodes, "nodes")
    call op_decl_set(4, edges, "edges")
    call op_decl_map(edges, nodes, 1, edge_nodes, p_edge_nodes, "edge_nodes")
    call op_decl_dat(nodes, 1, "real(8)", x, p_x, "x")
    call op_decl_dat(edges, 1, "real(8)", w, p_w, "w")
    call op_par_loop_2(assign_k, edges, &
        op_arg_dat(p_x, 1, p_edge_nodes, 1, "real(8)", OP_INC), &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ))
#ifdef INFO_DIM
    call op_par_loop_3(extreme_k, edges, &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ), &
        op_arg_gbl(mx, 2, "real(8)", OP_MAX), &
        op_arg_info(loc, 1, "integer(4)", 2))
#endif
#ifdef IMPLIED_SAVE
    call op_par_loop_1(save_k, edges, &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_RW))
#endif
#ifdef SLICE
    call op_decl_dat(edges, 4, "real(8)", q, p_q, "q")
    call op_par_loop_1(slice_k, edges, &
        op_arg_dat(p_q, -1, OP_ID, 4, "real(8)", OP_RW))
#endif
#ifdef CONFLICTING_DAT
    call op_par_loop_2(assign_k, edges, &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_RW), &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(4)", OP_READ))
#endif
#ifdef VECTOR_MAP
    call op_par_loop_2(assign_k, edges, &
        op_arg_dat(p_x, -2, p_edge_nodes, 1, "real(8)", OP_INC), &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ))
#endif
    call op_exit()

contains

    subroutine assign_k(x, w)
        real(8) :: x, w
        x = w + 1.0_8
    end subroutine assign_k
#ifdef IMPLIED_SAVE

    subroutine save_k(w)
        real(8) :: w
        real(8) :: s = 2.0_8
        w = w + s
    end subroutine save_k
#endif
#ifdef SLICE

    subroutine slice_k(q)
        real(8), dimension(4) :: q
        q(1:2) = q(3:4)
    end subroutine slice_k
#endif
#ifdef INFO_DIM

    subroutine extreme_k(w, mx, loc)
        real(8) :: w, mx(2)
        integer(4) :: loc(1)
        if (w > mx(1)) then
            mx(1) = w
            loc(1) = 1
        end if
    end subroutine extreme_k
#endif
end program rejects

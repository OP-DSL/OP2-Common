! A loop the cuda_f2c target warns about and runs through its seq fallback:
! an OP_INC argument assigned rather than incremented.
program rejects
    use OP2_FORTRAN_DECLARATIONS
    use OP2_FORTRAN_REFERENCE

    implicit none

    integer(4) :: edge_nodes(4) = (/ 0, 1, 2, 3 /)
    real(8) :: x(4) = 0.0_8, w(4) = 1.0_8
    type(op_set) :: nodes, edges
    type(op_map) :: p_edge_nodes
    type(op_dat) :: p_x, p_w

    call op_init(0)
    call op_decl_set(4, nodes, "nodes")
    call op_decl_set(4, edges, "edges")
    call op_decl_map(edges, nodes, 1, edge_nodes, p_edge_nodes, "edge_nodes")
    call op_decl_dat(nodes, 1, "real(8)", x, p_x, "x")
    call op_decl_dat(edges, 1, "real(8)", w, p_w, "w")
    call op_par_loop_2(assign_k, edges, &
        op_arg_dat(p_x, 1, p_edge_nodes, 1, "real(8)", OP_INC), &
        op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ))
    call op_exit()

contains

    subroutine assign_k(x, w)
        real(8) :: x, w
        x = w + 1.0_8
    end subroutine assign_k
end program rejects

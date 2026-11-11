! The translator corpus: each loop of corpus_kernels against its kernel called
! directly, element by element, on the same data.

program corpus_tests
  use op2_fortran_declarations
  use op2_fortran_reference
  use op2_fortran_rt_support

  use corpus_kernels

  implicit none

  integer(4), parameter :: nedges = 96
  integer(4), parameter :: nnodes = 40

  type(op_set) :: edges, nodes
  type(op_map) :: pedge
  type(op_dat) :: p_x4, p_x3, p_w, p_k, p_y4, p_y3, p_y2, p_node3, p_acc0, p_acc1

  integer(4) :: edge_nodes(2 * nedges)
  real(8) :: x4(4 * nedges), x3(3 * nedges), w(nedges), node3(3 * nnodes)
  integer(4) :: k(nedges)
  real(8) :: y4(4 * nedges), y3(3 * nedges), y2(2 * nedges), acc(3 * nnodes)
  real(8), allocatable :: expected(:), fetched(:)
  integer(4) :: e, n0, n1

  do e = 1, nedges
    edge_nodes(2 * e - 1) = mod(e, nnodes)
    edge_nodes(2 * e) = mod(7 * e + 3, nnodes)
    if (edge_nodes(2 * e) == edge_nodes(2 * e - 1)) &
      edge_nodes(2 * e) = mod(edge_nodes(2 * e) + 1, nnodes)
    x4(4 * e - 3:4 * e) = [real(mod(5 * e, 11), 8) / 10.0_8, real(mod(3 * e, 7), 8) / 6.0_8, &
                           real(mod(e, 5), 8) / 4.0_8, real(mod(2 * e, 9), 8) / 8.0_8 - 0.5_8]
    x3(3 * e - 2:3 * e) = x4(4 * e - 3:4 * e - 1)
    w(e) = real(mod(13 * e, 17), 8) / 16.0_8
    k(e) = mod(11 * e, 9)
  end do
  do e = 1, 3 * nnodes
    node3(e) = real(mod(e, 13), 8) / 12.0_8
  end do
  acc = 1.0_8
  y4 = 0.0_8
  y3 = 0.0_8
  y2 = 0.0_8

  call op_init_base(0, 0)

  call op_decl_set(nedges, edges, "edges")
  call op_decl_set(nnodes, nodes, "nodes")
  call op_decl_map(edges, nodes, 2, edge_nodes, pedge, "pedge")
  call op_decl_dat(edges, 4, "real(8)", x4, p_x4, "x4")
  call op_decl_dat(edges, 3, "real(8)", x3, p_x3, "x3")
  call op_decl_dat(edges, 1, "real(8)", w, p_w, "w")
  call op_decl_dat(edges, 1, "integer(4)", k, p_k, "k")
  call op_decl_dat(edges, 4, "real(8)", y4, p_y4, "y4")
  call op_decl_dat(edges, 3, "real(8)", y3, p_y3, "y3")
  call op_decl_dat(edges, 2, "real(8)", y2, p_y2, "y2")
  call op_decl_dat(nodes, 3, "real(8)", node3, p_node3, "node3")
  call op_decl_dat(nodes, 3, "real(8)", acc, p_acc0, "acc0")
  call op_decl_dat(nodes, 3, "real(8)", acc, p_acc1, "acc1")

  call op_par_loop_3(steps, edges, &
    op_arg_dat(p_x4, -1, OP_ID, 4, "real(8)", OP_READ), &
    op_arg_dat(p_k, -1, OP_ID, 1, "integer(4)", OP_READ), &
    op_arg_dat(p_y4, -1, OP_ID, 4, "real(8)", OP_WRITE))
  allocate(expected(4 * nedges), fetched(4 * nedges))
  do e = 1, nedges
    call steps(x4(4 * e - 3), k(e), expected(4 * e - 3))
  end do
  call op_fetch_data(p_y4, fetched)
  call compare("steps", expected, fetched)
  deallocate(expected, fetched)

  call op_par_loop_2(logic, edges, &
    op_arg_dat(p_x3, -1, OP_ID, 3, "real(8)", OP_READ), &
    op_arg_dat(p_y3, -1, OP_ID, 3, "real(8)", OP_WRITE))
  allocate(expected(3 * nedges), fetched(3 * nedges))
  do e = 1, nedges
    call logic(x3(3 * e - 2), expected(3 * e - 2))
  end do
  call op_fetch_data(p_y3, fetched)
  call compare("logic", expected, fetched)

  call op_par_loop_3(kinds, edges, &
    op_arg_dat(p_x3, -1, OP_ID, 3, "real(8)", OP_READ), &
    op_arg_dat(p_k, -1, OP_ID, 1, "integer(4)", OP_READ), &
    op_arg_dat(p_y3, -1, OP_ID, 3, "real(8)", OP_WRITE))
  do e = 1, nedges
    call kinds(x3(3 * e - 2), k(e), expected(3 * e - 2))
  end do
  call op_fetch_data(p_y3, fetched)
  call compare("kinds", expected, fetched)
  deallocate(expected, fetched)

  call op_par_loop_3(increments, edges, &
    op_arg_dat(p_acc0, 1, pedge, 3, "real(8)", OP_INC), &
    op_arg_dat(p_acc1, 2, pedge, 3, "real(8)", OP_INC), &
    op_arg_dat(p_w, -1, OP_ID, 1, "real(8)", OP_READ))
  allocate(expected(6 * nnodes), fetched(3 * nnodes))
  expected = 1.0_8
  do e = 1, nedges
    n0 = edge_nodes(2 * e - 1)
    n1 = edge_nodes(2 * e)
    call increments(expected(3 * n0 + 1), expected(3 * nnodes + 3 * n1 + 1), w(e))
  end do
  call op_fetch_data(p_acc0, fetched)
  call compare("increments", expected(1:3 * nnodes), fetched)
  call op_fetch_data(p_acc1, fetched)
  call compare("increments", expected(3 * nnodes + 1:), fetched)
  deallocate(expected, fetched)

  call op_par_loop_3(control, edges, &
    op_arg_dat(p_node3, 2, pedge, 3, "real(8)", OP_READ), &
    op_arg_dat(p_k, -1, OP_ID, 1, "integer(4)", OP_READ), &
    op_arg_dat(p_y2, -1, OP_ID, 2, "real(8)", OP_WRITE))
  allocate(expected(2 * nedges), fetched(2 * nedges))
  do e = 1, nedges
    call control(node3(3 * edge_nodes(2 * e) + 1), k(e), expected(2 * e - 1))
  end do
  call op_fetch_data(p_y2, fetched)
  call compare("control", expected, fetched)
  deallocate(expected, fetched)

  call op_exit()

contains

  ! Stop unless every value of a loop's result matches the direct calls'.
  subroutine compare(loop, expected, fetched)
    character(len=*), intent(in) :: loop
    real(8), intent(in) :: expected(:), fetched(:)
    integer(4) :: i

    do i = 1, size(expected)
      if (abs(fetched(i) - expected(i)) > 1.0e-12_8 * max(1.0_8, abs(expected(i)))) then
        write(*,*) "ERROR: ", loop, " value", i, ":", fetched(i), "expected", expected(i)
        call op_exit()
        stop 1
      end if
    end do
    write(*,*) loop, " passed"
  end subroutine compare
end program corpus_tests

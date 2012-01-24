module mass
  implicit none
  integer, parameter :: num_ele = 2, num_nodes = 4, num_dim = 2
  real, parameter :: PI = 4 * atan(1.0d0)
contains
  subroutine mass_kernel(A, x, i, j, q)
    real, intent(out) :: A
    real, dimension(3,2), intent(in) :: x
    integer, intent(in) :: i, j, q

    real, dimension(2,2) :: J
    real, dimension(3) :: w
    real :: detJ

    ! Calculate Jacobian
    double J(1,1) = x(2,1) - x(1,1)
    double J(1,2) = x(3,1) - x(1,1)
    double J(2,1) = x(2,2) - x(1,2)
    double J(2,2) = x(3,2) - x(1,2)

    ! Calculate determinant of Jacobian
    detJ = J(1,1)*J(2,2) - J(1,2)*(2,1)

    ! Quadrature weights
    w = (/ 0.166667, 0.166667, 0.166667 /)

    ! Values of basis functions at  quadrature points
    CG1 = (/ (/ 0.666667, 0.166667, 0.166667 /),
             (/ 0.166667, 0.666667, 0.166667 /),
             (/ 0.166667, 0.166667, 0.666667 /) /)

    ! Local assembly
    A += CG1(i,q) * CG1(j,q) * detJ * w(q);

  end subroutine mass_kernel
end module mass

program main
  use mass
  use op2
  implicit none

  integer, dimension(:), allocatable :: p_elem_node
  real, dimension(:), allocatable :: p_xn
  type(op_dat) :: xn
  type(op_mat) :: mat
  type(op_sparsity) :: sparsity
  type(op_set) :: nodes, elements
  type(op_map) :: elem_node
  integer :: i
  real :: val

  ! Set up
  call op_init(1)

  allocate(p_elem_node(3 * num_ele))
  allocate(p_xn(num_nodes,num_dim))

  ! Create element -> node mapping
  p_elem_node(1) = 1
  p_elem_node(2) = 2
  p_elem_node(3) = 4
  p_elem_node(4) = 3
  p_elem_node(5) = 4
  p_elem_node(6) = 2

  ! Create coordinates
  p_xn(1) = (/ 0.0, 0.0 /)
  p_xn(2) = (/ 1.0, 0.0 /)
  p_xn(3) = (/ 1.0, 1.0 /)
  p_xn(4) = (/ 0.0, 1.0 /)

  ! Initialise OP2 data structures
  call op_decl_set(num_nodes, nodes, "nodes")
  call op_decl_set(num_ele, elements, "elements")

  call op_decl_map(elements, nodes, 3, p_elem_node, elem_node, "elem_node")

  call op_decl_dat(nodes, 2, xn, p_xn, "xn")

  call op_decl_sparsity(elem_node, elem_node, sparsity)

  call op_decl_mat(sparsity, mat)

  ! Initialise matrix
  op_zero(mat)

  ! Assemble matrix
  call op_par_loop(mass_kernel, (elements, 3, 3, 3) &
       op_arg(mat, op_i(1), elem_node, op_i(2), elem_node, op_i(3), OP_ID, OP_INC), &
       op_arg(xn, -3, elem_node, OP_READ))

  ! Tidy up
  deallocate(p_elem_node)
  deallocate(p_xn)

  call op_exit()
end program main

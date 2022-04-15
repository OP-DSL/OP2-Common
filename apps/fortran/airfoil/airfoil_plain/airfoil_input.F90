module airfoil_input
    implicit none
    private

    public :: read_input

    integer(4), parameter :: file_id = 1
    character(*), parameter :: file_name = "new_grid.dat"

contains

    subroutine read_input(nnode, ncell, nedge, nbedge, x, cell, edge, ecell, bedge, becell, bound)
        integer(4), intent(out) :: nnode, ncell, nedge, nbedge
        real(8), dimension(:), allocatable, intent(out) :: x
        integer(4), dimension(:), allocatable, intent(out) :: cell, edge, ecell, bedge, becell, bound

        integer(4) :: i

        open(file_id, file=file_name)

        read(file_id, *) nnode, ncell, nedge, nbedge

        allocate(x(2 * nnode))

        allocate(cell(4 * ncell))
        allocate(edge(2 * nedge))
        allocate(ecell(2 * nedge))
        allocate(bedge(2 * nbedge))
        allocate(becell(nbedge))
        allocate(bound(nbedge))

        do i = 1, nnode
            read(file_id, *) x(2 * (i - 1) + 1), x(2 * (i - 1) + 2)
        end do

        do i = 1, ncell
            read(file_id, *) cell(4 * (i - 1) + 1), cell(4 * (i - 1) + 2), cell(4 * (i - 1) + 3), cell(4 * (i - 1) + 4)
        end do

        do i = 1, nedge
            read(file_id, *) edge(2 * (i - 1) + 1), edge(2 * (i - 1) + 2), ecell(2 * (i - 1) + 1), ecell(2 * (i - 1) + 1)
        end do

        do i = 1, nbedge
            read(file_id, *) bedge(2 * (i - 1) + 1), bedge(2 * (i - 1) + 2), becell(i), bound(i)
        end do

        close(file_id)
    end subroutine

end module

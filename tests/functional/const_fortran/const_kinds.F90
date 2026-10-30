! An application kind, as Hydra's come from a kinds module: the translator
! cannot see it, so the build names it with --kind rk=8.
module const_kinds
    implicit none
    private

    public :: rk

    integer, parameter :: rk = 8
end module

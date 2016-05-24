module cudaConfigurationParams

  integer(4) :: OP_Warpsize = 32

contains

integer(4) function getPartitionSize (name, size)

  implicit none

  character(len=*) :: name
  integer(4) :: size

  getPartitionSize = 128

end function

integer(4) function getBlockSize (name, size)

  implicit none

  character(len=*) :: name
  integer(4) :: size

  getBlockSize = 128

end function


end module

!
!
!
!subroutine writeOutputToFile ( res, resSize )
!
!	! formal parameter
!	integer(4) :: resSize
!	real(8), dimension(resSize) :: res
!	
!
!	integer(4) :: OUT_FILE_ID = 20
!	integer(4) :: i
!
!	! open file
!	open ( OUT_FILE_ID, file = 'result.txt' )
!
!	do i = 1, resSize
!		write ( OUT_FILE_ID, * ) res(i)
!	end do
!
!	close ( OUT_FILE_ID )
!
!end subroutine writeOutputToFile
!
!
!
!
!
!subroutine writeRealDataToFile ( dataIn, dataSize, filename )
!
!	! formal parameter
!	integer(4) :: dataSize
!	real(8), dimension(dataSize) :: dataIn
!	character(len=20) :: filename
!
!	integer(4) :: OUT_FILE_ID = 30
!	integer(4) :: i
!
!	! open file
!	open ( OUT_FILE_ID, file = filename )
!
!	do i = 1, dataSize
!		write ( OUT_FILE_ID, * ) dataIn(i)
!	end do
!
!	close ( OUT_FILE_ID )
!
!end subroutine writeRealDataToFile
!
!subroutine writeIntDataToFile ( dataIn, dataSize, filename )
!
!	! formal parameter
!	integer(4) :: dataSize
!	integer(4), dimension(dataSize) :: dataIn
!	character(len=20) :: filename
!
!	integer(4) :: OUT_FILE_ID = 30
!	integer(4) :: i
!
!	! open file
!	open ( OUT_FILE_ID, file = filename )
!
!	do i = 1, dataSize
!		write ( OUT_FILE_ID, * ) dataIn(i)
!	end do
!
!	close ( OUT_FILE_ID )
!
!end subroutine writeIntDataToFile

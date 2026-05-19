module idx_kernels
  implicit none
  private

  public :: write_direct_idx, write_indirect_idx, write_mixed_idx

contains
  subroutine write_direct_idx(dat, idx)
    real(8), intent(out) :: dat
    integer(4), intent(in) :: idx

    dat = idx
  end subroutine write_direct_idx

  subroutine write_indirect_idx(dat, idx0, idx1, idx2)
    real(8), dimension(3), intent(out) :: dat
    integer(4), intent(in) :: idx0
    integer(4), intent(in) :: idx1
    integer(4), intent(in) :: idx2

    dat(1) = idx0
    dat(2) = idx1
    dat(3) = idx2
  end subroutine write_indirect_idx

  subroutine write_mixed_idx(dat, direct_idx, idx0, idx1, idx2)
    real(8), dimension(4), intent(out) :: dat
    integer(4), intent(in) :: direct_idx
    integer(4), intent(in) :: idx0
    integer(4), intent(in) :: idx1
    integer(4), intent(in) :: idx2

    dat(1) = direct_idx
    dat(2) = idx0
    dat(3) = idx1
    dat(4) = idx2
  end subroutine write_mixed_idx
end module idx_kernels

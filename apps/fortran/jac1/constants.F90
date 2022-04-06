module OP2_CONSTANTS

#ifdef OP2_WITH_CUDAFOR
    use cudafor
    real(8), constant :: alpha_OP2
#endif

    real(8) :: alpha

end module

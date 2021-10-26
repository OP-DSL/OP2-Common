  ifdef DEBUG
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O0 -MMD -MP
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O0 -MMD -MP #-g -Wextra
  else
  CCFLAGS  = -std=c99 -fPIC -DUNIX -Wall -g -O3 -MMD -MP
  CXXFLAGS = -fPIC -DUNIX -Wall -g -O3 -MMD -MP #-g -Wextra
  endif
  CXX      = g++
  MPICXX   = $(MPICPP_PATH)
  MPIFLAGS = $(CXXFLAGS)

  FC           = gfortran
  CC           = gcc
  CFLAGS       = -g -O2 -std=c99 -fPIC -Wall -pedantic -pipe $(INC)
  FFLAGS       = -O2 -Jmod -fPIC -Wall -pedantic -pipe -g -DOP2_ARG_POINTERS -ffixed-line-length-none -ffree-line-length-none

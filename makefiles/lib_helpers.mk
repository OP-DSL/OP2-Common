# Generate helper variables OP2_LIB_SEQ, OP2_LIB_MPI_CUDA, ...
define OP2_LIB_template =
OP2_LIB_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_$(1) $(2)
OP2_LIB_FOR_$(call UPPERCASE,$(1)) := $(OP2_LIB) -lop2_for_$(1) $(3)
endef

OP2_LIB_EXTRA :=
OP2_LIB_EXTRA_MPI :=

OP2_LIB_EXTRA_MPI := $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(KAHIP_LIB)
OP2_LIB_FOR_EXTRA_MPI := $(PARMETIS_LIB) $(PTSCOTCH_LIB) $(KAHIP_LIB)

ifeq ($(OP2_LIBS_WITH_HDF5),true)
  OP2_LIB_EXTRA += -lop2_hdf5 $(HDF5_SEQ_LIB)
  OP2_LIB_EXTRA_MPI += $(HDF5_PAR_LIB)

  OP2_LIB_FOR_EXTRA += -lop2_for_hdf5 $(HDF5_SEQ_LIB)
  OP2_LIB_FOR_EXTRA_MPI += $(HDF5_PAR_LIB)
endif

$(foreach lib,$(OP2_LIBS_SINGLE_NODE),$(eval $(call OP2_LIB_template,$(lib),\
  $(OP2_LIB_EXTRA),$(OP2_LIB_FOR_EXTRA))))

$(foreach lib,$(OP2_LIBS_MPI),$(eval $(call OP2_LIB_template,$(lib),\
  $(OP2_LIB_EXTRA_MPI),$(OP2_LIB_FOR_EXTRA_MPI))))

OP2_LIB_CUDA += $(CUDA_LIB)
OP2_LIB_MPI_CUDA += $(CUDA_LIB)

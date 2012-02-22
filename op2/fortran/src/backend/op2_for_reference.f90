! This module defines the Fortran interface towards C reference implementation
! of op_par_loop functions

module OP2_Fortran_Reference

  use OP2_Fortran_Declarations

  interface

        ! debug C functions (to obtain similar output file that can be diff-ed
    subroutine op_par_loop_2_f ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1  &
                             & ) BIND(C,name='op_par_loop_2')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(op_set_core) :: set
      type(op_dat_core) :: data0, data1
      integer(kind=c_int), value :: itemSel0, itemSel1, access0, access1
      type(op_map_core) :: map0, map1

    end subroutine op_par_loop_2_f


    subroutine op_par_loop_5_F ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1, &
                               & data2, itemSel2, map2, access2, &
                               & data3, itemSel3, map3, access3, &
                               & data4, itemSel4, map4, access4 &
                             & ) BIND(C,name='op_par_loop_5')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(op_set_core) :: set
      type(op_dat_core) :: data0, data1, data2, data3, data4
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4
      integer(kind=c_int), value :: access0, access1, access2, access3, access4
      type(op_map_core) :: map0, map1, map2, map3, map4

    end subroutine op_par_loop_5_F


    subroutine op_par_loop_6_F ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1, &
                               & data2, itemSel2, map2, access2, &
                               & data3, itemSel3, map3, access3, &
                               & data4, itemSel4, map4, access4, &
                               & data5, itemSel5, map5, access5  &
                             & ) BIND(C,name='op_par_loop_6')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(op_set_core) :: set
      type(op_dat_core) :: data0, data1, data2, data3, data4, data5
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5
      integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5
      type(op_map_core) :: map0, map1, map2, map3, map4, map5

    end subroutine op_par_loop_6_F


    subroutine op_par_loop_8_F ( subroutineName, set, &
                               & data0, itemSel0, map0, access0, &
                               & data1, itemSel1, map1, access1, &
                               & data2, itemSel2, map2, access2, &
                               & data3, itemSel3, map3, access3, &
                               & data4, itemSel4, map4, access4, &
                               & data5, itemSel5, map5, access5, &
                               & data6, itemSel6, map6, access6, &
                               & data7, itemSel7, map7, access7  &
                             & ) BIND(C,name='op_par_loop_8')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(op_set_core) :: set
      type(op_dat_core) :: data0, data1, data2, data3, data4, data5, data6, data7
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5, itemSel6, itemSel7
      integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5, access6, access7
      type(op_map_core) :: map0, map1, map2, map3, map4, map5, map6, map7

    end subroutine op_par_loop_8_F

    subroutine op_par_loop_12_F ( subroutineName, set, &
                                & data0, itemSel0, map0, access0, &
                                & data1, itemSel1, map1, access1, &
                                & data2, itemSel2, map2, access2, &
                                & data3, itemSel3, map3, access3, &
                                & data4, itemSel4, map4, access4, &
                                & data5, itemSel5, map5, access5, &
                                & data6, itemSel6, map6, access6, &
                                & data7, itemSel7, map7, access7,  &
                                & data8, itemSel8, map8, access8,  &
                                & data9, itemSel9, map9, access9,  &
                                & data10, itemSel10, map10, access10,  &
                                & data11, itemSel11, map11, access11  &
                              & ) BIND(C,name='op_par_loop_12')

      use, intrinsic :: ISO_C_BINDING

      import :: op_set_core, op_map_core, op_dat_core

      interface
        subroutine subroutineName () BIND(C)
        end subroutine subroutineName
      end interface

      type(op_set_core) :: set
      type(op_dat_core) :: data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11
      integer(kind=c_int), value :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, &
                                    itemSel5, itemSel6, itemSel7, itemSel8, itemSel9, itemSel10, itemSel11
      integer(kind=c_int), value :: access0, access1, access2, access3, access4, access5,&
                                    access6, access7, access8, access9, access10, access11
      type(op_map_core) :: map0, map1, map2, map3, map4, map5, map6, map7, map8, map9, map10, map11

    end subroutine op_par_loop_12_F

  end interface

  contains

  subroutine op_par_loop_2 ( subroutineName, set, &
                             & data0, itemSel0, map0, access0, &
                             & data1, itemSel1, map1, access1 &
                           & )

    external subroutineName

    type(op_set) :: set
    type(op_dat) :: data0, data1
    integer(kind=c_int) :: itemSel0, itemSel1
    integer(kind=c_int) :: access0, access1
    type(op_map) :: map0, map1

    integer(kind=c_int) :: itemSelC0, itemSelC1

    ! selector are used in C++ to address correct map field, hence must be converted from 1->N style to 0->N-1 one
    itemSelC0 = itemSel0 - 1
    itemSelC1 = itemSel1 - 1

    ! warning: look at the -1 on itemSels: it is used to access C++ arrays!
    call op_par_loop_2_f ( subroutineName, set%setPtr, &
                         & data0%dataPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%dataPtr, itemSelC1, map1%mapPtr, access1 &
                       & )

  end subroutine op_par_loop_2


  subroutine op_par_loop_5 ( subroutineName, set, &
                           & data0, itemSel0, map0, access0, &
                           & data1, itemSel1, map1, access1, &
                           & data2, itemSel2, map2, access2, &
                           & data3, itemSel3, map3, access3, &
                           & data4, itemSel4, map4, access4  &
                         & )

    external subroutineName

    type(op_set) :: set
    type(op_dat) :: data0, data1, data2, data3, data4
    integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4
    integer(kind=c_int) :: access0, access1, access2, access3, access4
    type(op_map) :: map0, map1, map2, map3, map4

    integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4

    ! see above
    itemSelC0 = itemSel0 - 1
    itemSelC1 = itemSel1 - 1
    itemSelC2 = itemSel2 - 1
    itemSelC3 = itemSel3 - 1
    itemSelC4 = itemSel4 - 1

    ! warning: look at the -1 on itemSels: it is used to access C++ arrays!
    call op_par_loop_5_f ( subroutineName, set%setPtr, &
                         & data0%dataPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%dataPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%dataPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%dataPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%dataPtr, itemSelC4, map4%mapPtr, access4  &
                       & )

  end subroutine op_par_loop_5

  subroutine op_par_loop_6 ( subroutineName, set, &
                           & data0, itemSel0, map0, access0, &
                           & data1, itemSel1, map1, access1, &
                           & data2, itemSel2, map2, access2, &
                           & data3, itemSel3, map3, access3, &
                           & data4, itemSel4, map4, access4, &
                           & data5, itemSel5, map5, access5  &
                         & )

    external subroutineName

    type(op_set) :: set
    type(op_dat) :: data0, data1, data2, data3, data4, data5
    integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5
    integer(kind=c_int) :: access0, access1, access2, access3, access4, access5
    type(op_map) :: map0, map1, map2, map3, map4, map5

    integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4, itemSelC5

    itemSelC0 = itemSel0 - 1
    itemSelC1 = itemSel1 - 1
    itemSelC2 = itemSel2 - 1
    itemSelC3 = itemSel3 - 1
    itemSelC4 = itemSel4 - 1
    itemSelC5 = itemSel5 - 1

    ! warning: look at the -1 on itemSels: it is used to access C++ arrays!
    call op_par_loop_6_f ( subroutineName, set%setPtr, &
                         & data0%dataPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%dataPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%dataPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%dataPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%dataPtr, itemSelC4, map4%mapPtr, access4, &
                         & data5%dataPtr, itemSelC5, map5%mapPtr, access5  &
                       & )

  end subroutine op_par_loop_6

  subroutine op_par_loop_8 ( subroutineName, set, &
                           & data0, itemSel0, map0, access0, &
                           & data1, itemSel1, map1, access1, &
                           & data2, itemSel2, map2, access2, &
                           & data3, itemSel3, map3, access3, &
                           & data4, itemSel4, map4, access4, &
                           & data5, itemSel5, map5, access5, &
                           & data6, itemSel6, map6, access6, &
                           & data7, itemSel7, map7, access7  &
                         & )

    external subroutineName
!     type(c_funptr) :: subroutineName
    type(op_set) :: set
    type(op_dat) :: data0, data1, data2, data3, data4, data5, data6, data7
    integer(kind=c_int) :: itemSel0, itemSel1, itemSel2, itemSel3, itemSel4, itemSel5, itemSel6, itemSel7
    integer(kind=c_int) :: access0, access1, access2, access3, access4, access5, access6, access7
    type(op_map) :: map0, map1, map2, map3, map4, map5, map6, map7

    integer(kind=c_int) :: itemSelC0, itemSelC1, itemSelC2, itemSelC3, itemSelC4, itemSelC5, itemSelC6, itemSelC7

    itemSelC0 = itemSel0 - 1
    itemSelC1 = itemSel1 - 1
    itemSelC2 = itemSel2 - 1
    itemSelC3 = itemSel3 - 1
    itemSelC4 = itemSel4 - 1
    itemSelC5 = itemSel5 - 1
    itemSelC6 = itemSel6 - 1
    itemSelC7 = itemSel7 - 1

    ! warning: look at the -1 on itemSels: it is used to access C++ arrays!
    call op_par_loop_8_f ( subroutineName, set%setPtr, &
                         & data0%dataPtr, itemSelC0, map0%mapPtr, access0, &
                         & data1%dataPtr, itemSelC1, map1%mapPtr, access1, &
                         & data2%dataPtr, itemSelC2, map2%mapPtr, access2, &
                         & data3%dataPtr, itemSelC3, map3%mapPtr, access3, &
                         & data4%dataPtr, itemSelC4, map4%mapPtr, access4, &
                         & data5%dataPtr, itemSelC5, map5%mapPtr, access5, &
                         & data6%dataPtr, itemSelC6, map6%mapPtr, access6, &
                         & data7%dataPtr, itemSelC7, map7%mapPtr, access7  &
                       & )

  end subroutine op_par_loop_8

end module OP2_Fortran_Reference


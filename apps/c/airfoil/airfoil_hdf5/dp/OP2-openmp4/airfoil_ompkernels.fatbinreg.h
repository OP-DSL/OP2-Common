// Prototypes                                                    
                                                                    
  static void __xlcuf_register_(void **cubin_handle);             
  static void __sti____cudaRegisterAll_()                         
              __attribute__((__constructor__));                     
  //void __cudaRegisterLinkedBinary_(void (*)(void **),void*,void*,void (*)(void *));
                                                                    
  // Function definitions                                           
                                                                    
  static void __xlcuf_register_(void **cubin_handle)              
  {                                                                 
  }                                                                 
                                                                    
  static void __sti____cudaRegisterAll_()                         
  {                                                                 
    //__cudaRegisterLinkedBinary_9_pq_f102_f_m(                     
    __cudaRegisterLinkedBinary(                                   
      (void (*)(void **)) (__xlcuf_register_),                    
      (void *) &__FATIDNAME(__NV_MODULE_ID),                        
      (void *) &__module_id_str,                                    
      (void (*)(void *)) &____nv_dummy_param_ref);                  
  } 
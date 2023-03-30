#pragma once


#ifdef DEBUG_VERBOSE 

#define DWRITE(...) {\
    fprintf(stderr, __VA_ARGS__);\
    fflush(stderr);\
}

#else

#define DWRITE(...) {\
}/* do nothing */

#endif 


#define PDWRITE(...) {\
    if(_rank==0){\
        DWRITE(__VA_ARGS__);\
    }\
}
#ifdef FLOAT_TYPE
typedef float ValueType;
#define VALUESTR "float"
#define TOLERANCE 1e-5
#else
typedef double ValueType;
#define VALUESTR "double"
#define TOLERANCE 1e-12
#endif

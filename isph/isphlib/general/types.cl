// types based on floating point precision
#if FP == 64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double scalar;
typedef double2 scalar2;
typedef double4 scalar4;
#else // default 32-bit precision
typedef float scalar;
typedef float2 scalar2;
typedef float4 scalar4;
#endif

// types based on dimensions of simulation
#if DIM == 3
typedef scalar4 vector;
#define vector_to_scalar4(X) X
#define scalar4_to_vector(X) X

#else // default 2D

typedef scalar2 vector;
#define vector_to_scalar4(X) (scalar4)(X.x,X.y,0,0)
#define scalar4_to_vector(X) X.xy
#endif

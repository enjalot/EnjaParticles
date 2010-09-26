// HOW TO INCLUDE WHEN COMPILING? ?

#ifndef _CL_MACROS_H_
#define _CL_MACROS_H_


//---------------------------------------------------------------------- 
// Offsets into var_sorted array

//enum {DENS=0, POS, VEL, ACC, FOR};
#define POS 0
#define VEL 1
#define FOR 2
#define ACC 3
#define DEN 4


// copied from SPHSimLib code
#ifdef USE_TEX
//#define FETCH(a, t, i) tex1Dfetch(t##_tex, i)
#define FETCH(t, i) tex1Dfetch(t##_tex, i)
#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_MATRIX3(a,t,i) tex1DfetchMatrix3(t##_tex,i)
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
#else
//#define FETCH(a, t, i) a.t[i]
#define FETCH(t, i) t[i]
#define FETCH_VAR(t, i, ivar) t[i+ivar*numParticles]
#define FETCH_VEL(t, i) t[i+VEL*numParticles]
#define FETCH_DENS(t, i) t[i+DENS*numParticles]
#define FETCH_FOR(t, i) t[i+FOR*numParticles]
#define FETCH_ACC(t, i) t[i+ACC*numParticles]
#define FETCH_POS(t, i) t[i+POS*numParticles]

//#define FETCH_NOTEX(a, t, i) a.t[i]
#define FETCH_NOTEX(t, i) t[i]
//#define FETCH_FLOAT3(a,t,i) make_float3(FETCH(a,t,i))
#define FETCH_FLOAT3(t,i) make_float3(FETCH(t,i))
#define FETCH_MATRIX3(a,t,i) a.t[i]
#define FETCH_MATRIX3_NOTEX(a,t,i) a.t[i]
//#define FETCH(a, t, i) (a + __mul24(i,sizeof(a)) + (void*)offsetof(a, t))
#endif




#endif

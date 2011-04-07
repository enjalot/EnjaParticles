#ifndef _DEBUG_MACROS_H_
#define _DEBUG_MACROS_H_


#define VERBOSE


#ifdef VERBOSE
#define DBG_PRINT(x) {double x_x = cublasSdot(ntot, x, 1, x, 1); \
                  char* ss = #x; \
                  printf(".. DEBUG .. <%s,%s>= %g\n", ss, ss,  x_x); }
#define DBG_PRINT2(x, msg) {double x_x = cublasSdot(ntot, x, 1, x, 1); \
                  char* ss = #x; \
                  printf(".. DEBUG .. %s, <%s,%s>= %g\n", msg, ss, ss,  x_x); }
#define DBG_PRINTdot(x,y) {double x_x = cublasSdot(ntot, x, 1, y, 1); \
                  char* ss = #x; \
                  char* tt = #y; \
                  printf(".. DEBUG .. <%s,%s>= %g\n", ss, tt,  x_x); }
#else
#define DBG_PRINT(x) 
#define DBG_PRINT2(x, msg) 
#define DBG_PRINTdot(x,y)
#endif


//#ifndef _DEBUG_MACROS_H_
#endif


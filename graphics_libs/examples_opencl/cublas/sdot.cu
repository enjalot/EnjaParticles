//#include "cublasP.h"
#define CUBLAS_SDOT_LOG_THREAD_COUNT    (7)
#define CUBLAS_SDOT_THREAD_COUNT        (1 << CUBLAS_SDOT_LOG_THREAD_COUNT)
#define CUBLAS_SDOT_CTAS                (80)

__global__ void sdot_gld_main (int n, const float* sx, int incx,
float* sy, int incy, float* result)
{
extern __shared__ float partialSum[];

#undef fetchx
#undef fetchy
#define fetchx(i)  sx[i]
#define fetchy(i)  sy[i]

    unsigned int i, tid, totalThreads, ctaStart;
    float sum = 0.0f;
    tid = threadIdx.x;
    totalThreads = gridDim.x * CUBLAS_SDOT_THREAD_COUNT;
    ctaStart = CUBLAS_SDOT_THREAD_COUNT * blockIdx.x;

    if ((incx == incy) && (incx > 0)) {
        /* equal, positive, increments */
        if (incx == 1) {
            /* both increments equal to 1 */
            for (i = ctaStart + tid; i < n; i += totalThreads) {
                sum += fetchy(i) * fetchx(i);
            }
        } else {
            /* equal, positive, non-unit increments. */
            for (i = ctaStart + tid; i < n; i += totalThreads) {
                sum += fetchy(i*incx) * fetchx(i*incx);
            }
        }
    } else {
        /* unequal or nonpositive increments */
        int ix = ((incx < 0) ? ((1 - n) * incx) : 0);
        int iy = ((incy < 0) ? ((1 - n) * incy) : 0);
        for (i = ctaStart + tid; i < n; i += totalThreads) {
            sum += fetchy(iy+i*incy) * fetchx(ix+i*incx);
        }
    }

    partialSum[tid] = sum;

    for (i = (CUBLAS_SDOT_THREAD_COUNT >> 1); i > 0; i >>= 1) {
        __syncthreads(); 
        if (tid < i) {
            partialSum[tid] += partialSum[tid + i];
		} 
    }

    __syncthreads(); 
    if (tid == 0) {
        result[blockIdx.x] = partialSum[tid];
    }
}


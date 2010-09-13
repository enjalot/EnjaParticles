# 1 "a.cpp"
# 1 "<built-in>"
# 1 "<command line>"
# 1 "a.cpp"
# 17 "a.cpp"
texture<float, 1, cudaReadModeElementType> KP_t;
texture<float, 1, cudaReadModeElementType> KM_t;
texture<float, 1, cudaReadModeElementType> LP_t;
texture<float, 1, cudaReadModeElementType> LM_t;
texture<float, 1, cudaReadModeElementType> MP_t;
texture<float, 1, cudaReadModeElementType> MM_t;
# 58 "a.cpp"
__constant__ int nx;
__constant__ int ny;
__constant__ int nz;
__constant__ int nb_zblocks;
__constant__ int zblocksz;
__constant__ int bx;
__constant__ int cx;






# 1 "inverse_tex_mul.cu" 1
# 9 "inverse_tex_mul.cu"
__device__ void inverse_diag_3d_z_mul_kernel(float* y, int i_block );
__device__ void inverse_diag_3d_all_z_mul_kernel(float* y );



__device__ void inverse_diag_3d_all_z_mul_kernel(float* y )
{



 for (int i_block=0; i_block < nb_zblocks; i_block++) {

  inverse_diag_3d_z_mul_kernel(y, i_block );

  __syncthreads();
 }
}

__device__ void inverse_diag_3d_z_mul_kernel(float* y, int zblock )
{
# 44 "inverse_tex_mul.cu"
 int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
 int yi = (blockIdx.y * blockDim.y) + threadIdx.y;

 int zi = (zblock * zblocksz) + threadIdx.z;


 __syncthreads();

  float kp = tex1Dfetch(KP_t, (xi));
  float km = tex1Dfetch(KM_t, (xi));
  float lp = tex1Dfetch(LP_t, (yi));
  float lm = tex1Dfetch(LM_t, (yi));
  float mp = tex1Dfetch(MP_t, (zi));
  float mm = tex1Dfetch(MM_t, (zi));

 __syncthreads();
# 69 "inverse_tex_mul.cu"
 float a;
 a = (km + kp);
 a = a * (lp + lm);
 a = a * (mp + mm);






 y[xi+nx*(yi+ny*zi)] = 1./a;
# 101 "inverse_tex_mul.cu"
}

__global__ void inverse_diag_3d_mul_kernel(float* y )
{
 inverse_diag_3d_all_z_mul_kernel(y );
}
# 72 "a.cpp" 2
# 1 "inverse_tex_add.cu" 1
# 9 "inverse_tex_add.cu"
__device__ void inverse_diag_3d_z_kernel(float* y, int i_block );
__device__ void inverse_diag_3d_all_z_kernel(float* y );



__device__ void inverse_diag_3d_all_z_kernel(float* y )
{



 for (int i_block=0; i_block < nb_zblocks; i_block++) {

  inverse_diag_3d_z_kernel(y, i_block );

  __syncthreads();
 }
}

__device__ void inverse_diag_3d_z_kernel(float* y, int zblock )
{
# 44 "inverse_tex_add.cu"
 int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
 int yi = (blockIdx.y * blockDim.y) + threadIdx.y;

 int zi = (zblock * zblocksz) + threadIdx.z;


 __syncthreads();

  float kp = tex1Dfetch(KP_t, (xi));
  float km = tex1Dfetch(KM_t, (xi));
  float lp = tex1Dfetch(LP_t, (yi));
  float lm = tex1Dfetch(LM_t, (yi));
  float mp = tex1Dfetch(MP_t, (zi));
  float mm = tex1Dfetch(MM_t, (zi));

 __syncthreads();
# 69 "inverse_tex_add.cu"
 float a;
 a = km + kp;
 a = a + lp + lm;
 a = a + mp + mm;






 y[xi+nx*(yi+ny*zi)] = 1./a;
# 101 "inverse_tex_add.cu"
}

__global__ void inverse_diag_3d_kernel(float* y )
{
 inverse_diag_3d_all_z_kernel(y );
}
# 73 "a.cpp" 2
# 1 "polar_3d_kernel_opencl.cu" 1
# 15 "polar_3d_kernel_opencl.cu"
__device__ void read_row(int row, __global float* arr, __local float* sh );
__device__ void read_row_polar_3d(int row, __global float* arr, __local float* sh );
__device__ void read_plane_polar_3d(int row, __global float* arr, __local float* sha, int zblock );
__device__ void copy_two_z_planes(__local float* sh );


__device__ void matrix_vec_one_z_block(float* x, float* y, int zblock, __local float* sh );
__device__ void matrix_vec_all_z_blocks(float* x, float* y, __local float* sh );
# 33 "polar_3d_kernel_opencl.cu"
__global__ void matrix_vec_polar_3d(float* x, float* y )




{
 extern __shared__ float sh[];
# 48 "polar_3d_kernel_opencl.cu"
 matrix_vec_all_z_blocks(x, y, sh );
}


__device__ void matrix_vec_all_z_blocks(float* x, float* y, __local float* sh )
{
 for (int i_block=0; i_block < nb_zblocks; i_block++) {
  matrix_vec_one_z_block(x, y, i_block, sh );
  __syncthreads();
 }
}

__device__ void matrix_vec_one_z_block(float* x, float* y, int zblock, __local float* sh )
{
# 70 "polar_3d_kernel_opencl.cu"
 volatile int xi = (blockIdx.x * blockDim.x) + threadIdx.x;
 volatile int yi = (blockIdx.y * blockDim.y) + threadIdx.y;
 volatile int zi = (zblock * blockDim.z) + threadIdx.z;

 __syncthreads();





        int nbt = threadIdx.x*threadIdx.y*threadIdx.z;
        __syncthreads();
# 91 "polar_3d_kernel_opencl.cu"
 if (zblock == 0) {



  read_plane_polar_3d(threadIdx.z, x, sh, zblock );
 } else {
  copy_two_z_planes(sh );
 }
 __syncthreads();

 if (threadIdx.z < 2) {


  read_plane_polar_3d(threadIdx.z+blockDim.z, x, sh, zblock );
 }

 __syncthreads();
# 119 "polar_3d_kernel_opencl.cu"
 int ii = (threadIdx.x+1)+bx*(threadIdx.y+1)+cx*(threadIdx.z+1);






 __syncthreads();


 volatile float a;

 float kp = tex1Dfetch(KP_t, (xi));
 float km = tex1Dfetch(KM_t, (xi));
 float lp = tex1Dfetch(LP_t, (yi));
 float lm = tex1Dfetch(LM_t, (yi));
 float mp = tex1Dfetch(MP_t, (zi));
 float mm = tex1Dfetch(MM_t, (zi));





 a = kp*(sh[ii+1]-sh[ii]) - km*(sh[ii]-sh[ii-1]);
 a = a + lp*(sh[ii+bx]-sh[ii]) - lm*(sh[ii]-sh[ii-bx]);
 a = a + mp*(sh[ii+cx]-sh[ii]) - mm*(sh[ii]-sh[ii-cx]);
# 159 "polar_3d_kernel_opencl.cu"
 __syncthreads();



 int glob = xi+nx*(yi+ny*zi);

 y[glob] = a;
# 184 "polar_3d_kernel_opencl.cu"
}

__device__ void read_plane_polar_3d(int plane, float* sol, __local float* sha, int zblock )
{
# 198 "polar_3d_kernel_opencl.cu"
 volatile int glob_plane_offset = plane-1 + zblock*blockDim.z;





 int loc_plane_offset = cx * plane;

 __local float* sha_loc = sha+loc_plane_offset;





 if (glob_plane_offset < 0) {
  sha_loc[1+threadIdx.x+bx*(1+threadIdx.y)] = 0.;
  return;
 }
 if (glob_plane_offset == nz) {
  sha_loc[1+threadIdx.x+bx*(1+threadIdx.y)] = 0.;
  return;
 }


 glob_plane_offset = glob_plane_offset*nx*ny;
 __global float* sol_glob = sol+glob_plane_offset;

 __syncthreads();


 read_row_polar_3d(threadIdx.y, sol_glob, sha_loc );
 __syncthreads();



 if (threadIdx.y < 2) {


  read_row_polar_3d(threadIdx.y+blockDim.y, sol_glob, sha_loc );
 }
 __syncthreads();
}

__device__ void read_row_polar_3d(int row, __global float* arr, __local float* sha )
{






 volatile int loc_row;


 volatile int glob_row = row-1 + blockDim.y * blockIdx.y;

 loc_row = bx * row;
 __local float* sha_loc = sha + loc_row;

 volatile int x = threadIdx.x + blockIdx.x * blockDim.x;
# 267 "polar_3d_kernel_opencl.cu"
 if ((glob_row < 0)) {
  sha_loc[1+threadIdx.x] = 0.;
  return;
 }

 if ((glob_row == ny)) {
  sha_loc[1+threadIdx.x] = 0.;
  return;
 }






 loc_row = bx * row;



 __global float* sol_row = arr + nx*glob_row;




 sha_loc[1+threadIdx.x] = sol_row[x];
# 300 "polar_3d_kernel_opencl.cu"
 if (threadIdx.x == 15) {
  if (blockIdx.x != 0) {

   sha_loc[0] = sol_row[blockIdx.x*blockDim.x - 1];
  } else {
   sha_loc[0] = 0.0;
  }
 }
# 318 "polar_3d_kernel_opencl.cu"
 if (threadIdx.x == 0) {



  if (blockIdx.x != (gridDim.x-1) ) {
   sha_loc[1+blockDim.x] = sol_row[blockIdx.x*blockDim.x + blockDim.x];
  } else {

   sha_loc[1+blockDim.x] = 0.0;
  }
 }

}



__device__ void copy_two_z_planes(__local float* sh )
{


 volatile int base = threadIdx.x + bx * threadIdx.y + cx * threadIdx.z;
 sh[base] = sh[base + cx*2];




 if (threadIdx.x < 2) {
  base = blockDim.x + threadIdx.x + bx * threadIdx.y + cx * threadIdx.z;
  sh[base] = sh[base + cx*2];
 }



 __syncthreads();

 base = threadIdx.x + bx * (2+threadIdx.y) + cx * threadIdx.z;
 sh[base] = sh[base + cx*2];




 if (threadIdx.x < 2) {
  base = blockDim.x + threadIdx.x + bx * (2+threadIdx.y) + cx * threadIdx.z;
  sh[base] = sh[base + cx*2];
 }
}
# 74 "a.cpp" 2

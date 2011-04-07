#ifndef _CLOCKS_H_
#define _CLOCKS_H_

// I no  longer use timege
//GE::Time clock_ssdot;
//GE::Time clock_ssdot_cpu;
//GE::Time clock_ssdot_gpu;
GE::Time clock_sdot("sdot", -1, 5);
GE::Time clock_sdot_cpu("sdot_cpu", -1, 5);
GE::Time clock_sdot_gpu("sdot_gpu_blocked", -1, 5);

GE::Time clock_mat_mul("matmul", -1, 5);
GE::Time clock_scal("scal", -1, 5);
GE::Time clock_scal_params("scal_params", -1, 5);
GE::Time clock_mat_vec("mat_vec", -1, 5);
GE::Time clock_saxpy("saxpy", -1, 5);
GE::Time clock_scopy("scopy", -1, 5);
GE::Time clock_svecvec("svecvec", -1, 5);

GE::Time clock_scale_cpu("scal_cpu", -1, 5);
GE::Time clock_scopy_cpu("scopy_cpu", -1, 5);
GE::Time clock_saxpy_cpu("saxpy_cpu", -1, 5);
GE::Time clock_mat_mul_cpu("matmul_cpu", -1, 5);

GE::Time clock_inv_mat_vec_cpu("inverseDiag3DKernel_cpu", -1, 5);
GE::Time clock_inv_mat_vec_gpu("inverseDiag3DKernel_gpu", -1, 5);
GE::Time clock_inv_mat_vec("inverseDiag3DKernel", -1, 5);

GE::Time clock_scopy_gpu("scopy_gpu_blocked", -1, 5);
GE::Time clock_scale_gpu("scale_gpu_blocked", -1, 5);
GE::Time clock_saxpy_gpu("saxpy_gpu_blocked", -1, 5);
GE::Time clock_mat_mul_gpu("matmul_gpu_blocked", -1, 5);

GE::Time clock_mat_vec_cpu("mat_vec_blocked", -1, 5);
GE::Time clock_mat_vec_gpu("mat_vec_blocked", -1, 5);


extern GE::Time clock_dbg;

#endif

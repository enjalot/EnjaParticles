FILE(REMOVE_RECURSE
  "./sph_cuda_generated_CudaMax.cu.o"
  "./sph_cuda_generated_CudaUtils.cu.o"
  "./sph_cuda_generated_SimBase.cu.o"
  "./sph_cuda_generated_SimDEM.cu.o"
  "./sph_cuda_generated_SimSimpleSPH.cu.o"
  "./sph_cuda_generated_SimSnowSPH.cu.o"
  "./sph_cuda_generated_UniformGrid.cu.o"
  "./sph_cuda_generated_cuPrintf.cu.o"
  "./sph_cuda_generated_srts_radix_sort.cu.o"
  "./sph_cuda_generated_srts_verifier.cu.o"
  "./sph_cuda_generated_timer.cu.o"
  "./sph_cuda_generated_checkCudaError.cu.o"
  "libsph_cuda.pdb"
  "libsph_cuda.a"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/sph_cuda.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)

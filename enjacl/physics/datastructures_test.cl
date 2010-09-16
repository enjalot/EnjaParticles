# 1 "datastructures_test.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "datastructures_test.cpp"
# 17 "datastructures_test.cpp"
__kernel void datastructures(
     __constant int numParticles,
     __constant int nb_vars,

     __global float4* dParticles,
     __global float4* dParticlesSorted,

        __global uint* sort_hashes,
        __global uint* sort_indexes,
        __global uint* cell_indexes_start,
        __global uint* cell_indexes_end,
     __local uint* sharedHash
     )
{
 uint index = get_global_id(0);



 if (index >= numParticles) return;





 uint hash = sort_hashes[index];





 uint tid = get_local_id(0);

 sharedHash[tid+1] = hash;

 if (index > 0 && tid == 0) {

  sharedHash[0] = sort_hashes[index-1];
 }


 barrier(CLK_LOCAL_MEM_FENCE);
# 67 "datastructures_test.cpp"
 if ((index == 0 || hash != sharedHash[tid]) )
 {
  cell_indexes_start[hash] = index;
  if (index > 0) {
   cell_indexes_end[sharedHash[tid]] = index;
  }
 }

 if (index == numParticles - 1)
 {
  cell_indexes_end[hash] = index + 1;
 }

 uint sortedIndex = sort_indexes[index];






 for (int j=0; j < nb_vars; j++) {
  dParticlesSorted[index+j*numParticles] = dParticles[sortedIndex+j*numParticles];




 }
}

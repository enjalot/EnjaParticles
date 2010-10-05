# 1 "datastructures_test.cpp"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "datastructures_test.cpp"






# 1 "cl_macros.h" 1
# 8 "datastructures_test.cpp" 2


__kernel void datastructures(
     int num,
     int nb_vars,
     __global float4* vars_unsorted,
     __global float4* vars_sorted,
        __global uint* sort_hashes,
        __global uint* sort_indices,
        __global uint* cell_indices_start,
        __global uint* cell_indices_end,
     __local uint* sharedHash
     )
{
 uint index = get_global_id(0);


 if (index >= num) return;

 uint hash = sort_hashes[index];





 uint tid = get_local_id(0);


 sharedHash[tid+1] = hash;

 if (index > 0 && tid == 0) {

  sharedHash[0] = sort_hashes[index-1];
 }


 barrier(CLK_LOCAL_MEM_FENCE);
# 53 "datastructures_test.cpp"
 if ((index == 0 || hash != sharedHash[tid]) )
 {
  cell_indices_start[hash] = index;
  if (index > 0) {
   cell_indices_end[sharedHash[tid]] = index;
  }
 }


 if (index == num - 1) {
  cell_indices_end[hash] = index + 1;
 }

 uint sorted_index = sort_indices[index];
# 81 "datastructures_test.cpp"
 vars_sorted[index+1*num] = vars_unsorted[sorted_index+1*num];
 vars_sorted[index+2*num] = vars_unsorted[sorted_index+2*num];
 vars_sorted[index+0*num].x = vars_unsorted[sorted_index+0*num].x;

}

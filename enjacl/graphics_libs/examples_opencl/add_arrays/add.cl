

__kernel void add(__global float* a, __global float* b, __global float* res, const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count) {
       res[i] = a[i] + b[i];
	}
}


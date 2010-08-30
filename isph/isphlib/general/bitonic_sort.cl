__kernel void bitonicSortStep
(
	__global uint* keysIn , 
	__global uint* valuesIn ,
	uint phase,
	uint sstep, 
	uint inv, 
	uint ssize, 
    uint stride,
	uint numElements, 
	uint maxVal
)
{
	uint globalId = get_global_id(0);
	
	uint comparerId = globalId & ( (ssize >> 1) - 1);
	uint sequenceId = globalId  >> (sstep - 1); 
	uint I1 = (ssize)*(sequenceId) + comparerId;
	uint I2 = I1 + stride;
	uint dir = (globalId & (0x01 << (phase-1)))>>(phase-1)^inv;
    
	if (I1 >= numElements) return;
	if (I2 >= numElements) return;
	
	uint key1 = keysIn[I1];  
	uint key2 = keysIn[I2];

	if ((key1 > key2) != dir )
	{	
		keysIn[I1] = key2;
		keysIn[I2] = key1;
		
		uint tmpVal = valuesIn[I1];
		valuesIn[I1] = valuesIn[I2];
		valuesIn[I2] = tmpVal;
	}
}

#include <stdlib.h>
#include "ArrayFloat.h"

enum CUDA_CREATE {LIN_MEM_1D, LIN_MEM_2D, ARRAY_2D};

//----------------------------------------------------------------------
//----------------------------------------------------------------------
void testFunc()
{
	ArrayFloat* arr = new ArrayFloat(2,3,4);
	arr->setTo(5.);
	printf("arr(1,1,1) = %f\n", (*arr)(1,1,1));

	ArrayFloat h(2,3,4);
	h.setTo(5.);
	printf("h(1,1,1) = %f\n", h(1,1,1));
}
//----------------------------------------------------------------------
void testCudaCreate(CUDA_CREATE array_type)
{
	switch (array_type) {
		case LIN_MEM_1D:
			break;
		case LIN_MEM_2D:
			break;
		case ARRAY_2D:
			ArrayFloat arr(2,3,4);
			arr(3,4,5) = 10.;
			arr.createCudaArray();
			arr.copyCudaArrayToDevice();
			arr.copyCudaArrayToHost();
			break;
	}
}
//----------------------------------------------------------------------
int main()
{
	CUDA_CREATE array_type = ARRAY_2D;

	testFunc();
	//testCudaCreate(array_type);
	exit(0);
}
//----------------------------------------------------------------------


#include "gpubitonicsort.h"
#include "isph.h"


using namespace std;
using namespace isph;


GpuBitonicSort::GpuBitonicSort(cl_context GPUContext,
							   cl_command_queue CommandQue,
  							   unsigned int numElements) :
					           cxGPUContext(GPUContext),
					           cqCommandQueue(CommandQue)

{
   mNumElements = numElements;
   // Need to find closest power of 2
   unsigned int tmp1 = iLog2(mNumElements);
   unsigned long tmp2 = (0x01 << tmp1);
   if ( mNumElements == tmp2 )  
   {
        mNumElementsP2 =  mNumElements;
        mP2Exponent = tmp1;  
   }
   else
   {
        mNumElementsP2 = (tmp2 << 1);
        mP2Exponent = tmp1 + 1;
   }
   mNumThreads = numElements >> 1;
   // Find number of phases it depends on power of 2 element count
   mNumPhases = iLog2(mNumElementsP2);
   mMaxVal = 0x01 << 31;
   
   // Load and compile OpenCL Kernel
   cl_int ciErrNum;
   std::string source = Utils::LoadCLSource("general/bitonic_sort.cl");
   const char* cBitonicSort = source.c_str();
   size_t szKernelLength = source.length();
   cpProgram = clCreateProgramWithSource(cxGPUContext, 1, &cBitonicSort , &szKernelLength, &ciErrNum);
   ciErrNum =  clBuildProgram(cpProgram, 0, NULL, "-cl-mad-enable", NULL, NULL);

   if (ciErrNum != CL_SUCCESS)
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(ciErrNum));

   ckBitonicSortStep = clCreateKernel(cpProgram, "bitonicSortStep", &ciErrNum);
   	if (ciErrNum != CL_SUCCESS)
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(ciErrNum));
}

GpuBitonicSort::~GpuBitonicSort()
{
	clReleaseKernel(ckBitonicSortStep);
	clReleaseProgram(cpProgram);
}

//------------------------------------------------------------------------
// Sorts input arrays of unsigned integer keys and (optional) values
// 
// @param pData       pData of keys, value pairs 
//------------------------------------------------------------------------
void GpuBitonicSort::sort(cl_mem d_keys,
			              cl_mem d_values) 
{

    // Cycle trough each phase/step
	for (unsigned int p=1; p <= mNumPhases; p++)
		for (unsigned int s=p; s >= 1; s--)
		{
           // At each step we need to calculate the number of comparator/threads
           // We calculate the number of sequences encompassing all real elements
	
	       //cout  <<  "Phase: " << phase << " Step: " << step << endl;
   	       // Calculate sequence size = 2^step
		   unsigned int ssize = 0x01 << s;
           //cout  <<  "Bitonic Sequence size: " << ssize << endl;

		   // Calculate comparison stride = size/2
		   unsigned int stride = ssize >> 1;
	       //cout  <<  "Comparison stride: " << stride << endl;

          // Number of padding sequences (only padding elements) = (mNumElementsP2 - mNumElements)/2^step 
		  unsigned int nPaddingSeq  = ((mNumElementsP2 - mNumElements) >> s);
		  // Number of sequence where computations are performed
          unsigned int nComputSeq   =   (0x01 << (mP2Exponent - s) ) - nPaddingSeq;
          // Threads 
		  mNumThreads  =   (nComputSeq  * ssize) >> 1;
          // We need to invert dir if the last comparator has a dir = 1
          unsigned inv = 0x0; 
	      if ( ( ( (mNumThreads - 1) & ( 0x01 << (p - 1) ) ) >> (p - 1) ) == 1 )
	      {
		   inv = 0x1;
	      }

		  bitonicStep(d_keys,d_values, p, s, inv, ssize, stride);
     	  //cpProgram->Variable("CELLS_HASH")->ReadTo(ids);
     	  //cpProgram->Variable("HASHES_PARTICLE")->ReadTo(hashes);

		}
}

//----------------------------------------------------------------------------
// Perform one step of the bitonic sort.
//----------------------------------------------------------------------------
void GpuBitonicSort::bitonicStep(cl_mem d_keys,
			                     cl_mem d_values,
								 unsigned int phase,
								 unsigned int step, 
								 unsigned int inv, 
								 unsigned int ssize, 
								 unsigned int stride)
{
	size_t globalWorkSize[1] = {mNumThreads};
	size_t localWorkSize[1] = {1};

	cl_int ciErrNum;
	ciErrNum  = clSetKernelArg(ckBitonicSortStep, 0, sizeof(cl_mem),  (void*)&d_keys);
	ciErrNum |= clSetKernelArg(ckBitonicSortStep, 1, sizeof(cl_mem),  (void*)&d_values);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 2, sizeof(cl_uint), (void*)&phase);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 3, sizeof(cl_uint), (void*)&step);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 4, sizeof(cl_uint), (void*)&inv);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 5, sizeof(cl_uint), (void*)&ssize);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 6, sizeof(cl_uint), (void*)&stride);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 7, sizeof(cl_uint), (void*)&mNumElements);
    ciErrNum |= clSetKernelArg(ckBitonicSortStep, 8, sizeof(cl_uint), (void*)&mMaxVal);
    ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckBitonicSortStep, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (ciErrNum != CL_SUCCESS)
		Log::Send(Log::Error, CLSystem::Instance()->ErrorDesc(ciErrNum));
}

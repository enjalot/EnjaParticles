
#include "cpubitonicsort.h"
#include "log.h"


using namespace isph;

CpuBitonicSort::CpuBitonicSort(unsigned int numElements) 
{
   mNumElements = numElements;
   // Need to find closest power of 2
   unsigned int tmp1 = iLog2(mNumElements);
   unsigned int tmp2 = (0x01 << tmp1);
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
}

CpuBitonicSort::~CpuBitonicSort()
{
}

//------------------------------------------------------------------------
// Sorts input arrays of unsigned integer keys and (optional) values
// 
// @param pData       pData of keys, value pairs 
//------------------------------------------------------------------------
void CpuBitonicSort::sort(Vec<2,unsigned int>* pData) 
{
	for (unsigned int p=1; p <= mNumPhases ; p++)
		for (unsigned int s=p; s >= 1; s--)
		{
           // At each step we need to calculate the number of comparator/threads
           // We calculate the number of sequences encompassing all real elements
           //  
	       //cout  <<  "Phase: " << p << " Step: " << s << endl;
   	       // Calculate sequence size = 2^step
		   unsigned int ssize = 0x01 << s;
           //cout  <<  "Bitonic Sequence size: " << ssize << endl;
	       // Calculate comparison stride = size/2
		   unsigned int stride = ssize >> 1;
	       //cout  <<  "Comparison stride: " << stride << endl;

		  unsigned int nPaddingSeq  = ((mNumElementsP2 - mNumElements) >> s);
          unsigned int nComputSeq   =   (0x01 << (mP2Exponent - s) ) - nPaddingSeq;
          unsigned int mNumThreadsO  =   (nComputSeq  * ssize) >> 1;
		  //cout<< "mNumThreads(1):  " << mNumThreads << endl ;
          unsigned int nRealSeq     =   mNumElements >> s;
          mNumThreads  = (nRealSeq * ssize) >> 1;
		  //cout<< "mNumThreads(2):  " << mNumThreads << endl;
          unsigned int nElementInMixed = mNumElements & (ssize -1); 
		  if (nElementInMixed>stride) mNumThreads  += nElementInMixed-stride; 
		  //cout<< "mNumThreads(3):  " << mNumThreads << endl;
          
		  // We need to invert dir if the last comparator has a dir = 1
          unsigned inv = 0x0; 
	      if ( ( ( (mNumThreadsO - 1) & ( 0x01 << (p - 1) ) ) >> (p - 1) ) == 1 )
	      {
		   inv = 0x1;
	      }

		  bitonicStep(pData, p, s, inv, ssize, stride);
		}
}

//----------------------------------------------------------------------------
// Perform one step of the bitonic sort.
//----------------------------------------------------------------------------
void CpuBitonicSort::bitonicStep(Vec<2,unsigned int> *pData, 
								 unsigned int phase,
								 unsigned int step, 
								 unsigned int inv, 
								 unsigned int ssize, 
								 unsigned int stride)
{

	for (unsigned int globalId = 0; globalId < mNumThreads; globalId++)
	{
      Vec<2,unsigned int> element1;
	  Vec<2,unsigned int> element2; 
      unsigned int comparerId = globalId & ( (ssize >> 1) - 1);
	  unsigned int sequenceId = globalId  >> (step - 1); 
	  unsigned int I1 = (ssize)*(sequenceId) + comparerId;
      unsigned int I2 = I1 + stride;

	  if (I2 > (mNumElements - 1)) return;
      
	  unsigned int dir = (globalId & (0x01 << (phase-1)))>>(phase-1)^inv;

	  //cout  <<  "Thread: " << globalId << " Sequence Id: " << sequenceId  << " Comparer Id: " << comparerId  << " Dir: " << dir;
	  //cout	<< " I1: " << I1 << " I2: " << I2 << endl;
	  //if (I1> (mNumElements - 1)) 
	  //{
	  //  Vec<2,unsigned int> element(0,mMaxVal);
	  //	element1 = element;
	  //}
	  //else
	  //{
	  //  element1 = pData[I1];
	  //}
	  element1 = pData[I1];
    
	  //if (I2> (mNumElements - 1)) 
	  //{
	  //  Vec<2,unsigned int> element(0,mMaxVal);
	  //	element2 = element;
	  //}
	  //else
	  //{
   	  //  element2 = pData[I2];
	  //}
      element2 = pData[I2];

	  if ((element1.y > element2.y)!= dir )
	  {
		  unsigned int tmp1 = element1.x;
		  element1.x = element2.x;
          element2.x = tmp1;
		  unsigned int tmp2 = element1.y;
		  element1.y = element2.y;
          element2.y = tmp2;
		  pData[I1] = element1;
          pData[I2] = element2;
	  }
	}
}

#include "../GE_SPH.h"

namespace rtps {

//----------------------------------------------------------------------
unsigned int GE_SPH::zhash_cpu(int i, int j, int k)
{
	int* ix = cl_xindex->getHostPtr();
	int* iy = cl_yindex->getHostPtr();
	int* iz = cl_zindex->getHostPtr();

	printf("ixyz= %d, %d, %d\n", ix[i], iy[j], iz[k]);
	return (ix[i] | iy[j] | iz[k]);
}
//----------------------------------------------------------------------
// based on 1024^3 grid
void GE_SPH::zindices()
{
	cl_xindex = new BufferGE<int>(ps->cli, 1024);
	cl_yindex = new BufferGE<int>(ps->cli, 1024);
	cl_zindex = new BufferGE<int>(ps->cli, 1024);

	int* ix = cl_xindex->getHostPtr();
	int* iy = cl_yindex->getHostPtr();
	int* iz = cl_zindex->getHostPtr();

	// 1024 take 10 bits
	// 0 --> 0
	// 1 --> 100
    // 10 --> 100 000
	// 11 --> 100 100

	int* mask = new int[10];

	for (int i=0; i < 10; i++) {
		mask[i] = 1 << i;
		printf("mask[%d] = %d\n", i, mask[i]);
	}

	unsigned int bit[10];
	int d = 0xd;
	printf("d= %d (base 10)\n", d);
	printf("d= %x (base 16)\n", d);
	printf("d= %o (base octal)\n", d);

	// x           y            z
	// 0 -> 2    0 -> 1       0 -> 0
	// 1 -> 5    1 -> 4       1 -> 3
	// 2 -> 8    2 -> 7       2 -> 6
	// 3 -> 11

	for (int val=0; val < 1024; val++) {
		unsigned int x,y,z;
		bitshifts(mask, val, x, y, z);
		ix[val] = x;
		iy[val] = y;
		iz[val] = z;
		//bitshifts(mask, val, ix[val], iy[val], iz[val]);
		printf("bits[%d]: %d, %d, %d\n", val, x, y, z);
		//printf("bits[%d]: %d, %d, %d\n", val, ix[val], iy[val], iz[val]);
	}

	/*  // WORKS
	for (int i=0; i < 10; i++) {
		printf("----\n");
		unsigned int bit = mask[i] & d;
		unsigned bx = bit << (2*(i+1)); // for x
		unsigned by = bit << (1+2*i); // for y
		unsigned bz = bit << (2*i); // for z
		printf("bit=mask[%d] & d: %d\n", i, mask[i] & d);
		printf("bit<<(2*i) = %d\n", bx);
		printf("bit<<(2*i) = %d\n", by);
		printf("bit<<(2*i) = %d\n", bz);
	}
	*/

	delete [] mask;
}
//----------------------------------------------------------------------
void GE_SPH::bitshifts(int* mask, int d, 
   unsigned int& bx, unsigned int& by, unsigned int& bz)
{
	unsigned int x;
	unsigned int y;
	unsigned int z;

	bx = by = bz = 0;

	for (int i=0; i < 10; i++) {
		unsigned int bit = mask[i] & d;
		x = bit << (2*(i+1)); // for x
		y = bit << (1+2*i); // for y
		z = bit << (2*i); // for z
		bx |= x;
		by |= y;
		bz |= z;
	}
}
//----------------------------------------------------------------------
}

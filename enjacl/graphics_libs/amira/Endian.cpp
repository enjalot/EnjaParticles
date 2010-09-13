#include <stdio.h>
#include <stdlib.h>
#include <sys/utsname.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <string.h>
#include <iostream>
#include "Endian.h"

// These routines needed because AMIRA always assumes output is in
// BIGENDIAN (network) byte order.
// if endian=L_ENDIAN, then
//   the write routine will reverse byte order
//   the read routine will reverse byte order back.
// To disable the byte ordering garbage, just set endian=B_ENDIAN
// by calling "disableEndian()"
Endian::Endian()
{
         // L_ENDIAN is proper choice when running on SPOCK or KIRK
         // and .... visualizing on SPOCK or KIRK
	if (1==1) {
		endian = L_ENDIAN;
	} else {
		endian = B_ENDIAN;
	}

	// Only use this class on systems where floats and ints are 4 bytes

	//printf("sizeof(float) = %d\n", sizeof(float));
	//printf("sizeof(double) = %d\n", sizeof(double));
	//printf("sizeof(int) = %d\n", sizeof(int));
	//printf("sizeof(long) = %d\n", sizeof(long));

	if (sizeof(float) != 4 || sizeof(MSLONG) != 4) {
		printf("int or float don't have a size of 4 bytes\n");
		exit(1);
	}
}
//----------------------------------------------------------------------
Endian::~Endian()
{}
//----------------------------------------------------------------------
void Endian::convertToLittleEndian(float* f, int n)
// assume size of float is 4 bytes
// convert to Little Endian from Big Endian
// bigEndian is network format, also Unix format
{
	if (isBigEndian()) return;

	char* c = (char*) f;
	char d1;

	for (int i=0; i < 4*n; i+=4) {
		d1     = c[i];
		c[i]   = c[i+3];
		c[i+3] = d1;
		d1     = c[i+1];
		c[i+1] = c[i+2];
		c[i+2] = d1;
	}
}
//----------------------------------------------------------------------
void Endian::convertToLittleEndian(int* f, int n)
// assume size of float is 4 bytes
// convert to Little Endian from Big Endian
// bigEndian is network format, also Unix format
// "ntohl" converts a u_long from TCP/IP network order to host byte order
{
	if (isBigEndian()) return;

	for (int i=0; i < n; i++) {
		f[i] = ntohl(f[i]);
	}
}
//----------------------------------------------------------------------
void Endian::convertFromLittleEndian(float* f, int n)
// assume size of float is 4 bytes
// convert to Little Endian from Big Endian
// bigEndian is network format, also Unix format
{
	if (isBigEndian()) return;

	char* c = (char*) f;
	char d1;

	for (int i=0; i < 4*n; i+=4) {
		d1     = c[i];
		c[i]   = c[i+3];
		c[i+3] = d1;
		d1     = c[i+1];
		c[i+1] = c[i+2];
		c[i+2] = d1;
	}
}
//----------------------------------------------------------------------
void Endian::convertFromLittleEndian(int* f, int n)
// assume size of float is 4 bytes
// convert to Little Endian from Big Endian
// bigEndian is network format, also Unix format
// "htonl" converts a u_long from host order to TCP/IP network byte order 
{
	if (isBigEndian()) return;

	for (int i=0; i < n; i++) {
		f[i] = htonl(f[i]);
	}
}
//----------------------------------------------------------------------
size_t Endian::fread (int* ptr, size_t size, size_t nitems, FILE *stream)
{
	MSLONG n = ::fread (ptr, size, nitems, stream);
	convertToLittleEndian(ptr, nitems);
	return n;
}
//----------------------------------------------------------------------
size_t Endian::fread (float *ptr, size_t size, size_t nitems, FILE *stream)
{
	MSLONG n = ::fread (ptr, size, nitems, stream);
	convertToLittleEndian(ptr, nitems);
	return n;
}
//----------------------------------------------------------------------
size_t Endian::fwrite (int* ptr, size_t size, size_t nitems, FILE *stream)
{
	int* tmp = new int [nitems];
	memcpy(tmp, ptr, 4*nitems); 
	convertFromLittleEndian(tmp, nitems);
	MSLONG n = ::fwrite (tmp, size, nitems, stream);
	delete [] tmp;
	return n;
}
//----------------------------------------------------------------------
size_t Endian::fwrite (float *ptr, size_t size, size_t nitems, FILE *stream)
{
	float* tmp = new float [nitems];
	memcpy(tmp, ptr, 4*nitems); 
	convertFromLittleEndian(tmp, nitems);
	MSLONG n = ::fwrite (tmp, size, nitems, stream);
	delete [] tmp;
	return n;
}
//----------------------------------------------------------------------
#if STANDALONE
main()
{
	Endian Myendian;

	printf("is little endian: %d\n", Myendian.isLittleEndian());
	printf("is big endian: %d\n", Myendian.isBigEndian());
}
//----------------------------------------------------------------------
#endif

#ifndef _MYENDIAN_H_
#define _MYENDIAN_H_

#ifndef MSLONG

#ifdef GORDON_FOURBYTEINT
#define MSLONG int
#else
#define MSLONG long
#endif

#endif

class Endian {
public:
	Endian();
	~Endian();

	enum {L_ENDIAN=0, B_ENDIAN};

	// figure out a way to use templates so that routine
	// automatically become no-ops when system is bigEndian
	// bigEndian is network byte order; bigEndian is necessary
        // byte order for AMIRA files?

         // called by fread
	void convertToLittleEndian(float* f, int n);
	void convertToLittleEndian(int* f, int n);

         // called by fwrite
	void convertFromLittleEndian(float* f, int n);
	void convertFromLittleEndian(int* f, int n);

	inline int isLittleEndian() {return (endian==L_ENDIAN);}
	inline int isBigEndian() {return (endian==B_ENDIAN);}
        inline int disableConvert() {endian=B_ENDIAN;}
        inline int enableConvert() {endian=L_ENDIAN;}

	size_t fread (int *ptr, size_t size, size_t nitems, FILE *stream); 
	size_t fread (float *ptr, size_t size, size_t nitems, FILE *stream); 
	size_t fwrite (int *ptr, size_t size, size_t nitems, FILE *stream); 
	size_t fwrite (float *ptr, size_t size, size_t nitems, FILE *stream); 

private:
	int endian;
	char sysname[32];
};

#endif

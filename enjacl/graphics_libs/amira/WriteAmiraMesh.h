//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteAmiraMesh.h,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteAmiraMesh.h,v $
// $Revision: 2.2 $
// $State: Exp $
 
//================================================================================

#ifndef _WRITEAMIRAMESH_H_
#define _WRITEAMIRAMESH_H_

#ifdef GORDON_FOURBYTEINT
#define MSLONG int
#else
#define MSLONG long
#endif




#include <string.h>
#include <stdio.h>
#include "Endian.h"

class Vec3;

class WriteAmiraMesh {
private:
	int nx, ny, nz;
	int ntot;
	int nbComponents;
	float bbx, bby, bbz;  // bounding box dimensions
	float** data;
	char* fileName;
	FILE* fd;
	Endian endian;

public:
	WriteAmiraMesh(int nx_, int ny_, int nz_, int nbComponents_);
	WriteAmiraMesh(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName_);
	WriteAmiraMesh();
	~WriteAmiraMesh();
	void init(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName_);
	//void setFileName(char *name) {fileName = strdup(name);}
	void writeData(float* vx, float* vy, float* vz);
	void writeData(float* vx, char* name);
	void writeCurvilinearData(float* grid, float* scalar, char* name);
	void writeDataASCII(float* scalar, char* name);
	void writeDataASCII(float* vector, int nbComponents, char* name);
	int getPts() {return ntot;}
	void WriteHeader();
	void WriteHeaderScalar(const char* name, const int nb, const char* type=0);
	void WriteScalar(int* data, int nb);
	void WriteScalar(float* data, int nb);
	void WriteScalar(double* data, int nb);

    void writeRectilinearData(float* vx, char* name,const char* fileName_,
       float* x,float* y,float* z);
    void readRectilinearData(float** vx, char* name,const char* fileName_,
       float** x,float** y,float** z);
    void writeRectilinearData(float* vx, float* vy, float* vz, 
       char* name,const char* fileName_,
       float* x,float* y,float* z);
    void readRectilinearData(float** vx, float** vy, float** vz, 
       char* name,const char* fileName_,
       float** x,float** y,float** z);
	void CloseFile();
};

#endif

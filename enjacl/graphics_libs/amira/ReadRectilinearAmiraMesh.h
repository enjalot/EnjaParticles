//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/ReadAmiraMesh.h,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: ReadAmiraMesh.h,v $
// $Revision: 2.2 $
// $State: Exp $
 
//================================================================================

#ifndef _READRECTILINEARAMIRAMESH_H_
#define _READRECTILINEARAMIRAMESH_H_

#ifdef GORDON_FOURBYTEINT
#define MSLONG int
#else
#define MSLONG long
#endif

//Needed spirit tools from the Boost library
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/actor/assign_actor.hpp>
#include <boost/spirit/actor/insert_key_actor.hpp>



#include <vector>
#include <string.h>
#include <stdio.h>
#include "Endian.h"

class Vec3;

class ReadRectilinearAmiraMesh {
private:
	int nx, ny, nz;
	int ntot;
	const char* fileName;
	FILE* fd;
	Endian endian;
	//float *x, *y, *z;
	//float *vx, *vy, *vz;
	boost::spirit::rule<> ui;
	boost::spirit::rule<> ui_lattice;
	boost::spirit::rule<> ui_coord;
	boost::spirit::rule<> at_coord;
	boost::spirit::rule<> at_lattice;
	int int_lattice, int_coord;

// What happens when I delete this object? If external arrays point to internal 
// arrays there could be problems. Therefore, it is not a good idea to retrieve
// pointers via argument, unless we do NOT destroy arrays generated when the array 
// is destroyed. 


public:
	ReadRectilinearAmiraMesh(const char* fileName_=0);
	ReadRectilinearAmiraMesh();
	~ReadRectilinearAmiraMesh();

	// get grid dimensions and allocate/initialize grid arrays
	// read the grid data into x,y,z
	void readHeader();

	// Do not use 
	#if 0
	int* getGrid(float** x, float** y, float** z) {
		*x = this->x;
		*y = this->y;
		*z = this->z;
	}
	#endif

	void getDims(int* nx, int* ny, int* nz) {
		*nx = this->nx;
		*ny = this->ny;
		*nz = this->nz;
	}

	// read grid
	// memory is allocated in calling program
	void readGrid(float* x , float* y, float* z);

	// Allocate scalar array in calling program
    void read(float* scalar);

	// Allocate vector array components in calling program because this can be done 
	// in one of multiple ways
    void read(float* vx, float* vy, float* vz);
    void read(float* vx, float* vy, float* vz, float* vel);
	void read(std::vector<Vec3>& vel);
	void read(std::vector<Vec3>& vel, float* vel1);

	void closeFile();

private:
	void init(const char* fileName_);
};

#endif

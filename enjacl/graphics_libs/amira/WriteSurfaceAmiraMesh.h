//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteSurfaceAmiraMesh.h,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteSurfaceAmiraMesh.h,v $
// $Revision: 2.2 $
// $State: Exp $
 
//================================================================================

#ifndef _WRITESURFACEAMIRAMESH_H_
#define _WRITESURFACEAMIRAMESH_H_

#ifdef GORDON_FOURBYTEINT
#define MSLONG int
#else
#define MSLONG long
#endif




#include <string.h>
#include <stdio.h>
#include "Endian.h"
#include <Vec3.h>
#include <Vec3i.h>
#include <vector>

class Vec3;
class Vec3i;

class WriteSurfaceAmiraMesh {
private:
	int nbVertices;
	int nbTriangles;;
	int nbTets;
	int nbComponents;

	std::vector<Vec3> vertices; // size of vector should be nbNodes;
	std::vector<Vec3i> triangles; // size of vector should be nbTriangles

	const char* fileName;
	FILE* fd;
	Endian endian;

public:
	/// nbComponents=0: only store geometry
	WriteSurfaceAmiraMesh(std::vector<Vec3>& nodes, std::vector<Vec3i> triangles,  int nbComponents_, const char* fileName);
	//WriteSurfaceAmiraMesh(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName_);
	WriteSurfaceAmiraMesh();
	~WriteSurfaceAmiraMesh();
	//void setFileName(char *name) {fileName = strdup(name);}
	void writeData();

	void writeHeader();
	void writeHeaderField(const char* name, const int nb, const char* type=0);

	void closeFile();

private:
	void init(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName_);
};

#endif

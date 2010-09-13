//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteSurfaceAmiraMesh.cpp,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteSurfaceAmiraMesh.cpp,v $
// $Revision: 2.2 $
// $State: Exp $

//================================================================================

//Needed spirit tools from the Boost library
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/actor/assign_actor.hpp>
#include <boost/spirit/actor/insert_key_actor.hpp>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "WriteSurfaceAmiraMesh.h"
#include "Endian.h"

#include "Vec3.h"

using namespace std;

char* skip_lines_until(const char* strg);





WriteSurfaceAmiraMesh::WriteSurfaceAmiraMesh(vector<Vec3>& vertices, vector<Vec3i> triangles,  int nbComponents_, const char* fileName)
{
	this->vertices = vertices;
	this->triangles = triangles;
	this->nbVertices = vertices.size();
	this->nbTriangles = triangles.size();
	this->fileName = fileName;

	if (fileName) {
    	fd = fopen(fileName, "w");
    	if (!fd) {
        	printf("file %s cannot be opened for writing\n", fileName);
        	exit(1);
		}
    }
}
//----------------------------------------------------------------------
WriteSurfaceAmiraMesh::WriteSurfaceAmiraMesh()
{
}
//----------------------------------------------------------------------
WriteSurfaceAmiraMesh::~WriteSurfaceAmiraMesh()
{
		;
}
//----------------------------------------------------------------------
void WriteSurfaceAmiraMesh::init(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName)
{
}
//----------------------------------------------------------------------
void WriteSurfaceAmiraMesh::writeData()
{
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName);
        exit(1);
    }

    fprintf(fd, "\n@1\n");

    for (int i=0; i < nbVertices; i++) {
		endian.fwrite(&vertices[i][0], sizeof(float), 3, fd);
	}

    fprintf(fd, "\n@2\n");

    for (int i=0; i < nbTriangles; i++) {
		Vec3i& v = triangles[i];
		endian.fwrite(v.getVec(), sizeof(int), 3, fd);
		//endian.fwrite(&triangles[i][0], sizeof(int), 3, fd);
	}
}
//----------------------------------------------------------------------
void WriteSurfaceAmiraMesh::writeHeaderField(const char* name, const int nb, const char* type)
{
	if (type) {
    	fprintf(fd, "Field { %s %s } = @%1d\n", type, name, nb);
	}
	else {
    	fprintf(fd, "Field { float %s } = @%1d\n", name, nb);
	}
}
//----------------------------------------------------------------------
//void WriteSurfaceAmiraMesh::writeHeaderScalar(char* name, int nb)
//{
    //fprintf(fd, "Lattice { float %s } = @%1d", name, nb);
//}
//----------------------------------------------------------------------
void WriteSurfaceAmiraMesh::writeHeader()
{
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Nodes %d", nbVertices);
    fprintf(fd, "define Triangles  %d", nbTriangles);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " }\n");
	fprintf(fd, "Nodes { float[3] Coordinates } = @1\n");
	fprintf(fd, "Triangles { int[3] Nodes } = @2\n");
}
//----------------------------------------------------------------------
void WriteSurfaceAmiraMesh::closeFile()
{
	fclose(fd);
}
//----------------------------------------------------------------------
char* skip_lines_until(const char* strg)
{
}
//----------------------------------------------------------------------

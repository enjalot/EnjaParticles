//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteAmiraMesh.cpp,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteAmiraMesh.cpp,v $
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
#include "WriteAmiraMesh.h"
#include "Endian.h"

#include "Vec3.h"

using namespace std;
using namespace boost::spirit;


char* skip_lines_until(const char* strg);





WriteAmiraMesh::WriteAmiraMesh(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName_)
{
    init(nx_, ny_, nz_, nbComponents_, fileName_);
    //printf("after init\n");
}
//----------------------------------------------------------------------
WriteAmiraMesh::WriteAmiraMesh(int nx_, int ny_, int nz_, int nbComponents_)
{
    init(nx_, ny_, nz_, nbComponents_, 0);
}
//----------------------------------------------------------------------
WriteAmiraMesh::WriteAmiraMesh()
{
}
//----------------------------------------------------------------------
WriteAmiraMesh::~WriteAmiraMesh()
{
		;
		//delete fileName;
}
//----------------------------------------------------------------------
void WriteAmiraMesh::init(int nx_, int ny_, int nz_, int nbComponents_, const char* fileName)
{
    nx = nx_;
    ny = ny_;
    nz = nz_;
    ntot = nx*ny*nz;
    nbComponents = nbComponents_;

    bbx = 1.0; // we fix the bounding box dimension to 1 in x direction
    bby = float(ny)/float(nx);
    bbz = float(nz)/float(ny);

	if (fileName) {
    	fd = fopen(fileName, "w");
    	if (!fd) {
        	printf("file %s cannot be opened for writing\n", fileName);
        	exit(1);
		}
    }
}
//----------------------------------------------------------------------
void WriteAmiraMesh::writeData(float* vx, float* vy, float* vz)
{
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName);
        exit(1);
    }
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"uniform\",\n");
    fprintf(fd, " BoundingBox 0 1 0 1 0 0.25 \n");
    fprintf(fd, " }\n");
    fprintf(fd, "Lattice { float[3] VectorField } = @1");
    fprintf(fd, "\n@1\n");

    {for (int i=0; i < ntot; i++) {
        //printf("i= %d\n", i);
        //printf("i=%d, vx,vy,vz= %f, %f, %f\n", i, vx[i], vy[i], vz[i]);
        endian.fwrite(vx+i, sizeof(float), 1, fd);
        endian.fwrite(vy+i, sizeof(float), 1, fd);
        endian.fwrite(vz+i, sizeof(float), 1, fd);
    }}
}
//----------------------------------------------------------------------
void WriteAmiraMesh::writeCurvilinearData(float* grid, float* scalar, char* name)
// grid:    size nbPoints*3
// scalar:  size nbPoints*3
{
// Grid is a 3D grid

    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"curvilinear\",\n");
    //fprintf(fd, " BoundingBox 0 1 0 1 0 1 \n");
    fprintf(fd, " }\n");
	fprintf(fd, "Lattice { float ScalarField }= @1\n");
	fprintf(fd, "Lattice { float[3] Coordinates }= @2\n");

    //fprintf(fd, "Lattice { float %s } = @1", name);

    fprintf(fd, "\n@1\n");
    endian.fwrite(scalar, sizeof(float), ntot, fd);

    fprintf(fd, "\n@2\n");
    endian.fwrite(grid, sizeof(float), 3*ntot, fd);
}
//---------------------------------------------------------------
void WriteAmiraMesh::writeData(float* scalar, char* name)
{
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"uniform\",\n");
    fprintf(fd, " BoundingBox 0 1 0 1 0 1 \n");
    fprintf(fd, " }\n");
    fprintf(fd, "Lattice { float %s } = @1", name);
    fprintf(fd, "\n@1\n");

    endian.fwrite(scalar, sizeof(float), ntot, fd);
    //fwrite(scalar, sizeof(float), ntot, fd);
}
//---------------------------------------------------------------
void WriteAmiraMesh::writeDataASCII(float* vector, int nbComponents, char* name)
{
    fprintf(fd, "# AmiraMesh ASCII\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"uniform\",\n");
    fprintf(fd, " BoundingBox 0 %f 0 %f 0 %f \n", bbx,bby,bbz);
    fprintf(fd, " }\n");
    if (nbComponents==1) 
      fprintf(fd, "Lattice { float %s } = @1", name);
    else     
      fprintf(fd, "Lattice { float[%d] %s } = @1", nbComponents, name);
    fprintf(fd, "\n@1\n");

    int i,c,ind;
    
    for (i=0; i < ntot; i++) {
      ind = nbComponents*i;
      float *v=&vector[ind];
      for (c=0; c<nbComponents; c++) {
	fprintf(fd,"%f ",v[c]);
	
      }
      fprintf(fd,"\n");
    }

// 		float* v = &vector[3*i];
// 		fprintf(fd, "%f %f %f\n", v[0], v[1], v[2]);
// 		//if (i<25) printf("%f %f %f\n", v[0], v[1], v[2]);
// 	}
}
//---------------------------------------------------------------
void WriteAmiraMesh::writeDataASCII(float* scalar, char* name)
{
    fprintf(fd, "# AmiraMesh ASCII\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"uniform\",\n");
    fprintf(fd, " BoundingBox 0 1 0 1 0 1 \n");
    fprintf(fd, " }\n");
    fprintf(fd, "Lattice { float %s } = @1", name);
    fprintf(fd, "\n@1\n");

	for (int i=0; i < ntot; i++) {
		//printf("i,scal= %d, %f\n", i, scalar[i]);
		fprintf(fd, "%f\n", scalar[i]);
	}
	
    //endian.fwrite(scalar, sizeof(float), ntot, fd);
    //fwrite(scalar, sizeof(float), ntot, fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::WriteHeaderScalar(const char* name, const int nb, const char* type)
{
	if (type) {
    	fprintf(fd, "Lattice { %s %s } = @%1d\n", type, name, nb);
	}
	else {
    	fprintf(fd, "Lattice { float %s } = @%1d\n", name, nb);
	}
}
//----------------------------------------------------------------------
//void WriteAmiraMesh::WriteHeaderScalar(char* name, int nb)
//{
    //fprintf(fd, "Lattice { float %s } = @%1d", name, nb);
//}
//----------------------------------------------------------------------
void WriteAmiraMesh::WriteHeader()
{
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"uniform\",\n");
    fprintf(fd, " BoundingBox 0 1 0 1 0 1 \n");
    fprintf(fd, " }\n");
}
//----------------------------------------------------------------------
void WriteAmiraMesh::WriteScalar(double* data, int nb)
{
	printf("not implemented\n");
	int n = 1;
    fprintf(fd, "\n@%1d\n", n);
    //endian.fwrite(data, sizeof(double), nb, fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::WriteScalar(float* data, int nb)
{
	int n = 1;
    fprintf(fd, "\n@%1d\n", n);
    endian.fwrite(data, sizeof(float), nb, fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::WriteScalar(int* data, int nb)
{
	int n = 1;
    fprintf(fd, "\n@%1d\n", n);
    endian.fwrite(data, sizeof(int), nb, fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::CloseFile()
{
	fclose(fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::readRectilinearData(float** vx, float** vy, float** vz,
  char* name,const char* fileName_,float** x,float** y,float** z)
{
    Endian endian;
	bool res;

    rule<> ui = (str_p("define Lattice") >> +blank_p >>
                int_p[assign_a(nx)] >> +blank_p >>
                int_p[assign_a(ny)] >> +blank_p >>
                int_p[assign_a(nz)] >> *anychar_p);
    rule<> ui1 = (*blank_p >> str_p("@1") >> *anychar_p);
    rule<> ui2 = (*blank_p >> str_p("@2") >> *anychar_p);


    printf("writeRectilinearData, filename= %s\n", fileName_);
    FILE* fd = fopen(fileName_, "rb");
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName_);
        exit(1);
    }

    char line[80];
    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

        res = parse(line, ui).full;

        res = parse(line, ui1).full;
        if (res == true) {
			printf("@1 found ----\n");
            printf("nx,ny,nz= %d, %d, %d\n", nx, ny, nz);
            *x = new float[nx];
            *y = new float[ny];
            *z = new float[nz];
            // READ BINARY
            endian.fread(*x, sizeof(float), nx, fd);
            endian.fread(*y, sizeof(float), ny, fd);
            endian.fread(*z, sizeof(float), nz, fd);
            for (int i=0; i < 5; i++) {
                printf("x,y,z= %f, %f, %f\n", (*x)[i], (*y)[i], (*z)[i]);
            }
        }

        res = parse(line, ui2).full;
        if (res == true) {
			printf("@2 found ----\n");
            int ntot= nx*ny*nz;
			*vx = new float [ntot];
			*vy = new float [ntot];
			*vz = new float [ntot];
            // READ BINARY
    		for (long j=0; j < ntot; j++) {
        		endian.fread(*vx+j, sizeof(float), 1, fd);
        		endian.fread(*vy+j, sizeof(float), 1, fd);
        		endian.fread(*vz+j, sizeof(float), 1, fd);
			}
            for (int i=0; i < 5; i++) {
                printf("vx,vy,vz= %f, %f, %f\n", (*vx)[i], (*vy)[i], (*vz)[i]);
            }
		}
    }

    fclose(fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::writeRectilinearData(float* vx, float* vy, float* vz,
  char* name,const char* fileName_,float* x,float* y,float* z)
{

    if (sizeof(MSLONG)!=4) {
     printf("long does not have a size of 4 bytes");
     exit(1);
    }
	Endian endian;

    printf("writeRectilinearData, filename= %s\n", fileName_);
    FILE* fd = fopen(fileName_, "w");
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName_);
        exit(1);
    }
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "define Coordinates %d \n", nx+ny+nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"rectilinear\",\n");
    fprintf(fd, " }\n");
    fprintf(fd, "Coordinates { float xyz } = @1\n");
    fprintf(fd, "Lattice { float %s } = @2\n", name);

    fprintf(fd, "\n@1\n");
    endian.fwrite(x, sizeof(float), nx, fd);
    endian.fwrite(y, sizeof(float), ny, fd);
    endian.fwrite(z, sizeof(float), nz, fd);

    ntot = nx*ny*nz;
    fprintf(fd, "\n@2\n");
    printf("ntot= %d\n", ntot);
    printf("writeRectilinearData (vx,vy,vz) \n");
    {for (MSLONG i=0; i < ntot; i++) {
        endian.fwrite(vx+i, sizeof(float), 1, fd);
        endian.fwrite(vy+i, sizeof(float), 1, fd);
        endian.fwrite(vz+i, sizeof(float), 1, fd);
    }}
    fclose(fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::readRectilinearData(float** scalar, char* name,
  const char* fileName_,float** x,float** y,float** z)
{
    Endian endian;
	bool res;

    rule<> ui = (str_p("define Lattice") >> +blank_p >>
                int_p[assign_a(nx)] >> +blank_p >>
                int_p[assign_a(ny)] >> +blank_p >>
                int_p[assign_a(nz)] >> *anychar_p);
    rule<> ui1 = (*blank_p >> str_p("@1") >> *anychar_p);
    rule<> ui2 = (*blank_p >> str_p("@2") >> *anychar_p);


    printf("writeRectilinearData, filename= %s\n", fileName_);
    FILE* fd = fopen(fileName_, "rb");
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName_);
        exit(1);
    }

    char line[80];
    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

        res = parse(line, ui).full;

        res = parse(line, ui1).full;
        if (res == true) {
            printf("nx,ny,nz= %d, %d, %d\n", nx, ny, nz);
            *x = new float[nx];
            *y = new float[ny];
            *z = new float[nz];
            // READ BINARY
            endian.fread(*x, sizeof(float), nx, fd);
            endian.fread(*y, sizeof(float), ny, fd);
            endian.fread(*z, sizeof(float), nz, fd);
            for (int i=0; i < 5; i++) {
                printf("x,y,z= %f, %f, %f\n", (*x)[i], (*y)[i], (*z)[i]);
            }
        }

        res = parse(line, ui2).full;
        if (res == true) {
            int ntot= nx*ny*nz;
            *scalar = new float[ntot];
            // READ BINARY
            endian.fread(*scalar, sizeof(float), ntot, fd);
            for (int i=0; i < 5; i++) {
                printf("scalar= %f\n", (*scalar)[i]);
            }
        }
    }

    fclose(fd);
}
//----------------------------------------------------------------------
void WriteAmiraMesh::writeRectilinearData(float* scalar, char* name,
  const char* fileName_,float* x,float* y,float* z)
{
    Endian endian;

    printf("writeRectilinearData, filename= %s\n", fileName_);
    FILE* fd = fopen(fileName_, "w");
    if (!fd) {
        printf("file %s cannot be opened for writing\n", fileName_);
        exit(1);
    }
    fprintf(fd, "# AmiraMesh BINARY\n");
    fprintf(fd, "# Dimensions in x, y, z, directions\n");
    fprintf(fd, "define Lattice %d %d %d\n", nx, ny, nz);
    fprintf(fd, "define Coordinates %d \n", nx+ny+nz);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, " CoordType \"rectilinear\",\n");
    fprintf(fd, " }\n");
    fprintf(fd, "Coordinates { float xyz } = @1\n");
    fprintf(fd, "Lattice { float %s } = @2\n", name);

    fprintf(fd, "\n@1\n");
    endian.fwrite(x, sizeof(float), nx, fd);
    endian.fwrite(y, sizeof(float), ny, fd);
    endian.fwrite(z, sizeof(float), nz, fd);

    fprintf(fd, "\n@2\n");
    ntot = nx*ny*nz;
    printf("ntot= %d\n", ntot);
    endian.fwrite(scalar, sizeof(float), ntot, fd);

    fclose(fd);
}
//----------------------------------------------------------------------
char* skip_lines_until(const char* strg)
{
}

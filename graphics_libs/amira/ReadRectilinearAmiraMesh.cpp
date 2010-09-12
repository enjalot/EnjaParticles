//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/ReadRectilinearAmiraMesh.cpp,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: ReadRectilinearAmiraMesh.cpp,v $
// $Revision: 2.2 $
// $State: Exp $

//================================================================================


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "ReadRectilinearAmiraMesh.h"
#include "Endian.h"

#include "Vec3.h"
#include <string>

using namespace std;
using namespace boost::spirit;


char* skip_lines_until(const char* strg);


//----------------------------------------------------------------------
ReadRectilinearAmiraMesh::ReadRectilinearAmiraMesh(const char* fileName_)
{
    init(fileName_);
}
//----------------------------------------------------------------------
ReadRectilinearAmiraMesh::ReadRectilinearAmiraMesh()
{
	init(0);
}
//----------------------------------------------------------------------
ReadRectilinearAmiraMesh::~ReadRectilinearAmiraMesh()
{
	closeFile();
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::init(const char* fileName)
{
    nx = -1;
    ny = -1;
    nz = -1;
	fd = 0;
	this->fileName = fileName;

	printf("===== open file %s ==== \n", fileName);

	if (fileName) {
    	fd = fopen(fileName, "rb");
    	if (!fd) {
        	printf("file %s cannot be opened for reading\n", fileName);
        	exit(1);
		}
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::closeFile()
{
	if (fd) {
		fclose(fd);
		fd = 0;
	}
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::readHeader()
{
	int_coord   = -1;
	int_lattice = -1;

    ui =   (str_p("define Lattice") >> +blank_p >>
                int_p[assign_a(nx)] >> +blank_p >>
                int_p[assign_a(ny)] >> +blank_p >>
                int_p[assign_a(nz)]);
	at_coord   = (ch_p('@') >> int_p);
	at_lattice = (ch_p('@') >> int_p);

	std::string lat;
	std::string coord;

    //ui_lattice = (*blank_p >> "Lattice"     >> *anychar_p >> str_p("@") >> int_p[assign_a(int_lattice)] >> *anychar_p);

	// Does not work because anychar_p is as long as possible so consumes the entire line
    //ui_coord   = (*anychar_p >> str_p("Coordinates" )); // NOT OK (DO NOT UNDERSTAND)

    ui_coord     = (*blank_p >> str_p("Coordinates") >> *~ch_p('@') >> at_coord[assign_a(coord)] ); // OK
    ui_lattice   = (*blank_p >> str_p("Lattice")     >> *~ch_p('@') >> at_lattice[assign_a(lat)] ); 

	// I REALLY SHOULD SEARCH
	// Coordinates { float xyz } = @1
	// Lattice { float ...} = @2
	// Search for Lattice, then search for @[integer]
	// Search for Coordinates, then search for @[integer]

	bool res;

    char line[80];
	int count = 0;

    for (int i=0; ; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

		// partial hit. Entire string need not be consumed
        res = parse(line, ui).hit;
		if (res == true) {
			count++;
		}

        res = parse(line, ui_coord).hit;
        if (res == true) {
			//printf("coord hit, coord= %s\n", coord.c_str());
			at_coord   = (str_p(coord.c_str()) >> ch_p('\n'));
			//printf("*** ASSIGN at_coord ***\n");
			count++;
		}

        res = parse(line, ui_lattice).hit;
        if (res == true) {
			//printf("lattice hit, lattice= %s\n", lat.c_str());
			at_lattice = (str_p(lat.c_str()) >> ch_p('\n'));
			//printf("*** ASSIGN at_lattice ***\n");
			count++;
		}
		// at this point, nx,ny,nz are calculate as are the "@1", "@2" that delimit
		// the scalar and vector data
		if (count == 3) break;
	}

	// Store location in the file (do a seek)
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::read(std::vector<Vec3>& vel, float* veloc)
{
    Endian endian;
	bool res = false;
	Vec3 v;
	//printf("vel size: %d\n", vel->size()); exit(0);

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

		//printf("readvec, line= %s\n", line);
        res = parse(line, at_lattice).hit;
		//if (line[0] == '@' && line[1] == '2') {
			//printf("MANUAL: velocity, @2 found, line= %s\n", line);
		//}

        if (res == true) {
			//printf("velocity, @2 found ----\n");
            int ntot= nx*ny*nz;
            // READ BINARY
        	endian.fread(veloc, sizeof(float), 3*ntot, fd);

    		for (long j=0; j < ntot; j++) {
			    v.setValue(veloc[3*j], veloc[3*j+1], veloc[3*j+2]);
				vel.push_back(v);
			}
			break;
		}
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::read(std::vector<Vec3>& vel)
{
    Endian endian;
	bool res = false;
	Vec3 v;
	//printf("vel size: %d\n", vel->size()); exit(0);

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

		//printf("readvec, line= %s\n", line);
        res = parse(line, at_lattice).hit;
		//if (line[0] == '@' && line[1] == '2') {
			//printf("MANUAL: velocity, @2 found, line= %s\n", line);
		//}

        if (res == true) {
			//printf("velocity, @2 found ----\n");
            int ntot= nx*ny*nz;
			float* veloc = new float [3*ntot];
            // READ BINARY
        	endian.fread(veloc, sizeof(float), 3*ntot, fd);

    		for (long j=0; j < ntot; j++) {
			    v.setValue(veloc[3*j], veloc[3*j+1], veloc[3*j+2]);
				vel.push_back(v);
			}
			delete [] veloc;
			break;
		}
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::read(float* vx, float* vy, float* vz, float* vel)
// vel is storage provided by caller. 
{
    Endian endian;
	bool res = false;

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

		//printf("readvec, line= %s\n", line);
        res = parse(line, at_lattice).hit;
		//if (line[0] == '@' && line[1] == '2') {
			//printf("MANUAL: velocity, @2 found, line= %s\n", line);
		//}

        if (res == true) {
			//printf("velocity, @2 found ----\n");
            int ntot= nx*ny*nz;
            // READ BINARY
        	endian.fread(vel, sizeof(float), 3*ntot, fd);

    		for (long j=0; j < ntot; j++) {
				vx[j] = vel[3*j];
				vy[j] = vel[3*j+1];
				vz[j] = vel[3*j+2];
        		//endian.fread(vx+j, sizeof(float), 1, fd);
        		//endian.fread(vy+j, sizeof(float), 1, fd);
        		//endian.fread(vz+j, sizeof(float), 1, fd);
			}
			//printf("*** RectLin:: velxyz[0] = %f, %f, %f\n", vel[0], vel[1], vel[2]);

            //for (int i=0; i < 50; i++) {
                //printf("vx,vy,vz= %f, %f, %f\n", vx[i], vy[i], vz[i]);
            //}

			break;
		}
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::read(float* vx, float* vy, float* vz)
{
    Endian endian;
	bool res = false;

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

		//printf("readvec, line= %s\n", line);
        res = parse(line, at_lattice).hit;
		//if (line[0] == '@' && line[1] == '2') {
			//printf("MANUAL: velocity, @2 found, line= %s\n", line);
		//}

        if (res == true) {
			//printf("velocity, @2 found ----\n");
            int ntot= nx*ny*nz;
			float* vel = new float [3*ntot];
            // READ BINARY
        	endian.fread(vel, sizeof(float), 3*ntot, fd);

    		for (long j=0; j < ntot; j++) {
				vx[j] = vel[3*j];
				vy[j] = vel[3*j+1];
				vz[j] = vel[3*j+2];
        		//endian.fread(vx+j, sizeof(float), 1, fd);
        		//endian.fread(vy+j, sizeof(float), 1, fd);
        		//endian.fread(vz+j, sizeof(float), 1, fd);
			}
			//printf("*** RectLin:: velxyz[0] = %f, %f, %f\n", vel[0], vel[1], vel[2]);
			delete [] vel;

            //for (int i=0; i < 50; i++) {
                //printf("vx,vy,vz= %f, %f, %f\n", vx[i], vy[i], vz[i]);
            //}

			break;
		}
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::read(float* scalar)
{
    Endian endian;
	bool res;

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;

        res = parse(line, at_lattice).hit;
        if (res == true) {
			//printf("scalar, @2 found\n");
            int ntot= nx*ny*nz;
            // READ BINARY
            endian.fread(scalar, sizeof(float), ntot, fd);
            //for (int i=0; i < 300; i++) {
                //printf("scalar= %f\n", scalar[i]);
            //}
			break;
        }
    }
}
//----------------------------------------------------------------------
void ReadRectilinearAmiraMesh::readGrid(float* x, float* y, float* z)
{
    Endian endian;
	bool res;

    char line[80];

    for (int i=0;; i++) {
        char* c = fgets(line, 80, fd);
        // EOF
        if (c == 0) break;
		res = parse(line, at_coord).hit;
		if (res == true) {
			//printf("@1 found, nxyz= %d, %d, %d\n", nx, ny, nz);
			// somehow, grid is not properly allocated
			//x = new float [nx];
            // READ BINARY
            endian.fread(x, sizeof(float), nx, fd);
            endian.fread(y, sizeof(float), ny, fd);
            endian.fread(z, sizeof(float), nz, fd);
			break;
        }
    }
}
//----------------------------------------------------------------------

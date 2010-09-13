// Currently write files in ASCII because problem with Endian on my mac (Leopard)

//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteLineSet.cpp,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteLineSet.cpp,v $
// $Revision: 2.2 $
// $State: Exp $

//================================================================================

//Needed spirit tools from the Boost library
//#include <boost/spirit/core.hpp>
//#include <boost/spirit/actor/push_back_actor.hpp>
//#include <boost/spirit/actor/assign_actor.hpp>
//#include <boost/spirit/actor/insert_key_actor.hpp>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "write_lineset.h"
#include "Endian.h"

#include "Vec3.h"

using namespace std;
//using namespace boost::spirit;


char* skip_lines_until(const char* strg);



//----------------------------------------------------------------------
WriteLineSet::WriteLineSet(int nb_vertices, int nb_line_indices, const char* fileName_)
{
	init(nb_vertices, nb_line_indices, fileName_);
}
//----------------------------------------------------------------------
WriteLineSet::WriteLineSet()
{
}
//----------------------------------------------------------------------
WriteLineSet::~WriteLineSet()
{
		;
}
//----------------------------------------------------------------------
void WriteLineSet::init(int nb_vertices, int nb_line_indices, const char* fileName)
{
	this->nb_vertices = nb_vertices;
	this->nb_line_indices = nb_line_indices;

	if (fileName) {
    	fd = fopen(fileName, "w");
    	if (!fd) {
        	printf("file %s cannot be opened for writing\n", fileName);
        	exit(1);
		}
    }
}
//----------------------------------------------------------------------
void WriteLineSet::writeHeader(const char* scal_name)
{
	writeHeader();
	char line[80];
	// It appears tat I must use Data and a name for the scalar
	sprintf(line, "Vertices { float Data } = @3");
	//sprintf(line, "Vertices { float %s } = @3", scal_name);
    fprintf(fd, "%s\n", line);
}
//----------------------------------------------------------------------
void WriteLineSet::writeHeader(const char* scal1_name, const char* scal2_name)
{
	writeHeader();
	char line[80];
	sprintf(line, "Vertices { float %s } = @3", scal1_name);
    fprintf(fd, "%s\n", line);
	sprintf(line, "Vertices { float %s } = @4", scal2_name);
}
//----------------------------------------------------------------------
void WriteLineSet::writeHeader(const char* scal1_name, const char* scal2_name, const char* scal3_name)
{
	writeHeader();
	char line[80];
	sprintf(line, "Vertices { float %s } = @3", scal1_name);
    fprintf(fd, "%s\n", line);
	sprintf(line, "Vertices { float %s } = @4", scal2_name);
    fprintf(fd, "%s\n", line);
	sprintf(line, "Vertices { float %s } = @5", scal3_name);
    fprintf(fd, "%s\n", line);
}
//----------------------------------------------------------------------
void WriteLineSet::writeHeader()
{
    fprintf(fd, "# AmiraMesh BINARY 1.0\n");
    //fprintf(fd, "# AmiraMesh ASCII 1.0\n");
    fprintf(fd, "define Lines %d\n", nb_line_indices);
    fprintf(fd, "define Vertices %d\n", nb_vertices);
    fprintf(fd, "Parameters {\n");
    fprintf(fd, "    ContentType \"HxLineSet\"\n"); 
    fprintf(fd, "}\n");
    fprintf(fd, "Vertices { float[3] Coordinates } = @1\n");
    fprintf(fd, "Lines { int LineIdx } = @2\n");
}
//----------------------------------------------------------------------
void WriteLineSet::write(float* vertices, int* lines)
{
    Endian endian;
    fprintf(fd, "\n@1\n");
	endian.fwrite(vertices, sizeof(float), 3*nb_vertices, fd);
	//for (int i=0; i < 3*nb_vertices; i++) {
		//fprintf(fd, "%f ", vertices[i]);
	//}
    fprintf(fd, "\n@2\n");
	endian.fwrite(lines, sizeof(int), nb_line_indices, fd);
	//for (int i=0; i < nb_line_indices; i++) {
		//fprintf(fd, "%d ", lines[i]);
	//}
}
//----------------------------------------------------------------------
void WriteLineSet::write(float* scalar, const char* marker)
{
    Endian endian;
	printf("WRITE Marker %s to file\n", marker);
	fprintf(fd, "\n%s\n", marker);
	endian.fwrite(scalar, sizeof(float), nb_vertices, fd);

	//for (int i=0; i < nb_vertices; i++) {
		//fprintf(fd, "%f ", scalar[i]);
	//}

	//printf("nb_vertices= %d\n", nb_vertices);
	//for (int i=0; i < nb_vertices; i++) {
		//printf("scalar[%d] = %f\n", i, scalar[i]);
	//}
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void WriteLineSet::closeFile()
{
	fclose(fd);
}
//----------------------------------------------------------------------

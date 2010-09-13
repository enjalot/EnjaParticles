
// Currently write files in ASCII because problem with Endian on my mac (Leopard)

//================================================================================

// $Author: erlebach $
// $Date: 2002/01/05 15:18:28 $
// $Header: /home/erlebach/D2/src/CVS/computeCP/WriteLineSet.h,v 2.2 2002/01/05 15:18:28 erlebach Exp $
// Sticky tag, $Name:  $
// $RCSfile: WriteLineSet.h,v $
// $Revision: 2.2 $
// $State: Exp $
 
//================================================================================

#ifndef _WRITELINESET_H_
#define _WRITELINESET_H_

#ifdef GORDON_FOURBYTEINT
#define MSLONG int
#else
#define MSLONG long
#endif

#include <stdio.h>

class WriteLineSet {
private:
	char* fileName;
	FILE* fd;
	int nb_vertices;
	int nb_line_indices; // includes -1 to indicate end of line (EOL)

public:
// Assume not additional data is stored with the lines
	WriteLineSet(int nb_vertices, int nb_line_indices, const char* fileName=0);
	WriteLineSet();
	~WriteLineSet();
	void init(int nb_vertices, int nb_line_indices, const char* fileName_=0);
	void writeHeader();
	void writeHeader(const char* scal1_name, const char* scal2_name, const char* scal3_name);
	void writeHeader(const char* scal1_name, const char* scal2_name);
	void writeHeader(const char* scal_name);
	void write(float* vertices, int* lines);
	void write(float* scalar, const char* marker);

	void closeFile();
};

#endif

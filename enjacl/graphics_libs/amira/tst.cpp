//Needed spirit tools
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>
#include <boost/spirit/actor/assign_actor.hpp>
#include <boost/spirit/actor/insert_key_actor.hpp>

//Other needed classes
#include <iostream>
#include <vector>
#include <utility>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace boost::spirit;


//#include "GuiParser.h"

int main(int argc, char** argv) 
{
	FILE* fd = fopen("a.dat", "r");
	int nx, ny, nz;
	bool res;

	rule<> ui = (str_p("Input data") >> +blank_p >> 
				int_p[assign_a(nx)] >> +blank_p >> 
				int_p[assign_a(ny)] >> +blank_p >> 
				int_p[assign_a(nz)] >> *anychar_p);
	rule<> ui1 = (*blank_p >> str_p("@1") >> *anychar_p);
	rule<> ui2 = (*blank_p >> str_p("@2") >> *anychar_p);

	float* x;
	float* y;
	float* z;
	float* scal;

	char line[80];
	for (int i=0;; i++) {
		char* c = fgets(line, 80, fd);
		// EOF
		if (c == 0) break;

		res = parse(line, ui).full;

		res = parse(line, ui1).full;
		if (res == true) {
			printf("res 2 = true\n");
			printf("nx,ny,nz= %d, %d, %d\n", nx, ny, nz);
			x = new float[nx];
			y = new float[ny];
			z = new float[nz];
			// READ BINARY
			fread(x, sizeof(float), nx, fd);
			fread(y, sizeof(float), ny, fd);
			fread(z, sizeof(float), nz, fd);
			for (int i=0; i < 5; i++) {
				printf("x,y,z= %f, %f, %f\n", x[i], y[i], z[i]);
			}
		}

	//	printf("=== line= %s\n", line);
		res = parse(line, ui2).full;
		if (res == true) {
			printf("res 2 = true\n");
			int ntot= nx*ny*nz;
			scal = new float[ntot];
			// READ BINARY
			fread(scal, sizeof(float), ntot, fd);
		}

	}
	exit(0);
}


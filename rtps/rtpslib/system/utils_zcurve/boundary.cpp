#include <stdio.h>

//----------------------------------------------------------------------
int main()
{
	double K = 10000.;
	double z0 = 0.05;
	double g = -9.8;

	// solve: 
	// d/dt2(z) = K (z-z0) + 9.8
	// d/dt(x) = v
	// d/dt(v) = K (z-z0) + 9.8

	double dt = 1.e-4;

	double z = z0;
	double v = -5.;

	double z1;

	int nbiter = 10000;

	for (int i=0; i < nbiter; i++) {
		v = v - dt*( K*(z-z0) + g);
		z1 = z + dt * v;

		printf("z,v= %f, %f\n", z, v);

		if (z1 > z) break;
		if (z < 0.) break;

		z = z1;
	}

	return 0;
}
//----------------------------------------------------------------------

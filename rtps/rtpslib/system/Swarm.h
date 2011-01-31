#ifndef RTPS_SWARM_H_INCLUDED
#define RTPS_SWARM_H_INCLUDED


#include "../RTPS.h"
#include "System.h"

#include "../opencl/Buffer.h"

namespace rtps {

class Swarm: public System{

public:
        Swarm(RTPS *rtps, int n);
        ~Swarm();
       
   	RTPS *ps;	
 
	std::vector<float4> positions;
        std::vector<float4> velocities;
        std::vector<float4> colors;

	rtps::Buffer<float4> cl_position;
    	rtps::Buffer<float4> cl_color;
	
        void update();
	void FlockIt_CPU();

protected:
       float separationdist;
       float searchradius;
       float maxspeed;

private:
	int* flockmates;	// array with flockmates
	float d_closestFlockmate;	// distance to the closest boid, need it to compute Separation
	float ID_closestFlockmate;	// ID of the closest boid, need it to compute Separation
	int numFlockmates;	// number of flockmates found
  	float4 acc;		// acceleration

	int SearchFlockmates(int i);
	float4 Separation(int boid_ID);
	float4 Alignment();
	float4 Cohesion(int boid_ID);

};
}
#endif

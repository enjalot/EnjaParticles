/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/



#include <vector>
#include "structs.h"
using namespace std;
using namespace rtps;

class Boids
{
public:
	typedef vector<float4> VF;
	typedef vector<int> VI;

private:
	float wcoh;
	float wsep;
	float walign;

	float dim;  // dimension of domain (a square)

	float DESIRED_SEPARATION; // 2.;
	float NEIGHBOR_RADIUS; // 2.;
	float MAX_FORCE; // 10.;
	float MAX_SPEED; // 10.;

	VF pos;
	VF vel;
	VF acc;


public:
	Boids(VF& pos);
	~Boids();
	void neighbors(vector<float4>& pos, int which, VI& neighbors);
	float4 avg_value(VI& neigh, VF& val); 
	float4 avg_separ(VI& neigh, VF& pos, int i);
	void set_ic(VF pos, VF vel, VF acc);
	void update();
	void setDomainSize(float dim) {this->dim = dim;}
	float getDomainSize() { return dim; }
	float getDesiredSeparation() { return DESIRED_SEPARATION; }

	VF& getPos() { return pos; }
	VF& getVel() { return vel; }
	VF& getAcc() { return acc; }
};

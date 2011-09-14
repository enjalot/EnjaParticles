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


#include "boids.h"

//----------------------------------------------------------------------
Boids::Boids(VF& pos_) : pos(pos_)
{
	DESIRED_SEPARATION = 20.;
	NEIGHBOR_RADIUS = 5.;
	MAX_FORCE = 10.;
	MAX_SPEED = 0.7; 

	wcoh = 0.;
	wsep = 0.;
	walign = 0.;

	// pure separation
	// with particles on a rectangle, upper right corner separates. That must be an error
	// probably correct: symmetries are maintained
	wcoh = 0.007; // makes particles implode (must be a mistake?)

	// not quite correct. There is some asymmetry
	wsep = 0.1; // must be very strong compared to wcoh

	// might be slight error: lower left corner
	walign = .10;  // particles end in a steady configuration

	//printf("constructor: pos.size= %d\n", pos.size());
}
//----------------------------------------------------------------------
Boids::~Boids()
{
}
//----------------------------------------------------------------------
void Boids::neighbors(vector<float4>& pos, int which, VI& neigh) 
// h is the search radius
{
	//printf("neigh size= %d\n", neigh.size());
	float dist;
	float h = NEIGHBOR_RADIUS;

		for (int j=0; j < pos.size(); j++) {
			if (which == j) continue;
			float4 d = pos[which] - pos[j];
			dist = d.length(); // uses a sqrt (inefficient)
			if (dist <= h) neigh.push_back(j);
		}
	//printf("sz= %d\n", neigh.size());
	//return neigh;
}
//----------------------------------------------------------------------
float4 Boids::avg_value(VI& neigh, VF& val) //, float4& posi)
// usable by velocity and position
{
	float4 avg = float4(0.,0.,0.,0.);
	for (int k=0; k < neigh.size(); k++) {
		avg = avg + val[neigh[k]];
	}
	avg = neigh.size() > 0 ? avg/neigh.size() : avg;
	return avg;
}
//----------------------------------------------------------------------
float4 Boids::avg_separ(VI& neigh, VF& pos, int i)
// which is the current boid
{
	float desired_separ = DESIRED_SEPARATION;
	float4 steer = float4(0.,0.,0.,1.);
	int count = 0;

	// The way this works: only edge particles should expand outward, and then 
	// the flock slowly expands

	pos[i].print("***** pos *****");
	pos[i].printd("***** pos_d *****");

	printf("neigh size= %d\n", neigh.size());
	//if (neigh.size() > 10) exit(0);

	for (int k=0; k < neigh.size(); k++) {
		// vector pointing from neighbor to local boid
		float4 diff = pos[i] - pos[neigh[k]];
		//pos[neigh[k]].print("neighbor");
		//float4 diff = pos[neigh[k]] - pos[i];
		float d = diff.length();
		//printf("sep= %f, desired sep= %f\n", d, desired_separ);
		if (d < desired_separ) {
			diff = normalize3(diff);
			diff = diff / d;
			steer = steer + diff;
			count++;
		}
	}
	if (count > 0) {
		//printf("count= %d\n", count);
		steer = steer / count;
	}
	//steer.print("st");
	return steer;
}
//----------------------------------------------------------------------
void Boids::set_ic(Boids::VF pos, Boids::VF vel, Boids::VF acc)
{
	this->pos = pos;
	this->vel = vel;
	this->acc = acc;
}
//----------------------------------------------------------------------
void Boids::update()
{
	float h = NEIGHBOR_RADIUS;
	float desired_sep = DESIRED_SEPARATION;


	printf("========== ENTER UPDATE ===============\n");
	printf("pos.size= %d\n", pos.size());
	for (int i=0; i < pos.size(); i++) {
		VI neigh;
		neighbors(pos, i, neigh); 
		//VI neigh = neighbors(pos, i); // return might have error on linux!

		float4 sep = avg_separ(neigh, pos, i);
		//printf("size= %d\n", neigh.size()); // nb neighbors sometimes reaches 30!
		float4 coh = avg_value(neigh, pos) - pos[i];
		coh = normalize3(coh);

		float4 align = avg_value(neigh, vel) - vel[i];
		float align_mag = align.length();
		//float4 align_norm = normalize3(align);
		//if (align_mag > MAX_FORCE) {
			//align = align_norm*MAX_FORCE;
		//}
		align = normalize3(align);

		//printf("------\n");
		//sep.print("sep");
		//coh.print("coh");
		//align.print("align");

		acc[i] = acc[i] + wcoh*coh +wsep*sep + walign*align;
		float acc_mag = acc[i].length();
		float4 acc_norm = normalize3(acc[i]);
		// MAX_SPEED is crucial
		if (acc_mag > MAX_SPEED) { 
			acc[i] = acc_norm*MAX_SPEED;
			;
		}
		acc[i].w = 1.;

		acc_mag = acc[i].length();
		//printf("acc_mag= %f\n", acc_mag);


		//vel[i] = vel[i] + acc[i];
		float4 v = float4(-pos[i].y, pos[i].x, 0, 0.);
		v = v*.00;
		vel[i] = v + acc[i];
		pos[i] = pos[i] + vel[i];
		//pos[i] = pos[i] + acc[i] + vel[i];
		//acc[i].print("acc");
		if (pos[i].x >= dim)  pos[i].x = -dim;
		if (pos[i].x <= -dim) pos[i].x =  dim;
		if (pos[i].y >= dim)  pos[i].y = -dim;
		if (pos[i].y <= -dim) pos[i].y =  dim;
	}
	//exit(0);
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//----------------------------------------------------------------------

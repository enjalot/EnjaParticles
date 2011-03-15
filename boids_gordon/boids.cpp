#include "boids.h"

//----------------------------------------------------------------------
Boids::Boids(VF& pos_) : pos(pos_)
{
	DESIRED_SEPARATION = 20.;
	NEIGHBOR_RADIUS = 20.;
	MAX_FORCE = 10.;
	MAX_SPEED = .1;

	wcoh = 0.;
	wsep = 0.;
	walign = 0.;

	wcoh = 0.0;
	wsep = 1.;
	walign = 1.;
}
//----------------------------------------------------------------------
Boids::~Boids()
{
}
//----------------------------------------------------------------------
vector<int> Boids::neighbors(vector<float4>& pos, int which) 
// h is the search radius
{
	VI neigh;
	float dist;
	float h = NEIGHBOR_RADIUS;

		for (int j=0; j < pos.size(); j++) {
			if (which == j) continue;
			float4 d = pos[which] - pos[j];
			dist = d.length();
			if (dist < h) neigh.push_back(j);
		}
	//printf("sz= %d\n", neigh.size());
	return neigh;
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

	//printf("neigh size= %d\n", neigh.size());
	//if (neigh.size() > 10) exit(0);

	for (int k=0; k < neigh.size(); k++) {
		// vector pointing from neighbor to local boid
		float4 diff = pos[i] - pos[neigh[k]];
		//float4 diff = pos[neigh[k]] - pos[i];
		float d = diff.length();
		//printf("d= %f, desired sep= %f\n", d, desired_separ);
		if (d < desired_separ) {
			diff = normalize3(diff);
			diff = diff / d;
			steer = steer + diff;
			count++;
		}
		if (count > 0) {
			//printf("count= %d\n", count);
			steer = steer / count;
		}
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


	for (int i=0; i < pos.size(); i++) {
		VI neigh = neighbors(pos, i);

		float4 sep = avg_separ(neigh, pos, i);
		float4 coh = avg_value(neigh, pos) - pos[i];

		float4 align = avg_value(neigh, vel) - vel[i];
		float align_mag = align.length();
		float4 align_norm = normalize3(align);
		if (align_mag > MAX_FORCE) {
			align = align_norm*MAX_FORCE;
		}

		//printf("------\n");
		//sep.print("sep");
		//coh.print("coh");
		//align.print("align");

		acc[i] = acc[i] + wcoh*coh +wsep*sep + walign*align;
		float acc_mag = acc[i].length();
		float4 acc_norm = normalize3(acc[i]);
		if (acc_mag > MAX_SPEED) { 
			acc[i] = acc_norm*MAX_SPEED;
		}
		acc[i].w = 1.;

		acc_mag = acc[i].length();
		//printf("acc_mag= %f\n", acc_mag);


		//vel[i] = vel[i] + acc[i];
		float4 v = float4(-pos[i].y, pos[i].x, 0, 0.);
		v = v*.01;
		vel[i] = v + acc[i];
		pos[i] = pos[i] + vel[i];
		//pos[i] = pos[i] + acc[i] + vel[i];
		//acc[i].print("acc");
		if (pos[i].x > dim)  pos[i].x = -dim;
		if (pos[i].x < -dim) pos[i].x =  dim;
		if (pos[i].y > dim)  pos[i].y = -dim;
		if (pos[i].y < -dim) pos[i].y =  dim;
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

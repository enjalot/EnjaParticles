#include "boids.h"

//----------------------------------------------------------------------
Boids::Boids(VF& pos_, int dim_, float wcoh_, float wsep_, float walign_) : pos(pos_), dim(dim_), wcoh(wcoh_), wsep(wsep_), walign(walign_)
{
	DESIRED_SEPARATION = 12.;
	NEIGHBOR_RADIUS = 30.;  // search grid
	MAX_FORCE = 10.;
	MAX_SPEED = 3.; 

	// pure separation
	// with particles on a rectangle, upper right corner separates. That must be an error
	// probably correct: symmetries are maintained
	//wcoh = 0.030; //0.015; // makes particles implode (must be a mistake?)

	// not quite correct. There is some asymmetry
	//wsep = .3; // must be very strong compared to wcoh

	// might be slight error: lower left corner
	//walign = 0.3; //.03;  // particles end in a steady configuration

	//printf("constructor: pos.size= %d\n", pos.size());

	//dim = getDomainSize(); //how you gonna call this in a constructor yo? //HYUNYA
	xmin = -dim;
	xmax =  dim;
	ymin = -dim;
	ymax =  dim;
	float h = NEIGHBOR_RADIUS;
	nx = (int) ((xmax - xmin) / h);
	ny = (int) ((ymax - ymin) / h);
	dx = (int) ((xmax - xmin) / h);
	dy = (int) ((ymax - xmin) / h);
	int nb_cells = nx*ny;
	printf("nx,ny= %d, %d\n", nx, ny);
	neigh_list = new VI [nb_cells]; 

	int num = pos.size();
	vel_coh.resize(num);
	vel_sep.resize(num);
	vel_align.resize(num);
	pos_real.resize(num);

	for (int i=0; i < num; i++) {
		pos_real[i] = pos[i];
	}
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

	//pos[i].print("***** pos *****");
	//pos[i].printd("***** pos_d *****");

	//printf("neigh size= %d\n", neigh.size());
	//if (neigh.size() > 10) exit(0);

	for (int k=0; k < neigh.size(); k++) {
		//printf("neighbor: %d\n", k);
		//pos[neigh[k]].print("neighbor");
		// vector pointing from neighbor to local boid
		float4 diff = pos[i] - pos[neigh[k]];
		//diff.print("diff");
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
		steer = normalize3(steer);
	}
	//float steer_lg = steer.length();
	//if (steer_lg > 1.e-6) steer.print("st");
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


	//printf("========== ENTER UPDATE ===============\n");
	//printf("pos.size= %d\n", pos.size());
	for (int i=0; i < pos.size(); i++) {
		VI neigh;
		//neighbors(pos_real, i, neigh); 
		neighbors(pos, i, neigh); 
		//neigh = neighbors(pos, i); // return might have error on linux!

		//float4 sep = avg_separ(neigh, pos_real, i);
		float4 sep = avg_separ(neigh, pos, i);
		//sep.print("sep");
		//printf("size= %d\n", neigh.size()); // nb neighbors sometimes reaches 30!
		//float4 coh = avg_value(neigh, pos_real) - pos[i];
		float4 coh = avg_value(neigh, pos) - pos[i];
		coh = normalize3(coh);

		float4 align = avg_value(neigh, vel) - vel[i];
		//float align_mag = align.length();
		//float4 align_norm = normalize3(align);
		//if (align_mag > MAX_FORCE) {
			//align = align_norm*MAX_FORCE;
		//}
		align = normalize3(align);
		
        //printf("------\n");
		//sep.print("sep");
		//coh.print("coh");
		//align.print("align");

		vel_coh[i] = wcoh*coh;
		vel_sep[i] = wsep*sep;
		vel_align[i] = walign*align;

		//float mag = vel_sep[i].length();
		//if (mag > 1.e-8) vel_sep[i].print("sep");

		acc[i] = vel[i] + vel_coh[i] + vel_sep[i] + vel_align[i];
		//acc[i] = acc[i] + wcoh*coh +wsep*sep + walign*align;
		float acc_mag = acc[i].length();
		//acc[i].print("acc");
		float4 acc_norm = normalize3(acc[i]);
		// MAX_SPEED is crucial
		if (acc_mag > MAX_SPEED) { 
			acc[i] = acc_norm*MAX_SPEED;
			;
		}
		acc[i].w = 1.;


		//acc_mag = acc[i].length();
		//printf("acc_mag= %f\n", acc_mag);


		//vel[i] = vel[i] + acc[i];
		float4 v = float4(-3.*pos[i].y, pos[i].x, 0, 0.);
		v = v*.00;
		vel[i] = v + acc[i];
	}

	for (int i=0; i < pos.size(); i++) {
		pos_real[i] = pos_real[i] + vel[i];
		pos[i] = pos[i] + vel[i];

		// pos[i] used for plotting
		if (pos[i].x >= dim)  {
		//exit(0);
			pos[i].x = -2*dim + pos[i].x;
		} else if (pos[i].x <= -dim) {
		//exit(0);
			pos[i].x =  2*dim - pos[i].x;
		}

		if (pos[i].y >= dim) {  
		//exit(0);
			pos[i].y = -2*dim + pos[i].y;
		} else if (pos[i].y <= -dim) {
		//exit(0);
			pos[i].y =  2*dim - pos[i].y;
		}
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

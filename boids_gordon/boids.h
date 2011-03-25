
#include <vector>
#include "ArrayT.h"
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

	float xmin, xmax;
	float ymin, ymax;
	int nx, ny; // nb cells in x,y
	float dx, dy;

	float DESIRED_SEPARATION; // 2.;
	float NEIGHBOR_RADIUS; // 2.;
	float MAX_FORCE; // 10.;
	float MAX_SPEED; // 10.;

	VF pos;
	VF pos_real; // no adjustment for periodic Boundary-Conditions (BC)
	VF vel;
	VF acc;

public:
	VF vel_coh;
	VF vel_sep;
	VF vel_align;

public:
    // wcoh   = 0.03
    // wsep   = 0.30
    // walign = 0.10
	Boids(VF& pos, int dim=300, float wcoh=.0, float wsep=0.3, float walign=.10);
	~Boids();
	void neighbors(vector<float4>& pos, int which, VI& neighbors);
	float4 avg_value(VI& neigh, VF& val); 
	float4 avg_separ(VI& neigh, VF& pos, int i);
	void set_ic(VF pos, VF vel, VF acc);
	void update();
	void setDomainSize(float dim) {this->dim = dim;}
	float getDomainSize() { return dim; }
	float getDesiredSeparation() { return DESIRED_SEPARATION; }

	VI* neigh_list; 

	VF& getPos() { return pos; }
	VF& getVel() { return vel; }
	VF& getAcc() { return acc; }
};

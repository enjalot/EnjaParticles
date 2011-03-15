
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
	VI neighbors(vector<float4>& pos, int which);
	float4 avg_value(VI& neigh, VF& val); 
	float4 avg_separ(VI& neigh, VF& pos, int i);
	void set_ic(VF pos, VF vel, VF acc);
	void update();
	void setDomainSize(float dim) {this->dim = dim;}

	VF& getPos() { return pos; }
	VF& getVel() { return vel; }
	VF& getAcc() { return acc; }
};

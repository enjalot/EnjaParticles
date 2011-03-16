/* File : flock.i */
%module flock
%{
/* Put headers and other declarations here */
#define SWIG_FILE_WITH_INIT
#include "boids.h"
#include "structs.h"
#include "domain/IV.h"
%}

%include "typemaps.i"
%include "std_vector.i"
%include "cstring.i"

/** Import decl details for float4 */
%include "float4.i"

/** Swig cant provide std::vector of any type. We need to
    declare the types we want available */
namespace std
{
    %template(intvec) vector<int>; 
    %template(float4vec) vector<rtps::float4>; 
    %template(vecvec) vector<vector<int> >; 
}

%include "domain/IV.h"

/** Grabbed from boids.h. Note I removed definitions. */
using namespace std;
class Boids
{
	typedef vector<rtps::float4> VF;
	typedef vector<int> VI;
public:
	Boids(VF& pos);
	~Boids();
	void neighbors(vector<rtps::float4>& pos, int which, VI& neighbors);
	rtps::float4 avg_value(VI& neigh, VF& val); 
	rtps::float4 avg_separ(VI& neigh, VF& pos, int i);
	void set_ic(VF pos, VF vel, VF acc);
	void update();
	void setDomainSize(float dim);
	float getDomainSize();
	float getDesiredSeparation();
	VF& getPos();
	VF& getVel();
	VF& getAcc();
};

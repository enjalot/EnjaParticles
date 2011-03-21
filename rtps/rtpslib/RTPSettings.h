#ifndef RTPS_RTPSETTINGS_H_INCLUDED
#define RTPS_RTPSETTINGS_H_INCLUDED

#include "domain/Domain.h"

namespace rtps{

class RTPSettings
{
public:
    //decide which system to use
    enum SysType {Simple, SPH, FLOCK};
    SysType system;


    RTPSettings();
    RTPSettings(SysType system, int max_particles, float dt);
    RTPSettings(SysType system, int max_particles, float dt, Domain grid);
    
    //collision
    RTPSettings(SysType system, int max_particles, float dt, Domain grid, bool tri_collision);

    //flock
    RTPSettings(SysType system, int max_particles, float dt, Domain grid, float maxspeed, float mindist, float searchradius, float color[]);

    //maximum number of particles a system can hold
    int max_particles;
    
    //the bounding domain of the system
    Domain grid;

    //time step per iteration
    float dt;

    //triangle collision?
    bool tri_collision;

    // max speed of the boids
    float max_speed;

    // desired separation distance of the boids
    float min_dist;

    // radius to search for flockmates
    float search_radius;

    // color of the flock
    float4 color;

// Added by GE
private:
	float render_radius_scale;
	float render_blur_scale;
	int render_type;
	bool use_glsl;
	bool use_alpha_blending;

public:
	float getRadiusScale() { return render_radius_scale; }
	void setRadiusScale(float scale) {
		render_radius_scale = scale;
	}

	float getBlurScale() { return render_blur_scale; }
	void setBlurScale(float scale) {
		render_blur_scale = scale;
	}

	int getRenderType() { return render_type; }
	void setRenderType(int type) {
		render_type = type;
	}

	int getUseAlphaBlending() { return use_alpha_blending; }
	int setUseAlphaBlending(bool use_alpha) { use_alpha_blending = use_alpha; }

	int getUseGLSL() { return use_glsl; }
	int setUseGLSL(bool use_glsl) { 
		this->use_glsl = use_glsl; }
};

}

#endif

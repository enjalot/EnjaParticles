#ifndef RTPS_SYSTEM_H_INCLUDED
#define RTPS_SYSTEM_H_INCLUDED

#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
#else
    //OpenGL stuff
    #include <GL/gl.h>
#endif

#include "../domain/Domain.h"
#include "ForceField.h"

#include<stdio.h>
namespace rtps {

class System
{
public:
    virtual void update() = 0;
    
    virtual ~System(){};
    
    virtual Domain getGrid(){ return grid; };
    virtual int getNum(){ return num; };
    virtual void setNum(int nn){num = nn;};//should this be public
    virtual GLuint getPosVBO() { return pos_vbo; };
    virtual GLuint getColVBO() { return col_vbo; };

    virtual int addBox(int nn, float4 min, float4 max, bool scaled){ return 0;};
    virtual void addBall(int nn, float4 center, float radius, bool scaled){};

    virtual void addForceField(ForceField ff){};
    
    GLuint ghost_vbo;
    int nb_ghosts;
    int max_ghosts;

protected:
    //number of particles
    int num; 
    //maximum number of particles (for array allocation)
    int max_num;

    GLuint pos_vbo;
    GLuint col_vbo;
    //flag is true if the system's constructor creates the VBOs for the system
    bool managed;

    Domain grid;

};

}

#endif

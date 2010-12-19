#ifndef RTPS_SYSTEM_H_INCLUDED
#define RTPS_SYSTEM_H_INCLUDED


#include "importgl.h"
#include "../domain/Domain.h"

namespace rtps {

class System
{
public:
    virtual void update() = 0;
    
    
    virtual Domain getGrid(){ return grid; };
    virtual int getNum(){ return num; };
    virtual GLuint getPosVBO() { return pos_vbo; };
    virtual GLuint getColVBO() { return col_vbo; };

    virtual int addBox(int nn, float4 min, float4 max, bool scaled){ return 0;};
    virtual void addBall(int nn, float4 center, float radius, bool scaled){};


protected:
    //number of particles
    int num; 
    int max_num;

    GLuint pos_vbo;
    GLuint col_vbo;
    //flag is true if the system's constructor creates the VBOs for the system
    bool managed;

    Domain grid;

};

}

#endif

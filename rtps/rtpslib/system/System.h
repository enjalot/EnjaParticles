#ifndef RTPS_SYSTEM_H_INCLUDED
#define RTPS_SYSTEM_H_INCLUDED

#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
#else
    //OpenGL stuff
    #include <GL/gl.h>
#endif


namespace rtps {

class System
{
public:
    System(){};
    //virtual ~System();
    virtual void update() = 0;

    //number of particles
    int num; 


    virtual GLuint getPosVBO();
    virtual GLuint getColVBO();
};

}

#endif

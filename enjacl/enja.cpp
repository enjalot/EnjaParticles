#include <stdio.h>


#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    //OpenGL stuff
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #include <GLUT/glut.h>
    #include <OpenGL/CGLCurrent.h> //is this really necessary?
#else
    //OpenGL stuff
    #include <GL/glx.h>
#endif


#include "enja.h"
//#include "incopencl.h"
#include "util.h"


#include<math.h>
#include<stdlib.h>
#include<time.h>
#include <sstream>
#include <iomanip>


//we include our cl programs with STRINGIFY macro trick
//then store them in this list for usage
#include "physics/lorenz.cl"
#include "physics/gravity.cl"
#include "physics/vfield.cl"

#ifdef OPENCL_SHARED
//#include "physics/collision_ge.cl"
//#include "physics/collision_ge_a.cl"
// Version for collision against triangles with shared memory
//#include "physics/collision_ge_b.cl"
// Version for collision against bounding boxes
#include "physics/collision_ge_bb.cl"
#else
#include "physics/collision.cl"
#endif
//#include "physics/transform.cl"
#include "physics/position.cl"

const std::string EnjaParticles::sources[] = {
        lorenz_program_source,
        gravity_program_source,
        vfield_program_source,
        collision_program_source,
        position_program_source
    };


float EnjaParticles::rand_float(float mn, float mx)
{
	float r = random() / (float) RAND_MAX;
	return mn + (mx-mn)*r;
}

int EnjaParticles::init(AVec4 g, AVec4 v, AVec4 c, int n)
{
    // This is the main initialization function for our particle systems
    // AVec4* g is the array of generator points (this initializes our system)
    // AVec4* c is the array of color values 
    
    //this should be configurable. how many updates do we run per frame:
    updates = 4;
    dt = .001;
    glsl = false;
    blending = false;
    point_scale = 1.0f;

    //we pack radius and life into generator and velocity arrays
    for(int i=0; i < n; i++)
    {
        //initialize the radii array with random values between 1.0 and particle_radius
        //g[i].w = 1.0f + particle_radius*drand48();
        //initialize the particle life array with random values between 0 and 1
        v[i].w = drand48();
        //v[i].w = 1.0f;
    }

    num = n;
    vert_gen= g;
    velo_gen = v;
    velocities = v;
    colors = c;

    //index array so we can draw transparent items correctly
    std::vector<int> ind(n);
    for(int i = 0; i < n; i++)
    {
        ind[i] = i;
    }
    indices = ind;

    //initialize our vbos
    vbo_size = sizeof(Vec4) * n;
    v_vbo = createVBO(&vert_gen[0], vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    c_vbo = createVBO(&colors[0], vbo_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    i_vbo = createVBO(&ind[0], sizeof(int) * n, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    

    //we initialize our timers, they only time every 5th call
    ts[0] = new GE::Time("update", 5);
    ts[1] = new GE::Time("render", 5);
    ts[2] = new GE::Time("total render", 5);


    return 1;
}


EnjaParticles::EnjaParticles(int s, int n)
{
    //printf("default constructor\n");
    
    particle_radius = 5.0f;
    
    system = s;
    //init system
    AVec4 g(n);

    float f = 0;
    for(int i=0; i < n; i++)
    {
        f = (float)i;
		float rad = rand_float(0.5, 2.);
        g[i].x = 0.0 + rad*cos(2.*M_PI*(f/n));  //with lorentz this looks more interesting
        g[i].y = 0.0 + rad*sin(2.*M_PI*(f/n));
        g[i].z = 0.f;
        g[i].w = 1.f;
    }

    AVec4 c(n);
    for(int i=0; i < n; i++)
    {
        c[i].x = 1.0;   //Red
        c[i].y = 0.0;   //Green
        c[i].z = 0.0;   //Blue
        c[i].w = 1.0;   //Alpha
    }
    //initialize the velocities array, by default we just set to 0
    AVec4 v(n);
    f = 0.f;
    for(int i=0; i < n; i++)
    {
        f = (float)i;
        //v[i].x = 0.0 + .5*cos(2.*M_PI*(f/n));  //with lorentz this looks more interesting
        //v[i].z = 3.f;
        //v[i].y = 0.0 + .5*sin(2.*M_PI*(f/n));
        v[i].x = 0.5f; //.01 * (1. - 2.*drand48()); // between -.02 and .02
        v[i].y = 0.5f; //.05 * drand48();
        v[i].z = 0.f; //.01 * (1. - 2.*drand48());
        v[i].w = 0.f;
    }


    srand48(time(NULL));

    //printf("before init call\n");

    //init particle system
    init(g, v, c, n);

    //printf("before opencl call\n");

    //init opencl
    int success = init_cl();
    
}

//Take in vertex vert_gen as well as velocity vert_gen that are len elements long
//This is to support generating particles from Blender Mesh objects
EnjaParticles::EnjaParticles(int s, AVec4 g, AVec4 v, int len, int n, float radius)
{
    system = s;
    if(radius <1.0f)
        radius = 1.0f;
    particle_radius = radius;

    AVec4 _vert_gen(n);
    AVec4 _velo_gen(n);
 
    AVec4 c(n);
    
    srand48(time(NULL));
    int j;
    for(int i=0; i < n; i++)
    {
        //fill the vert_gen
        j = (int)(drand48()*len); //randomly get a vertex/velocity from a generator
        _vert_gen[i] = g[j];
        _velo_gen[i] = v[j];

        //handle the colors
        c[i].x = 1.0;   //Red
        c[i].y = 0.0;   //Green
        c[i].z = 0.0;   //Blue
        c[i].w = 1.0;   //Alpha
    }

   

    //init particle system
    init(_vert_gen, _velo_gen, c, n);

    //init opencl
    int success = init_cl();
}

/* lets implement this when we need it
EnjaParticles::EnjaParticles(int s, AVec4* g, AVec4* v, AVec4* c, int n)
{
    system = s;
    //init particle system
    init(g, v, c, n);

    //init opencl
    int success = init_cl();
}
*/

EnjaParticles::~EnjaParticles()
{

    printf("Release! num=%d\n", num);
    ts[0]->print();
    ts[1]->print();
    ts[2]->print();
    delete ts[0];
    delete ts[1];
    delete ts[2];

    ts_cl[0]->print();
    ts_cl[1]->print();
    ts_cl[2]->print();
    ts_cl[3]->print();
    delete ts_cl[0];
    delete ts_cl[1];
    delete ts_cl[2];
    delete ts_cl[3];
    
/*
    if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
*/
    if(v_vbo)
    {
        glBindBuffer(1, v_vbo);
        glDeleteBuffers(1, (GLuint*)&v_vbo);
        v_vbo = 0;
    }
    if(c_vbo)
    {
        glBindBuffer(1, c_vbo);
        glDeleteBuffers(1, (GLuint*)&c_vbo);
        c_vbo = 0;
    }
    if(i_vbo)
    {
        glBindBuffer(1, i_vbo);
        glDeleteBuffers(1, (GLuint*)&i_vbo);
        i_vbo = 0;
    }
/*
    if(cl_vbos[0])clReleaseMemObject(cl_vbos[0]);
    if(cl_vbos[1])clReleaseMemObject(cl_vbos[1]);
    if(cl_vbos[2])clReleaseMemObject(cl_vbos[2]);
    if(cl_vert_gen)clReleaseMemObject(cl_vert_gen);
    if(cl_velo_gen)clReleaseMemObject(cl_velo_gen);
    if(cl_velocities)clReleaseMemObject(cl_velocities);
    //if(cl_life)clReleaseMemObject(cl_life);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
    if(cdDevices)delete(cdDevices);
*/
}


int EnjaParticles::getVertexVBO()
{
    return v_vbo;
}

int EnjaParticles::getColorVBO()
{
    return c_vbo;
}


int EnjaParticles::getNum()
{
    return num;
}

float EnjaParticles::getFPS()
{
    return 1000.f / ts[2]->getAverage();    //1 second divided by total render time 
}

std::string EnjaParticles::printReport()
{
    //print timings with all library options
    //helper function for printing timings for graphing
    std::stringstream ss;
    //ts[0] is total time for all update calls, ts[2] is total rendering time
    // num, render ms, cl ms, /*updates*/, glsl, alpha blending
    ss << num << " & " << ts[2]->getAverage() << " & " << ts[0]->getAverage() << " & " /*<< updates << " & "*/ << glsl << " & " << blending << " \\" << std::ends;
    return ss.str();
}


std::string* EnjaParticles::getReport()
{
    //helper function for printing timings to screen
    std::stringstream ss1;
    std::stringstream ss2;
    std::string* s = new std::string[2];
    ss1 << std::fixed << std::setprecision(6);
    ss1 << "Average Render Time (per frame): " << ts[2]->getAverage() << std::ends;
    s[0] = ss1.str();
    ss2 << std::fixed << std::setprecision(6);
    ss2 << "Average OpenCL Time (per frame): " << ts_cl[0]->getAverage() + ts_cl[1]->getAverage() + ts_cl[2]->getAverage() << std::ends;
    s[1] = ss2.str();
    return s;
}


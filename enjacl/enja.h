#ifndef ENJA_PARTICLES_H_INCLUDED
#define ENJA_PARTICLES_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS

#include <string>
#include <vector>
#include "incopencl.h"
#include "timege.h"
#include <stdio.h>

//should switch to blender's library, or just pass in the texture from blender


typedef struct Vec4
{
    float x;
    float y;
    float z;
    float w;

    Vec4(){};
    Vec4(float xx, float yy, float zz, float ww):
        x(xx),
        y(yy),
        z(zz),
        w(ww)
    {}
	void set(float xx, float yy, float zz, float ww=1.) {
		x = xx;
		y = yy;
		z = zz;
		w = ww;
	}
} Vec4;

// size: 4*4 = 16 floats
// shared memory = 65,536 bytes = 16,384 floats
//               = 1024 triangles
typedef struct Triangle
{
    Vec4 verts[3];
    Vec4 normal;    //should pack this in verts array
	//float dummy; // for more efficient global -> shared
} Triangle;

typedef struct Box
{
	float xmin, xmax;
	float ymin, ymax;
	float zmin, zmax;
} Box;

typedef std::vector<Vec4> AVec4;

class EnjaParticles
{
public:

    int update();   //update the particle system
    int cpu_update();   //update the particle system using cpu code
    int render(); //render calls update then renders the particles

    int getVertexVBO(); //get the vertices vbo id
    int getColorVBO(); //get the color vbo id
    int getNum(); //get the number of particles

    float getFPS(); //get the calculated frames per second for entire rendering process
    std::string* getReport(); //get string report of timings
    std::string printReport(); //get string report of timings with library options

    //constructors: will probably have more as more options are added
    //choose system and number of particles
    EnjaParticles(int system, int num);
    //specify initial positions and velocities, with arrays on length len, and number of particles
    EnjaParticles(int system, AVec4 generators, AVec4 velocities, int len, int num, float radius);
    //EnjaParticles(int system, Vec4* generators, Vec4* velocities, Vec4* colors, int num);
    
    //extra properties of the system
    bool collision;         //enable collision detection
    int updates;            //number of times to update per frame
    float dt;
    float particle_radius;  
    Vec4 transform[4];          //store the blender transformation matrix
    double gl_transform[16];    //store the opengl matrix of the object

    //rendering options
    void use_glsl();        //not the best way, call this before rendering and it sets up glsl program
    bool blending;          //use alpha blending
    float point_scale;      //scale for rendering glsl particles

    ~EnjaParticles();

    enum {LORENZ, GRAVITY, VFIELD, COLLISION, POSITION, TRANSFORM};
    static const std::string sources[];

    //keep track of transformation from blender
    /*
    float translation[3];
    Vec4 rotation[3];
    Vec4 invrotation[3];
    */
    int n_triangles;
    int n_boxes;
    //handle triangles for collision detection
    void loadTriangles(std::vector<Triangle>);
    void loadBoxes(std::vector<Box>, std::vector<Triangle>, 
				   std::vector<int> tri_offsets);
    //Triangle faceToTriangle(Vec4 face[4]);

//private:
    //particles
    int num;                //number of particles
    int system;             //what kind of system?
    AVec4 vert_gen;       //vertex generators
    AVec4 velo_gen;       //velocity generators
    AVec4 velocities;       //for cpu version only
    AVec4 colors;
    std::vector<int> indices;
    //float* life;  //life is packed into velocity.w

    int init(AVec4 vert_gen, AVec4 velo_gen, AVec4 colors, int num);

    //opencl
    std::vector<cl::Device> devices;
    cl::Context context;
    cl::CommandQueue queue;

    cl::Program transform_program;  //keep track of blender transforms
    cl::Program vel_update_program;  //integrate the velocities of the particles
    cl::Program collision_program;  //check for collisions
    cl::Program pos_update_program;     //update the positions

    cl::Kernel transform_kernel; //kernel for updating with blender transformations
    cl::Kernel vel_update_kernel;
    cl::Kernel collision_kernel;
    cl::Kernel pos_update_kernel;

	// true if all objects have been loaded to the GPU
	bool are_objects_loaded;

	float rand_float(float mn, float mx);

    unsigned int deviceUsed;
    
    cl_int err;
    cl::Event event;


    //cl::Buffer vbo_cl;
    std::vector<cl::Memory> cl_vbos;  //0: vertex vbo, 1: color vbo, 2: index vbo
    cl::Buffer cl_vert_gen;  //want to have the start points for reseting particles
    cl::Buffer cl_velo_gen;  //want to have the start velocities for reseting particles
    cl::Buffer cl_velocities;  //particle velocities
    
    cl::Buffer cl_transform;  //transformation matrix from blender
    //cl::Buffer cl_life;        //keep track where in their life the particles are (packed into velocity.w now)
    int v_vbo;   //vertices vbo
    int c_vbo;   //colors vbo
    int i_vbo;   //index vbo
    unsigned int vbo_size; //size in bytes of the vbo
    
    //for collisions
    cl::Buffer cl_triangles;  //particle velocities
    cl::Buffer cl_boxes;  //particle velocities
    cl::Buffer cl_tri_offsets;  //triangle offsets (relates to box list)

    //timers
    GE::Time *ts[3];    //library timers (update, render, total)
    GE::Time *ts_cl[2]; //opencl timers (cl update routine, execute kernel)

    int init_cl();
    int setup_cl(); //helper function that initializes the devices and the context
    cl::Program loadProgram(std::string kernel_source);
    void popCorn(); // sets up the kernel and pushes data
    
    //opengl
    void drawArrays();      //seperate out the opengl glDrawArrays call
    int compileShaders();
    int glsl_program;   //should be GLuint
    bool glsl;
    //int loadTexture(std::vector<unsigned char> image, int width, int height);
    int loadTexture();
    GLuint gl_tex;     //particle texture handle
    
    int loadWebCam();
    //CvCapture* capture;
    void* capture; //hack for blender since it can't include opencv headers for some reason
    bool loadedcam;
};

#endif


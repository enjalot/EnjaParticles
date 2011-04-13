#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#ifdef WIN32
#define _USE_MATH_DEFINES
#include <cmath>
#endif

#include <string>

#include <RTPS.h>
#include <System.h>
#include <Kernel.h>
#include <Buffer.h>

#include <Domain.h>
#include <SPHSettings.h>


#include <Prep.h>
#include <Hash.h>
#include <BitonicSort.h>
//#include <DataStructures.h>
#include <CellIndices.h>
#include <Permute.h>
#include <Density.h>
#include <Force.h>
#include <Collision_wall.h>
#include <Collision_triangle.h>
#include <LeapFrog.h>
#include <Lifetime.h>
#include <Euler.h>

//#include "../util.h"
#include <Hose.h>

//#include <timege.h>
#include <timer_eb.h>

#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif

namespace rtps
{

    class RTPS_EXPORT SPH : public System
    {
    public:
        SPH(RTPS *ps, int num);
        ~SPH();

        void update();
        //wrapper around IV.h addRect
        int addBox(int nn, float4 min, float4 max, bool scaled, float4 color=float4(1.0f, 0.0f, 0.0f, 1.0f));
        //wrapper around IV.h addSphere
        void addBall(int nn, float4 center, float radius, bool scaled);
        //wrapper around Hose.h 
        void addHose(int total_n, float4 center, float4 velocity, float radius, float4 color=float4(1.0, 0.0, 0.0, 1.0f));
        void sprayHoses();

        virtual void render();

        void loadTriangles(std::vector<Triangle> triangles);

        void testDelete();
        int cut; //for debugging DEBUG

        EB::TimerList timers;
        int setupTimers();
        void printTimers();
        void pushParticles(vector<float4> pos, float4 velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));
        void pushParticles(vector<float4> pos, vector<float4> velo, float4 color=float4(1.0, 0.0, 0.0, 1.0));

        std::vector<float4> getDeletedPos();
        std::vector<float4> getDeletedVel();

    protected:
        virtual void setRenderer();
    private:
        //the particle system framework
        RTPS* ps;
        RTPSettings* settings;

        //SPHSettings* sphsettings;
        SPHParams sphp;
        GridParams grid_params;
        GridParams grid_params_scaled;
        Integrator integrator;
        float spacing; //Particle rest distance in world coordinates

        int nb_var;

        std::vector<float4> deleted_pos;
        std::vector<float4> deleted_vel;


        //keep track of hoses
        std::vector<Hose> hoses;

        //needs to be called when particles are added
        void calculateSPHSettings();
        void setupDomain();
        void prepareSorted();
        //void popParticles();

        //This should be in OpenCL classes
        Kernel k_scopy;

        std::vector<float4> positions;
        std::vector<float4> colors;
        std::vector<float4> velocities;
        std::vector<float4> veleval;

        std::vector<float>  densities;
        std::vector<float4> forces;
        std::vector<float4> xsphs;

        Buffer<float4>      cl_position_u;
        Buffer<float4>      cl_position_s;
        Buffer<float4>      cl_color_u;
        Buffer<float4>      cl_color_s;
        Buffer<float4>      cl_velocity_u;
        Buffer<float4>      cl_velocity_s;
        Buffer<float4>      cl_veleval_u;
        Buffer<float4>      cl_veleval_s;

        Buffer<float>       cl_density_s;
        Buffer<float4>      cl_force_s;
        Buffer<float4>      cl_xsph_s;

        //Neighbor Search related arrays
        //Buffer<float4>      cl_vars_sorted;
        //Buffer<float4>      cl_vars_unsorted;
        //Buffer<float4>      cl_cells; // positions in Ian code
        Buffer<unsigned int>         cl_cell_indices_start;
        Buffer<unsigned int>         cl_cell_indices_end;
        //Buffer<int>                  cl_vars_sort_indices;
        Buffer<unsigned int>         cl_sort_hashes;
        Buffer<unsigned int>         cl_sort_indices;
        //Buffer<unsigned int>         cl_unsort;
        //Buffer<unsigned int>         cl_sort;

        //Buffer<Triangle>    cl_triangles;

        //Two arrays for bitonic sort (sort not done in place)
        //should be moved to within bitonic
        Buffer<unsigned int>         cl_sort_output_hashes;
        Buffer<unsigned int>         cl_sort_output_indices;

        Bitonic<unsigned int> bitonic;

        //Parameter structs
        Buffer<SPHParams>   cl_sphp;
        Buffer<GridParams>  cl_GridParams;
        Buffer<GridParams>  cl_GridParamsScaled;

        Buffer<float4>      clf_debug;  //just for debugging cl files
        Buffer<int4>        cli_debug;  //just for debugging cl files


        //CPU functions
        void cpuDensity();
        void cpuPressure();
        void cpuViscosity();
        void cpuXSPH();
        void cpuCollision_wall();
        void cpuEuler();
        void cpuLeapFrog();

        void updateCPU();
        void updateGPU();

        //calculate the various parameters that depend on max_num of particles
        void calculate();
        //copy the SPH parameter struct to the GPU
        void updateSPHP();

        //Nearest Neighbors search related functions
        Prep prep;
        void call_prep(int stage);
        Hash hash;
        //DataStructures datastructures;
        CellIndices cellindices;
        Permute permute;
        void hash_and_sort();
        void bitonic_sort();
        Density density;
        Force force;
        void collision();
        CollisionWall collision_wall;
        CollisionTriangle collision_tri;
        void integrate();
        LeapFrog leapfrog;
        Euler euler;

        Lifetime lifetime;


        float Wpoly6(float4 r, float h);
        float Wspiky(float4 r, float h);
        float Wviscosity(float4 r, float h);

        //OpenCL helper functions, should probably be part of the OpenCL classes
        void loadScopy();
        void scopy(int n, cl_mem xsrc, cl_mem ydst); 

        //void sset_int(int n, int val, cl_mem xdst);

    };



}

#endif

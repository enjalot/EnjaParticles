#ifndef RTPS_SPH_H_INCLUDED
#define RTPS_SPH_H_INCLUDED

#include <string>

#include <RTPS.h>
#include <System.h>
#include <Kernel.h>
#include <Buffer.h>

#include <Domain.h>
#include <SPHSettings.h>

#include <Hash.h>
#include <BitonicSort.h>

//#include "../util.h"
#include <Hose.h>

//#include <timege.h>
#include <timer_eb.h>

#include "../rtps_common.h"


namespace rtps
{

    class RTPS_EXPORT SPH : public System
    {
    public:
        SPH(RTPS *ps, int num);
        ~SPH();

        void update();
        //wrapper around IV.h addRect
        int addBox(int nn, float4 min, float4 max, bool scaled);
        //wrapper around IV.h addSphere
        void addBall(int nn, float4 center, float radius, bool scaled);
        //wrapper around Hose.h 
        void addHose(int total_n, float4 center, float4 velocity, float radius);
        void sprayHoses();

        virtual void render();

        void loadTriangles(std::vector<Triangle> triangles);

        void testDelete();
        int cut; //for debugging DEBUG

        /*
        template <typename RT>
        RT GetSettingAs(std::string key, std::string defaultval = "0")
        {
            return sphsettings->GetSettingAs<RT>(key, defaultval);
        }
        template <typename RT>
        void SetSetting(std::string key, RT value)
        {
            sphsettings->SetSetting(key, value);
        }
        */

        EB::TimerList timers;
        int setupTimers();
        void printTimers();
        void pushParticles(vector<float4> pos, float4 velo);

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

        bool triangles_loaded; //keep track if we've loaded triangles yet

        //keep track of hoses
        std::vector<Hose> hoses;

        //needs to be called when particles are added
        void calculateSPHSettings();
        void setupDomain();
        void prepareSorted();
        //void popParticles();

        Kernel k_density, k_pressure, k_viscosity;
        Kernel k_collision_wall;
        Kernel k_collision_tri;
        Kernel k_euler, k_leapfrog;
        Kernel k_xsph;

        Kernel k_prep;
        Kernel k_hash;
        Kernel k_datastructures;
        Kernel k_neighbors;

        //This should be in OpenCL classes
        Kernel k_scopy;

        std::vector<float4> positions;
        std::vector<float4> colors;
        std::vector<float>  densities;
        std::vector<float4> forces;
        std::vector<float4> velocities;
        std::vector<float4> veleval;
        std::vector<float4> xsphs;

        Buffer<float4>      cl_position;
        Buffer<float4>      cl_color;
        Buffer<float>       cl_density;
        Buffer<float4>      cl_force;
        Buffer<float4>      cl_velocity;
        Buffer<float4>      cl_veleval;
        Buffer<float4>      cl_xsph;

        //Neighbor Search related arrays
        Buffer<float4>      cl_vars_sorted;
        Buffer<float4>      cl_vars_unsorted;
        Buffer<float4>      cl_cells; // positions in Ian code
        Buffer<unsigned int>         cl_cell_indices_start;
        Buffer<unsigned int>         cl_cell_indices_end;
        Buffer<int>         cl_vars_sort_indices;
        Buffer<unsigned int>         cl_sort_hashes;
        Buffer<unsigned int>         cl_sort_indices;
        Buffer<unsigned int>         cl_unsort;
        Buffer<unsigned int>         cl_sort;

        Buffer<Triangle>    cl_triangles;

        //Two arrays for bitonic sort (sort not done in place)
        Buffer<unsigned int>         cl_sort_output_hashes;
        Buffer<unsigned int>         cl_sort_output_indices;

        Bitonic<unsigned int> bitonic;

        //Parameter structs
        Buffer<SPHParams>   cl_sphp;
        Buffer<GridParams>  cl_GridParams;
        Buffer<GridParams>  cl_GridParamsScaled;

        //index neighbors. Maximum of 50
        //Buffer<int>         cl_index_neigh;

        //for keeping up with deleted particles
        Buffer<unsigned int> cl_num_changed;

        Buffer<float4>      clf_debug;  //just for debugging cl files
        Buffer<int4>        cli_debug;  //just for debugging cl files


        //still in use?
        Buffer<float4> cl_error_check;

        //these are defined in sph/ folder
        void loadCollision_wall();
        void loadCollision_tri();
        void loadEuler();
        void loadLeapFrog();

        //Nearest Neighbors search related kernels
        void loadPrep();
        //void loadHash();
        void loadBitonicSort();
        void loadDataStructures();
        void loadNeighbors();

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

        void calculate();
        //copy the SPH parameter struct to the GPU
        void updateSPHP();

        //Nearest Neighbors search related functions
        void prep(int stage);
        //void hash();
        Hash* hash;
        void printHashDiagnostics();
        void bitonic_sort();
        void buildDataStructures();
        void printDataStructuresDiagnostics();
        void neighborSearch(int choice);
        void collision();
        void collide_wall();
        void collide_triangles();
        void integrate();
        void euler();
        void leapfrog();

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

#ifndef RTPS_SIMPLE_H_INCLUDED
#define RTPS_SIMPLE_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "ForceField.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
#ifdef WIN32
    #if defined(rtps_EXPORTS)
        #define RTPS_EXPORT __declspec(dllexport)
    #else
        #define RTPS_EXPORT __declspec(dllimport)
	#endif 
#else
    #define RTPS_EXPORT
#endif
//#include "../util.h"

namespace rtps
{


    class RTPS_EXPORT Simple : public System
    {
    public:
        Simple(RTPS *ps, int num);
        ~Simple();

        void update();

        bool forcefields_enabled;
        int max_forcefields;

        //the particle system framework
        RTPS *ps;

        std::vector<float4> positions;
        std::vector<float4> colors;
        std::vector<float4> velocities;
        std::vector<float4> forces;
        std::vector<ForceField> forcefields;


        Kernel k_forcefield;
        Kernel k_euler;

        Buffer<float4> cl_position;
        Buffer<float4> cl_color;
        Buffer<float4> cl_force;
        Buffer<float4> cl_velocity;
        Buffer<ForceField> cl_forcefield;


        void loadForceField();
        void loadForceFields(std::vector<ForceField> ff);
        void loadEuler();

        void cpuForceField();
        void cpuEuler();


    };

}

#endif

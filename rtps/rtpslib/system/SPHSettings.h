#ifndef RTPS_SPHSETTINGS_H_INCLUDED
#define RTPS_SPHSETTINGS_H_INCLUDED

#include <stdlib.h>
#include <string>
#include <map>
#include <iostream>
#include <stdio.h>
#include <sstream>

#include <structs.h>
#include <Buffer.h>
#include <Domain.h>

namespace rtps
{

#ifdef WIN32
#pragma pack(push,16)
#endif
    //Struct which gets passed to OpenCL routines
	typedef struct SPHParams
    {
        float mass;
        float rest_distance;
        float smoothing_distance;
        float simulation_scale;
        
        //dynamic params
        float boundary_stiffness;
        float boundary_dampening;
        float boundary_distance;
        float K;        //gas constant
        
        float viscosity;
        float velocity_limit;
        float xsph_factor;
        float gravity; // -9.8 m/sec^2

        float friction_coef;
        //next 4 not used at the moment
        float restitution_coef;
        float shear;
        float attraction;

        float spring;
        //float surface_threshold;
        //constants
        float EPSILON;
        float PI;       //delicious
        //Kernel Coefficients
        float wpoly6_coef;
        
        float wpoly6_d_coef;
        float wpoly6_dd_coef; // laplacian
        float wspiky_coef;
        float wspiky_d_coef;

        float wspiky_dd_coef;
        float wvisc_coef;
        float wvisc_d_coef;
        float wvisc_dd_coef;


        //CL parameters
        int num;
        int nb_vars; // for combined variables (vars_sorted, etc.)
        int choice; // which kind of calculation to invoke
        int max_num;


        void print()
        {
            printf("----- SPHParams ----\n");
            printf("mass: %f\n", mass);
            printf("rest distance: %f\n", rest_distance);
            printf("smoothing distance: %f\n", smoothing_distance);
            printf("simulation_scale: %f\n", simulation_scale);
            printf("--------------------\n");

            /*
            printf("friction_coef: %f\n", friction_coef);
            printf("restitution_coef: %f\n", restitution_coef);
            printf("damping: %f\n", boundary_dampening);
            printf("shear: %f\n", shear);
            printf("attraction: %f\n", attraction);
            printf("spring: %f\n", spring);
            printf("gravity: %f\n", gravity);
            printf("choice: %d\n", choice);
            */
        }
    } SPHParams
#ifndef WIN32
	__attribute__((aligned(16)));
#else
		;
        #pragma pack(pop)
#endif

/*
    class SPHSettings
    {
        public:
            SPHSettings(Domain grid, int max_num);
            //SPHSettings(Buffer<SPHParams>& cl_sphp, Domain grid, int max_num);
            
            //update the struct by going through the map and setting the values
            bool updateSPHP(SPHParams& sphp);   
            //tell the user if any settings have been changed since last update
            bool has_changed();

            void printSettings();

            // Return the value associate with KEY as the specified template parameter type
            // e.g.,
            //  int i = SPHSettings.GetSettingAs<int>("key");
            //  double d = SPHSettings.GetSettingAs<double>("key2");
            //  string s = SPHSettings.GetSettingAs<string>("key3");
            template <typename RT>
            RT GetSettingAs(std::string key, std::string defaultval = "0") 
            {
                std::cout << "[Project Settings] \t"; 
                if (settings.find(key) == settings.end()) 
                {
                    RT ret = ss_typecast<RT>(defaultval);
                    std::cout << key << " = " << ret;
                    std::cout << "\t<-- default value." << std::endl;
                    return ret;
                }
                std::cout << key << " = " << settings[key] << std::endl;
                return ss_typecast<RT>(settings[key]);
            }

            template <typename RT>
            void SetSetting(std::string key, RT value) {
                // TODO: change to stringstream for any type of input that is cast as string
                RT oldval = this->GetSettingAs<RT>(key);
                if (oldval != value)
                {
                    std::ostringstream oss; 
                    oss << value; 
                    settings[key] = oss.str(); 
                    changed = true;
                }
            }

        private:
            std::map<std::string, std::string> settings;
            //Buffer<SPHParams>   cl_sphp;
            Domain domain;

            bool changed;
            
            //calculates the base settings that depend on max_num
            void calculate(int max_num);

            // This routine is adapted from post on GameDev:
            // http://www.gamedev.net/community/forums/topic.asp?topic_id=190991
            // Should be safer to use this than atoi. Performs worse, but our
            // hotspot is not this part of the code.
            template<typename RT, typename _CharT, typename _Traits , typename _Alloc >
            RT ss_typecast( const std::basic_string< _CharT, _Traits, _Alloc >& the_string )
            {
                std::basic_istringstream< _CharT, _Traits, _Alloc > temp_ss(the_string);
                RT num;
                temp_ss >> num;
                return num;
            }


    };
*/



    enum Integrator
    {
        EULER, LEAPFROG
    };


}

#endif

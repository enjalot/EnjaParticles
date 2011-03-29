#include <SPHSettings.h>

namespace rtps
{
    SPHSettings::SPHSettings(Domain grid, int max_num)
    {
        //Buffer<SPHParams>& cl_sphp_, 
        //cl_sphp = cl_sphp_;
        
        domain = grid;

        calculate(max_num);

        //dynamic params
        SetSetting("Boundary Stiffness", 20000.0f);
        SetSetting("Boundary Dampening", 256.0f);
        SetSetting("Gas Constant", 15.0f);
        SetSetting("Viscosity", .01f);
        SetSetting("Velocity Limit", 600.0f);
        SetSetting("XSPH Factor", .1f);
        SetSetting("Gravity", -9.8f); // -9.8 m/sec^2
        SetSetting("Friction Kinetic", 0.0f);
        SetSetting("Friction Static", 0.0f);

        //next 4 not used at the moment
        SetSetting("Restitution", 0.0f);
        SetSetting("Shear", 0.0f);
        SetSetting("Attraction", 0.0f);
        SetSetting("Spring", 0.0f);

        //constants
        SetSetting("EPSILON", 1E-6);
        SetSetting("PI", M_PI);       //delicious

        //CL parameters
        SetSetting("Number of Particles", 0);
        SetSetting("Number of Variables", 10); // for combined variables (vars_sorted, etc.) //TO be depracated
        SetSetting("Choice", 0); // which kind of calculation to invoke //TO be depracated

        
    }

    void SPHSettings::calculate(int max_num)
    {
        /*!
        * The Particle Mass (and hence everything following) depends on the MAXIMUM number of particles in the system
        */

        float rho0 = 1000;                              //rest density [kg/m^3 ]
        //float mass = (128*1024.0)/max_num * .0002;    //krog's way
        float VP = 2 * .0262144 / max_num;              //Particle Volume [ m^3 ]
        //float VP = .0262144 / 16000;                  //Particle Volume [ m^3 ]
        float mass = rho0 * VP;                         //Particle Mass [ kg ]
        //constant .87 is magic
        float rest_distance = .87 * pow(VP, 1.f/3.f);   //rest distance between particles [ m ]
        //float rest_distance = pow(VP, 1.f/3.f);     //rest distance between particles [ m ]
        float smoothing_distance = 2.0f * rest_distance;//interaction radius


        float4 dmin = domain.getBndMin();
        float4 dmax = domain.getBndMax();
        //printf("dmin: %f %f %f\n", dmin.x, dmin.y, dmin.z);
        //printf("dmax: %f %f %f\n", dmax.x, dmax.y, dmax.z);
        float domain_vol = (dmax.x - dmin.x) * (dmax.y - dmin.y) * (dmax.z - dmin.z);
        //printf("domain volume: %f\n", domain_vol);


        //ratio between particle radius in simulation coords and world coords
        float simulation_scale = pow(.5 * VP * max_num / domain_vol, 1.f/3.f); 
        //float simulation_scale = pow(VP * 16000/ domain_vol, 1.f/3.f); 

       
        SetSetting("Maximum Number of Particles", max_num);
        SetSetting("Mass", mass);
        SetSetting("Rest Distance", rest_distance);
        SetSetting("Smoothing Distance", smoothing_distance);
        SetSetting("Simulation Scale", simulation_scale);


        float boundary_distance = .5f * rest_distance;
        SetSetting("Boundary Distance", boundary_distance);
        float spacing = rest_distance/ simulation_scale;
        SetSetting("Spacing", spacing);
 

        float pi = M_PI;
        float h9 = pow(smoothing_distance, 9.f);
        float h6 = pow(smoothing_distance, 6.f);
        float h3 = pow(smoothing_distance, 3.f);
        //Kernel Coefficients
        SetSetting("wpoly6", 315.f/64.0f/pi/h9 );
        SetSetting("wpoly6_d", -945.f/(32.0f*pi*h9) );  //doesn't seem right
        SetSetting("wpoly6_dd", -945.f/(32.0f*pi*h9) ); // laplacian
        SetSetting("wspiky", 15.f/pi/h6 );
        SetSetting("wspiky_d", -45.f/(pi*h6) );
        SetSetting("wspiky_dd", 15./(2.*pi*h3) );
        SetSetting("wvisc", 15./(2.*pi*h3) );
        SetSetting("wvisc_d", 15./(2.*pi*h3) ); //doesn't seem right
        SetSetting("wvisc_dd", 45./(pi*h6) );



    }
    
    bool SPHSettings::has_changed()
    {
        return changed;
    }

    bool SPHSettings::updateSPHP(SPHParams& sphp)
    {
        if(!changed) return changed; //nothing changed

        //update all the members of the sphp struct
        //sphp.grid_min = this->GetSettingAs<float4>; //GetSettingAs doesn't support float4
        //sphp.grid_max;
        sphp.mass = GetSettingAs<float>("Mass");
        sphp.rest_distance = GetSettingAs<float>("Rest Distance");
        sphp.smoothing_distance = GetSettingAs<float>("Smoothing Distance");
        sphp.simulation_scale = GetSettingAs<float>("Simulation Scale");
        
        //dynamic params
        sphp.boundary_stiffness = GetSettingAs<float>("Boundary Stiffness");
        sphp.boundary_dampening = GetSettingAs<float>("Boundary Dampening");
        sphp.boundary_distance = GetSettingAs<float>("Boundary Distance");
        sphp.K = GetSettingAs<float>("Gas Constant");        //gas constant
        sphp.viscosity = GetSettingAs<float>("Viscosity");
        sphp.velocity_limit = GetSettingAs<float>("Velocity Limit");
        sphp.xsph_factor = GetSettingAs<float>("XSPH Factor");
        sphp.gravity = GetSettingAs<float>("Gravity"); // -9.8 m/sec^2
        sphp.friction_coef = GetSettingAs<float>("Friction");
        sphp.restitution_coef = GetSettingAs<float>("Restitution");

        //next 3 not used at the moment
        sphp.shear = GetSettingAs<float>("Shear");
        sphp.attraction = GetSettingAs<float>("Attraction");
        sphp.spring = GetSettingAs<float>("Spring");
        //sphp.surface_threshold;

        //constants
        sphp.EPSILON = GetSettingAs<float>("EPSILON");
        sphp.PI = GetSettingAs<float>("PI");       //delicious
        //Kernel Coefficients
        sphp.wpoly6_coef = GetSettingAs<float>("wpoly6");
        sphp.wpoly6_d_coef = GetSettingAs<float>("wpoly6_d");
        sphp.wpoly6_dd_coef = GetSettingAs<float>("wpoly6_dd"); // laplacian
        sphp.wspiky_coef = GetSettingAs<float>("wspiky");
        sphp.wspiky_d_coef = GetSettingAs<float>("wspiky_d");
        sphp.wspiky_dd_coef = GetSettingAs<float>("wspiky_dd");
        sphp.wvisc_coef = GetSettingAs<float>("wvisc");
        sphp.wvisc_d_coef = GetSettingAs<float>("wvisc_d");
        sphp.wvisc_dd_coef = GetSettingAs<float>("wvisc_dd");

        //CL parameters
        sphp.num = GetSettingAs<int>("Number of Particles");
        sphp.nb_vars = GetSettingAs<int>("Number of Variables"); // for combined variables (vars_sorted, etc.)
        sphp.choice = GetSettingAs<int>("Choice"); // which kind of calculation to invoke
        sphp.max_num = GetSettingAs<int>("Maximum Number of Particles");

/*
        //update the OpenCL buffer
        std::vector<SPHParams> vparams(0);
        vparams.push_back(sphp);
        cl_sphp.copyToDevice(vparams);
*/
        changed = false;
        return true;
    }


    void SPHSettings::printSettings()
    {
        printf("SPH Settings\n");
        typedef std::map <std::string, std::string> MapType;

        MapType::const_iterator end = settings.end();
        for(MapType::const_iterator it = settings.begin(); it != end; ++it)
        {
            printf("%s: %s\n", it->first.c_str(), it->second.c_str());
        }
    }

}

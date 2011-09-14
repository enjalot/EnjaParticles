/****************************************************************************************
* Real-Time Particle System - An OpenCL based Particle system developed to run on modern GPUs. Includes SPH fluid simulations.
* version 1.0, September 14th 2011
* 
* Copyright (C) 2011 Ian Johnson, Andrew Young, Gordon Erlebacher, Myrna Merced, Evan Bollig
* 
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* 
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
****************************************************************************************/


//#include <OUTERSettings.h>
#include <OUTER.h>

namespace rtps
{


    void OUTER::calculate()
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


        float4 dmin = grid->getBndMin();
        float4 dmax = grid->getBndMax();
        //printf("dmin: %f %f %f\n", dmin.x, dmin.y, dmin.z);
        //printf("dmax: %f %f %f\n", dmax.x, dmax.y, dmax.z);
        float domain_vol = (dmax.x - dmin.x) * (dmax.y - dmin.y) * (dmax.z - dmin.z);
        //printf("domain volume: %f\n", domain_vol);


        //ratio between particle radius in simulation coords and world coords
        float simulation_scale = pow(.5f * VP * max_num / domain_vol, 1.f/3.f); 
        //float simulation_scale = pow(VP * 16000/ domain_vol, 1.f/3.f); 

       
        settings->SetSetting("Maximum Number of Particles", max_num);
        settings->SetSetting("Mass", mass);
        settings->SetSetting("Rest Distance", rest_distance);
        settings->SetSetting("Smoothing Distance", smoothing_distance);
        settings->SetSetting("Simulation Scale", simulation_scale);


        //float boundary_distance = .5f * rest_distance;
        float boundary_distance =  smoothing_distance;
        settings->SetSetting("Boundary Distance", boundary_distance);
        float spacing = rest_distance/ simulation_scale;
        settings->SetSetting("Spacing", spacing);
 

        float pi = M_PI;
        float h9 = pow(smoothing_distance, 9.f);
        float h6 = pow(smoothing_distance, 6.f);
        float h3 = pow(smoothing_distance, 3.f);
        //Kernel Coefficients
        settings->SetSetting("wpoly6", 315.f/64.0f/pi/h9 );
        settings->SetSetting("wpoly6_d", -945.f/(32.0f*pi*h9) );  //doesn't seem right
        settings->SetSetting("wpoly6_dd", -945.f/(32.0f*pi*h9) ); // laplacian
        settings->SetSetting("wspiky", 15.f/pi/h6 );
        settings->SetSetting("wspiky_d", -45.f/(pi*h6) );
        settings->SetSetting("wspiky_dd", 15./(2.*pi*h3) );
        settings->SetSetting("wvisc", 15./(2.*pi*h3) );
        settings->SetSetting("wvisc_d", 15./(2.*pi*h3) ); //doesn't seem right
        settings->SetSetting("wvisc_dd", 45./(pi*h6) );

        //dynamic params
        if(!settings->Exists("Gravity"))
            settings->SetSetting("Gravity", -9.8f); // -9.8 m/sec^2
        settings->SetSetting("Gas Constant", 15.0f);
        settings->SetSetting("Viscosity", .01f);
        settings->SetSetting("Velocity Limit", 600.0f);
        settings->SetSetting("XOUTER Factor", .1f);
        settings->SetSetting("Friction Kinetic", 0.0f);
        settings->SetSetting("Friction Static", 0.0f);
        settings->SetSetting("Boundary Stiffness", 20000.0f);
        settings->SetSetting("Boundary Dampening", 256.0f);


        //next 4 not used at the moment
        settings->SetSetting("Restitution", 0.0f);
        settings->SetSetting("Shear", 0.0f);
        settings->SetSetting("Attraction", 0.0f);
        settings->SetSetting("Spring", 0.0f);

        //constants
        settings->SetSetting("EPSILON", 1E-6);
        settings->SetSetting("PI", M_PI);       //delicious

        //CL parameters
        settings->SetSetting("Number of Particles", 0);
        settings->SetSetting("Number of Variables", 10); // for combined variables (vars_sorted, etc.) //TO be depracated
        settings->SetSetting("Choice", 0); // which kind of calculation to invoke //TO be depracated


    }
   
    void OUTER::updateOUTERP()
    {

        //update all the members of the sphp struct
        //sphp.grid_min = this->settings->GetSettingAs<float4>; //settings->GetSettingAs doesn't support float4
        //sphp.grid_max;
        sphp.mass = settings->GetSettingAs<float>("Mass");
        sphp.rest_distance = settings->GetSettingAs<float>("Rest Distance");
        sphp.smoothing_distance = settings->GetSettingAs<float>("Smoothing Distance");
        sphp.simulation_scale = settings->GetSettingAs<float>("Simulation Scale");
        
        //dynamic params
        sphp.boundary_stiffness = settings->GetSettingAs<float>("Boundary Stiffness");
        sphp.boundary_dampening = settings->GetSettingAs<float>("Boundary Dampening");
        sphp.boundary_distance = settings->GetSettingAs<float>("Boundary Distance");
        sphp.K = settings->GetSettingAs<float>("Gas Constant");        //gas constant
        sphp.viscosity = settings->GetSettingAs<float>("Viscosity");
        sphp.velocity_limit = settings->GetSettingAs<float>("Velocity Limit");
        sphp.xsph_factor = settings->GetSettingAs<float>("XOUTER Factor");
        sphp.gravity = settings->GetSettingAs<float>("Gravity"); // -9.8 m/sec^2
        sphp.friction_coef = settings->GetSettingAs<float>("Friction");
        sphp.restitution_coef = settings->GetSettingAs<float>("Restitution");

        //next 3 not used at the moment
        sphp.shear = settings->GetSettingAs<float>("Shear");
        sphp.attraction = settings->GetSettingAs<float>("Attraction");
        sphp.spring = settings->GetSettingAs<float>("Spring");
        //sphp.surface_threshold;

        //constants
        sphp.EPSILON = settings->GetSettingAs<float>("EPSILON");
        sphp.PI = settings->GetSettingAs<float>("PI");       //delicious
        //Kernel Coefficients
        sphp.wpoly6_coef = settings->GetSettingAs<float>("wpoly6");
        sphp.wpoly6_d_coef = settings->GetSettingAs<float>("wpoly6_d");
        sphp.wpoly6_dd_coef = settings->GetSettingAs<float>("wpoly6_dd"); // laplacian
        sphp.wspiky_coef = settings->GetSettingAs<float>("wspiky");
        sphp.wspiky_d_coef = settings->GetSettingAs<float>("wspiky_d");
        sphp.wspiky_dd_coef = settings->GetSettingAs<float>("wspiky_dd");
        sphp.wvisc_coef = settings->GetSettingAs<float>("wvisc");
        sphp.wvisc_d_coef = settings->GetSettingAs<float>("wvisc_d");
        sphp.wvisc_dd_coef = settings->GetSettingAs<float>("wvisc_dd");

        //CL parameters
        sphp.num = settings->GetSettingAs<int>("Number of Particles");
        sphp.nb_vars = settings->GetSettingAs<int>("Number of Variables"); // for combined variables (vars_sorted, etc.)
        sphp.choice = settings->GetSettingAs<int>("Choice"); // which kind of calculation to invoke
        sphp.max_num = settings->GetSettingAs<int>("Maximum Number of Particles");

        //update the OpenCL buffer
        std::vector<OUTERParams> vparams(0);
        vparams.push_back(sphp);
        cl_sphp.copyToDevice(vparams);

        settings->updated();
    }


}

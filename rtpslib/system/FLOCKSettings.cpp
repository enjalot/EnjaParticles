#include <FLOCK.h>

namespace rtps{

    void FLOCK::calculate(){
        
        // SETTINGS
        float rest_distance = .05f;
    
        //messing with smoothing distance, making it really small to remove interaction still results in weird force values
        float smoothing_distance = 2.0f * rest_distance;

        float4 dmin = grid->getBndMin();
        float4 dmax = grid->getBndMax();
        float domain_vol = (dmax.x - dmin.x) * (dmax.y - dmin.y) * (dmax.z - dmin.z);
        float VP = 2 * .0262144 / max_num;              //Particle Volume [ m^3 ]
        
        float simulation_scale = pow(.5f * VP * max_num / domain_vol, 1.f/3.f) * 5.f; 
        // must be less than smoothing_distance
        float spacing = rest_distance/ simulation_scale;
    
        // SIMULATION SETTINGS
        settings->SetSetting("Rest Distance", rest_distance);
        settings->SetSetting("Smoothing Distance", smoothing_distance);
        settings->SetSetting("Simulation Scale", simulation_scale);

        // SPACING
        settings->SetSetting("Spacing", spacing);

        // BOID SETTINGS
        settings->SetSetting("Min Separation Distance", 1.f);
        settings->SetSetting("Searching Radius", 1.f);
        settings->SetSetting("Max Speed", 5.f);
        settings->SetSetting("Angular Velocity", 0.f);

        // BOID WEIGHTS
        settings->SetSetting("Separation Weight", 1.50f);
        settings->SetSetting("Alignment Weight", 0.75f);
        settings->SetSetting("Cohesion Weight", 0.5f);
        settings->SetSetting("Goal Weight", 0.f);
        settings->SetSetting("Avoid Weight", 0.f);
        settings->SetSetting("LeaderFollowing Weight", 0.f);
       
        // BOID RULE'S SETTINGS 
        settings->SetSetting("Slowing Distance", 0.025f);
        
        settings->SetSetting("Maximum Number of Particles", max_num);
        settings->SetSetting("Number of Particles", 0);
    }

    void FLOCK::updateFLOCKP(){
        
        // CL SETTINGS
        flock_params.max_num = settings->GetSettingAs<int>("Maximum Number of Particles");
        flock_params.num = settings->GetSettingAs<int>("Number of Particles");

        // SIMULATION SETTINGS
        flock_params.rest_distance = settings->GetSettingAs<float>("Rest Distance");
        flock_params.smoothing_distance = settings->GetSettingAs<float>("Smoothing Distance");
        flock_params.simulation_scale = settings->GetSettingAs<float>("Simulation Scale");

        // SPACING
        spacing = settings->GetSettingAs<float>("Spacing");

        // BOID SETTINGS
        flock_params.min_dist = 0.5f * flock_params.smoothing_distance * settings->GetSettingAs<float>("Min Separation Distance");
        flock_params.search_radius = 0.8f * flock_params.smoothing_distance * settings->GetSettingAs<float>("Searching Radius");
        flock_params.max_speed = settings->GetSettingAs<float>("Max Speed");
        flock_params.ang_vel = settings->GetSettingAs<float>("Angular Velocity");
        // BOID WEIGHTS
        flock_params.w_sep = settings->GetSettingAs<float>("Separation Weight");
        flock_params.w_align = settings->GetSettingAs<float>("Alignment Weight");
        flock_params.w_coh = settings->GetSettingAs<float>("Cohesion Weight");
        flock_params.w_goal = settings->GetSettingAs<float>("Goal Weight");
        flock_params.w_avoid = settings->GetSettingAs<float>("Avoid Weight");
        flock_params.w_leadfoll = settings->GetSettingAs<float>("LeaderFollowing Weight");
        
        // BOID RULE'S SETTINGS 
        flock_params.slowing_distance= settings->GetSettingAs<float>("Slowing Distance");

        // update the OpenCL buffer
        std::vector<FLOCKParameters> vparams(0);
        vparams.push_back(flock_params);
        cl_FLOCKParameters.copyToDevice(vparams);

        settings->updated();
    }

}

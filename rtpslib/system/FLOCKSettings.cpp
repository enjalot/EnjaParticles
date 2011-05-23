#include <FLOCK.h>

namespace rtps{

    void FLOCK::calculate(){
        
        // SETTINGS
        float rest_distance = .05f;
    
        //messing with smoothing distance, making it really small to remove interaction still results in weird force values
        float smoothing_distance = 2.0f * rest_distance;

        // float simulation_scale = pow(particle_vol * max_num / domain_vol, 1./3.); 
        float simulation_scale = 1.0f;

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

        // BOID WEIGHTS
        settings->SetSetting("Separation Weight", 1.50f);
        settings->SetSetting("Alignment Weight", 0.75f);
        settings->SetSetting("Cohesion Weight", 0.5f);
        settings->SetSetting("LeaderFollowing Weight", 0.f);
       
        // BOID RULE'S SETTINGS 
        settings->SetSetting("Slowing Distance", 0.025f);
        
        
        settings->SetSetting("Number of Particles", 0);
    }

    void FLOCK::updateFLOCKP(){
        
        // CL SETTINGS
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

        // BOID WEIGHTS
        flock_params.w_sep = settings->GetSettingAs<float>("Separation Weight");
        flock_params.w_align = settings->GetSettingAs<float>("Alignment Weight");
        flock_params.w_coh = settings->GetSettingAs<float>("Cohesion Weight");
        flock_params.w_leadfoll = settings->GetSettingAs<float>("LeaderFollowing Weight");
        
        // BOID RULE'S SETTINGS 
        flock_params.slowing_distance= settings->GetSettingAs<float>("Slowing Distance");

    //mymese debbug
#if 0    
    printf("***\ninside FLOCKSettings\n***\n"); 
    printf("smoth_dist: %f\n", flock_params.smoothing_distance);
    printf("radius: %f\n", flock_params.search_radius);
    printf("min dist: %f \n", flock_params.min_dist);
#endif
        // update the OpenCL buffer
        std::vector<FLOCKParameters> vparams(0);
        vparams.push_back(flock_params);
        cl_FLOCKParameters.copyToDevice(vparams);

        settings->updated();
    }

}
//#endif

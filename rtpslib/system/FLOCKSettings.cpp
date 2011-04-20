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
    
        // PARAMETERS 
        float grid_min = grid.getBndMin();
        float grid_max = grid.getBndMax();

        // SET THE SETTINGS
        settings->SetSetting("Maximum Number of Particles", max_num);
        
        // CL SETTINGS
        settings->SetSetting("Number of Particles", 0);
        settings->SetSetting("Number of Variables", 10);
        settings->SetSetting("Choice", 0);

        // SIMULATION SETTINGS
        settings->SetSetting("Rest Distance", rest_distance);
        settings->SetSetting("Smoothing Distance", smoothing_distance);
        settings->SetSetting("Simulation Scale", simulation_scale);

        // SPACING
        settings->SetSetting("Spacing", spacing);

        // GRID SIZE
        settings->SetSetting("Grid Min", grid_min);
        settings->SetSetting("Grid Max", grid_max);

        // BOID SETTINGS
        if(!settings->Exist("Min Separation Distance"))
            settings->SetSetting("Min Separation Distance", 1.f);
        if(!settings->Exist("Searching Radius"))
            settings->SetSetting("Searching Radius", 1.f);
        if(!settings->Exist("Max Speed"))
            settings->SetSetting("Max Speed", 5.f);

        // BOID WEIGHTS
        if(!settings->Exist("Separation Weight"))
            settings->SetSetting("Separation Weight", 1.50f);
        if(!settings->Exist("Alignment Weight"))
            settings->SetSetting("Alignment Weight", 0.75f);
        if(!settings->Exist("Cohesion Weight"))
            settings->SetSetting("Cohesion Weight", 0.5f);
    }

    void FLOCK::updateFLOCKP(){
        
        // SET THE SETTINGS
        flock_params.max_num = settigs->GetSettingAs<int>("Maximum Number of Particles");
        
        // CL SETTINGS
        flock_parms.num = settings->GetSettingAs<int>("Number of Particles");
        flock_params.nb_vars = settings->GetSettingAs<int>("Number of Variables");
        flock_params.choice = settings->GetSettingAs<int>("Choice");

        // SIMULATION SETTINGS
        flock_params.rest_distance = settings->GetSettingAs<float>("Rest Distance");
        flock_params.smoothing_distance = settings->GetSettingAs<float>("Smoothing Distance");
        flock_params.simulation_scale = settings->GetSettingAs<float>("Simulation Scale");

        // SPACING
        flock_params.spacing = settings->GetSettingAs<float>("Spacing");

        // GRID SIZE
        flock_params.grid_min = settings->GetSettingAs<float>("Grid Min");
        flock_params.grid_max = settings->GetSettingAs<float>("Grid Max");

        // BOID SETTINGS
        flock_params.min_dist = 0.5f * flock_params.smoothing_distance * settings->GetSettingAs<float>("Min Separation Distance");
        flock_params.search_radius = 0.8f * flock_params.smoothing_distance * settings->GetSettingAs<float>("Searching Radius");
        flock_params.max_speed = settings->GetSettingAs<float>("Max Speed");

        // BOID WEIGHTS
        flock_params.w_sep = settings->GetSettingAs<float>("Separation Weight");
        flock_params.w_algn = settings->GetSettingAs<float>("Alignment Weight");
        flock_params.w_coh = settings->GetSettingAs<float>("Cohesion Weight");
    
        // update the OpenCL buffer
        std::vector<FLOCKParameters> vparams(0);
        vparams.push_back(flock_params);
        cl_FLOCKParams.copyToDevice(vparams);

        settings->updated();
    }

}
#endif

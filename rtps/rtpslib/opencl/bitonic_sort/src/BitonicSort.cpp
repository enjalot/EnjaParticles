


namespace rtps

{

Bitonic::Bitonic()
{
    
}

void Bitonic::loadBitonic(int bitonic_type)
{

    std::string path(SIMPLE_CL_SOURCE_DIR);
    path += "/euler_cl.cl";
    k_euler = Kernel(cli, path, "bitonic");

    k_euler.setArg(0, cl_position.getDevicePtr());
    k_euler.setArg(1, cl_velocity.getDevicePtr());
    k_euler.setArg(2, cl_force.getDevicePtr());
    k_euler.setArg(3, cl_color.getDevicePtr());
    k_euler.setArg(4, ps->settings.dt); //time step


}

void Bitonic::Sort()
{


    scopy(num, cl_sort_output_hashes.getDevicePtr(), 
	             cl_sort_hashes.getDevicePtr());
	scopy(num, cl_sort_output_indices.getDevicePtr(), 
	             cl_sort_indices.getDevicePtr());
    

}

}

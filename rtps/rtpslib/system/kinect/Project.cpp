#include "../Kinect.h"
namespace rtps
{

void Kinect::loadProject()
{
    printf("create projection kernel\n");
    std::string path(KINECT_CL_SOURCE_DIR);
	path = path + "/project_cl.cl";
    k_project = Kernel(ps->cli, path, "project");

    printf("kernel made, set args\n");
    int args = 0;
    k_project.setArg(args++, cl_kinect.getDevicePtr()); 
    k_project.setArg(args++, cl_kinect_col.getDevicePtr());
	k_project.setArg(args++, cl_kinect_depth.getDevicePtr());
	k_project.setArg(args++, cl_kinect_rgb.getDevicePtr());
	k_project.setArg(args++, cl_pt.getDevicePtr());
	k_project.setArg(args++, cl_ipt.getDevicePtr());
    k_project.setArg(args++, 640);
	k_project.setArg(args++, 480);


}
void Kinect::projection()
{

	int ctaSize = 128; // work group size
    
	k_project.execute(640*480, ctaSize);
	
    ps->cli->queue.finish();
}

}

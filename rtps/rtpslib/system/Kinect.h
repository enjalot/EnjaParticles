#ifndef RTPS_KINECT_H_INCLUDED
#define RTPS_KINECT_H_INCLUDED

#include <string>

#include "../RTPS.h"
#include "System.h"
#include "ForceField.h"
#include "../opencl/Kernel.h"
#include "../opencl/Buffer.h"
//#include "../util.h"

#include <ntk/camera/kinect_grabber.h>
#include <ntk/camera/rgbd_processor.h>
#include <ntk/camera/calibration.h>
#include <ntk/utils/opencv_utils.h>
//#include <ntk/utils/eigen_utils.h>
//#include <ntk/utils/arg.h>
#include <ntk/geometry/pose_3d.h>
#include <Eigen/Core>
#include <Eigen/Geometry>


namespace rtps {


class Kinect : public System
{
public:
    Kinect(RTPS *ps, int num);
    ~Kinect();

    void update();
    void setupTimers();
    void printTimers();

    bool forcefields_enabled;
    int max_forcefields;

    //the particle system framework
    RTPS *ps;

    std::vector<float4> positions;
    std::vector<float4> colors;
    std::vector<float4> velocities;
    std::vector<float4> forces;
    std::vector<ForceField> forcefields;

    std::vector<float4> kinect_data;
    std::vector<float4> kinect_col;
    std::vector<float> kinect_depth;
    std::vector<uchar> kinect_rgb;




    Kernel k_forcefield;
    Kernel k_euler;
    Kernel k_project;

    Buffer<float4> cl_position;
    Buffer<float4> cl_color;
    Buffer<float4> cl_force;
    Buffer<float4> cl_velocity;
    Buffer<ForceField> cl_forcefield;

    Buffer<float4> cl_kinect;
    Buffer<float4> cl_kinect_col;
    Buffer<float> cl_kinect_depth;
    Buffer<uchar> cl_kinect_rgb;
    Buffer<float> cl_pt; //projection transforms
    Buffer<float> cl_ipt;
    

    void loadForceField();
    void loadForceFields(std::vector<ForceField> ff);
    void loadEuler();
    void loadProject();

    void projection();

    void cpuForceField();
    void cpuEuler();

    //kinect
    
    ntk::RGBDCalibration calibration;
    ntk::KinectGrabber grabber;
    ntk::RGBDProcessor processor;
    ntk::RGBDImage current_frame;
    cv::Mat3b mapped_color;

    //get what we need to do this part on the GPU
    cv::Mat1f pt; //project transform
    cv::Mat1f ipt; //inverse project transform

 
    enum {TI_KINECT=0, TI_KINECT_GPU, TI_KINECT_LOOP
          }; //2
    GE::Time* timers[3];

    
};

}

#endif

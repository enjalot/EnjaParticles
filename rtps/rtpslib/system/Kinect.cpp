#include <stdio.h>

#include <GL/glew.h>

#include "System.h"
#include "Kinect.h"


using namespace ntk;
using namespace cv;

namespace rtps {


Kinect::Kinect(RTPS *psfr, int n)
{
    max_num = n;
    num = max_num;
    //store the particle system framework
    ps = psfr;
    grid = ps->settings.grid;
    forcefields_enabled = true;
    max_forcefields = 100;

    printf("num: %d\n", num);
    positions.resize(max_num);
    colors.resize(max_num);
    forces.resize(max_num);
    velocities.resize(max_num);
    //forcefields.resize(max_forcefields);

    float4 min = grid.getBndMin();
    float4 max = grid.getBndMax();

    float spacing = .1; 
    //std::vector<float4> box = addRect(num, min, max, spacing, 1);
    //std::copy(box.begin(), box.end(), positions.begin());


    float4 center = float4(1,1,.1,1);
    float4 center2 = float4(2,2,.1,1);
    float4 center3 = float4(1,2,.1,1);

    forcefields.push_back( ForceField(center2, .5,.1) );
    forcefields.push_back( ForceField(center, .5, .1) );
    forcefields.push_back( ForceField(center3, .5, .1) );

    //forcefields.push_back( ForceField(center, 1., 20, 0, 0) );
        //forcefields.push_back( ForceField() );
    //forcefields.push_back( ForceField() );


    //std::fill(positions.begin(), positions.end(), float4(0.0f, 0.0f, 0.0f, 1.0f));
    std::fill(colors.begin(), colors.end(),float4(1.0f, 0.0f, 0.0f, 0.0f));
    std::fill(forces.begin(), forces.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    std::fill(velocities.begin(), velocities.end(),float4(0.0f, 0.0f, 0.0f, 0.0f));
    
    managed = true;
    pos_vbo = createVBO(&positions[0], positions.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("pos vbo: %d\n", pos_vbo);
    col_vbo = createVBO(&colors[0], colors.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("col vbo: %d\n", col_vbo);

#if GPU
    //vbo buffers
    printf("making cl_buffers\n");
    cl_position = Buffer<float4>(ps->cli, pos_vbo);
    cl_color = Buffer<float4>(ps->cli, col_vbo);
    printf("done with cl_buffers\n");
    //pure opencl buffers
    cl_force = Buffer<float4>(ps->cli, forces);
    cl_velocity = Buffer<float4>(ps->cli, velocities);;

    //could generalize this to other integration methods later (leap frog, RK4)
    printf("create euler kernel\n");
    loadEuler();
    

    printf("load forcefiels");
    loadForceField();   
    loadForceFields(forcefields);




    std::string calibstr(DATA_DIR);
    calibstr += "kinect_calibration.yml";
    printf("%s\n", calibstr.c_str());
    calibration.loadFromFile(calibstr.c_str());

    grabber.initialize();
    // Tell the grabber that we have calibration data.
    grabber.setCalibrationData(calibration);
    grabber.start();
    printf("grabber started\n");

    // New opencv window
    //namedWindow("color");
    //namedWindow("depth");
    //namedWindow("mapped_color");

    // Tell the processor to transform raw depth into meters using linear coefficients.
    processor.setFilterFlag(RGBDProcessor::ComputeKinectDepthBaseline, true);
    processor.setFilterFlag(RGBDProcessor::FilterMedian, true);
    processor.setFilterFlag(RGBDProcessor::NoAmplitudeIntensityUndistort, true);

   
    kinect_data.resize(640*480);
    kinect_col.resize(640*480);
    std::fill(kinect_col.begin(), kinect_col.end(),float4(0.0f, 0.0f, 0.0f, 1.0f));
    kin_vbo = createVBO(&kinect_data[0], kinect_data.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    kin_col_vbo = createVBO(&kinect_col[0], kinect_col.size()*sizeof(float4), GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
    printf("kinect vbo: %d\n", kin_vbo);
    printf("kinect color vbo: %d\n", kin_col_vbo);
    cl_kinect = Buffer<float4>(ps->cli, kin_vbo);
    cl_kinect_col = Buffer<float4>(ps->cli, kin_col_vbo);
 
    
    setupTimers();


#endif

}

Kinect::~Kinect()
{
    if(pos_vbo && managed)
    {
        glBindBuffer(1, pos_vbo);
        glDeleteBuffers(1, (GLuint*)&pos_vbo);
        pos_vbo = 0;
    }
    if(col_vbo && managed)
    {
        glBindBuffer(1, col_vbo);
        glDeleteBuffers(1, (GLuint*)&col_vbo);
        col_vbo = 0;
    }
}

void Kinect::update()
{
#ifdef CPU

    cpuForceField();
    //printf("calling cpuEuler\n");
    cpuEuler();

    //printf("pushing positions to gpu\n");
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float4), &positions[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, col_vbo);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float4), &colors[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glFinish();

    //printf("done pushing to gpu\n");


#endif
#ifdef GPU

    //call kernels
    //add timings
    glFinish();
    cl_position.acquire();
    cl_color.acquire();
   
    //k_forcefield.execute(num, 128);
    k_forcefield.execute(num);
    k_euler.execute(num);

    cl_position.release();
    cl_color.release();
#endif

    timers[TI_KINECT]->start();

    //while (true)
    //{
    //printf("wait for next frame\n");
    grabber.waitForNextFrame();
    grabber.copyImageTo(current_frame);
    processor.processImage(current_frame);

    //printf("before assert\n");
    ntk_assert(current_frame.calibration(), "Ensure there is calibration data in the image");
    //printf("after assert\n");
    cv::Size cfs = current_frame.depth().size();
    mapped_color.create(current_frame.depth().size());
    mapped_color = Vec3b(0,0,0);

    //printf("curr fame size %d %d ", cfs.width, cfs.height);
    //std::vector<float4> kinect_data(cfs.width * cfs.height);

    timers[TI_KINECT_LOOP]->start();
    // equivalent to for(int r = 0; r < im.rows; ++r) for (int c = 0; c < im.cols; ++c)
    for_all_rc(current_frame.depth())
    {
      float depth = current_frame.depth()(r,c);

      // Check for invalid depth.
      if (depth < 1e-5)
        continue;

      // Point in depth image.
      Point3f p_depth (c,r,depth);

      // Point in 3D metric space
      Point3f p3d;
      p3d = current_frame.calibration()->depth_pose->unprojectFromImage(p_depth);
      // Debug output: show p3d if the global debug level is >= 1
       //ntk_dbg_print(p3d, 1);
       //printf(" %f %f %f\n", p3d.x, p3d.y, p3d.z);
       //printf("index: %d\n", c+ r*cfs.width);
       kinect_data[r + c*cfs.height] = float4(p3d.x, p3d.y, p3d.z, 1.0f);

      // Point in color image
      Point3f p_rgb;
      p_rgb = current_frame.calibration()->rgb_pose->projectToImage(p3d);
      int r_rgb = p_rgb.y;
      int c_rgb = p_rgb.x;
      // Check if the pixel coordinates are valid and set the value.
      if (is_yx_in_range(current_frame.rgb(), r_rgb, c_rgb))
      {
        Vec3b mcol = current_frame.rgb()(r_rgb, c_rgb);
        mapped_color(r, c) = mcol;
        kinect_col[r_rgb + c_rgb*cfs.height] = float4(mcol[2]/255.f, mcol[1]/255.f, mcol[0]/255.f, 1.0f);
        //kinect_col[r + c*cfs.height] = float4(.5, .5, .5, 1.0f);
        //printf("color! %d %d %d\n", mcol[0], mcol[1], mcol[2]);
      }
    }

    timers[TI_KINECT_LOOP]->end();
    timers[TI_KINECT]->end();
    timers[TI_KINECT_GPU]->start();
   
    cl_kinect.acquire();
    cl_kinect_col.acquire();
   
    cl_kinect.copyToDevice(kinect_data);
    cl_kinect_col.copyToDevice(kinect_col);
   
    cl_kinect.release();
    cl_kinect_col.release();

    timers[TI_KINECT_GPU]->end();
    /*
    int fps = grabber.frameRate();
    cv::putText(current_frame.rgbRef(),
                cv::format("%d fps", fps),
                Point(10,20), 0, 0.5, Scalar(255,0,0,255));
    */
    // Display the image
    //imshow("color", current_frame.rgb());
   



}

void Kinect::setupTimers()
{
    //int print_freq = 20000;
    int print_freq = 1000; //one second
    int time_offset = 5;

    timers[TI_KINECT]     = new GE::Time("kinect frame grab", time_offset, print_freq);
    timers[TI_KINECT_GPU]     = new GE::Time("kinect gpu push", time_offset, print_freq);
    timers[TI_KINECT_LOOP]     = new GE::Time("kinect pixel loop", time_offset, print_freq);
 

}

void Kinect::printTimers()
{
    for(int i = 0; i < 3; i++) //switch to vector of timers and use size()
    {
        timers[i]->print();
    }
}

}

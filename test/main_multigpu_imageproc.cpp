

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
//#include <utils.h>
//#include <string.h>
//#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>


#include <GL/glew.h>
#if defined __APPLE__ || defined(MACOSX)
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
//OpenCL stuff
#endif
#include <RTPS.h>
#include "../rtpslib/render/util/stb_image.h"
#include "../rtpslib/render/util/stb_image_write.h"
//#include <omp.h>

using namespace rtps;

//int num_gpus = 1;
const int NDRANGE = 1;

int window_width = 640;
int window_height = 480;
int glutWindowHandle = 0;

int hindex; 

CL* cli;

const string cl_image_manip = "__kernel void negative(read_only image2d_t in_img, write_only image2d_t out_img)\n"
                           "{\n"
                           "    int xInd = get_global_id(0);\n"
                           "    int yInd = get_global_id(1);\n"
                           "    sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
                           "    uint4 col = read_imageui(in_img,smp, (int2)(xInd,yInd));\n"
                           "    uint4 newCol;\n"
                           "    newCol.xyz = (uint3)(255,255,255)-col.xyz;\n"
                           "    newCol.w = 255;\n"
                           "    write_imageui(out_img,(int2)(xInd,yInd), newCol);\n"
                           "}\n"
                           "__kernel void average(read_only image2d_t in_img, write_only image2d_t out_img, __constant int* filterwidth)\n"
                           "{\n"
                           "    int xInd = get_global_id(0);\n"
                           "    int yInd = get_global_id(1);\n"
                           "    sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
                           "    uint4 sum = (uint4)(0,0,0,0);\n"
                           "    for(int i = -*filterwidth; i<=*filterwidth; i++)\n"
                           "    {\n"
                           "        for(int j = -*filterwidth; j<*filterwidth; j++)\n"
                           "        {\n" 
                           "            sum += read_imageui(in_img,smp, (int2)(xInd+i,yInd+j));\n"
                           "        }\n" 
                           "    }\n"
                           "    sum.xyz=sum.xyz/(4*(*filterwidth)*(*filterwidth));\n"
                           "    sum.w=255;\n"
                           "    write_imageui(out_img,(int2)(xInd,yInd), sum);\n"
                           "}\n"
                           "__kernel void filter(read_only image2d_t in_img, write_only image2d_t out_img, __constant int* filterwidth, __constant float* filter)\n"
                           "{\n"
                           "    int xInd = get_global_id(0);\n"
                           "    int yInd = get_global_id(1);\n"
                           "    sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
                           "    int4 sum = (int4)(0,0,0,0);\n"
                           "    int offset = floor(*filterwidth/2.);\n"
                           "    for(int i = 0; i<=*filterwidth; i++)\n"
                           "    {\n"
                           "        for(int j = 0; j<*filterwidth; j++)\n"
                           "        {\n" 
                           "            sum.xyz += filter[i+(j*(*filterwidth))] * read_imagei(in_img,smp, (int2)(xInd+i-offset,yInd+j-offset)).xyz;\n"
                           "        }\n" 
                           "    }\n"
                           "    //sum.xyz=sum.xyz/(4*(*filterwidth)*(*filterwidth));\n"
                           "    //sum/= (*filterwidth) * (*filterwidth);\n"
                           "    sum.w=255;\n"
                           "    sum = clamp(sum,(int4)(0,0,0,0),(int4)(255,255,255,255));\n"
                           "    write_imageui(out_img,(int2)(xInd,yInd), abs(sum));\n"
                           "}\n"
                           "__kernel void sobel(read_only image2d_t in_img, write_only image2d_t out_img)\n"
                           "{\n"
                           "    int xInd = get_global_id(0);\n"
                           "    int yInd = get_global_id(1);\n"
                           "    sampler_t smp = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"
                           "    int4 sumx = (int4)(0,0,0,0);\n"
                           "    int4 sumy = (int4)(0,0,0,0);\n"
                           "    sumx.xyz += read_imagei(in_img,smp, (int2)(xInd+1,yInd-1)).xyz;\n"
                           "    sumy.xyz += read_imagei(in_img,smp, (int2)(xInd+1,yInd-1)).xyz;\n"
                           "    sumx.xyz +=2 * read_imagei(in_img,smp, (int2)(xInd+1,yInd)).xyz;\n"
                           "    sumy.xyz +=2 * read_imagei(in_img,smp, (int2)(xInd,yInd-1)).xyz;\n"
                           "    sumx.xyz += read_imagei(in_img,smp, (int2)(xInd+1,yInd+1)).xyz;\n"
                           "    sumy.xyz += read_imagei(in_img,smp, (int2)(xInd-1,yInd-1)).xyz;\n"
                           "    sumx.xyz -= read_imagei(in_img,smp, (int2)(xInd+1,yInd-1)).xyz;\n"
                           "    sumy.xyz -= read_imagei(in_img,smp, (int2)(xInd-1,yInd+1)).xyz;\n"
                           "    sumx.xyz -=2 * read_imagei(in_img,smp, (int2)(xInd+1,yInd)).xyz;\n"
                           "    sumy.xyz -=2 * read_imagei(in_img,smp, (int2)(xInd,yInd+1)).xyz;\n"
                           "    sumx.xyz -= read_imagei(in_img,smp, (int2)(xInd+1,yInd+1)).xyz;\n"
                           "    sumy.xyz -= read_imagei(in_img,smp, (int2)(xInd+1,yInd+1)).xyz;\n"
                           "    uint4 sum = convert_uint4(sqrt(convert_float4((sumx*sumx)+(sumy*sumy))));\n"
//                           "    int4 sum = ceil(g);\n"
                           "    sum.w=255;\n"
//                           "    sum = clamp(sum,(int4)(0,0,0,0),(int4)(255,255,255,255));\n"
                           "    write_imageui(out_img,(int2)(xInd,yInd), abs(sum));\n"
                           "}\n";


enum kern_name
{
KERNEL_NEGATIVE,
KERNEL_AVERAGE,
KERNEL_SOBEL,
KERNEL_FILTER
};
//timers
//GE::Time *ts[3];

void printFloatVector(vector<float> vec)
{
    printf("[");
    for(int i = 0; i<vec.size(); i++)
    {
        printf("%f,",vec[i]);
    }
    printf("\b]\n");
}

class CLProfiler
{
public:
    void addEvent(const char* name, int device_num, int num_dev, cl::Event& event)
    {
        cl_ulong start, end, queued, submit;
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &queued);
        event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &submit);
        double timing = (end - start) * 1.0e-6; 
        stringstream s;
        s<<name<< " # "<<device_num+1<<" of "<<num_dev;
        cl_timing& tmp = timings[s.str()]; 
        tmp.end_time = end * 1.0e-6;
        tmp.start_time = start * 1.0e-6;
        tmp.queue_time = queued * 1.0e-6;
        tmp.submit_time = submit * 1.0e-6;
        if(timing>tmp.max_time)
            tmp.max_time = timing;
        if(timing<tmp.min_time)
            tmp.min_time = timing;
        tmp.total_time += timing;
        tmp.num_times += 1;
    }
    void printAll()
    {
        for (map<string, cl_timing>::iterator i = timings.begin();
              i!=timings.end(); i++)
        {
            cout<<i->first<<":"<<endl;
            /*cout<<"\tMinimum Time:\t\t"<<*min_element(i->second.begin(),i->second.end())<<endl;
            cout<<"\tMaximum Time:\t\t"<<*max_element(i->second.begin(),i->second.end())<<endl;
            float total = 0.0;
            for(vector<float>::iterator j = i->second.begin(); j!=i->second.end(); j++)
                total+=*j;
            cout<<"\tAverage Time:\t\t"<<total/i->second.size()<<endl;
            cout<<"\tTotal Time:\t\t"<<total<<endl;
            cout<<"\tCount:\t\t"<<i->second.size()<<"\n"<<endl;**/
            cout<<setprecision(15)<<fixed;
            cout<<"\tSubmit Time:\t\t"<<i->second.submit_time<<endl;
            cout<<"\tQueue Time:\t\t"<<i->second.queue_time<<endl;
            cout<<"\tStart Time:\t\t"<<i->second.start_time<<endl;
            cout<<"\tEnd Time:\t\t"<<i->second.end_time<<endl;
            cout<<"\tMinimum Time:\t\t"<<i->second.min_time<<endl;
            cout<<"\tMaximum Time:\t\t"<<i->second.max_time<<endl;
            cout<<"\tAverage Time:\t\t"<<i->second.total_time/i->second.num_times<<endl;
            cout<<"\tTotal Time:\t\t"<<i->second.total_time<<endl;
            cout<<"\tCount:\t\t"<<i->second.num_times<<"\n"<<endl;
        }
    }
private:
    struct cl_timing
    {
        cl_timing()
        {
            min_time = numeric_limits<double>::max();
            max_time = numeric_limits<double>::min();
            total_time = start_time = end_time = queue_time = submit_time = 0.0;
            num_times = 0;
        }
        double min_time;
        double max_time;
        double total_time;
        double queue_time;
        double submit_time;
        double start_time;
        double end_time;
        int num_times;
    };
    map<string, cl_timing > timings;
};
void printEventInfo(const char* name, int device_num, cl::Event& event,map<string,vector<float> >& )
{
    
}
//================
//#include "materials_lights.h"

//----------------------------------------------------------------------
float rand_float(float mn, float mx)
{
    float r = rand() / (float) RAND_MAX;
    return mn + (mx-mn)*r;
}
//----------------------------------------------------------------------

//----------------------------------------------------------------------
int main(int argc, char** argv)
{

    //initialize glut
/*    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH
		//|GLUT_STEREO //if you want stereo you must uncomment this.
		);
    glutInitWindowSize(window_width, window_height);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);


    glutWindowHandle = glutCreateWindow("vector addition");*/
    //omp_set_num_threads(2);

    int num_runs = 10;
    int filterwidth = 3;
    //Sobel X filter.
    //float filter[]= {-1,0,1,-2, 0, 2, -1,0,1};
    //Sobel Y filter.
    float filter[]= {-1,-2,-1,0,0, 0, 1, 2, 1};
    
    kern_name kern = KERNEL_SOBEL;
    
    CLProfiler prof;

    //Argument 1 sets the number of times to run
    if(argc>1)
    {
        num_runs = atoi(argv[1]);
    }

    printf("Number of times to run %d\n",num_runs);

    int x,y,real_comp;

    string imgFile ="../sprites/atlantisapproach_nasa_4288.jpg";
    if(argc>2)
    {
        imgFile=argv[2];
    }
    int comp = 3;
    stbi_uc* img = stbi_load(imgFile.c_str(),&x,&y,&real_comp,comp);
    printf("img x = %d, y = %d, real_comp = %d\n",x,y,real_comp);
    
    for(int i =3;i<x*y*comp;i+=comp)
    {
        //set alpha channel to full. (This is arbitrary but necessary because of my lack of
        //understanding of image formats)
        img[i]=255;
    }

    size_t imgSize = sizeof(stbi_uc)*x*y*comp;

    stbi_uc* img_out = new stbi_uc[imgSize];

    cl::ImageFormat fmt;

    
    //RGB is strange with the requirements. I currently force 4 channels arbitrarily
    if(comp == 3)
    {
        fmt.image_channel_order = CL_RGB; 
        fmt.image_channel_data_type = CL_UNORM_INT_101010;
    }
    else if(comp==4)
    {
        fmt.image_channel_order = CL_RGBA;
        fmt.image_channel_data_type = CL_UNSIGNED_INT8;
    }
    else
    {
        printf("Image format is incorrect! Must be RGB or RGBA.\n");
    }

    cli = new CL();

    cl::Program::Sources src;
    src.push_back(pair<const char*, ::size_t>(cl_image_manip.c_str(), cl_image_manip.length()));

    vector<cl::Kernel> kernels;
    for(int i = 0; i<cli->devices.size();i++)
    {
        cl::Program prog(cli->context_vec[i],src);
        vector<cl::Device> dev;
        dev.push_back(cli->devices[i]);
        try
        {
            prog.build(dev);
        } 
        catch (cl::Error er)
        {
            printf("loadProgram::program.build\n");
            printf("source= %s\n", cl_image_manip.c_str());
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
        std::cout << "Build Status: " << prog.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cli->devices[i]) << std::endl;
        std::cout << "Build Options:\t" << prog.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cli->devices[i]) << std::endl;
        std::cout << "Build Log:\t " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cli->devices[i]) << std::endl;
    
        if(kern==KERNEL_NEGATIVE)
            kernels.push_back(cl::Kernel(prog,"negative"));
        else if(kern==KERNEL_AVERAGE)
            kernels.push_back(cl::Kernel(prog,"average"));
        else if(kern==KERNEL_FILTER)
            kernels.push_back(cl::Kernel(prog,"filter"));
        else if(kern==KERNEL_SOBEL)
            kernels.push_back(cl::Kernel(prog,"sobel"));
    }

    //Create timers to keep up with execution/memory transfer times.
    int num_timers = 3;

    EB::TimerList timers;
    timers["vect_add_cpu"] = new EB::Timer("Adding vectors on single CPU", 0);
    const char* timer_name_temp[] = {"buffer_write_%d_%ss","vect_add_%d_%ss","buffer_read_%d_%ss"};
    const char* timer_desc_temp[] = {"Writing buffers to %d %ss","manipulating img on %d %ss","Reading buffers from %d %ss"};

    char timer_name[num_timers*cli->devices.size()][256];
    for(int num_gpus = 1; num_gpus<=cli->devices.size(); num_gpus++)
    {
        for(int i = 0;i<num_timers; i++)
        {
            char timer_desc[256];
            sprintf(timer_name[i+(num_gpus-1)*num_timers],timer_name_temp[i],num_gpus,"GPU");
            sprintf(timer_desc,timer_desc_temp[i],num_gpus,"GPU");
            timers[timer_name[i+(num_gpus-1)*num_timers]] = new EB::Timer(timer_desc,0);
        }
    }

    //run simulation num_runs times to make sure we get a good statistical sampleing of run times.
    for(int j = 0; j<num_runs; j++)
    {
        int i;
        cout<<"run: "<<j+1<<" of "<< num_runs<<endl;
        
        //Run on 1 gpu, 2 gpus, ..., n gpus where n is the number of devices belonging to the cl context.
        for(int num_gpus = 1; num_gpus<=cli->devices.size(); num_gpus++)
        {
            int timer_num = 3*(num_gpus-1);
            cl::Event event_img_write[num_gpus],event_img_read[num_gpus],event_execute[num_gpus];

            //Create Buffers for each gpu.
            cl::Image2D cl_img_in[num_gpus];
            cl::Image2D cl_img_out[num_gpus];
            //if(kern == KERNEL_AVERAGE || kern == KERNEL_FILTER)
                cl::Buffer cl_filterwidth[num_gpus];
            //if(kern == KERNEL_FILTER)
                cl::Buffer cl_filter[num_gpus];

            //Set size and buffer properties for each of the buffer. Divide by num_gpus to evenly distribute
            //data accross them
            timers[timer_name[timer_num]]->start();
            #pragma omp parallel for private(i)// schedule(static,1)
            for(i = 0; i<num_gpus; i++)
            {
                try
                {
                    //can't assign an event even though underneath it will enqueue a command to write to GPU.
                    cl_img_in[i] = cl::Image2D(cli->context_vec[i],CL_MEM_READ_ONLY,fmt,x,y/num_gpus,0,NULL,NULL);
                    cl_img_out[i] = cl::Image2D(cli->context_vec[i],CL_MEM_WRITE_ONLY,fmt,x,y/num_gpus,0,NULL,NULL);
                    if(kern==KERNEL_AVERAGE || kern== KERNEL_FILTER)
                        cl_filterwidth[i] = cl::Buffer(cli->context_vec[i],CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(int),&filterwidth,NULL);
                    if(kern == KERNEL_FILTER)
                        cl_filter[i] = cl::Buffer(cli->context_vec[i],CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*filterwidth*filterwidth,&filter,NULL);
                    
                    cl::size_t<3> origin;
                    origin[0]=0;                   
                    origin[1]=0;
                    //origin[1]=i*(y/num_gpus);//+(i?1:0);                   
                    origin[2]=0;                   
//printf("origin (%d,%d,%d)\n",origin[0],origin[1],origin[2]);
                                                   
                    cl::size_t<3> region;              
                    region[0] = x;                 
                    //region[1] = (i+1)*(y/num_gpus);
                    region[1] = (y/num_gpus);
                    region[2] = 1;                 
//printf("region (%d,%d,%d)\n",region[0],region[1],region[2]);

                    cli->err = cli->queue[i].enqueueWriteImage(cl_img_in[i],CL_FALSE,origin,region,0,0,(void*)&img[(x*i*comp*(y/num_gpus))],NULL,&event_img_write[i]);
                    cli->queue[i].flush();
                    cli->queue[i].finish();
                }
                catch (cl::Error er)
                {
                    printf("j = %d, num_gpus = %d, i = %d\n",j,num_gpus,i);
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
            timers[timer_name[timer_num]]->stop();

            //Transfer our host buffers to each GPU then wait for it to finish before executing the kernel.
            //set the kernel arguments
            for(i=0;i<num_gpus; i++)
            {
                kernels[i].setArg(0,cl_img_in[i]);
                kernels[i].setArg(1,cl_img_out[i]);
                if(kern==KERNEL_AVERAGE || kern==KERNEL_FILTER)
                {
                    kernels[i].setArg(2,cl_filterwidth[i]);
                }
                if(kern==KERNEL_FILTER)
                    kernels[i].setArg(3,cl_filter[i]);
            }

            //Set the kernel arguments vec a,b,c and enqueue kernel.
            timers[timer_name[timer_num+1]]->start();
            #pragma omp parallel for private(i)//, schedule(static,1)
            for(i = 0; i<num_gpus; i++)
            {
                try
                {
                    cli->err = cli->queue[i].enqueueNDRangeKernel(kernels[i],cl::NullRange,  cl::NDRange(x,y/num_gpus),cl::NullRange , NULL, &event_execute[i]);
                }
                catch (cl::Error er)
                {
                    printf("j = %d, num_gpus = %d, i = %d\n",j,num_gpus,i);
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
            #pragma omp parallel for private(i)
            for(i = 0; i<num_gpus; i++)
            {
		        cli->queue[i].flush();
                cli->queue[i].finish();
            }
            timers[timer_name[timer_num+1]]->stop();
        
            timers[timer_name[timer_num+2]]->start();
            #pragma omp parallel for private(i)// schedule(static,1)
            for(i = 0; i<num_gpus; i++)
            {
                cl::Event event;
                try
                {
                    cl::size_t<3> origin;
                    origin[0]=0;                   
                    origin[1]=0;
                    //origin[1]=i*(y/num_gpus)+(i?1:0);                   
                    origin[2]=0;                   
//printf("origin (%d,%d,%d)\n",origin[0],origin[1],origin[2]);
                                                   
                    cl::size_t<3> region;              
                    region[0] = x;                 
                    region[1] = (y/num_gpus);
                    //region[1] = (i+1)*(y/num_gpus);
                    region[2] = 1;                 
//printf("region (%d,%d,%d)\n",region[0],region[1],region[2]);

                    cli->err = cli->queue[i].enqueueReadImage(cl_img_out[i],CL_FALSE,origin,region,0,0,(void*)&img_out[(x*i*comp*(y/num_gpus))/*+(i?2:0)*/],NULL,&event_img_read[i]);
                    cli->queue[i].flush();
                    cli->queue[i].finish();
                }
                catch (cl::Error er)
                {
                    printf("i = %d, num_gpus = %d\n",i,num_gpus);
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
            timers[timer_name[timer_num+2]]->stop();


            for(i=0;i<num_gpus;i++)
            {
                prof.addEvent("Image write on GPU",i,num_gpus,event_img_write[i]);
                prof.addEvent("Negative calculation on GPU",i,num_gpus,event_execute[i]);
                prof.addEvent("Image read on GPU",i,num_gpus,event_img_read[i]);
            }
        }
    }

    //stbi_write_jpg("outputfile.jpg",x,y,comp,img_out);
    stbi_write_png("outputfile.png",x,y,comp,img_out,0);

    timers.printAll();
    prof.printAll();

    delete cli;
    delete[] img;
    delete[] img_out;
    return 0;
}

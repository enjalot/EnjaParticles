
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
//#include <utils.h>
//#include <string.h>
//#include <string>
#include <sstream>
#include <iomanip>

#include <RTPS.h>

using namespace rtps;

int VECTOR_SIZE = 10000000;
//int NUM_GPUS = 1;
const int NDRANGE = 1;


int hindex; 

CL* cli;

const string cl_vect_add = "__kernel void vect_add(__global float* a, __global float* b, global float* c)"
                           "{"
                           "    int index = get_global_id(0);"
                           "    c[index] = a[index] + b[index];"
                           "}";

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

    if(argc>1)
    {
        VECTOR_SIZE = atoi(argv[1]);
    }

    printf("Vector size is %d\n",VECTOR_SIZE);
    EB::TimerList timers;
    timers["vect_add_cpu"] = new EB::Timer("Adding vectors on single CPU", 0);
    
    const int num_timers = 3;
    const char* timer_name_temp[] = {"buffer_write_%d_%ss","vect_add_%d_%ss","buffer_read_%d_%ss"};
    const char* timer_desc_temp[] = {"Writing buffers to %d %ss","Adding vectors on %d %ss","Reading buffers from %d %ss"};
    

    cli = new CL();

    vector<float> a_h(VECTOR_SIZE);
    vector<float> b_h(VECTOR_SIZE);
    vector<float> c_h(VECTOR_SIZE);

    for(int i = 0; i<VECTOR_SIZE; i++)
    {
        a_h[i]=b_h[i]=i;
    }

    for(int NUM_GPUS = 1; NUM_GPUS<=cli->devices.size(); NUM_GPUS++)
    {
        char timer_name[num_timers][256];
        for(int i = 0;i<num_timers; i++)
        {
            char timer_desc[256];
            sprintf(timer_name[i],timer_name_temp[i],NUM_GPUS,"GPU");
            sprintf(timer_desc,timer_desc_temp[i],NUM_GPUS,"GPU");
            timers[timer_name[i]] = new EB::Timer(timer_desc,0);
        }
        try
        {
            cl::Buffer a_d[NUM_GPUS];
            cl::Buffer b_d[NUM_GPUS];
            cl::Buffer c_d[NUM_GPUS];
            for(int i = 0; i<NUM_GPUS; i++)
            {
                a_d[i] = cl::Buffer(cli->context,CL_MEM_READ_ONLY,(a_h.size()/NUM_GPUS)*sizeof(float));
                b_d[i] = cl::Buffer(cli->context,CL_MEM_READ_ONLY,(b_h.size()/NUM_GPUS)*sizeof(float));
                c_d[i] = cl::Buffer(cli->context,CL_MEM_WRITE_ONLY,(c_h.size()/NUM_GPUS)*sizeof(float));
            }
    
            //printFloatVector(a_h);
            //printFloatVector(b_h);
            cl::Event event;
            timers[timer_name[0]]->start();
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cli->err = cli->queue[i].enqueueWriteBuffer(a_d[i], CL_FALSE, 0, (a_h.size()/NUM_GPUS)*sizeof(float), &a_h[i*(a_h.size()/NUM_GPUS)], NULL, &event);
                cli->err = cli->queue[i].enqueueWriteBuffer(b_d[i], CL_FALSE, 0/*i*(b_h.size()/NUM_GPUS)*sizeof(float)*/, (b_h.size()/NUM_GPUS)*sizeof(float), &b_h[i*(b_h.size()/NUM_GPUS)], NULL, &event);
            }
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cli->queue[i].finish();
            }
            timers[timer_name[0]]->stop();
    
            cl::Program::Sources src;
            src.push_back(pair<const char*, ::size_t>(cl_vect_add.c_str(), cl_vect_add.length()));
            cl::Program prog(cli->context,src);
            prog.build(cli->devices);
            cl::Kernel kernel(prog,"vect_add");
            
            timers[timer_name[1]]->start();
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cl_ulong start, end;
                float timing = -1.0f;
        
                kernel.setArg(0,a_d[i]);
                kernel.setArg(1,b_d[i]);
                kernel.setArg(2,c_d[i]);
                try
                {
                    cli->err = cli->queue[i].enqueueNDRangeKernel(kernel,cl::NullRange,  cl::NDRange(NDRANGE), cl::NullRange, NULL, &event);
                }
                catch (cl::Error er)
                {
                    printf("err: work group size: %d\n", NDRANGE);
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cli->queue[i].finish();
            }
            timers[timer_name[1]]->stop();
        
            timers[timer_name[2]]->start();
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cli->err = cli->queue[i].enqueueReadBuffer(c_d[i], CL_FALSE, 0/*i*(c_h.size()/NUM_GPUS)*sizeof(float)*/, (c_h.size()/NUM_GPUS)*sizeof(float), &c_h[i*(c_h.size()/NUM_GPUS)], NULL, &event);
            }
            for(int i = 0; i<NUM_GPUS; i++)
            {
                cli->queue[i].finish();
            }
            timers[timer_name[2]]->stop();
    
            //printFloatVector(c_h);
            //initialize the OpenGL scene for rendering
        }
        catch (cl::Error er)
        {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    for(int NUM_CPUS = 1; NUM_CPUS<=cli->cpu_devices.size(); NUM_CPUS++)
    {
        char timer_name[num_timers][256];
        for(int i = 0;i<num_timers; i++)
        {
            char timer_desc[256];
            sprintf(timer_name[i],timer_name_temp[i],NUM_CPUS,"CPU");
            sprintf(timer_desc,timer_desc_temp[i],NUM_CPUS,"CPU");
            timers[timer_name[i]] = new EB::Timer(timer_desc,0);
        }
        try
        {
            cl::Buffer a_d[NUM_CPUS];
            cl::Buffer b_d[NUM_CPUS];
            cl::Buffer c_d[NUM_CPUS];
            for(int i = 0; i<NUM_CPUS; i++)
            {
                a_d[i] = cl::Buffer(cli->cpu_context,CL_MEM_READ_ONLY,(a_h.size()/NUM_CPUS)*sizeof(float));
                b_d[i] = cl::Buffer(cli->cpu_context,CL_MEM_READ_ONLY,(b_h.size()/NUM_CPUS)*sizeof(float));
                c_d[i] = cl::Buffer(cli->cpu_context,CL_MEM_WRITE_ONLY,(c_h.size()/NUM_CPUS)*sizeof(float));
            }
    
            //printFloatVector(a_h);
            //printFloatVector(b_h);
            cl::Event event;
            timers[timer_name[0]]->start();
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cli->err = cli->cpu_queue[i].enqueueWriteBuffer(a_d[i], CL_FALSE, 0, (a_h.size()/NUM_CPUS)*sizeof(float), &a_h[i*(a_h.size()/NUM_CPUS)], NULL, &event);
                cli->err = cli->cpu_queue[i].enqueueWriteBuffer(b_d[i], CL_FALSE, 0/*i*(b_h.size()/NUM_CPUS)*sizeof(float)*/, (b_h.size()/NUM_CPUS)*sizeof(float), &b_h[i*(b_h.size()/NUM_CPUS)], NULL, &event);
            }
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cli->cpu_queue[i].finish();
            }
            timers[timer_name[0]]->stop();
    
            cl::Program::Sources src;
            src.push_back(pair<const char*, ::size_t>(cl_vect_add.c_str(), cl_vect_add.length()));
            cl::Program prog(cli->cpu_context,src);
            prog.build(cli->cpu_devices);
            cl::Kernel kernel(prog,"vect_add");
            
            timers[timer_name[1]]->start();
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cl_ulong start, end;
                float timing = -1.0f;
        
                kernel.setArg(0,a_d[i]);
                kernel.setArg(1,b_d[i]);
                kernel.setArg(2,c_d[i]);
                try
                {
                    cli->err = cli->cpu_queue[i].enqueueNDRangeKernel(kernel,cl::NullRange,  cl::NDRange(NDRANGE), cl::NullRange, NULL, &event);
                }
                catch (cl::Error er)
                {
                    printf("err: work group size: %d\n", NDRANGE);
                    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
                }
            }
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cli->cpu_queue[i].finish();
            }
            timers[timer_name[1]]->stop();
        
            timers[timer_name[2]]->start();
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cli->err = cli->cpu_queue[i].enqueueReadBuffer(c_d[i], CL_FALSE, 0/*i*(c_h.size()/NUM_CPUS)*sizeof(float)*/, (c_h.size()/NUM_CPUS)*sizeof(float), &c_h[i*(c_h.size()/NUM_CPUS)], NULL, &event);
            }
            for(int i = 0; i<NUM_CPUS; i++)
            {
                cli->cpu_queue[i].finish();
            }
            timers[timer_name[2]]->stop();
    
            //printFloatVector(c_h);
            //initialize the OpenGL scene for rendering
        }
        catch (cl::Error er)
        {
            printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
        }
    }

    timers["vect_add_cpu"]->start();
    for(int i = 0; i<a_h.size(); i++)
    {
        c_h[i]=a_h[i]+b_h[i];
    }
    timers["vect_add_cpu"]->stop();

    timers.printAll();

    delete cli;
    return 0;
}


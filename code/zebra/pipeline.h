#ifndef PIPELINE_H
#define PIPELINE_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class Pipeline 
{
    public:
    void initialise( int platform, int device );
    int add_blackwhite( int w, int h, int comp, int platform, int device );
    int add_whitepoint( int w, int h, int comp, int platform, int device, int r, int g, int b );
    int add_gaussian( int w, int h, int comp, int platform, int device, int kernel_size );
    int set_image( unsigned char* img_original, unsigned char* result, int w, int h, int comp );
    int run(unsigned char* img_original, unsigned char* result, int w, int h, int comp, int platform, int device, int kernel_size);

    private:
    cl_program load_program( QString file_name, cl_context context, cl_device_id device );
    cl_command_queue create_command_queue( cl_context context, cl_device_id device );
    cl_context createContext();
    double createFilter(double *gKernel, int size);

    cl_context context;
    cl_program program;
    cl_command_queue command_queue;

    cl_mem gray_cl;
    cl_mem img_cl;

    std::vector<cl_kernel> kernels;

};
#endif

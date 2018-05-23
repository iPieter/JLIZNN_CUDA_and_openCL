#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>
#include <Qfile>
#include <QLatin1Literal>
#include <QTextStream>

#include "pipeline.h"

cl_program Pipeline::load_program( QString file_name, cl_context context, cl_device_id device )
{
    cl_int err;

    QFile qfile( file_name );
    if (!qfile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout << "Invalid file" << std::endl;
        exit(-1);
    }

    QTextStream in(&qfile);

    std::string file;
    while (!in.atEnd()) {
        file += in.readLine().toUtf8().constData();
        file += "\n";
    }

    const char *f = file.c_str();

    cl_program program = clCreateProgramWithSource( context, 1, (const char **)(&f), nullptr, &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Error creating program: " << err << std::endl;
    }

    err = clBuildProgram( program, 0, NULL, NULL, NULL, NULL );

    if( err != CL_SUCCESS )
    {
        std::cout << "Error building program: " << err << std::endl;
    
        if (err == CL_BUILD_PROGRAM_FAILURE) 
        {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

            // Allocate memory for the log
            char *log = (char *) malloc(log_size);

            // Get the log
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

            // Print the log
            printf("%s\n", log);

            exit(-1);
        }
    }

    std::cout << "Created program" << std::endl;

    return program;
}

cl_command_queue  Pipeline::create_command_queue( cl_context context, cl_device_id device )
{
    char* value;
    size_t valueSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);
    free(value);

    cl_int err;
    cl_command_queue result = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Error creating command queue: " << err << std::endl;
    }
    
    std::cout << "Created command queue" << std::endl;

    return result;
}

cl_context Pipeline::createContext()
{
    cl_int errNum = 0;
    cl_uint numPlatforms = 0;
    cl_platform_id firstPlatformId = 0;
    cl_context context = NULL;

    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (CL_SUCCESS != errNum || numPlatforms == 0)
    {
        fprintf(stderr, "Failed to find any OpenCL platforms\n");
        return NULL;
    }

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };

    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &errNum);

    if (CL_SUCCESS != errNum)
    {
     fprintf(stderr, "Failed to create an OpenCL GPU or CPU context\n");
            return NULL;
        
    }

    return context;
}

double Pipeline::createFilter(double *gKernel, int size)
{
   //generate pascals triangle
   double *row = new double[size];

   size--;

   row[0] = 1.0; //First element is always 1
   double sum = 1.0;
   for(int i=1; i<size/2+1; i++){ //Progress up, until reaching the middle value
       row[i] = row[i-1] * (size-i+1)/i;
       sum += row[i];
   }
   for(int i=size/2+1; i<=size; i++){ //Copy the inverse of the first part
       row[i] = row[size-i];
       sum += row[i];
   }

   size++;

   // generate kernel
   for (int x = 0; x < size; ++x)
       for (int y = 0; y < size; ++y)
           gKernel[x + y * size] = row[x] * row[y];

   delete [ ] row;
   row = NULL;

   return sum;
}

int Pipeline::add_gaussian( int w, int h, int comp, int platform, int device, int kernel_size )
{
    int err  = 0;
    cl_kernel blur_kernel = clCreateKernel( program, "kernel_test", &err );
    kernels.push_back( blur_kernel );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't create blur kernel" << err << std::endl;
    }

    const int size = w*h;

    double *mask = new double[kernel_size * kernel_size];
    double sf = createFilter( mask, kernel_size );
    sf *= sf;

    cl_mem mask_cl = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(double) * kernel_size * kernel_size, (void *)mask, NULL );

    err |= clSetKernelArg( blur_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( blur_kernel, 1, sizeof(cl_mem), &img_cl );
    err |= clSetKernelArg( blur_kernel, 2, sizeof(int), (const void *)&size );
    err |= clSetKernelArg( blur_kernel, 3, sizeof(int), (const void *)&w );
    err |= clSetKernelArg( blur_kernel, 4, sizeof(cl_mem), &mask_cl );
    err |= clSetKernelArg( blur_kernel, 5, sizeof(int), (const void *)&kernel_size );
    err |= clSetKernelArg( blur_kernel, 6, sizeof(double), (const void *)&sf );
    err |= clSetKernelArg( blur_kernel, 7, sizeof(double), (const void *)&comp );

    return err;
}


int Pipeline::set_image( unsigned char* img_original, unsigned char* result, int w, int h, int comp )
{
    int err = 0;
    gray_cl = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned char) * w * h * comp, img_original, NULL );
    img_cl = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(unsigned char) * w * h * comp, result, NULL );
    
    std::cout << "Setting buffer args" << std::endl;

    return err;
}

void Pipeline::initialise( int platform, int device )
{
    cl_uint platform_id_count = 0;
    clGetPlatformIDs( 0, nullptr, &platform_id_count );
    std::vector<cl_platform_id> platform_ids(platform_id_count);
    clGetPlatformIDs( platform_id_count, platform_ids.data (), nullptr );

    cl_uint device_id_count = 0;
    clGetDeviceIDs( platform_ids [0], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_id_count );
    std::vector<cl_device_id> device_ids( device_id_count );
    clGetDeviceIDs( platform_ids [0], CL_DEVICE_TYPE_ALL, device_id_count, device_ids.data (), nullptr );


    context = createContext();

    program = load_program(":/test.cl", context, device_ids[platform] );
    command_queue = create_command_queue( context, device_ids[device] );
}

int Pipeline::run(unsigned char* img_original, unsigned char* result, int w, int h, int comp, int platform, int device, int kernel_size)
{
   
    if( !img_original )
    {
        return -1;
    }

    int err = 0;
    cl_kernel gray_kernel = clCreateKernel( program, "gray", &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't create gray kernel" << err << std::endl;
    }

    cl_kernel red_kernel = clCreateKernel( program, "red", &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't create red kernel" << err << std::endl;
    }


    //float *img = (float *)malloc( w * h * comp * sizeof(unsigned char));
    //memcpy( img, img_original, w * h * comp );

    std::cout << "Creating buffers" << std::endl;

    std::cout<< "Size: " << w * h * comp << std::endl;

    std::cout << "Setting buffer args" << std::endl;

    err |= clSetKernelArg( gray_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( gray_kernel, 1, sizeof(cl_mem), &img_cl );

    err |= clSetKernelArg( red_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( red_kernel, 1, sizeof(cl_mem), &img_cl );


    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't set kernel args" << std::endl;
    }

    std::cout << "Enqueuing kernel" << std::endl;

    //size_t globalWorkSize[] = { (size_t)h, (size_t)w };
    size_t globalWorkSize[] = { (size_t)(h/8)*8, (size_t)(w/8)*8 };
    //size_t localWorkSize[] = {8, 8};

    //timing
    cl_event event;

    //err = clEnqueueNDRangeKernel( command_queue, gray_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
    //err = clEnqueueNDRangeKernel( command_queue, red_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
    for ( auto kernel : kernels )
        err = clEnqueueNDRangeKernel( command_queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, &event );
   
    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't enqueue buffer:" << err << std::endl;
    }

    std::cout << "Waiting for queue to finish" << std::endl;

    clWaitForEvents(1, &event);

    err = clFinish( command_queue );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't finish command queue: " << err << std::endl;     
    }

    std::cout << "Reading buffer" << std::endl;
    
    err = clEnqueueReadBuffer( command_queue, img_cl, CL_TRUE, 0, w * h * comp * sizeof(unsigned char), result, 0, NULL, NULL );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't read buffer: " << err << std::endl;
    }

    std::cout << "Writing image" << std::endl;

    //printf( "%d \n", stbi_write_png("output.png", w, h, comp, (void *)img, 0 ) );

    clReleaseContext( context );
    clReleaseCommandQueue( command_queue );
    clReleaseMemObject( gray_cl );
    clReleaseKernel( gray_kernel );
    clReleaseProgram( program );

    //free( img );
    cl_ulong time_submit;
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(time_submit), &time_submit, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double ns = time_end-time_start;
    double overhead = time_start-time_submit;
    printf("OpenCl Execution time is: %0.6f milliseconds, width %0.6f overhead \n",ns / 1000000.0, overhead / 1000000.0);


    return 0;
}

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

cl_program load_program( QString file_name, cl_context context, cl_device_id device )
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

cl_command_queue create_command_queue( cl_context context, cl_device_id device )
{
    char* value;
    size_t valueSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device: %s\n", value);
    free(value);

    cl_int err;
    cl_command_queue result = clCreateCommandQueue( context, device, 0, &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Error creating command queue: " << err << std::endl;
    }
    
    std::cout << "Created command queue" << std::endl;

    return result;
}

cl_context CreateContext()
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

int run(unsigned char* img_original, int w, int h, int comp)
{
    cl_uint platform_id_count = 0;
    clGetPlatformIDs( 0, nullptr, &platform_id_count );
    std::vector<cl_platform_id> platform_ids(platform_id_count);
    clGetPlatformIDs( platform_id_count, platform_ids.data (), nullptr );

    cl_uint device_id_count = 0;
    clGetDeviceIDs( platform_ids [0], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_id_count );
    std::vector<cl_device_id> device_ids( device_id_count );
    clGetDeviceIDs( platform_ids [0], CL_DEVICE_TYPE_ALL, device_id_count, device_ids.data (), nullptr );


    cl_context context = CreateContext();

    cl_program program = load_program(":/test.cl", context, device_ids[2] );
    cl_command_queue command_queue = create_command_queue( context, device_ids[2] );

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

    cl_kernel blur_kernel = clCreateKernel( program, "kernel_test", &err );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't create blur kernel" << err << std::endl;
    }

    float *img = (float *)malloc( w * h * comp * sizeof(unsigned char));
    memcpy( img, img_original, w * h * comp );

    std::cout << "Creating buffers" << std::endl;

    std::cout<< "Size: " << w * h * comp << std::endl;

    err = 0;
    cl_mem gray_cl = clCreateBuffer( context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned char) * w * h * comp, img_original, NULL );
    cl_mem img_cl = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(unsigned char) * w * h * comp, img, NULL );
    
    std::cout << "Setting buffer args" << std::endl;

    err |= clSetKernelArg( gray_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( gray_kernel, 1, sizeof(cl_mem), &img_cl );

    err |= clSetKernelArg( red_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( red_kernel, 1, sizeof(cl_mem), &img_cl );

    const int size = w*h;

    //const int KERNEL_OFFSET = 5;
    /*
    const int blur[25]= {
        1, 4, 6, 4, 1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4, 6, 4, 1
    };*/
    const int KERNEL_OFFSET = 13;
    const int blur[KERNEL_OFFSET * KERNEL_OFFSET] = {
        1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 
12, 144, 792, 2640, 5940, 9504, 11088, 9504, 5940, 2640, 792, 144, 12, 
66, 792, 4356, 14520, 32670, 52272, 60984, 52272, 32670, 14520, 4356, 792, 66, 
220, 2640, 14520, 48400, 108900, 174240, 203280, 174240, 108900, 48400, 14520, 2640, 220, 
495, 5940, 32670, 108900, 245025, 392040, 457380, 392040, 245025, 108900, 32670, 5940, 495, 
792, 9504, 52272, 174240, 392040, 627264, 731808, 627264, 392040, 174240, 52272, 9504, 792, 
924, 11088, 60984, 203280, 457380, 731808, 853776, 731808, 457380, 203280, 60984, 11088, 924, 
792, 9504, 52272, 174240, 392040, 627264, 731808, 627264, 392040, 174240, 52272, 9504, 792, 
495, 5940, 32670, 108900, 245025, 392040, 457380, 392040, 245025, 108900, 32670, 5940, 495, 
220, 2640, 14520, 48400, 108900, 174240, 203280, 174240, 108900, 48400, 14520, 2640, 220, 
66, 792, 4356, 14520, 32670, 52272, 60984, 52272, 32670, 14520, 4356, 792, 66, 
12, 144, 792, 2640, 5940, 9504, 11088, 9504, 5940, 2640, 792, 144, 12,  
1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1
    };



    cl_mem mask_cl = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * KERNEL_OFFSET * KERNEL_OFFSET, (void *)&blur, NULL );

    err |= clSetKernelArg( blur_kernel, 0, sizeof(cl_mem), &gray_cl );
    err |= clSetKernelArg( blur_kernel, 1, sizeof(cl_mem), &img_cl );
    err |= clSetKernelArg( blur_kernel, 2, sizeof(int), (const void *)&size );
    err |= clSetKernelArg( blur_kernel, 3, sizeof(int), (const void *)&w );
    err |= clSetKernelArg( blur_kernel, 4, sizeof(cl_mem), &mask_cl );
    err |= clSetKernelArg( blur_kernel, 5, sizeof(int), (const void *)&KERNEL_OFFSET );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't set kernel args" << std::endl;
    }

    std::cout << "Enqueuing kernel" << std::endl;

    size_t globalWorkSize[] = { (size_t)(h/8 + 1)*8, (size_t)(w/8 + 1)*8 };
    size_t localWorkSize[] = { 8, 8 };

    //err = clEnqueueNDRangeKernel( command_queue, gray_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
    //err = clEnqueueNDRangeKernel( command_queue, red_kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
    err = clEnqueueNDRangeKernel( command_queue, blur_kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't enqueue buffer:" << err << std::endl;
    }

    std::cout << "Waiting for queue to finish" << std::endl;

    err = clFinish( command_queue );

    if( err != CL_SUCCESS )
    {
        std::cout << "Couldn't finish command queue: " << err << std::endl;     
    }

    std::cout << "Reading buffer" << std::endl;
    
    err = clEnqueueReadBuffer( command_queue, img_cl, CL_TRUE, 0, w * h * comp * sizeof(unsigned char), img, 0, NULL, NULL );

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

    free( img );

    return 0;
}

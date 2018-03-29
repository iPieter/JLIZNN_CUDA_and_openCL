#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cuda.h>
#include <cuda_runtime.h>

#include "std_image.h"
#include "std_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 32

__global__ void to_gray( unsigned char *img, int array_size )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < array_size )
	{
		img[ i * 3 + 1 ] = img[ i * 3 ];
		img[ i * 3 + 2 ] = img[ i * 3 ];
	}
}

void gpu( unsigned char *img, int array_size )
{
	int n_blocks = array_size / BLOCK_SIZE + (array_size % BLOCK_SIZE == 0 ? 0 : 1);

	cudaEvent_t start, stop;
	float ms;

	cudaEventCreate(&stop);
	cudaEventCreate(&start);
	cudaEventRecord(start);

	unsigned char *a_d;

	cudaMalloc((void **) &a_d, array_size * sizeof(unsigned char));

	cudaMemcpy( a_d, img, array_size * sizeof(unsigned char), cudaMemcpyHostToDevice );

	cudaEvent_t swap_start, swap_stop;
	cudaEventCreate(&swap_start);
	cudaEventCreate(&swap_stop);

	cudaEventRecord(swap_start);
	to_gray <<< n_blocks, BLOCK_SIZE >>>( a_d, array_size / 3 );
	cudaEventRecord(swap_stop);
	cudaEventSynchronize(swap_stop);
	cudaEventElapsedTime( &ms, swap_start, swap_stop );
	printf("%f;", ms );

	cudaMemcpy( img, a_d, array_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );

	cudaFree(a_d);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &ms, start, stop );
	printf("%f\n", ms );
}

int main()
{
	int w;
	int h;
	int comp;
	unsigned char* image = stbi_load( "test.jpg", &w, &h, &comp, STBI_rgb );

	if(image == NULL)
	{
		printf("Couldn't load image");
		return 0;
	}

	printf( "Loaded img: %d, %d, %d \n", w, h, comp );

	clock_t start = clock();
	int i;
	for( i = 0; i < w * h * comp; i += 3)
	{
		image[i + 1] = image[i];
		image[i + 2] = image[i];
	}
	printf("CPU: %d\n", (clock() - start) *1000 / CLOCKS_PER_SEC );

	gpu( image, w * h * 3 );

	printf( "%d", stbi_write_bmp("output.bmp", w, h, comp, (void *)image ) );

	stbi_image_free(image);

	return 0;
}

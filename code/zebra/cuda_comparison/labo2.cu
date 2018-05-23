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

#define TEST_INDEX(i,size) ((i >= 0 && i < size) ? true : false)

__global__ void edge( unsigned char *img_original, unsigned char *img, int w, int h, int array_size )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int kernel_y[3][3] = {
		{ 1, 1, 1},
		{ 1, 1, 1},
		{ 1, 1, 1}
	};

	const int KERNEL_SIZE = 5;

	const int kernel_x[KERNEL_SIZE][KERNEL_SIZE] = {
		{ 1, 4, 6, 4, 1},
		{ 4, 16, 24, 16, 4},
		{ 6, 24, 36, 24, 6},
		{ 4, 16, 24, 16, 4},
		{ 1, 4, 6, 4, 1}
	};

	if( x + y * w < array_size * 3 )
	{
		int sum_r_x = 0;
		int sum_g_x = 0;
		int sum_b_x = 0;

		//int sum_r_y = 0;
		//int sum_g_y = 0;
		//int sum_b_y = 0;

		for( int i = 0; i < KERNEL_SIZE; i++ )
		{
			for( int j = 0; j < KERNEL_SIZE; j++ )
			{
				int index = (x + i - KERNEL_SIZE / 2 + (y + j - KERNEL_SIZE /2) * w) * 3;
				if( TEST_INDEX(index, array_size * 3))
				{
					sum_r_x += kernel_x[i][j] * img_original[index];
					sum_g_x += kernel_x[i][j] * img_original[index + 1];
					sum_b_x += kernel_x[i][j] * img_original[index + 2];

					//sum_r_y += kernel_y[i][j] * img_original[index];
					//sum_g_y += kernel_y[i][j] * img_original[index + 1];
					//sum_b_y += kernel_y[i][j] * img_original[index + 2];
				}
			}
		}

		//img[ (x + y * w) * 3 + 0 ] = (unsigned char)(sqrt((float)(sum_r_x * sum_r_x + sum_r_y * sum_r_y )));
		//img[ (x + y * w) * 3 + 1 ] = (unsigned char)(sqrt((float)(sum_g_x * sum_g_x + sum_g_y * sum_g_y )));
		//img[ (x + y * w) * 3 + 2 ] = (unsigned char)(sqrt((float)(sum_b_x * sum_b_x + sum_b_y * sum_b_y )));

		img[ (x + y * w) * 3 + 0 ] = (unsigned char)(sum_r_x / 256);
		img[ (x + y * w) * 3 + 1 ] = (unsigned char)(sum_g_x / 256);
		img[ (x + y * w) * 3 + 2 ] = (unsigned char)(sum_b_x / 256);
	}
}

void gpu( unsigned char *img, int w, int h, int array_size )
{
	dim3 threads_per_block(8,8);
	dim3 n_blocks( w / threads_per_block.x, h / threads_per_block.y);

	cudaEvent_t start, stop;
	float ms;

	cudaEventCreate(&stop);
	cudaEventCreate(&start);
	cudaEventRecord(start);

	unsigned char *a_d;
	cudaMalloc((void **) &a_d, array_size * sizeof(unsigned char));
	cudaMemcpy( a_d, img, array_size * sizeof(unsigned char), cudaMemcpyHostToDevice );

	unsigned char *a_c;
	cudaMalloc((void **) &a_c, array_size * sizeof(unsigned char));
	cudaMemcpy( a_c, img, array_size * sizeof(unsigned char), cudaMemcpyHostToDevice );

	cudaEvent_t swap_start, swap_stop;
	cudaEventCreate(&swap_start);
	cudaEventCreate(&swap_stop);

	cudaEventRecord(swap_start);
	edge <<< n_blocks, threads_per_block >>>( a_c, a_d, w, h, array_size / 3 );
	cudaEventRecord(swap_stop);
	cudaEventSynchronize(swap_stop);
	cudaEventElapsedTime( &ms, swap_start, swap_stop );
	printf("%f;", ms );

	cudaMemcpy( img, a_d, array_size * sizeof(unsigned char), cudaMemcpyDeviceToHost );

	cudaFree(a_d);
	cudaFree(a_c);

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
	unsigned char* image = stbi_load( "middel.jpg", &w, &h, &comp, STBI_rgb );

	unsigned char* img_original = (unsigned char *)malloc( sizeof(unsigned char) * w * h * comp);
	memcpy( img_original, image, w*h*3*sizeof(unsigned char) );

	if(image == NULL)
	{
		printf("Couldn't load image");
		return 0;
	}

	printf( "Loaded img: %d, %d, %d \n", w, h, comp );


	clock_t start = clock();

	const int kernel_e_y[3][3] = {
		{ -1, 0, 1},
		{ -1, 0, 1},
		{ -1, 0, 1}
	};

	const int kernel_e_x[3][3] = {
		{ -1, 0, 1},
		{ -2, 0, 2},
		{ -1, 0, 1}
	};

	const int KERNEL_SIZE = 5;

	const int kernel_x[KERNEL_SIZE][KERNEL_SIZE] = {
		{ 1, 4, 6, 4, 1},
		{ 4, 16, 24, 16, 4},
		{ 6, 24, 36, 24, 6},
		{ 4, 16, 24, 16, 4},
		{ 1, 4, 6, 4, 1}
	};

	for( int x = 0; x < w; x++ )
	{
		for( int y = 0; y < h; y++ )
		{
			if( x + y * w < w*h * 3 )
			{
				int sum_r_x = 0;
				int sum_g_x = 0;
				int sum_b_x = 0;

				int sum_r_y = 0;
				int sum_g_y = 0;
				int sum_b_y = 0;

				for( int i = 0; i < KERNEL_SIZE; i++ )
				{
					for( int j = 0; j < KERNEL_SIZE; j++ )
					{
						int index = (x + i - KERNEL_SIZE / 2 + (y + j - KERNEL_SIZE /2) * w) * 3;
						if( TEST_INDEX(index, w*h * 3))
						{
							sum_r_x += kernel_x[i][j] * img_original[index];
							sum_g_x += kernel_x[i][j] * img_original[index + 1];
							sum_b_x += kernel_x[i][j] * img_original[index + 2];

							//sum_r_y += kernel_y[i][j] * img_original[index];
							//sum_g_y += kernel_y[i][j] * img_original[index + 1];
							//sum_b_y += kernel_y[i][j] * img_original[index + 2];
						}
					}
				}

				//img[ (x + y * w) * 3 + 0 ] = (unsigned char)(sqrt((float)(sum_r_x * sum_r_x + sum_r_y * sum_r_y )));
				//img[ (x + y * w) * 3 + 1 ] = (unsigned char)(sqrt((float)(sum_g_x * sum_g_x + sum_g_y * sum_g_y )));
				//img[ (x + y * w) * 3 + 2 ] = (unsigned char)(sqrt((float)(sum_b_x * sum_b_x + sum_b_y * sum_b_y )));

				image[ (x + y * w) * 3 + 0 ] = (unsigned char)(sum_r_x / 256);
				image[ (x + y * w) * 3 + 1 ] = (unsigned char)(sum_g_x / 256);
				image[ (x + y * w) * 3 + 2 ] = (unsigned char)(sum_b_x / 256);
			}
		}
	}

	printf("CPU: %d\n", (clock() - start) *1000000 / CLOCKS_PER_SEC );


	gpu( img_original, w ,h, w * h * 3 );

	//printf( "%d", stbi_write_bmp("output.bmp", w, h, comp, (void *)image ) );
	printf( "%d", stbi_write_png("output.png", w, h, comp, (void *)img_original, 0 ) );

	stbi_image_free(image);

	free(img_original);

	return 0;
}

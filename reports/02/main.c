#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 4

__global__ void swap_gpu( int *a, int array_size )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if( i < array_size / 2 )
	{
		a[i] 					^= a[ array_size - i - 1];
		a[ array_size - i - 1]  ^= a[i];
		a[i] 					^= a[ array_size - i - 1];
	}
}

void swap_cpu( int *a, int array_size )
{
	int i;
	for( i = 0; i < array_size / 2 ; i++ )
	{
		a[i] 					^= a[ array_size - i - 1];
		a[ array_size - i - 1]  ^= a[i];
		a[i] 					^= a[ array_size - i - 1];
	}
}

void print( int *a, int array_size )
{
	int i;
	for( i = 0; i < array_size; i++ )
		printf( "%04d%c", a[i], i == array_size - 1 ? ' ' : ',' );
	printf("\n");
}

void fill( int *a, int array_size )
{
	int i;
	for( i = 0; i < array_size; i++ )
	{
		a[i] = i;
	}
}

void test_gpu( int array_size )
{
	int n_blocks = array_size / BLOCK_SIZE + (array_size % BLOCK_SIZE == 0 ? 0 : 1);

	cudaEvent_t start, stop;
	float ms;

	cudaEventCreate(&stop);
	cudaEventCreate(&start);
	cudaEventRecord(start);

	int *a_h = (int *)malloc( sizeof(int) * array_size );

	fill( a_h, array_size );

	int *a_d;

	cudaMalloc((void **) &a_d, array_size * sizeof(int));

	cudaMemcpy( a_d, a_h, array_size * sizeof(int), cudaMemcpyHostToDevice );

	cudaEvent_t swap_start, swap_stop;
	cudaEventCreate(&swap_start);
	cudaEventCreate(&swap_stop);

	cudaEventRecord(swap_start);
	swap_gpu <<< n_blocks, BLOCK_SIZE >>>( a_d, array_size );
	cudaEventRecord(swap_stop);
	cudaEventSynchronize(swap_stop);
	cudaEventElapsedTime( &ms, swap_start, swap_stop );
	printf("%f;", ms );

	cudaMemcpy( a_h, a_d, array_size * sizeof(int), cudaMemcpyDeviceToHost );

	free(a_h);
	cudaFree(a_d);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &ms, start, stop );
	printf("%f\n", ms );
}

void test_cpu( int array_size )
{
	clock_t start = clock();

	int *a = (int *)malloc( sizeof(int) * array_size );
	fill( a, array_size );

	clock_t swap_start = clock();
	swap_cpu( a, array_size );
	printf("%d;", (clock() - swap_start) * 1000 / CLOCKS_PER_SEC );

	free(a);

	printf("%d;", (clock() - start) * 1000 / CLOCKS_PER_SEC );
}

int main()
{
	int i;
	for( i = 0; i < 29; i++ )
	{
		printf( "%d;", 1 << i );
		test_cpu( 1 << i );
		test_gpu( 1 << i );
	}

	return 0;
}

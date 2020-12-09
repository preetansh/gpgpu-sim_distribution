#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void
benchmark_1_kernel(int* finished, int* total) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	volatile bool done = false;

	int array_id = index/64;

	while (!done) {
		if (atomicCAS(&finished[array_id], 0, 1) == 0) {
	    	__threadfence();
	    	done = true;
	    	total[array_id] += 1;
	    	__threadfence();
	    	atomicExch(&finished[array_id], 0);
		}
	}
	
	done = false;
}

__global__ void
benchmark_2_kernel(int* finished, int* total) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int array_id = index/64;

	total[array_id] += 1;
}


int main(int argc, char *argv[])
{
  	printf("Simple Atomic Add kernel\n");

  	int* finished;
  	int* total;
  	int warpsNum = 24;

	finished = (int *) malloc(warpsNum * sizeof(int));
	// *finished = 0;
	for (int i = 0; i < warpsNum; i++) {
		finished[i] =  0;
	}
	total = (int *) malloc(warpsNum * sizeof(int));
	for (int i = 0; i < warpsNum; i++) {
		total[i] =  0;
	}
	
	int* device_finished;
	int* device_total;
	cudaMalloc(&device_finished, warpsNum * sizeof(int));
	cudaMalloc(&device_total, warpsNum * sizeof(int));
	cudaMemcpy(device_finished, finished, warpsNum * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_total, total, warpsNum * sizeof(int), cudaMemcpyHostToDevice);

	// compute number of blocks and threads per block
    const int threadsPerBlock = 256;
    const int blocks = 6;

	benchmark_1_kernel<<<blocks, threadsPerBlock>>>(device_finished, device_total);

	cudaDeviceSynchronize();
    cudaMemcpy(finished, device_finished, warpsNum * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(total, device_total, warpsNum * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < warpsNum; i++) {
    	printf("Final finished %d %d %d\n", i, finished[i], total[i]);
    }

    cudaFree(device_finished);
    cudaFree(device_total);

    free(finished);
    free(total);
}

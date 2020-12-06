#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void
baseline_atomic_add_kernel(int* finished) {
	int i = 0;
	while (i < 10) {
		// use atomic CAS to set finished to 0
        atomicAdd(finished, 1);
        i++;
	}
}

__global__ void
baseline_atomic_cas_kernel(int* finished) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	printf("Spinning before : %d\n", index);
	while (atomicCAS(finished, 0, 1) != 0) {
		// printf("Spinning inside : %d\n", index);
	}
	// printf("Spinning outside : %d\n", index);
    atomicExch(finished, 0);
}

__global__ void
baseline_atomic_cas_2_kernel(int* finished, int* total) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	bool done = false;
	printf("Spinning before : %d\n", index);
	while (!done) {
		printf("Spinning inside : %d\n", index);
		if (atomicCAS(finished, 0, 1) == 0) {
	    	__threadfence();
			printf("Spinning acq : %d\n", index);
	    	done = true;
	    	*total += 1;
	    	__threadfence();
	    	atomicExch(finished, 0);
		}
	}
	printf("Spinning finished : %d\n", index);
}


int main(int argc, char *argv[])
{
  	printf("Simple Atomic Add kernel\n");

  	int* finished;
  	int* total;
	finished = (int *) malloc(sizeof(int));
	(*finished) = 0;
	total = (int *) malloc(sizeof(int));
	(*total) = 0;
	int* device_finished;
	int* device_total;
	cudaMalloc(&device_finished, 1 * sizeof(int));
	cudaMalloc(&device_total, 1 * sizeof(int));
	cudaMemcpy(device_finished, finished, 1 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_total, total, 1 * sizeof(int), cudaMemcpyHostToDevice);

	// compute number of blocks and threads per block
    const int threadsPerBlock = 64;
    const int blocks = 1;

	baseline_atomic_cas_2_kernel<<<blocks, threadsPerBlock>>>(device_finished, device_total);

	cudaDeviceSynchronize();
    cudaMemcpy(finished, device_finished, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(total, device_total, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Final finished %d %d\n", *finished, *total);
}
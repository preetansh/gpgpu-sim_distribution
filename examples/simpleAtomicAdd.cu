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


int main(int argc, char *argv[])
{
  	printf("Simple Atomic Add kernel\n");

  	int* finished;
	finished = (int *) malloc(sizeof(int));
	(*finished) = 1;
	int* device_finished;
	cudaMalloc(&device_finished, 1 * sizeof(int));
	cudaMemcpy(device_finished, finished, 1 * sizeof(int), cudaMemcpyHostToDevice);

	// compute number of blocks and threads per block
    const int threadsPerBlock = 1;
    const int blocks = 1;

	baseline_atomic_add_kernel<<<blocks, threadsPerBlock>>>(device_finished);

	cudaDeviceSynchronize();
    cudaMemcpy(finished, device_finished, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Final finished %d\n", *finished);
}
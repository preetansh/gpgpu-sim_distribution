#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>

typedef struct ht_node {
	struct ht_node *next_;
	int val_;
} ht_node_t;

typedef struct ht {
	ht_node_t **buckets_;
	int *locks_;
	unsigned int num_buckets_;
} ht_t;

__device__ 
size_t hash_fn(int val) {
	static int seed = 13;
	return (val ^ seed) * seed;
}

__global__ void
benchmark_1_kernel(ht_t* hashtable, ht_node_t *to_insert, int* total, int num_to_process) {
	int index = (blockIdx.x * blockDim.x + threadIdx.x)*num_to_process;
	for(size_t i = 0;i < num_to_process;i++){
		unsigned int ins_array_ind = index + i;
		// printf("inserting %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, ins_array_ind, *total);
		if(ins_array_ind >= *total){
			// printf("too big %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
			return;
		}
		// printf("going in %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
		unsigned int hash_val = hash_fn(to_insert[ins_array_ind].val_) % hashtable->num_buckets_;
		int *lock = &(hashtable->locks_[hash_val]);

		volatile bool done = false;

		while (!done) {
			// printf("trying to get with %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
			if (atomicCAS(lock, 0, 1) == 0) {
				// printf("got with %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
				__threadfence();
				done = true;
				ht_node_t *node = &to_insert[ins_array_ind];
				// printf("trying to insert %d\n", node->val_);
				node->next_ = hashtable->buckets_[hash_val];
				hashtable->buckets_[hash_val] = node;
				// printf("inserted %d\n", node->val_);
				__threadfence();
				atomicExch(lock, 0);
			}
		}
		// printf("inserted %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, ins_array_ind, *total);
		// printf("done with %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
	}
}

__global__ void
checker_kernel(ht_t* hashtable, ht_node_t *to_insert, int* total, bool *found, int num_to_process) {
	int index = (blockIdx.x * blockDim.x + threadIdx.x)*num_to_process;
	for(size_t i = 0;i < num_to_process;i++){
		unsigned int ins_array_ind = index + i;
		// printf("checking %d %d %d %d %d\n", blockIdx.x , blockDim.x , threadIdx.x, index, *total);
		if(ins_array_ind >= *total){
			return;
		}
		unsigned int hash_val = hash_fn(to_insert[ins_array_ind].val_) % hashtable->num_buckets_;

		int looking_for = to_insert[ins_array_ind].val_;
		ht_node_t *node = hashtable->buckets_[hash_val];
		found[ins_array_ind] = false;
		while(node != NULL){
			if(node->val_ == looking_for){
				found[ins_array_ind] = true;
				// printf("found %d\n", looking_for);
				break;
			}
			node = node->next_;
		}
	}
}


int main(int argc, char *argv[])
{
  	printf("Simple Atomic Add kernel\n");

  	// int* finished;
  	// int* total;
  	// int warpsNum = 4;

	// finished = (int *) malloc(warpsNum * sizeof(int));
	// // *finished = 0;
	// for (int i = 0; i < warpsNum; i++) {
	// 	finished[i] =  0;
	// }
	// total = (int *) malloc(warpsNum * sizeof(int));
	// for (int i = 0; i < warpsNum; i++) {
	// 	total[i] =  0;
	// }

	const int num_buckets = 10;

	ht_t host_ht;
	host_ht.num_buckets_ = num_buckets;
	host_ht.locks_ = NULL;
	host_ht.buckets_ = NULL;
	
	ht_t* hashtable;
	ht_node_t **buckets;
	int *locks;
	cudaMalloc(&buckets, num_buckets*(sizeof(ht_node_t*)));
	cudaMemset(buckets, 0, num_buckets*sizeof(ht_node_t*));

	cudaMalloc(&locks, num_buckets*(sizeof(int)));
	cudaMemset(locks, 0, num_buckets*sizeof(int));

	host_ht.locks_ = locks;
	host_ht.buckets_ = buckets;

	cudaMalloc(&hashtable, sizeof(ht_t));
	cudaMemcpy(hashtable, &host_ht, sizeof(ht_t), cudaMemcpyHostToDevice);
	

	// int array[] = {0,1,2,3,4,5,6,7,8,9};
	std::vector<int> insert_vals;
	// insert_vals.insert(insert_vals.begin(), array,&array[sizeof(array)/ sizeof(*array)]);
	for(int i= 0;i < 100000;i++){
		insert_vals.push_back(i);
	}

	std::vector<ht_node_t> insert_nodes;
	for(int i = 0;i < insert_vals.size();i++){
		ht_node_t ins;
		ins.val_ = insert_vals[i];
		ins.next_ = NULL;
		insert_nodes.push_back(ins);
	}

	ht_node_t *to_insert;
	int* total;
	cudaMalloc(&to_insert, sizeof(ht_node_t) * insert_vals.size());
	cudaMemcpy(to_insert, insert_nodes.data(), insert_nodes.size() * sizeof(ht_node_t), cudaMemcpyHostToDevice);

	cudaMalloc(&total, sizeof(int));
	int size = insert_vals.size();
	cudaMemcpy(total, &size, sizeof(int), cudaMemcpyHostToDevice);

	// compute number of blocks and threads per block
    const int threadsPerBlock = 256;
    const int blocks = 10;

	benchmark_1_kernel<<<blocks, threadsPerBlock>>>(hashtable, to_insert, 
		total, insert_nodes.size()/threadsPerBlock + 1);

	cudaDeviceSynchronize();
    // cudaMemcpy(finished, device_finished, warpsNum * sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(total, device_total, warpsNum * sizeof(int), cudaMemcpyDeviceToHost);

	bool *found_array = new bool[insert_vals.size()];
	for(int i = 0;i < insert_vals.size();i++){
		found_array[i] = false;
	}

	bool *dev_found_array;

	cudaMalloc(&dev_found_array, sizeof(bool)*insert_vals.size());
	cudaMemset(dev_found_array, 0, sizeof(bool)*insert_vals.size());
	checker_kernel<<<blocks, threadsPerBlock>>>(hashtable, to_insert, total, dev_found_array,
		insert_nodes.size()/threadsPerBlock + 1);

	cudaDeviceSynchronize();
	cudaMemcpy(found_array, dev_found_array, sizeof(bool)*insert_vals.size(), cudaMemcpyDeviceToHost);
	size_t num_found = 0;
	for(int i = 0;i < insert_vals.size();i++){
		if(!found_array[i]){
			printf("DIDNT FIND %d\n", insert_vals[i]);
		}else{
			num_found++;
		}
	}

	if(num_found == insert_vals.size()){
		printf("FOUND EVERYTHING\n");
	}

	cudaFree(dev_found_array);


    // for (int i = 0; i < warpsNum; i++) {
    // 	printf("Final finished %d %d %d\n", i, finished[i], total[i]);
    // }

    cudaFree(hashtable);
	cudaFree(locks);
	cudaFree(buckets);
	cudaFree(to_insert);
	cudaFree(total);

    // free(finished);
    // free(total);
}

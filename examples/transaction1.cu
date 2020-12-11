#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>

#define ATOMIC_SYNC

#define NUM_ACCOUNTS 8000
#define NUM_TRANSACTIONS 122880

#define THREADS_PER_BLOCK_X 256
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 1

#define BLOCKS_PER_GRID_X	90
#define BLOCKS_PER_GRID_Y	1
#define BLOCKS_PER_GRID_Z	1

#define THREADS_PER_BLOCK	(THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y * THREADS_PER_BLOCK_Z)
#define TOTAL_THREADS		(THREADS_PER_BLOCK * BLOCKS_PER_GRID_X * BLOCKS_PER_GRID_Y * BLOCKS_PER_GRID_Z)

// these macros are for use in the shader!
#define BLOCK_ID			( blockIdx.x + (BLOCKS_PER_GRID_X * blockIdx.y) + (BLOCKS_PER_GRID_X * BLOCKS_PER_GRID_Y * blockIdx.z) )
#define THREAD_ID			( (THREADS_PER_BLOCK * BLOCK_ID) + threadIdx.x + (THREADS_PER_BLOCK_X * threadIdx.y) + (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y * threadIdx.z) )

struct account
{
	int lock;
	int balance;
};

struct transaction
{
	int amount;
	int src_account;
	int dest_account;
};



__global__ void interac_atomic( account* __restrict__ accounts, transaction *__restrict__ transactions, int numTransactions)
{
	int id = THREAD_ID; 
	for(int index = id; index < numTransactions; index += TOTAL_THREADS)
	{
		transaction* action = &transactions[index];
		account* src = &accounts[action->src_account];
		account* dest = &accounts[action->dest_account];
		
		// sanity check
		if(action->src_account == action->dest_account)
		{
			continue;
		}
	
		// acquire locks
      	account* lock1;
      	account* lock2; 
      	if (src > dest) {
        	lock1 = src; 
         	lock2 = dest;
      	} else {
         	lock2 = src; 
         	lock1 = dest;
      	}
	
    	// do transaction
        volatile int done = 0;
        while(!done){
            if(atomicCAS(&lock1->lock, 0, 1) == 0){
                if(atomicCAS(&lock2->lock,0,1) ==0){
               	    // do transaction
              		src->balance -= action->amount;
               		dest->balance += action->amount;
    
               		// release locks
               	   	atomicExch(&lock2->lock, 0);
               		atomicExch(&lock1->lock, 0);

			        done=1;
				   	continue;
                }
                atomicExch(&lock1->lock,0);
            }
        }
	}
}

void interac_gold(account* __restrict__  accounts, transaction* __restrict__ transactions, int num_transactions)
{
	for(int i = 0; i < num_transactions; ++i)
	{
		transaction* action = &transactions[i];
		account* src = &accounts[action->src_account];
		account* dest = &accounts[action->dest_account];
		
		src->balance -= action->amount;
		dest->balance += action->amount;
	}
}
	
int main(int argc, const char** argv)
{
    printf("Initializing...\n");
    srand(2009);  // set seed for rand()

    // allocate host memory for accounts
    unsigned int accounts_size = sizeof(account) * NUM_ACCOUNTS;
	unsigned int transactions_size = sizeof(transaction) * NUM_TRANSACTIONS;
    account* host_accounts = (account*)malloc(accounts_size);
	account* gold_accounts = (account*)malloc(accounts_size);
	transaction* host_transactions = (transaction*)malloc(transactions_size);

	// create random account balances
    for (int i = 0; i < NUM_ACCOUNTS; ++i)
	{
		host_accounts[i].lock = 0;
        host_accounts[i].balance = (int) fmod((float)rand(),100.0f);
		
		gold_accounts[i].lock = 0;
		gold_accounts[i].balance = host_accounts[i].balance;
	}
	
	// create random transaction pairs
	for (int i = 0; i < NUM_TRANSACTIONS; ++i)
	{
		host_transactions[i].amount = (int) fmod((float)rand(),50.0f);
		host_transactions[i].src_account = rand() % NUM_ACCOUNTS;	
		host_transactions[i].dest_account = rand() % NUM_ACCOUNTS;

		// make sure src != dest
		while(host_transactions[i].src_account == host_transactions[i].dest_account)
		{
			host_transactions[i].dest_account = rand() % NUM_ACCOUNTS;
		}
	}

    // allocate device memory
    account* device_accounts;
	transaction* device_transactions;
    cudaMalloc((void**) &device_accounts, accounts_size);
    cudaMalloc((void**) &device_transactions, transactions_size);

    // copy host memory to device
    cudaMemcpy(device_accounts, host_accounts, accounts_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_transactions, host_transactions, transactions_size, cudaMemcpyHostToDevice);
    
    // setup execution parameters
	dim3 block_size(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z);
	dim3 grid_size(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y, BLOCKS_PER_GRID_Z);
	
	printf("Beginning kernel execution...\n");

    // execute the kernel
    interac_atomic<<< grid_size, block_size >>>(device_accounts, device_transactions, NUM_TRANSACTIONS);
	
    cudaDeviceSynchronize();

    // copy result from device to host
    cudaMemcpy(host_accounts, device_accounts, accounts_size, cudaMemcpyDeviceToHost);

	
	printf("Computing gold results...\n");

    interac_gold(gold_accounts, host_transactions, NUM_TRANSACTIONS);
	
	printf("Comparing results...\n");

    // check result
	bool success = true;
    for (int i = 0; i < NUM_ACCOUNTS; ++i)
	{
		if(gold_accounts[i].balance != host_accounts[i].balance)
		{
			success = false;
			printf("Difference found in account %d: Gold = %d, Kernel = %d\n", i, gold_accounts[i].balance, host_accounts[i].balance);
		}
	}
	
	printf("Test %s\n", (success ? "PASSED! All account balances were correct." : "FAILED!"));

    // clean up memory
    free(host_accounts);
	free(gold_accounts);
	free(host_transactions);
    cudaFree(device_accounts);
	cudaFree(device_transactions);
	
}

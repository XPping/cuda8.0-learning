#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>
#include <cstdlib>
#include <vector>
using namespace std;


#define CHECK_CUDAERROR(err) \
	if(err != cudaSuccess)	\
		exit(1);

const int block_size = 16;

// Sum ################################################################
__global__ void sumReduceSharedKernel(int *in, int *out, int length)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int tmp[block_size];
	tmp[threadIdx.x] = in[threadIdx.x];
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) { // blockDim.x(BLOCK_SIZE) must equal to the length
		if (threadIdx.x < stride) {
			tmp[threadIdx.x] += tmp[threadIdx.x + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		out[blockIdx.x] = tmp[x];
	}
}
void sumReduceShared(int *in, int *out, int length)
{
	int size = length * sizeof(int);
	int *d_in, *d_out;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_in, size));
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_out, size));

	dim3 gridDim((length - 1) / block_size + 1, 1, 1);
	dim3 blockDim(block_size, 1, 1);
	sumReduceSharedKernel << <gridDim, blockDim >> > (d_in, d_out, length);

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	cudaFree(d_in); cudaFree(d_out);
}
void test_sumReduceShared() {
	int in[8] = { 3, 1, 7, 0, 4, 1, 6, 3 };
	int out[8] = { 0 };

	sumReduceShared(in, out, 8);

	printf("sum reduce shared: \n");
	printf("input: \n");
	for (int i = 0; i < 8; i++)
		printf("%d ", in[i]);
	printf("output: \n");
	for (int i = 0; i < 8; i++)
		printf("%d ", out[i]);
	printf("\n");
}
// Hist ################################################################
__global__ void sharedHistKernel(int *buffer, int *hist, long size)
{
	__shared__ int H[256];
	if (threadIdx.x < 256) H[threadIdx.x] = 0;
	__syncthreads();

	int i = blockIdx.x * blockDim.x + threadIdx.x; // stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	while (i < size) {
		atomicAdd(&(H[buffer[i]]), 1);
		i += stride;
	}
	__syncthreads();
	if (threadIdx.x < 256) {
		atomicAdd(&(hist[threadIdx.x]), H[threadIdx.x]);
	}
}

void calHist(int *in, int *hist, long length, long total_length)
{
	int size = length * sizeof(int);
	int total_size = total_length * sizeof(int);
	int *d_in, *d_hist;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_in, size));
	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_hist, total_size));

	dim3 gridDim((length - 1) / block_size + 1, 1, 1);
	dim3 blockDim(block_size, 1, 1);
	//histKernel << <gridDim, blockDim >> > (d_buffer, d_hist, length);

	// must one thread deal one val of buffer. Like 27 >= total_length
	sharedHistKernel << <dim3(1, 1, 1), dim3(27, 1, 1) >> > (d_in, d_hist, length);

	cudaMemcpy(hist, d_hist, total_size, cudaMemcpyDeviceToHost);

	cudaFree(d_in); cudaFree(d_hist);
}
void test_calHist() {
	printf("Hist: \n");
	int in[22] = {
		1,3,5,8,11,
		13,14,16,13,3,
		22,25,22,23,13,
		5,1,0,14,25, 1, 5 };
	int hist[26] = { 0 };
	calHist(in, hist, 22, 26);
	printf("input: \n");
	for (int i = 0; i < 22; i++) {
		printf("%d ", in[i]);
	}
	printf("\n");
	printf("output: \n");
	for (int i = 0; i < 26; i++)
		printf("%d, ", hist[i]);
	printf("\n");
}

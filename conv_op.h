#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#define CHECK_CUDAERROR(err) \
	if(err != cudaSuccess)	\
		exit(1);

// Conv1d ################################################################
__global__ void conv1dKernel(int *in, int *kernel, int *out, int kernel_size, int length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	int start = idx - (kernel_size / 2);
	for (int j = 0; j < kernel_size; j++) {
		if (start + j >= 0 && start + j < length) {
			sum += in[start + j] * kernel[j];
		}
	}
	out[idx] = sum;
}
const int conv1d_width = 16;
void  conv1d(int *in, int *kernel, int *out, int kernel_size, int length) {
	int size_in = length * sizeof(int);
	int size_kernel = kernel_size * sizeof(int);
	int *d_in, *d_kernel, *d_out;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_in, size_in));
	cudaMemcpy(d_in, in, size_in, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_kernel, size_kernel));
	cudaMemcpy(d_kernel, kernel, size_kernel, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_out, size_in));

	dim3 gridDim((length - 1) / conv1d_width + 1, 1, 1);
	dim3 blockDim(conv1d_width, 1, 1);
	conv1dKernel << <gridDim, blockDim >> > (d_in, d_kernel, d_out, kernel_size, length);
	cudaMemcpy(out, d_out, size_in, cudaMemcpyDeviceToHost);

	cudaFree(d_in); cudaFree(d_kernel); cudaFree(d_out);
}
void test_conv1d() {
	int a[11] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
	int b[5] = { 3, 4, 5, 4, 3 };
	int c[11] = { 0 };
	conv1d(a, b, c, 5, 11);
	printf("conv1d: \n");
	printf("Input: ");
	for (int i = 0; i < 11; i++)
		printf("%d ", a[i]);
	printf("\n");
	printf("kernel: ");
	for (int i = 0; i < 5; i++)
		printf("%d ", b[i]);
	printf("\n");
	printf("output: ");
	for (int i = 0; i < 11; i++)
		printf("%d ", c[i]);
	printf("\n");
}
// tiled const conv2d ################################################################
const int KERNEL_SIZE = 5;
const int TILE_WIDTH = 16;
const int BLOCK_WIDTH = TILE_WIDTH + KERNEL_SIZE - 1;

__constant__ int cudaKernelData[KERNEL_SIZE * KERNEL_SIZE];

__global__ void tiledConstantConv2dKernel(int* in, const int* __restrict__ kernel, int* out,
	int channels, int width, int height)
{
	__shared__ int Ns[BLOCK_WIDTH][BLOCK_WIDTH];

	int radius = KERNEL_SIZE / 2;

	for (int k = 0; k < channels; k++) {
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int destY = dest / BLOCK_WIDTH; // col of shared memory
		int destX = dest % BLOCK_WIDTH; // row of shared memory
		int srcY = blockIdx.y * TILE_WIDTH + destY - radius; // row index to fetch data
		int srcX = blockIdx.x * TILE_WIDTH + destX - radius; // col index to fetch data

		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			Ns[destY][destX] = in[(srcY * width + srcX) * channels + k];
		else
			Ns[destY][destX] = 0;

		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / BLOCK_WIDTH;
		destX = dest % BLOCK_WIDTH;
		srcY = blockIdx.y * TILE_WIDTH + destY - radius;
		srcX = blockIdx.x * TILE_WIDTH + destX - radius;
		if (destY < BLOCK_WIDTH) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				Ns[destY][destX] = in[(srcY * width + srcX) * channels + k];
			else
				Ns[destY][destX] = 0;
		}

		__syncthreads();

		int sum = 0;
		int y, x;
		for (y = 0; y < KERNEL_SIZE; y++)
			for (x = 0; x < KERNEL_SIZE; x++)
				sum += Ns[threadIdx.y + y][threadIdx.x + x] * cudaKernelData[y*KERNEL_SIZE + x];
		y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (y < height && x < width)
			out[(y*width + x)*channels + k] = sum;
		__syncthreads();
	}
}

void tiledConstantConv2d(int *in, int *kernel, int *out, int channels, int width, int height)
{
	int *d_in, *d_out;
	cudaMalloc((void **)&d_in, channels * width * height * sizeof(int));
	cudaMemcpy(d_in, in, channels * width * height * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(cudaKernelData, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(int));

	cudaMalloc((void **)&d_out, channels * width * height * sizeof(int));

	dim3 gridDim((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1, 1);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	tiledConstantConv2dKernel << <gridDim, blockDim >> > (d_in, cudaKernelData, d_out, channels, width, height);
	cudaMemcpy(out, d_out, channels * width * height * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(cudaKernelData);
	cudaFree(d_out);
}
void test_tiledConstantConv2d() {
	int kernel[KERNEL_SIZE * KERNEL_SIZE] = { -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1,
		-1, -1, 8, -1, -1,
		-1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1 };
	Mat img = imread("test.jpg", CV_LOAD_IMAGE_UNCHANGED);
	int channels, width, height;
	channels = img.channels(); 
	width = img.cols; 
	height = img.rows;
	int *in = (int *)malloc(channels * width * height * sizeof(int));
	int *in_iter = in;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				*in_iter = (int)img.at<Vec3b>(i, j)[k];
				in_iter++;
				//in[i*width*channels + j*channels + k] = img.at<Vec3b>(i, j)[k];
			}
		}
	}
	int *out = (int *)malloc(channels * width * height * sizeof(int));
	tiledConstantConv2d(in, kernel, out, channels, width, height);

	int maxV = 0, minV = 100000;
	for (int i = 0; i < width*height*channels; i++) {
		maxV = maxV > out[i] ? maxV : out[i];
		minV = minV < out[i] ? minV : out[i];
	}
	for (int i = 0; i < width*height*channels; i++) {
		float tmp = ((float)out[i] - minV) / (maxV - minV);
		out[i] = (int)(tmp * 255);
	}

	Mat img2(height, width, CV_8UC3);
	int jj = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channels; k++) {
				img2.at<Vec3b>(i, j)[k] = (unsigned char)out[jj++];
			}
		}
	}
	namedWindow("my", CV_WINDOW_AUTOSIZE);
	imshow("my", img2);
	waitKey(0);
	//destroyAllWindows("my");
}
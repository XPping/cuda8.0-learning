#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>


#define CHECK_CUDAERROR(err) \
	if(err != cudaSuccess)	\
		exit(1);

// Vector add ################################################################
__global__ void vectorAddKernel(int *A, int *B, int *C, int n) {
	// A + B = C, n is length
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) C[idx] = A[idx] + B[idx];
}
void vectorAdd(const int *A, const int *B, int *C, int n) {
	int size = n * sizeof(int);
	int *d_A, *d_B, *d_C;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_A, size));
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_B, size));
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_C, size));
	// Call kernel
	vectorAddKernel << <ceil(n / 256.0), 256 >> > (d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
void test_vectorAdd() {
	const int size = 10;
	const int a[size] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
	const int b[size] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
	int c[size] = { 0 };
	vectorAdd(a, b, c, size);
	for (int i = 0; i < size; i++) {
		printf("%d ", a[i]);
	}
	printf("+ ");
	for (int i = 0; i < size; i++) {
		printf("%d ", b[i]);
	}
	printf("= ");
	for (int i = 0; i < size; i++) {
		printf("%d ", c[i]);
	}
	printf("\n");
}

// Matrix Mul ################################################################
// A: m*n, B: n*k, C: m*k
__global__ void matrixMulKernel(int *A, int *B, int *C, int m, int n, int k) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if ((row < m) && (col < k)) {
		int sum = 0;
		for (int i = 0; i < n; i++) {
			sum += A[row*n + i] * B[col + i*k];
		}
		C[row*k + col] = sum;
	}
}
void matrixMul(const int *A, const int *B, int *C, int m, int n, int k) {
	int size_A = m * n * sizeof(int);
	int size_B = n * k * sizeof(int);
	int size_C = m * k * sizeof(int);
	int *d_A, *d_B, *d_C;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_A, size_A));
	cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_B, size_B));
	cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_C, size_C));

	//dim3 DimGrid((k - 1) / 2 + 1, (m - 1) / 3 + 1, 1);
	//dim3 DimBlock(2, 3, 1);
	dim3 DimGrid((m - 1) / 3 + 1, (k - 1) / 2 + 1, 1);
	dim3 DimBlock(3, 2, 1);
	matrixMulKernel << <DimGrid, DimBlock >> > (d_A, d_B, d_C, m, n, k);
	cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
void test_matrixMul() {
	printf("Matrix mul: \n");
	int A[5 * 3] = { 0 };
	int B[3 * 4] = { 0 };
	int C[5 * 4] = { 0 };

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 3; j++) {
			A[i * 3 + j] = i * 3 + j;
			printf("%d ", A[i * 3 + j]);
		}
		printf("\n");
	}
	printf("*\n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			B[i * 4 + j] = i * 4 + j;
			printf("%d ", B[i * 4 + j]);
		}
		printf("\n");
	}
	printf("=\n");
	matrixMul(A, B, C, 5, 3, 4);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%d ", C[i * 4 + j]);
		}
		printf("\n");
	}
}

// tiled Matrix Mul faster than Matrix Mul ################################################################
// A: m*n, B: n*k, C: m*k
const int tiled_width_mul = 3;
const int tiled_height_mul = 2;
__global__ void tiledMatrixMulKernel(int *A, int *B, int *C, int m, int n, int k) {
	__shared__ int ds_A[tiled_width_mul][tiled_height_mul];
	__shared__ int ds_B[tiled_width_mul][tiled_height_mul];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int row = bx * blockDim.x + tx;
	int col = by * blockDim.y + ty;

	int sum = 0;
	for (int t = 0; t < (n - 1) / tiled_height_mul + 1; t++) {
		if ((row < m) && (t*tiled_height_mul + ty < n)) {
			ds_A[tx][ty] = A[row*n + t*tiled_height_mul + ty];
		}
		else {
			ds_A[tx][ty] = 0;
		}
		if ((col < k) && (t*tiled_height_mul + tx < n)) {
			ds_B[tx][ty] = B[(t*tiled_height_mul + tx)*k + col];
		}
		else {
			ds_B[tx][ty] = 0;
		}
		__syncthreads();
		for (int i = 0; i < tiled_height_mul; i++) {
			sum += ds_A[tx][i] * ds_B[i][ty];
		}
		__syncthreads();
	}
	if ((row < m) && (col < k)) {
		C[row*k + col] = sum;
	}
}
void tiledMatrixMul(const int *A, const int *B, int *C, int m, int n, int k) {
	int size_A = m * n * sizeof(int);
	int size_B = n * k * sizeof(int);
	int size_C = m * k * sizeof(int);
	int *d_A, *d_B, *d_C;
	CHECK_CUDAERROR(cudaMalloc((void **)&d_A, size_A));
	cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_B, size_B));
	cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
	CHECK_CUDAERROR(cudaMalloc((void **)&d_C, size_C));

	//dim3 DimGrid((k - 1) / 2 + 1, (m - 1) / 3 + 1, 1);
	//dim3 DimBlock(2, 3, 1);
	dim3 DimGrid((m - 1) / tiled_width_mul + 1, (k - 1) / tiled_height_mul + 1, 1);
	dim3 DimBlock(tiled_width_mul, tiled_height_mul, 1);
	tiledMatrixMulKernel << <DimGrid, DimBlock >> > (d_A, d_B, d_C, m, n, k);
	cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
void test_tiledMatrixMul() {
	printf("Tiled matrix mul: \n");
	int A[5 * 3] = { 0 };
	int B[3 * 4] = { 0 };
	int C[5 * 4] = { 0 };

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 3; j++) {
			A[i * 3 + j] = i * 3 + j;
			printf("%d ", A[i * 3 + j]);
		}
		printf("\n");
	}
	printf("*\n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			B[i * 4 + j] = i * 4 + j;
			printf("%d ", B[i * 4 + j]);
		}
		printf("\n");
	}
	printf("=\n");
	tiledMatrixMul(A, B, C, 5, 3, 4);
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 4; j++) {
			printf("%d ", C[i * 4 + j]);
		}
		printf("\n");
	}
}
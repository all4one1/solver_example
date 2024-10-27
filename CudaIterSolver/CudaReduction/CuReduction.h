#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

struct CudaReduction
{
	//GPU-Grid points and Node (total) points
	unsigned int* Gp = nullptr, * Np = nullptr;
	unsigned int steps = 0, threads = 1024;

	double* res_array = nullptr, *final_res = nullptr;
	double res = 0;
	double** arr = nullptr;

	CudaReduction(double* device_ptr, unsigned int N, unsigned int thr = 1024);
	CudaReduction(unsigned int N, unsigned int thr = 1024);
	CudaReduction();
	~CudaReduction();

	void print_check();
	double reduce();
	double reduce(double* device_ptr);
	double reduce_test(cudaStream_t stream = 0);
	static double reduce(double* device_ptr, unsigned int N, unsigned int thr = 1024);

	void auto_test();
};

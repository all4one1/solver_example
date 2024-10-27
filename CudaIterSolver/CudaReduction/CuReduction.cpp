#include "CuReduction.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>


__device__ double res_dev;

__global__ void init_test(double* data, unsigned int n)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < n)
	{
		data[i] = i + 1;
	}
}
__global__ void gpu_print(double* f)
{
	printf("message: %f", f[0]);
	printf("\n");
}
__global__ void reduction_abs_sum(double* data, unsigned int n, double* reduced, bool last = false) {
	extern __shared__ double shared[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		shared[tid] = abs(data[i]);
	}
	else
	{
		shared[tid] = 0.0;
	}

	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			shared[tid] += shared[tid + s];
		}

		__syncthreads();
	}


	if (tid == 0) {
		reduced[blockIdx.x] = shared[0];
		if (last) res_dev = shared[0];
	}


}

CudaReduction::CudaReduction()
{

}

CudaReduction::CudaReduction(unsigned int N, unsigned int thr)
{
	threads = thr;

	unsigned int GN = N;
	while (true)
	{
		steps++;
		GN = (unsigned int)ceil(GN / (threads + 0.0));
		if (GN == 1)  break;
	}
	GN = N;

	Gp = new unsigned int[steps];
	Np = new unsigned int[steps];
	arr = new double* [steps + 1];

	for (unsigned int i = 0; i < steps; i++)
		Gp[i] = GN = (unsigned int)ceil(GN / (threads + 0.0));
	Np[0] = N;
	for (unsigned int i = 1; i < steps; i++)
		Np[i] = Gp[i - 1];

	//if (steps == 1) std::cout << "Warning: a small array of data" << std::endl;
	(steps != 1) ? cudaMalloc((void**)&res_array, sizeof(double) * Np[1]) : cudaMalloc((void**)&res_array, sizeof(double));

	cudaMalloc((void**)&final_res, sizeof(double));
}

CudaReduction::CudaReduction(double* device_ptr, unsigned int N, unsigned int thr)
{
	threads = thr;

	unsigned int GN = N;
	while (true)
	{
		steps++;
		GN = (unsigned int)ceil(GN / (threads + 0.0));
		if (GN == 1)  break;
	}
	GN = N;

	Gp = new unsigned int[steps];
	Np = new unsigned int[steps];
	arr = new double* [steps + 1];

	for (unsigned int i = 0; i < steps; i++)
		Gp[i] = GN = (unsigned int)ceil(GN / (threads + 0.0));
	Np[0] = N;
	for (unsigned int i = 1; i < steps; i++)
		Np[i] = Gp[i - 1];

	//if (steps == 1) std::cout << "Warning: a small array of data" << std::endl;
	(steps != 1) ? cudaMalloc((void**)&res_array, sizeof(double) * Np[1]) : cudaMalloc((void**)&res_array, sizeof(double));


	arr[0] = device_ptr;
	for (unsigned int i = 1; i <= steps; i++)
		arr[i] = res_array;

	cudaMalloc((void**)&final_res, sizeof(double));
}


CudaReduction::~CudaReduction()
{
	cudaFree(res_array); res_array = nullptr;
	delete[] Gp; Gp = nullptr;
	delete[] Np; Np = nullptr;
	delete[] arr; arr = nullptr;
	cudaFree(final_res); final_res = nullptr;
}


void CudaReduction::print_check()
{
	gpu_print << <1, 1 >> > (res_array);
}

void CudaReduction::auto_test()
{
	double* ptr_d;
	int N = 1234;

	cudaMalloc((void**)&ptr_d, N * sizeof(double));
	init_test << <1024, 1024 >> > (ptr_d, N);

	std::cout << "Exact value = " << N * (N + 1) / 2 << std::endl;
	std::cout << "Cuda result = " << CudaReduction::reduce(ptr_d, N, 128) << std::endl;
}


double CudaReduction::reduce(double* device_ptr)
{
	arr[0] = device_ptr;
	for (unsigned int i = 1; i <= steps; i++)
		arr[i] = res_array;

	for (unsigned int i = 0; i < steps; i++)
	{
		reduction_abs_sum << < Gp[i], threads, 1024 * sizeof(double) >> > (arr[i], Np[i], arr[i + 1]);
	}
	cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

	return res;
}

double CudaReduction::reduce_test(cudaStream_t stream)
{
	for (unsigned int i = 0; i < steps; i++)
	{
		reduction_abs_sum << < Gp[i], threads, 1024 * sizeof(double), stream>> > (arr[i], Np[i], arr[i + 1], i == steps - 1);
	}
	//cudaMemcpyFromSymbol(&res, res_dev, sizeof(double), 0, cudaMemcpyDeviceToHost);
	return 0.0;
}


double CudaReduction::reduce()
{
	for (unsigned int i = 0; i < steps; i++)
	{
		reduction_abs_sum << < Gp[i], threads, 1024 * sizeof(double) >> > (arr[i], Np[i], arr[i + 1]);
	}
	cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
	return res;
}

double CudaReduction::reduce(double* device_ptr, unsigned int N, unsigned int thr)
{
	CudaReduction temp(device_ptr, N, thr);
	return temp.reduce();
}


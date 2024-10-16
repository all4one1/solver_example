//#include "ReductionModified.cuh"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <cuda.h>
//#include <iostream>
//
//using CuCG::CudaReductionM;
//
//
//CudaReductionM::CudaReductionM()
//{
//
//}
//
//CudaReductionM::CudaReductionM(unsigned int N, unsigned int thr)
//{
//	threads = thr;
//
//	unsigned int GN = N;
//	while (true)
//	{
//		steps++;
//		GN = (unsigned int)ceil(GN / (threads + 0.0));
//		if (GN == 1)  break;
//	}
//	GN = N;
//
//	Gp = new unsigned int[steps];
//	Np = new unsigned int[steps];
//	arr = new double* [steps + 1];
//
//	for (unsigned int i = 0; i < steps; i++)
//		Gp[i] = GN = (unsigned int)ceil(GN / (threads + 0.0));
//	Np[0] = N;
//	for (unsigned int i = 1; i < steps; i++)
//		Np[i] = Gp[i - 1];
//
//	//if (steps == 1) std::cout << "Warning: a small array of data" << std::endl;
//	(steps != 1) ? cudaMalloc((void**)&res_array, sizeof(double) * Np[1]) : cudaMalloc((void**)&res_array, sizeof(double));
//}
//
//CudaReductionM::~CudaReductionM()
//{
//	cudaFree(res_array); res_array = nullptr;
//	delete[] Gp; Gp = nullptr;
//	delete[] Np; Np = nullptr;
//	delete[] arr; arr = nullptr;	
//	delete[] second; second = nullptr;
//}
//
//
//double CudaReductionM::reduce(double* v1, double *v2, bool withCopy)
//{
//	arr[0] = v1;
//	second = v2;
//	for (unsigned int i = 1; i <= steps; i++)
//		arr[i] = res_array;
//
//	for (unsigned int i = 0; i < steps; i++)
//		CuCG::dot_product << < Gp[i], threads, 1024 * sizeof(double) >> > (arr[i], arr[i], Np[i], arr[i + 1], i == steps - 1, CuCG::extra_action::compute_buffer);
//
//	if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
//
//	return res;
//}



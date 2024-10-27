#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

#include "CuGraph.h"

namespace cg = cooperative_groups;
//struct CudaLaunchSetup
//{
//	dim3 Grid3D, Block3D, Grid1D, Block1D;
//	unsigned int thread_x = 8, thread_y = 8, thread_z = 8, thread_1D = 1024;
//
//	CudaLaunchSetup(unsigned int N, unsigned int nx = 1, unsigned int ny = 1, unsigned nz = 1)
//	{
//		Grid3D = dim3(
//			(unsigned int)ceil((nx + 1.0) / thread_x),
//			(unsigned int)ceil((ny + 1.0) / thread_y),
//			(unsigned int)ceil((nz + 1.0) / thread_z));
//		Block3D = dim3(thread_x, thread_y, thread_z);
//
//		Grid1D = dim3((unsigned int)ceil((N + 0.0) / thread_1D));
//		Block1D = thread_1D;
//
//	};
//};
//struct SparseMatrixCuda
//{
//	/*		Compressed Sparse Row	 */
//
//	int Nfull = 0;	// the input (linear) size of a matrix
//	int nval = 0;	// number of non-zero elements
//	int nrow = 0;	// number of rows
//	size_t bytesVal = 0;
//	size_t bytesCol = 0;
//	size_t bytesRow = 0;
//	double* val = nullptr;
//	int* col = nullptr;
//	int* row = nullptr;
//
//	SparseMatrixCuda() {};
//	SparseMatrixCuda(int N, int nv, double* v, int* c, int* r) : Nfull(N), nval(nv)
//	{
//		nrow = N + 1;
//		bytesVal = nval * sizeof(double);
//		bytesCol = nval * sizeof(int);
//		bytesRow = nrow * sizeof(int);
//
//		cudaMalloc((void**)&val, sizeof(double) * nval);
//		cudaMalloc((void**)&col, sizeof(int) * nval);
//		cudaMalloc((void**)&row, sizeof(int) * nrow);
//
//		cudaMemcpy(val, v, bytesVal, cudaMemcpyHostToDevice);
//		cudaMemcpy(col, c, bytesCol, cudaMemcpyHostToDevice);
//		cudaMemcpy(row, r, bytesRow, cudaMemcpyHostToDevice);
//	}
//	~SparseMatrixCuda() {};
//};

namespace CuCG
{
	enum class KernelCoefficient
	{
		beta_and_omega, alpha_and_omega, alpha, omega
	};
	enum class ExtraAction { NONE, compute_rs_new_and_beta, compute_alpha, compute_omega, compute_buffer, compute_rs_old };


	__device__ double rs_old = 1, rs_new = 1, alpha = 1, beta = 1, buffer = 1, buffer2 = 1, omega = 1;

	__global__ void check()
	{
		printf("device: \n");
		printf("rs_old = %f, rs_new = %f, buffer = %f \n", rs_old, rs_new, buffer);
		printf("alpha = %f, beta = %f, omega = %f \n", alpha, beta, omega);
	}
	__global__ void check2(double *f, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			printf("%i %f \n", i, f[i]);
		}
	}

	__device__ void extra_action__(double res, ExtraAction action)
	{
		switch (action)
		{
		case ExtraAction::compute_rs_new_and_beta:
			rs_new = res;
			beta = (rs_new / rs_old) * (alpha / omega);
			rs_old = rs_new;
			break;
		case ExtraAction::compute_alpha:
			alpha = rs_new / res;
			break;
		case ExtraAction::compute_omega:
			omega = buffer / res;
			break;
		case ExtraAction::compute_buffer:
			buffer = res;
			break;
		case ExtraAction::compute_rs_old:
			rs_old = res;
			break;
		default:
			break;
		}
	}

	template <class T, unsigned int blockSize>
	__global__ void reduce5(T* g_idata, T* second, T* g_odata, unsigned int n, bool first, bool last, ExtraAction action) {
		// Handle to thread block group
		cg::thread_block cta = cg::this_thread_block();
		extern __shared__ double sdata[];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

		T mySum = (i < n) ? g_idata[i] * (first ? second[i] : 1) : 0;
		if (i + blockSize < n) mySum += g_idata[i + blockSize] * (first ? second[i + blockSize] : 1);

		sdata[tid] = mySum;
		cg::sync(cta);

		// do reduction in shared mem
		if ((blockSize >= 512) && (tid < 256)) {
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		cg::sync(cta);

		if ((blockSize >= 256) && (tid < 128)) {
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		cg::sync(cta);

		if ((blockSize >= 128) && (tid < 64)) {
			sdata[tid] = mySum = mySum + sdata[tid + 64];
		}

		cg::sync(cta);

		cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

		if (cta.thread_rank() < 32) {
			// Fetch final intermediate sum from 2nd warp
			if (blockSize >= 64) mySum += sdata[tid + 32];
			// Reduce final warp using shuffle
			for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
				mySum += tile32.shfl_down(mySum, offset);
			}
		}

		// write result for this block to global mem
		if (cta.thread_rank() == 0)
		{
			g_odata[blockIdx.x] = mySum;
			if (last)
			{
				extra_action__(mySum, action);
			}
		}
	}

	__global__ void dot_product(double* v1, double* v2, unsigned int n, double* reduced, bool first, bool last, ExtraAction action) {
		extern __shared__ double shared[];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < n)
		{
			if (first) shared[tid] = v1[i] * v2[i];
			else shared[tid] = v1[i];
		}
		else
			shared[tid] = 0.0;

		__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
		{
			if (tid < s)
				shared[tid] += shared[tid + s];
			__syncthreads();
		}
		if (tid == 0)
		{
			reduced[blockIdx.x] = shared[0];
			if (last)
			{
				extra_action__(shared[0], action);
				//switch (action)
				//{
				//case ExtraAction::compute_rs_new_and_beta:
				//	rs_new = shared[0];
				//	beta = (rs_new / rs_old) * (alpha / omega);
				//	rs_old = rs_new;
				//	break;
				//case ExtraAction::compute_alpha:
				//	alpha = rs_new / shared[0];
				//	break;
				//case ExtraAction::compute_omega:
				//	omega = buffer / shared[0];
				//	break;
				//case ExtraAction::compute_buffer:
				//	buffer = shared[0];
				//	break;
				//case ExtraAction::compute_rs_old:
				//	rs_old = shared[0];
				//	break;
				//default:
				//	break;
				//}

			}
		}
	}

	struct CudaReductionM
	{
		std::vector<unsigned int> grid_v;
		std::vector<unsigned int> N_v;

		#define def_threads 512
		unsigned int N = 0;
		unsigned int steps = 0, threads = def_threads, smem = sizeof(double) * def_threads;

		double* res_array = nullptr;
		double res = 0;
		double** arr = nullptr;
		double* second = nullptr;

		CudaReductionM(unsigned int N, unsigned int thr = 1024);
		CudaReductionM();
		~CudaReductionM();

		double reduce(double* v1, double* v2, bool withCopy = true, ExtraAction action = ExtraAction::NONE);
		CuGraph make_graph(double* v1, double* v2, bool withCopy, ExtraAction action);
	};

	CudaReductionM::CudaReductionM() {}
	CudaReductionM::CudaReductionM(unsigned int N, unsigned int thr)
	{
		bool doubleRead = false;
		if (thr < 64)
		{
			std::cout << "more threads needed " << std::endl;
			threads = 64;
		}

		unsigned int temp_ = N;
		threads = thr;
		N_v.push_back(N);

		steps = 0;
		while (true)
		{
			steps++;
			if (doubleRead) temp_ = (temp_ + (threads * 2 - 1)) / (threads * 2);
			else temp_ = (temp_ + threads - 1) / threads;

			grid_v.push_back(temp_);
			N_v.push_back(temp_);
			if (temp_ == 1)  break;
		}

		if (res_array != nullptr) cudaFree(res_array);
		cudaMalloc((void**)&res_array, sizeof(double) * N_v[1]);

		if (arr != nullptr) delete[] arr;
		arr = new double* [steps + 1];
	}
	CudaReductionM::~CudaReductionM()
	{
		cudaFree(res_array); res_array = nullptr;
		delete[] arr; arr = nullptr;
		grid_v.clear();
		N_v.clear();
	}

	double CudaReductionM::reduce(double* v1, double* v2, bool withCopy, ExtraAction action)
	{
		arr[0] = v1;	second = v2;
		for (unsigned int i = 1; i <= steps; i++)
			arr[i] = res_array;

		switch (threads)
		{
		case(512):
			for (unsigned int i = 0; i < steps; i++)	reduce5<double, 512> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
			break;
		case(256):
			for (unsigned int i = 0; i < steps; i++)	reduce5<double, 256> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
			break;
		case(128):
			for (unsigned int i = 0; i < steps; i++)	reduce5<double, 128> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
			break;
		case(64):
			for (unsigned int i = 0; i < steps; i++)	reduce5<double, 64> << <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
			break;
		default:
			break;
		}

		//for (unsigned int i = 0; i < steps; i++)	reduce5<double, 256><< <grid_v[i], threads, smem >> > (arr[i], second, arr[i + 1], N_v[i], i == 0, i == steps - 1, action);
			//dot_product << < grid_v[i], threads, smem >> > (arr[i], second, N_v[i], arr[i + 1], i == 0, i == steps - 1, action);
		
		if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

		return res;
	}

	CuGraph CudaReductionM::make_graph(double* v1, double *v2, bool withCopy, ExtraAction action)
	{
		arr[0] = v1;	second = v2;
		for (unsigned int i = 1; i <= steps; i++)
			arr[i] = res_array;

		void* kernel;

		if (threads == 512) kernel = reinterpret_cast<void*>(&reduce5<double, 512>);
		if (threads == 256) kernel = reinterpret_cast<void*>(&reduce5<double, 256>);
		if (threads == 128) kernel = reinterpret_cast<void*>(&reduce5<double, 128>);
		if (threads == 64)  kernel = reinterpret_cast<void*>(&reduce5<double, 64>);
		
		CuGraph graph;
		for (unsigned int i = 0; i < steps; i++)
		{
			bool first = (i == 0);
			bool last = (i == steps - 1);
			void* args[] = { &arr[i], &second, &arr[i + 1], &N_v[i], &first, &last, &action };
			graph.add_kernel_node(threads, grid_v[i], kernel, args, smem);
		}
		if (withCopy) graph.add_copy_node(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);
		return graph;
	}


	__global__ void hadamard_product(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] * v2[i];
		}
	}
	__global__ void matrix_dot_vector(double* res, SparseMatrixCuda M, double* vm, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = M.row[i]; j < M.row[i + 1]; j++)
			{
				s += M.val[j] * vm[M.col[j]];
			}
			res[i] = s;
		}
	}


	__global__ void vector_add_vector(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] + v2[i];
		}
	}
	__global__ void vector_substract_vector(double* res, double* v1, double* v2, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v1[i] - v2[i];
		}
	}
	__global__ void scalar_dot_vector(double* res, double scalar, double* v, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = scalar * v[i];
		}
	}

	__global__ void vector_minus_matrix_dot_vector(double* res, double* v, SparseMatrixCuda M, double* vm, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = M.row[i]; j < M.row[i + 1]; j++)
			{
				s += M.val[j] * vm[M.col[j]];
			}
			res[i] = v[i] - s;
		}
	}

	__global__ void vector_minus_vector(double* res, double* v1, double* v2, unsigned int N, KernelCoefficient choice)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double coef; 
			switch (choice)
			{
			case CuCG::KernelCoefficient::alpha:
				coef = alpha;
				break;
			case CuCG::KernelCoefficient::omega:
				coef = omega;
				break;
			default:
				break;
			}
			res[i] = v1[i] - coef * v2[i];
		}
	}

	__global__ void vector_add_2vectors(double* res, double* v, double* vs1, double* vs2, unsigned int N, KernelCoefficient choice)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double coef1, coef2;
			switch (choice)
			{
			case KernelCoefficient::beta_and_omega:
				coef1 = beta;
				coef2 = -beta * omega;
				break;
			case KernelCoefficient::alpha_and_omega:
				coef1 = alpha;
				coef2 = omega;
				break;
			default:
				break;
			}
			res[i] = v[i] + coef1 * vs1[i] + coef2 * vs2[i];
		}
	}
	__global__ void vector_set_to_vector(double* res, double* v, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = v[i];
		}
	}
	__global__ void vector_set_to_value(double* res, double scalar, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			res[i] = scalar;
		}
	}
}




void CUDA_BICGSTAB(unsigned int N, double* x, double* x0, double* b, SparseMatrixCuda& A, CudaLaunchSetup kernel_setting)
{
	CuCG::CudaReductionM CR(N, 512);
	double eps = 1e-8;
	double rs_host = 1;
	unsigned int k = 0;
	unsigned int Nbytes = N * sizeof(double);
	//#define device_single_double(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, sizeof(double));  cudaMemset(ptr, 0, sizeof(double)); 
	#define device_double_ptr(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
	device_double_ptr(r);
	device_double_ptr(r_hat);
	device_double_ptr(p);
	device_double_ptr(t);
	device_double_ptr(s);
	device_double_ptr(v);
	
	cudaMemset(x, 0, Nbytes);
	//cudaMemset(v, 0, Nbytes);
	//cudaMemset(p, 0, Nbytes);

	#define KERNEL(func) func<<< kernel_setting.Grid1D, kernel_setting.Block1D>>>
	// r = b - Ax
	KERNEL(CuCG::vector_minus_matrix_dot_vector)(r, b, A, x, N);
	// r_hat = r
	KERNEL(CuCG::vector_set_to_vector)(r_hat, r, N);
	// p = r
	KERNEL(CuCG::vector_set_to_vector)(p, r, N);

	// rs = r_hat * r
	CR.reduce(r_hat, r, true, CuCG::ExtraAction::compute_rs_old);

	auto single_iteration = [&]()
	{

		// rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
		CR.reduce(r_hat, r, false, CuCG::ExtraAction::compute_rs_new_and_beta);

		// p = r + beta * ( p - omega * v)
		KERNEL(CuCG::vector_add_2vectors)(p, r, p, v, N, CuCG::KernelCoefficient::beta_and_omega);
		
		// v = Ap
		KERNEL(CuCG::matrix_dot_vector)(v, A, p, N);

		// alpha = rs_new / (r_hat * v)
		CR.reduce(r_hat, v, false, CuCG::ExtraAction::compute_alpha);

		// s = r - alpha * v
		KERNEL(CuCG::vector_minus_vector)(s, r, v, N, CuCG::KernelCoefficient::alpha);

		// t = A * s
		KERNEL(CuCG::matrix_dot_vector)(t, A, s, N);

		// omega = (t * s) / (t * t)

		CR.reduce(t, s, false, CuCG::ExtraAction::compute_buffer);
		CR.reduce(t, t, false, CuCG::ExtraAction::compute_omega);

		// x = x + alpha * p + omega * s
		KERNEL(CuCG::vector_add_2vectors)(x, x, p, s, N, CuCG::KernelCoefficient::alpha_and_omega);

		// r = s - omega * t
		KERNEL(CuCG::vector_minus_vector)(r, s, t, N, CuCG::KernelCoefficient::omega);
	};


	while (true)
	{
		k++;	if (k > 1000000) break;

		single_iteration();

		// check exit by r^2
		if (k < 20 || k % 50 == 0)
		{
			rs_host = CR.reduce(r, r, true, CuCG::ExtraAction::NONE);
			if (k > 100000) break;
			//if (abs(rs_host) < eps) break;
		}

		//if (k == 20000) break;
		if (k % 1000 == 0) cout << k << " " << abs(rs_host) << endl;
	}

	cout << k << " " << abs(rs_host) << endl;
}

void CUDA_BICGSTAB_WITH_GRAPH(unsigned int N, double* x, double* x0, double* b, SparseMatrixCuda& A, CudaLaunchSetup kernel_setting)
{
	CuGraph graph;
	CuCG::KernelCoefficient action;
	unsigned int threads = kernel_setting.Block1D.x;
	unsigned int blocks = kernel_setting.Grid1D.x;


	CuCG::CudaReductionM CR(N, 256);
	double eps = 1e-8;
	double rs_host = 1;
	unsigned int k = 0;
	unsigned int Nbytes = N * sizeof(double);
	//#define device_single_double(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, sizeof(double));  cudaMemset(ptr, 0, sizeof(double)); 
	#define device_double_ptr(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
	device_double_ptr(r);
	device_double_ptr(r_hat);
	device_double_ptr(p);
	device_double_ptr(t);
	device_double_ptr(s);
	device_double_ptr(v);

	cudaMemset(x, 0, Nbytes);
	//cudaMemset(v, 0, Nbytes);
	//cudaMemset(p, 0, Nbytes);

	#define KERNEL(func) func<<< kernel_setting.Grid1D, kernel_setting.Block1D>>>
	// r = b - Ax
	KERNEL(CuCG::vector_minus_matrix_dot_vector)(r, b, A, x, N);
	// r_hat = r
	KERNEL(CuCG::vector_set_to_vector)(r_hat, r, N);
	// p = r
	KERNEL(CuCG::vector_set_to_vector)(p, r, N);

	// rs = r_hat * r
	CR.reduce(r_hat, r, false, CuCG::ExtraAction::compute_rs_old);
	

	// 1. rs_new = r_hat * r; 		// beta =  (rs_new / rs_old) * (alpha / omega)		// rs_old = rs_new
	graph.add_graph_as_node(CR.make_graph(r_hat, r, false, CuCG::ExtraAction::compute_rs_new_and_beta));
	
	// 2. p = r + beta * ( p - omega * v)
	{
		action = CuCG::KernelCoefficient::beta_and_omega;
		void* args[] = { &p, &r, &p, &v, &N, &action };
		graph.add_kernel_node(threads, blocks, CuCG::vector_add_2vectors, args);
	}

	// 3. v = Ap
	{
		void* args[] = { &v, &A, &p, &N };
		graph.add_kernel_node(threads, blocks, CuCG::matrix_dot_vector, args);
	}

	// 4. alpha = rs_new / (r_hat * v)
	graph.add_graph_as_node(CR.make_graph(r_hat, v, false, CuCG::ExtraAction::compute_alpha));

	// 5. s = r - alpha * v
	{ 
		action = CuCG::KernelCoefficient::alpha;
		void* args[] = { &s, &r, &v, &N, &action };
		graph.add_kernel_node(threads, blocks, CuCG::vector_minus_vector, args);
	}

	// 6. t = A * s
	{
		void* args[] = { &t, &A, &s, &N };
		graph.add_kernel_node(threads, blocks, CuCG::matrix_dot_vector, args);
	}

	// 7. omega = (t * s) / (t * t)
	graph.add_graph_as_node(CR.make_graph(t, s, false, CuCG::ExtraAction::compute_buffer));
	graph.add_graph_as_node(CR.make_graph(t, t, false, CuCG::ExtraAction::compute_omega));

	// 8. x = x + alpha * p + omega * s
	{
		action = CuCG::KernelCoefficient::alpha_and_omega;
		void* args[] = { &x, &x, &p, &s, &N, &action };
		graph.add_kernel_node(threads, blocks, CuCG::vector_add_2vectors, args);
	}

	// 9. r = s - omega * t
	{
		action = CuCG::KernelCoefficient::omega;
		void* args[] = { &r, &s, &t, &N, &action };
		graph.add_kernel_node(threads, blocks, CuCG::vector_minus_vector, args);
	}

	graph.instantiate();
	while (true)
	{
		k++;	if (k > 1000000) break;

		graph.launch();


		// check exit by r^2
		if (k < 20 || k % 50 == 0)
		{
			rs_host = CR.reduce(r, r, true, CuCG::ExtraAction::NONE);
			if (k > 100000) break;
			//if (abs(rs_host) < eps) break;
		}

		//if (k == 20000) break;
		if (k % 1000 == 0) cout << k << " " << abs(rs_host) << endl;
	}

	cout << k << " " << abs(rs_host) << endl;
}
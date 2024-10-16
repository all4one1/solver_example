#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>

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
	__device__ SparseMatrixCuda *A_dev;

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
				switch (action)
				{
				case ExtraAction::compute_rs_new_and_beta:
					rs_new = shared[0];
					beta = (rs_new / rs_old) * (alpha / omega);
					rs_old = rs_new;
					break;
				case ExtraAction::compute_alpha:
					alpha = rs_new / shared[0];
					break;
				case ExtraAction::compute_omega:
					omega = buffer / shared[0];
					break;
				case ExtraAction::compute_buffer:
					buffer = shared[0];
					break;
				case ExtraAction::compute_rs_old:
					rs_old = shared[0];
					break;
				default:
					break;
				}

			}
		}
	}

	struct CudaReductionM
	{
		//GPU-Grid points and Node (total) points
		unsigned int* Gp = nullptr, * Np = nullptr;
		unsigned int steps = 0, threads = 1024;

		double* res_array = nullptr;
		double res = 0;
		double** arr = nullptr;
		double* second = nullptr;

		CudaReductionM(unsigned int N, unsigned int thr = 1024);
		CudaReductionM();
		~CudaReductionM();

		double reduce(double* v1, double* v2, bool withCopy = true, ExtraAction action = ExtraAction::NONE);
	};

	CudaReductionM::CudaReductionM() {}
	CudaReductionM::CudaReductionM(unsigned int N, unsigned int thr)
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
	}
	CudaReductionM::~CudaReductionM()
	{
		cudaFree(res_array); res_array = nullptr;
		delete[] Gp; Gp = nullptr;
		delete[] Np; Np = nullptr;
		delete[] arr; arr = nullptr;
		//delete[] second; second = nullptr;
	}

	double CudaReductionM::reduce(double* v1, double* v2, bool withCopy, ExtraAction action)
	{
		arr[0] = v1;
		second = v2;
		for (unsigned int i = 1; i <= steps; i++)
			arr[i] = res_array;

		for (unsigned int i = 0; i < steps; i++)
		{
			dot_product << < Gp[i], threads, 1024 * sizeof(double) >> > (arr[i], second, Np[i], arr[i + 1], i == 0, i == steps - 1, action);
		}
		if (withCopy) cudaMemcpy(&res, res_array, sizeof(double), cudaMemcpyDeviceToHost);

		return res;
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
	__global__ void matrix_dot_vector(double* res, double* vm, unsigned int N)
	{
		//shared memory for s?
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = A_dev->row[i]; j < A_dev->row[i + 1]; j++)
			{
				s += A_dev->val[j] * vm[A_dev->col[j]];
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
	__global__ void vector_minus_matrix_dot_vector(double* res, double* v, double* vm, unsigned int N)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < N)
		{
			double s = 0;
			for (int j = A_dev->row[i]; j < A_dev->row[i + 1]; j++)
			{
				s += A_dev->val[j] * vm[A_dev->col[j]];
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


__global__ void accept(SparseMatrixCuda* A)
{
	CuCG::A_dev = A;
	//printf("check: %i \n", A->Nfull);
}
__global__ void showA()
{
	printf("check: %i \n", CuCG::A_dev->Nfull);
}

void CUDA_INIT(SparseMatrixCuda& A)
{
	SparseMatrixCuda* Asend;
	cudaMalloc((void**)&Asend, sizeof(SparseMatrixCuda));
	cudaMemcpy(Asend, &A, sizeof(SparseMatrixCuda), cudaMemcpyHostToDevice);
	accept << <1, 1 >> > (Asend);
	//showA << <1, 1 >> > ();
}


void CUDA_BICGSTAB(unsigned int N, double* x, double* x0, double* b, SparseMatrixCuda& A, CudaLaunchSetup kernel_setting)
{
	CuCG::CudaReductionM CR(N);
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



	//cout << "GPU print below " << endl;
	//CuCG::check2 << <1, 1 >> > (x, N);
	//CuCG::check << <1, 1 >> > ();
	//cudaDeviceSynchronize();
	//system("pause");

	//cudaFree(r);
	//cudaFree(r_hat);
	//cudaFree(p);
	//cudaFree(t);
	//cudaFree(s);
	//cudaFree(v);
}
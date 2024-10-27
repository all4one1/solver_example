#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <vector>


#include "CudaReduction/CuReduction.h"
#include "CudaExtras.h"


using std::cout;
using std::endl;
using std::ofstream;


struct CudaIterSolver
{
	int k = 0, write_i = 0, limit = 1000;
	double eps_iter = 1e-8;
	double res = 0, res0 = 0, eps = 0;
	CudaReduction CR;

	CudaIterSolver();
	CudaIterSolver(unsigned int N);



	void solveJacobi_testAsync(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel);

	void solveJacobi(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel);

	void solveJacobi_experimental(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel);

	void auto_test();
};


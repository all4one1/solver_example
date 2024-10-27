#include "CuSolver.h"

__global__ void swap_one(double* f_old, double* f_new, unsigned int N)
{
	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	if (l < N)	f_old[l] = f_new[l];
}
__global__ void kernelSolveJacobi(double* f, double* f0, double* b, int N, SparseMatrixCuda M)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	double s, diag;

	double w = 2.0 / 3.0;
	if (i < N)
	{
		s = 0;
		diag = 0;
		for (int j = M.row[i]; j < M.row[i + 1]; j++)
		{
			s += M.val[j] * f0[M.col[j]];
			if (M.col[j] == i) diag = M.val[j];
		}
		f[i] = f0[i] + (b[i] - s) / diag;
		//f[i] = f0[i] + (b[i] - s) / diag * w;
	}
}


__global__ void check(double* f, int N)
{
//	unsigned int l = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = 0; i < N; i++)
	{
		printf("%f ", f[i]);
	} printf("\n");
}



void CudaIterSolver::auto_test()
{
	//double A[6][6] =
	//{
	//	{ 30,3,4,0,0,0 },
	//	{ 4,22,1,3,0,0 },
	//	{ 5,7,33,6,7,0 },
	//	{ 0,1,2,42,3,3 },
	//	{ 0,0,2,11,52,2 },
	//	{ 0,0,0,3,9,26 },
	//};

	int nval = 24;
	int n = 6;
	double val[24] = { 30, 3, 4, 4, 22, 1, 3, 5, 7, 33, 6, 7, 1, 2, 42, 3, 3, 2, 11, 52, 2, 3, 9, 26 };
	int col[24] = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5 };
	int row[7] = { 0, 3, 7, 12, 17, 21, 24 };

	SparseMatrixCuda SMC(n, nval, val, col, row);


	double fh[6] = { 0, 0, 0, 0, 0, 0 };
	double* d, * d0, * b;
	cudaMalloc((void**)&d, sizeof(double) * n);
	cudaMalloc((void**)&d0, sizeof(double) * n);
	cudaMalloc((void**)&b, sizeof(double) * n);

	cudaMemcpy(d0, fh, sizeof(double) * n, cudaMemcpyHostToDevice);
	double bh[6] = { 1, 2, 3, 3, 2, 1 };
	cudaMemcpy(b, bh, sizeof(double) * n, cudaMemcpyHostToDevice);

	CudaLaunchSetup kernel(6);
	solveJacobi(d, d0, b, n, SMC, kernel);
	cudaMemcpy(fh, d, sizeof(double) * n, cudaMemcpyDeviceToHost);

	cout << "cuda test:   ";
	for (int i = 0; i < n; i++)
	{
		cout << fh[i] << " ";
	} cout << endl;

	double cg[6] =
	{ 0.1826929218e-1,
	0.7636750835e-1,
	0.5570467736e-1,
	0.6371099009e-1,
	0.2193724104e-1,
	0.2351661001e-1 };
	cout << "x should be: ";
	for (int i = 0; i < n; i++)
		cout << cg[i] << " ";
	cout << endl;


	cudaFree(d);
	cudaFree(d0);
	cudaFree(b);
}


CudaIterSolver::CudaIterSolver(){}

CudaIterSolver::CudaIterSolver(unsigned int N)
{
	CR = CudaReduction(N, 1024); 
}

//volatile 
__device__ double res_dev = 1, res0_dev = 0, eps_dev = 0;
__global__ void reduction(double* data, unsigned int n, double* reduced, bool last = false) {
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
		if (last)
		{
			res_dev = shared[0];
			eps_dev = abs(res_dev - res0_dev) / res0_dev;
			res0_dev = res_dev;
		}
	}


}

void CudaIterSolver::solveJacobi_testAsync(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel)
{
	CudaReduction CR(f, N, 1024);
	k = 0;
	eps = 1.0;
	res = 0.0;
	res0 = 0.0;

	cudaMemset(&eps_dev, 0, sizeof(double));

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaStream_t s1, s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	for (k = 0; k < 100000; k++)
	{
		//cudaEventRecord(start, s1);
		//cudaEventRecord(start);
		kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D, 0, s1>> > (f, f0, b, N, M);
		//cudaEventRecord(end);
		//cudaEventSynchronize(end);
		
		swap_one << < kernel.Grid1D, kernel.Block1D, 0, s1 >> > (f0, f, N);
		
		//res = CR.reduce();
		//res = CR.reduce_test();
		

		for (unsigned int i = 0; i < CR.steps; i++)
		{
			reduction << < CR.Gp[i], CR.threads, 1024 * sizeof(double), s1 >> > (CR.arr[i], CR.Np[i], CR.arr[i + 1], i == CR.steps - 1);
		}
		//cudaEventRecord(end, s1);

		//reduction << < CR.Gp[0], CR.threads, 1024 * sizeof(double), s1 >> > (CR.arr[0], CR.Np[0], CR.arr[1], false);
		//reduction << < CR.Gp[1], CR.threads, 1024 * sizeof(double), s1 >> > (CR.arr[1], CR.Np[1], CR.arr[2], true);

		cudaMemcpyFromSymbolAsync(&eps, eps_dev, sizeof(double), 0, cudaMemcpyDeviceToHost, s2);
		


		if (eps < eps_iter)
		{
			//cudaStreamSynchronize(s1);
			break;
		}
		
		//if (k % 1000 == 0) cout << "device k = " << k << ", eps = " << eps << endl;
		
	}
	cout << "device k = " << k << ", eps = " << eps << endl;
	//cout << "t = " << t << endl;
}

void CudaIterSolver::solveJacobi(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel)
{
	CudaReduction CR(f, N, 1024);
	k = 0;
	eps = 1.0;
	res = 0.0;
	res0 = 0.0;

	for (k = 0; k < 200000; k++)
	{

		kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);
		swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);

		res = CR.reduce();
		eps = abs(res - res0) / res0;
		res0 = res;

		if (eps < eps_iter)
		{
			break;
		}

		if (k % 1000 == 0) cout << "device k = " << k << ", eps = " << eps << endl;

	}
	cout << "device k = " << k << ", eps = " << eps << endl;
	//cout << "t = " << t << endl;
}

void CudaIterSolver::solveJacobi_experimental(double* f, double* f0, double* b, int N, SparseMatrixCuda& M, CudaLaunchSetup kernel)
{
	CudaReduction CR(f, N, 1024);
	k = 0;
	eps = 1.0;
	res = 0.0;
	res0 = 0.0;

	for (k = 0; k < 500000; k++)
	{
		if (k % 10000 == 0) cout << "device k = " << k << ", eps = " << eps << endl;
		if (k < 10)
		{
			kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);
			swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);

			res = CR.reduce();
			eps = abs(res - res0) / res0;
			res0 = res;

			if (eps < eps_iter)		break;
		}
		else
		{
			kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);
			swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);

			if (k % 1000 == 0)
			{
				res0 = CR.reduce();		

				kernelSolveJacobi << < kernel.Grid1D, kernel.Block1D >> > (f, f0, b, N, M);
				swap_one << < kernel.Grid1D, kernel.Block1D >> > (f0, f, N);
				k++;
				res = CR.reduce();

				eps = abs(res - res0) / res0;

				if (eps < eps_iter)		break;
			}
		}
		
		
	}
	cout << "device k = " << k << ", eps = " << eps << endl;
}

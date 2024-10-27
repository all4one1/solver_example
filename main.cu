
#include "CudaIterSolver/CuSolver.h"
#include "FromOuterSparse/SparseMatrix.h"
#include "CPUsolver.h"
#include "Extras.h"
#include <fstream>

//#include "TEST.h"
using namespace std;


__global__ void form_rhs(unsigned int N, double* b, double* f0, double coef)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		if (i == 0 || i == N - 1) return;

		b[i] = f0[i] * coef;
	}
}



#include "CG.h"
#include "CudaIterSolver/CuCG.h"

void finite_volume(SparseMatrix &SM, double *b, double a, unsigned int N)
{
	SM.add_line_with_map({ {0, 3 * a}, {1, -a} }, 0);
	for (unsigned int i = 1; i < N - 1; i++)
	{
		double left = -a;
		double center = 2 * a;
		double right = -a;
		SM.add_line_with_map({ { i - 1, left }, {i, center}, {i + 1, right} }, i);
	}
	SM.add_line_with_map({ {N - 2, -a}, {N - 1,  3 * a} }, N - 1);

	for (unsigned int i = 0; i < N; i++)
		b[i] = 0; 

	b[0] = 0; 
	b[N - 1] = 2 * a;
}
void finite_difference(SparseMatrix& SM, double* b, double a, unsigned int N)
{
	N = N - 1;
	SM.add_line_with_map({ {0, 1}}, 0);
	for (unsigned int i = 1; i <= N - 1; i++)
	{
		double left = -a;
		double center = 2 * a;
		double right = -a;
		SM.add_line_with_map({ { i - 1, left }, {i, center}, {i + 1, right} }, i);
	}
	SM.add_line_with_map({ { N, 1 } }, N);

	for (unsigned int i = 0; i <= N; i++)
		b[i] = 0;

	b[0] = 0;
	b[N] = 1;
}



int main()
{
	FuncTimer timer;

	int nx = 10000;
	int N = nx;
	size_t Nbytes = sizeof(double) * N;

	double Lx = 1.0;
	double hx = Lx / (nx);
	auto x_ = [hx](int i) {return 0.5 * hx + i * hx; };

	#define host_double_ptr(ptr) double *##ptr;   ptr = new double[N];  memset(ptr, 0, Nbytes);
	host_double_ptr(f_host);
	host_double_ptr(f0_host);
	host_double_ptr(b_host);

	#define device_double_ptr(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
	device_double_ptr(f_dev);
	device_double_ptr(f0_dev);
	device_double_ptr(b_dev);

	SparseMatrix SM(N);

	//double tau = 0.1;
	//double coef = 0;// 1.0 / tau;
	double a = 1.0 / pow(hx, 1);

	finite_volume(SM, b_host, a, N);
	//finite_difference(SM, b_host, a, N);


	//IterativeSolver hostSolver;	hostSolver.solveJacobi(f_host, f0_host, b_host, N, SM);
	//conjugate_gradient(N, f_host, f0_host, b_host, SM);
	timer.start("HOST");
	//BICGSTAB(N, f_host, f0_host, b_host, SM);
	timer.end("HOST");


	memset(f_host, 0, sizeof(double) * N);
	cudaMemcpy(b_dev, b_host, Nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(f0_dev, f_host, Nbytes, cudaMemcpyHostToDevice);
	//SM.save_full_matrix_with_rhs(4, b_host);
	SparseMatrixCuda SMC(SM.Nfull, SM.nval, SM.val.data(), SM.col.data(), SM.raw.data()); 
	CudaLaunchSetup kernel(SM.Nfull);


	timer.start("CUDA");
	CUDA_BICGSTAB_WITH_GRAPH(N, f_dev, f0_dev, b_dev, SMC, kernel);
	//CUDA_BICGSTAB(N, f_dev, f0_dev, b_dev, SMC, kernel);
	cudaDeviceSynchronize();
	timer.end("CUDA");

	//CudaIterSolver CUsolver;
	//CUsolver.solveJacobi(f_dev, f0_dev, b_dev, N, SMC, kernel);

	cudaMemcpy(f_host, f_dev, Nbytes, cudaMemcpyDeviceToHost);
	cout << f_host[N / 2] << " " << x_(N / 2) << endl;
	//for (int i = 0; i < N; i = i + 1) { cout << f_host[i] << " " << x_(i) << endl; }
	//int di = N < 1000 ? 1 : 10;
	ofstream w("result.dat");	for (int i = 0; i < N; i = i + (N < 1000 ? 1 : 10)) { w << x_(i) << " " << f_host[i] << endl; }
	
	timer.show_info();


	
	return 0;
}



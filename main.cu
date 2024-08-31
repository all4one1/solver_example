
#include "CudaIterSolver/CuSolver.h"
#include "FromOuterSparse/SparseMatrix.h"
#include "CPUsolver.h"
#include "Extras.h"
#include <fstream>
using namespace std;

int main()
{
	FuncTimer timer;
	CudaIterSolver CUsolver;
	//CUsolver.auto_test();

	//SparseMatrixCuda SMC(n, nval, sparse_matrix_elements, column, row);
	//CudaLaunchSetup kernel_settings(n);

	int nx = 500000;
	int N = nx + 1;
	size_t Nbytes = sizeof(double) * N;

	double Lx = 1.0;
	double hx = Lx / (nx);
	auto x_ = [hx](int i) {return i * hx; };

	#define host_double_ptr(ptr) double *##ptr;   ptr = new double[N];  memset(ptr, 0, Nbytes);
	host_double_ptr(f_host);
	host_double_ptr(f0_host);
	host_double_ptr(b_host);

	#define device_double_ptr(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
	device_double_ptr(f_dev);
	device_double_ptr(f0_dev);
	device_double_ptr(b_dev);

	SparseMatrix SM(N);
	
	SM.add_line_with_map({ { 0, 1 } }, 0);
	for (int i = 1; i < nx; i++)
	{
		double a = 1.0 / pow(hx, 2);
		double left = a;
		double center = -2 * a;
		double right = a;
		SM.add_line_with_map({ { i - 1, left }, {i, center}, {i + 1, right} }, i);
	}
	SM.add_line_with_map({ { nx, 1 } }, nx);






	b_host[0] = 0; // bc1
	b_host[nx] = 1; //bc2
	for (int i = 1; i < nx; i++)
	{
		b_host[i] = 0;// 2 * x_(i);
	}
	cudaMemcpy(b_dev, b_host, Nbytes, cudaMemcpyHostToDevice);
	//SM.save_full_matrix_with_rhs(4, b_host);

	IterativeSolver hostSolver;
	timer.start("CPU");
	//hostSolver.solveJacobi(f_host, f0_host, b_host, N, SM);
	timer.end("CPU");



	SparseMatrixCuda SMC(SM.Nfull, SM.nval, SM.val.data(), SM.col.data(), SM.raw.data());
	CudaLaunchSetup kernel(SM.Nfull);

	timer.start("CUDA");
	CUsolver.solveJacobi(f_dev, f0_dev, b_dev, N, SMC, kernel);
	timer.end("CUDA");


	cudaMemcpy(f_host, f_dev, Nbytes, cudaMemcpyDeviceToHost);
	ofstream w("result.dat");	for (int i = 0; i < N; i = i + 10)	{	w << x_(i) << " " << f_host[i] << endl;}







	timer.show_info();

	return 0;
}



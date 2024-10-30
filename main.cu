
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

void finite_volume(SparseMatrix &SM, double *b, double a, unsigned int N, double a_tau = 0)
{
	SM.add_line_with_map({ {0, 3 * a + a_tau}, {1, -a} }, 0);
	for (unsigned int i = 1; i < N - 1; i++)
	{
		double left = -a;
		double center = 2 * a + a_tau;
		double right = -a;
		SM.add_line_with_map({ { i - 1, left }, {i, center}, {i + 1, right} }, i);
	}
	SM.add_line_with_map({ {N - 2, -a}, {N - 1,  3 * a + a_tau} }, N - 1);

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

struct CH_EQUATION
{
	SparseMatrix SM;
	unsigned int N, N2, Nbytes, Nbytes2, stride;
	double eps = (1e-4) * 1;
	double M = 1, tau = 1e-5;
	double alpha = 0.1, beta = 0.1;
	double* C_host, * Mu_host, *F_host, * F0_host, *B_host;
	double* C_dev, * Mu_dev, * F_dev, * B_dev;
	double* x_;
	#define host_double_ptr2(ptr) ptr = new double[N2];  memset(ptr, 0, Nbytes2);
	#define device_double_ptr2(ptr) cudaMalloc((void**)&##ptr, Nbytes2);  cudaMemset(ptr, 0, Nbytes2); 

	CH_EQUATION(unsigned int N_, double a, double b) : N(N_), alpha(a), beta(b)
	{
		stride = N;
		N2 = N * 2;
		Nbytes = N * sizeof(double);
		Nbytes2 = Nbytes * 2;

		host_double_ptr2(F_host);		
		host_double_ptr2(F0_host);
		host_double_ptr2(B_host);
		x_ = new double[N];

		device_double_ptr2(F_dev);
		device_double_ptr2(B_dev);
		C_host = &F_host[0];
		Mu_host = &F_host[stride];
		C_dev = &F_dev[0];
		Mu_dev = &F_dev[stride];

		SM.resize(N2);

		finite_volume_CH(SM, B_host, alpha, beta, N, stride);
		form_rhs_on_host();
	}

	void finite_volume_CH(SparseMatrix& SM, double* b, double alpha, double beta, unsigned int N, unsigned int stride)
	{
		unsigned int n = N - 1;

		for (unsigned int i = 0; i < N; i++)
			SM(i, i) = 1.0;

		for (unsigned int i = 0; i < N; i++)
			SM(i, i + stride) = 2 * alpha;
		for (unsigned int i = 0; i < N - 1; i++)
			SM(i, i + 1 + stride) = -alpha;
		for (unsigned int i = 1; i < N; i++)
			SM(i, i - 1 + stride) = -alpha;

		SM(0, stride) = alpha;
		SM(0, 1 + stride) = -alpha;
		SM(n, n + stride) = alpha;
		SM(n, n + stride - 1) = -alpha;


		for (unsigned int i = 0; i < N; i++)
			SM(i + stride, i) = -2 * beta;
		for (unsigned int i = 0; i < N - 1; i++)
			SM(i + stride, i + 1) = beta;
		for (unsigned int i = 1; i < N; i++)
			SM(i + stride, i - 1) = beta;

		for (unsigned int i = 0; i < N; i++)
			SM(i + stride, i + stride) = 1.0;


		SM(stride, 0) = -beta;
		SM(stride, 1) = beta;
		SM(n + stride, n) = -beta;
		SM(n + stride, n - 1) = beta;

		for (unsigned int i = 0; i < N; i++)
			b[i] = 0;
	}

	void form_rhs_on_host()
	{
		for (int i = 0; i < N; i++)
		{
			B_host[i] = C_host[i];
			B_host[i + stride] = pow(C_host[i], 3) - C_host[i];
		}
	}

	void initial(double* f, int N, double hx, double Lx1 = -1.0, double Lx2 = 1.0)
	{
		double Pi = acos(-1.0);
		auto Gauss = [](double x, double mean, double sigma)
		{
			return 1.0 / (sigma * sqrt(2 * 3.14159)) * exp(-0.5 * pow((x - mean) / sigma, 2));
		};

		double mean = 0, sigma = 0.1;
		for (int i = 0; i < N; i++)
		{
			double x = Lx1 + i * hx + 0.5 * hx;
			x_[i] = x;
			double delta = 0.1;
			f[i] = tanh(x / delta);

			mean = -0.75;		sigma = 0.1;
			f[i] += 0.5 * Gauss(x, mean, sigma) / Gauss(mean, mean, sigma);
		}

	}
};

int main()
{
	FuncTimer timer;

	int nx = 300;
	int N = nx;
	size_t Nbytes = sizeof(double) * N;

	double X0 = -1.0;
	double Lx = 2.0;
	double hx = Lx / (nx);
	auto x_ = [hx, X0](int i) {return 0.5 * hx + i * hx + X0;};

	#define host_double_ptr(ptr) double *##ptr;   ptr = new double[N];  memset(ptr, 0, Nbytes);
	host_double_ptr(f_host);
	host_double_ptr(f0_host);
	host_double_ptr(b_host);

	#define device_double_ptr(ptr) double *##ptr;   cudaMalloc((void**)&##ptr, Nbytes);  cudaMemset(ptr, 0, Nbytes); 
	device_double_ptr(f_dev);
	device_double_ptr(f0_dev);
	device_double_ptr(b_dev);

	SparseMatrix SM(N);



	//ofstream w2("check.dat");	for (int i = 0; i < N; i++)		w2 << CH.x_[i] << " " << CH.C_host[i] << endl;


	


	double alpha = 0.06;
	double beta = 1;
	double tau = 0.1;
	double coef = 1.0 / tau * hx;
	double a = 1.0 / pow(hx, 1) * 0.1;
	double time = 0;
	size_t iter = 0;

	finite_volume(SM, b_host, a, N, coef);

	CH_EQUATION CH(N, alpha, beta);
	CH.initial(CH.C_host, N, hx);
	CH.form_rhs_on_host();

	tau = alpha * hx * hx;

	//finite_difference(SM, b_host, a, N);


	IterativeSolver hostSolver;	



	//conjugate_gradient(N, f_host, f0_host, b_host, SM);
	timer.start("HOST");
	//for (int k = 0; k < 1000; k++)
	while(true)
	{
		iter++;
		time += tau;
		//for (int i = 0; i < N; i = i + 1)			b_host[i] = f_host[i] * coef;		b_host[0] += 0; b_host[N - 1] += 2 * a;
		//hostSolver.solveJacobi(f_host, f0_host, b_host, N, SM);

		//BICGSTAB(N, f_host, f0_host, b_host, SM);


		CH.form_rhs_on_host();
		hostSolver.solveGS(CH.F_host, CH.F0_host, CH.B_host, CH.N2, CH.SM);
		//BICGSTAB(CH.N2, CH.F_host, CH.F0_host, CH.B_host, CH.SM);
		
		
		if (iter % 1000 == 0)
		{
			cout << time << " " << CH.C_host[N / 4] << endl;
			if (time > 1) break;
		}

	}
	timer.end("HOST");
	//for (int i = 0; i < N; i = i + 1) { cout << f_host[i] << " " << x_(i) << endl; }
	//ofstream w3("result.dat");	for (int i = 0; i < N; i = i + (N < 1000 ? 1 : 10)) { w3 << x_(i) << " " << f_host[i] << endl; }
	ofstream w3("result.dat");	for (int i = 0; i < N; i = i + (N < 1000 ? 1 : 10)) { w3 << x_(i) << " " << CH.C_host[i] << endl; }


	timer.show_info();
	return 0;
	memset(f_host, 0, sizeof(double) * N);
	cudaMemcpy(b_dev, b_host, Nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(f0_dev, f_host, Nbytes, cudaMemcpyHostToDevice);
	//SM.save_full_matrix_with_rhs(4, b_host);
	SparseMatrixCuda SMC(SM.Nfull, SM.nval, SM.val.data(), SM.col.data(), SM.raw.data()); 
	CudaLaunchSetup kernel(SM.Nfull);

	BiCGSTAB solver_cg(N, f_dev, f0_dev, b_dev, SMC, kernel);
	//solver_cg.make_graph(f_dev, f0_dev, b_dev, SMC);

	timer.start("CUDA");
	//solver_cg.solve_directly(f_dev, f0_dev, b_dev, SMC);
	solver_cg.solve_with_graph(f_dev, f0_dev, b_dev, SMC);

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



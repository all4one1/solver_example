#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>

#include "FromOuterSparse/SparseMatrix.h"

using std::cout;
using std::endl;
using std::ofstream;

struct IterativeSolver
{
	int k = 0, write_i = 0, limit = 1000;
	double eps_iter = 1e-6;
	ofstream w;
	IterativeSolver()
	{

	}

	void solveGS(double* f, double* f0, double* bb, int NN, SparseMatrix& M, int limit_ = 0)
	{
		if (limit_ != 0) limit = limit_;
		k = 0;
		//limit = 300;
		//for (k = 0; k < 100; k++)
		while (true)
		{
			k++;
			if (k > 20000)
			{
				//std::cout << "limit" << endl;
				break;
			}
			double s = 0;
			for (int j = 0; j < NN; j++)
			{
				s = M.line(j, f);
				f[j] = f[j] + (bb[j] - s) / M[j][j];
			}


			double max = 0;
			double dif;
			for (int i = 0; i < NN; i++)
			{
				dif = abs(f0[i] - f[i]);
				if (dif > max)
					max = dif;
			}
			for (int j = 0; j < NN; j++)
				f0[j] = f[j];

			if (max < eps_iter)	break;
			if (k % 1000 == 0) cout << "host k = " << k << endl;
		}
	}
	void solveJacobi(double* f, double* f0, double* bb, int NN, SparseMatrix& M)
	{
		k = 0;
		for (k = 0; k < 100000; k++)
		{
			double s = 0;
			for (int j = 0; j < NN; j++)
			{
				s = M.line(j, f0);
				f[j] = f0[j] + (bb[j] - s) / M[j][j];
			}


			double max = 0;
			double dif;
			for (int i = 0; i < NN; i++)
			{
				dif = abs(f0[i] - f[i]);
				if (dif > max)
					max = dif;
			}
			for (int j = 0; j < NN; j++)
				f0[j] = f[j];

			if (max < eps_iter)	break;

			if (k % 1000 == 0) cout << "host k = " << k << ", eps = " << max << endl;
		}
		
	}

	void write()
	{
		if (write_i == 0) w.open("iter_solver.dat");
		w << " " << k << endl;
		write_i++;
	}
	void auto_test()
	{
		double A[6][6] =
		{
			{ 30,3,4,0,0,0 },
			{ 4,22,1,3,0,0 },
			{ 5,7,33,6,7,0 },
			{ 0,1,2,42,3,3 },
			{ 0,0,2,11,52,2 },
			{ 0,0,0,3,9,26 },
		};
		double b[6] = { 1, 2, 3, 3, 2, 1 };
		int n = 6;
		double* x = new double[6];
		double* x0 = new double[6];
		//double** SM;
		//SM = new double* [n];
		//for (int i = 0; i < n; i++)
		//	SM[i] = new double[n];

		SparseMatrix SM(6);

		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				SM[i][j] = A[i][j];

		for (int i = 0; i < n; i++)
		{
			x[i] = x0[i] = 0;
		}

		solveGS(x, x0, b, n, SM);

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
		cout << endl << "auto test:   ";
		for (int i = 0; i < n; i++)
			cout << x[i] << " ";
		cout << endl;

		cout << "b should be: 1 2 3 3 2 1" << endl;
		cout << "auto test:   ";
		for (int i = 0; i < n; i++)
		{
			double s = 0;
			for (int j = 0; j < n; j++)
			{
				s += A[i][j] * x[j];
			}
			cout << s << " ";
		} cout << endl;
	}

};
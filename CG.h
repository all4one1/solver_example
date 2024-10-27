#pragma once


namespace mathsolver
{
	void vector_dot_vector(double* v1, double* v2, double& res, unsigned int N)
	{
		double s = 0;
		for (unsigned int i = 0; i < N; i++)
		{
			s += v1[i] * v2[i];
		}
		res = s;
	}

	void matrix_dot_vector(SparseMatrix& SM, double* v, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = SM.line(i, v);
		}
	}

	void vector_add_vector(double* v1, double* v2, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v1[i] + v2[i];
		}
	}
	void vector_minus_vector(double* v1, double* v2, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v1[i] - v2[i];
		}
	}

	void scalar_dot_vector(double* v1, double scalar, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = scalar * v1[i];
		}
	}

	void vector_plus_matrixDotVector(double* v, double coef, SparseMatrix& SM, double* vm, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v[i] + coef * SM.line(i, vm);
		}
	}

	void vector_plus_scalarDotVector(double* v, double coef, double* vs, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v[i] + coef * vs[i];
		}
	}
	void vector_plus_scalarDotVector_plus_scalarDotVector(double* v, double coef1, double* vs1, double coef2, double* vs2, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v[i] + coef1 * vs1[i] + coef2 * vs2[i];
		}
	}


	void vector_dot_matrixDotVector(double* vt, SparseMatrix& SM, double* vm, double* buffer, double& res, unsigned int N)
	{
		matrix_dot_vector(SM, vm, buffer, N);
		vector_dot_vector(vt, buffer, res, N);
	}

	void vector_copy_into_vector(double* v, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = v[i];
		}
	}
	void vector_set_to(double scalar, double* res, unsigned int N)
	{
		for (unsigned int i = 0; i < N; i++)
		{
			res[i] = scalar;
		}
	}
}


void conjugate_gradient(unsigned int N, double* x, double* x0, double* b, SparseMatrix& A)
{
	double rs_old = 0, rs_new = 0;
	double buffer = 0; // buffer2 = 0;
	double* buffer_ptr = new double[N];

	double* r0 = new double[N];
	double* r = new double[N];
	double* p0 = new double[N];
	double* p = new double[N];
	double alpha = 0, beta = 0;

	memset(x, 0, sizeof(double) * N);
	mathsolver::vector_plus_matrixDotVector(b, -1, A, x, r, N);
	mathsolver::vector_copy_into_vector(r, p, N); // p = r


	mathsolver::vector_dot_vector(r, r, rs_old, N);

	unsigned int k = 0;
	while (true)
	{
		k++; 
		if (k > 1000000) break;


		mathsolver::vector_dot_matrixDotVector(p, A, p, buffer_ptr, buffer, N);
		alpha = rs_old / buffer;

		mathsolver::vector_plus_scalarDotVector(x, alpha, p, x, N);
		mathsolver::vector_plus_matrixDotVector(r, -alpha, A, p, r, N);
		mathsolver::vector_dot_vector(r, r, rs_new, N);

		if (sqrt(rs_new) < 1e-6) break;
		
		//system("pause");

		beta = rs_new / rs_old * 1.0;
		mathsolver::vector_plus_scalarDotVector(r, beta, p, p, N);
		rs_old = rs_new;
		


		if (k % 1000 == 0)
		{
			std::cout << k << ", " << sqrt(rs_new) << std::endl;
		}
	}

	cout << "k = " << k << endl;
}

void BICGSTAB(unsigned int N, double* x, double* x0, double* b, SparseMatrix& A)
{
	double rs_old = 1, rs_new = 1;
	double alpha = 1, beta = 1, omega = 1;
	double buffer = 1, buffer2 = 1;

	double* r = new double[N];
	double* r_hat = new double[N];
	double* p = new double[N];
	double* t = new double[N];
	double* s = new double[N];
	double* Ap = new double[N];


	memset(x, 0, sizeof(double) * N);
	memset(Ap, 0, sizeof(double) * N);
	memset(p, 0, sizeof(double) * N);
	//mathsolver::vector_copy_into_vector(x0, x, N);

	mathsolver::vector_plus_matrixDotVector(b, -1, A, x, r, N); // r = b - Ax
	mathsolver::vector_copy_into_vector(r, r_hat, N); // r_hat = r
	mathsolver::vector_copy_into_vector(r, p, N); // p = r
	mathsolver::vector_dot_vector(r_hat, r, rs_old, N); // res = r_hat * r

	double eps = 1e-8;
	unsigned int k = 0;
	while (true)
	{
		k++;	if (k > 1000000) break;


		mathsolver::vector_dot_vector(r_hat, r, rs_new, N); 
		beta = (rs_new / rs_old) * (alpha / omega);
		rs_old = rs_new;



		mathsolver::vector_plus_scalarDotVector_plus_scalarDotVector(r, beta, p, -beta * omega, Ap, p, N);
		mathsolver::matrix_dot_vector(A, p, Ap, N);

		mathsolver::vector_dot_vector(r_hat, Ap, buffer, N);
		alpha = rs_new / buffer;

		mathsolver::vector_plus_scalarDotVector(r, -alpha, Ap, s, N);

		mathsolver::matrix_dot_vector(A, s, t, N);


		mathsolver::vector_dot_vector(t, s, buffer, N);
		mathsolver::vector_dot_vector(t, t, buffer2, N);
		omega = buffer / buffer2;

		mathsolver::vector_plus_scalarDotVector_plus_scalarDotVector(x, alpha, p, omega, s, x, N);
		mathsolver::vector_plus_scalarDotVector(s, -omega, t, r, N);
		
		mathsolver::vector_dot_vector(r, r, buffer, N);
		if (abs(buffer) < eps)
		{
			break;
		}
		if (k % 1000 == 0) cout << k << " " << abs(buffer) << endl;
	}

	cout << k << " " << abs(buffer) << endl;
}
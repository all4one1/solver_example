// this source file is for output, print, debug and other auxiliary stuff

#include "SparseMatrix.h"
#include <iostream>
#include <iomanip>
#include <fstream>
using std::cout;
using std::endl;
using std::ofstream;

void SparseMatrix::show_storage()
{
	double S = double(val.capacity() * 8 + col.capacity() * 4 + raw.capacity() * 4 + diag.capacity() * 8 + type.capacity() * 4);
	int num = int(val.capacity());
	cout << num << " elements, " << S / 1024 / 1024 << " MB approx. matrix memory usage" << " \n\n";
}


void SparseMatrix::save_full_matrix(int precision)
{

	ofstream out("coef.dat");
	ofstream out2("info.dat");
	out << std::fixed << std::setprecision(precision);
	for (unsigned int k = 0; k < raw.size() - 1; k++)
	{
		std::vector <double> line(Nfull);
		std::vector <int> t(Nfull);
		for (int i = 0; i < Nfull; i++)
		{
			line[i] = 0.0;
			//	t[i] = -1;
		}

		for (int j = raw[k]; j < raw[k + 1]; j++)
		{
			line[col[j]] = val[j];
			//	t[col[j]] = type[j];
		}

		for (int i = 0; i < Nfull; i++)
		{
			out << line[i] << " ";
			//	if (t[i] == -1) out2 << "_" << " ";
			//	else out2 << t[i] << " ";
		}
		out << endl;
		out2 << endl;
	}
}

void SparseMatrix::save_full_matrix_with_rhs(int precision, double *b)
{
	ofstream out("coef.dat");
	out << std::fixed << std::setprecision(precision);
	for (unsigned int k = 0; k < raw.size() - 1; k++)
	{
		std::vector <double> line(Nfull);
		std::vector <int> t(Nfull);
		for (int i = 0; i < Nfull; i++)
		{
			line[i] = 0.0;
		}

		for (int j = raw[k]; j < raw[k + 1]; j++)
		{
			line[col[j]] = val[j];
		}

		for (int i = 0; i < Nfull; i++)
		{
			out << line[i] << " ";
		}
		out << "	 " << b[k];
		out << endl;
	}
}


void SparseMatrix::save_full_matrix2()
{
	ofstream m("matrix.dat");
	for (int i = 0; i < Nfull; i++) {
		for (int j = 0; j < Nfull; j++)
		{
			m << get_element(i, j) << " ";
		}
		m << endl;
	}
}

void SparseMatrix::print_index_ij(int l)
{
	if (!(l < nval)) { cout << " out of range " << endl; return; }

	int j = col[l];
	int i;
	for (int q = 0; q < Nfull; q++)
	{
		if (l < raw[q + 1])
		{
			i = q;
			break;
		}
	}

	cout << "[" << i << "][" << j << "]" << endl;
}

void SparseMatrix::save_type()
{
	ofstream m("type.dat");

	for (int i = 0; i < Nfull; i++) {
		for (int j = 0; j < Nfull; j++)
		{
			std::string str = "_____";
			int t = get_type(i, j);
			if (t == center) str = "centr";
			if (t == south) str = "south";
			if (t == north) str = "north";
			if (t == west) str = "west_";
			if (t == east) str = "east_";

			m << str << " ";
		}
		m << endl;
	}
}

void SparseMatrix::print_sequently()
{
	//ofstream seq("seq.dat");
	cout << "Sequential (compressed) format: " << endl;
	for (int i = 0; i < nval; i++)
		cout << val[i] << " ";
	cout << endl;
	for (int i = 0; i < nval; i++)
		cout << col[i] << " ";
	cout << endl;

	for (int i = 0; i < Nfull + 1; i++)
		cout << raw[i] << " ";
	cout << endl;
}

void SparseMatrix::print_all()
{
	for (int i = 0; i < Nfull; i++) {
		for (int j = 0; j < Nfull; j++)
		{
			std::cout << get_element(i, j) << " ";
		}
		std::cout << endl;
	}
	cout << "Number of (non-zero) elements: " << nval << endl;
	[this]() {
		int count_zero = 0;
		for (auto it : val)
			if (abs(it) < 1e-15)
				count_zero++;	
		if (count_zero > 0)
			cout << "Number of zero elements: " << count_zero << endl;
	} ();
	

	

}
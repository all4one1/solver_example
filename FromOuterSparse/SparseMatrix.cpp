#include "SparseMatrix.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream> 
using std::cout;
using std::endl;
using std::ofstream;





SparseMatrix::SparseMatrix()
{
	raw.push_back(0);
#ifdef DEBUG
	printf("SparseMatrix constructor \n");
#endif // DEBUG

}

SparseMatrix::SparseMatrix(int n_)
{
	resize(n_);
}

SparseMatrix::~SparseMatrix()
{
#ifdef DEBUG
	printf("SparseMatrix destructor \n");
#endif // DEBUG
}

void SparseMatrix::add_one_next(double v, int i, int t)
{
	if (abs(v) < zero_threshold) return;
	type.push_back(t);
	val.push_back(v);
	col.push_back(i);
	nval++;
}
void SparseMatrix::insert_one(int place_in_vector, int matrix_row, 	
	double value, int column, int t)
{
	val.insert(val.begin() + place_in_vector, value);
	col.insert(col.begin() + place_in_vector, column);
	type.insert(type.begin() + place_in_vector, t);
	for (unsigned int i = matrix_row + 1; i < raw.size(); i++)
	{
		raw[i]++;
	}
	nval++;
}

void SparseMatrix::erase_one(int place_in_vector, int matrix_row)
{
	val.erase(val.begin() + place_in_vector);
	col.erase(col.begin() + place_in_vector);
	type.erase(type.begin() + place_in_vector);
	for (unsigned int i = matrix_row + 1; i < raw.size(); i++)
	{
		raw[i]--;
	}
	nval--;
}

void SparseMatrix::add_line_with_map(std::map<int, double> elements, int current_line)
{
	for (auto it = elements.begin(); it != elements.end(); ++it)
	{
		add_one_next(it->second, it->first);
	}
	endline(current_line);
}

void SparseMatrix::endline(int l)
{
	l++;
	raw[l] = nval;
}

void SparseMatrix::resize(int n_)
{
	//Nfull = n_;
	//val.clear(); 	
	//col.clear(); 
	//raw.clear();	
	//diag.clear();
	//raw.resize(Nfull + 1);
	//raw[0] = 0;
	//nraw = 0;
	//update_diag();

	*this = SparseMatrix();
	Nfull = n_;
	raw.resize(Nfull + 1);
	nraw = (int)raw.size();
	raw[0] = 0;
}
void SparseMatrix::reset()
{
	resize(Nfull);
}



void SparseMatrix::make_sparse_2d_laplace(int nx, int ny, double a)
{
	int l = -1;
	int off = nx;
	Nfull = nx * ny;
	raw.resize(Nfull + 1);
	raw[0] = 0;
	double ap, ae, aw, an, as;
	ap = 1 + 4 * a;
	ae = -a;
	aw = -a;
	an = -a;
	as = -a;



	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			l = i + off * j;

			ap = 1 + 4 * a;
			if (i == 0) ap -= a;
			if (i == nx - 1) ap -= a;
			if (j == 0) ap -= a;
			if (j == ny - 1) ap -= a;


			if (j > 0)


				add_one_next(as, l - off, south);

			if (i > 0)
				add_one_next(ae, l - 1, east);

			add_one_next(ap, l, center);

			if (i < nx - 1)
				add_one_next(aw, l + 1, west);

			if (j < ny - 1)
				add_one_next(an, l + off, north);

			endline(l);

		}
	}
	nval = (int)val.size();

	update_diag();
}
void SparseMatrix::make_sparse_from_double_array(int n_, double** M)
{
	Nfull = n_;
	resize(Nfull);
	for (int i = 0; i < Nfull; i++) {
		for (int j = 0; j < Nfull; j++)
		{
			if (M[i][j] != 0)
			{
				add_one_next(M[i][j], j);
			}
		}
		endline(i);
	}
	nval = (int)val.size();
}



double SparseMatrix::get_element(int ii, int jj)
{
	double v = 0;
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj) v = val[j];
	}
	return v;
}

int SparseMatrix::get_index(int ii, int jj)
{
	int id = -1;
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj) id = j;
	}
	return id;
}
void SparseMatrix::set_type(int ii, int jj, int t)
{
	int id = -1;
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj) id = j;
	}
	type[id] = t;
}
int SparseMatrix::get_type(int ii, int jj)
{
	int t = -1;
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj) t = type[j];
	}
	return t;
}
void SparseMatrix::update_diag()
{
	diag.resize(Nfull);
	for (int i = 0; i < Nfull; i++)
	{
		diag[i] = get_element(i, i);
	}
}



void SparseMatrix::update(int ii, int jj, double value)
{
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj)
		{
			val[jj] = value;
			return;
		}
	}
	cout << "SparseMatrix::update: warning! value (" << ii << ", " << jj << ") = " << value << endl;
}

// it can set zero value, which is not perfect
// perhaps, overload "=" operator is a solution
double& SparseMatrix::operator()(int ii, int jj)
{
	//cout << "index: " << ii << " " << jj << endl;
	if (!(ii < Nfull) || !(jj < Nfull))
	{
		cout << "bad case " << endl;
		double* v = new double; 	
		return *v;
	}

	int l1 = raw[ii];
	int l2 = raw[ii + 1];

	#ifdef DEBUG
		cout << "l1 = " << l1 << ", l2 = " << l2 << endl;
	#endif // DEBUG


	if (l2 == l1)
	{
		#ifdef DEBUG
			cout << "CASE 0: line does not have an element" << endl;
		#endif // DEBUG
		insert_one(l1, ii, 0.0, jj);
		return val[l1];
	}

	//before 
	if (jj >= 0 && jj < col[l1])
	{
		#ifdef DEBUG
			cout << "CASE 1: before" << endl;
		#endif // DEBUG
		insert_one(l1, ii, 0.0, jj);
		return val[l1];
	}

	//after
	if (jj < Nfull && jj > col[l2 - 1])
	{
		#ifdef DEBUG
			cout << "CASE 2: after" << endl;
		#endif // DEBUG
		insert_one(l2, ii, 0.0, jj);
		return val[l2];
	}


	//inside existing
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (col[j] == jj)
		{
			#ifdef DEBUG
				cout << "CASE 3: update existing ones" << endl;
			#endif // DEBUG
			return val[j];
		}
	}


	//inside non-existing
	for (int j = raw[ii]; j < raw[ii + 1]; j++)
	{
		if (jj > col[j] && jj < col[j + 1])
		{
			#ifdef DEBUG
				cout << "CASE 4: insert within a line" << endl;
			#endif // DEBUG
			insert_one(j + 1, ii, 0.0, jj);
			return val[j + 1];
		}
	}

	cout << "bad case - end " << endl;
	double* v = new double; 	
	return *v;
}


void SparseMatrix::erase_zeros()
{
	//очистка нулей после LU-разложения нужна тут
}

double SparseMatrix::line(int q, double* y)
{
	double s = 0.0;

	for (int j = raw[q]; j < raw[q + 1]; j++)
	{
		s += val[j] * y[col[j]];
	}
	return s;
}

double SparseMatrix::line1(int q, double* y)
{
	double s = 0.0;
	for (int j = raw[q]; j < raw[q + 1]; j++)
	{
		if (col[j] >= q) break;
		s += val[j] * y[col[j]];
	}
	return s;
}

double SparseMatrix::line2(int q, double* y)
{
	double s = 0.0;
	for (int j = raw[q]; j < raw[q + 1]; j++)
		//for (int j = raw[q + 1] - 1; j <= raw[q]; j--)
	{
		if (col[j] >= q + 1) {
			s += val[j] * y[col[j]];
		}
	}
	return s;
}


double SparseMatrix::max_element_abs()
{
	double max = 0;
	double v = 0;
	for (int i = 0; i < nval; i++)
	{
		v = abs(val[i]);
		if (v > max)
			max = v;
	}

	return max;
}

double SparseMatrix::get_diag(int l)
{
	return get_element(l,l);
}


void SparseMatrix::save_compressed_matrix(std::string filename)
{
	std::ofstream w(filename);
	w << Nfull << " " << nval << " " << nraw << endl;

	size_t n = val.size();
	for (size_t i = 0; i < n; i++)
	{
		w << val[i];
		if (i != (n - 1)) w << " ";
		else w << endl;
	}
	n = col.size();
	for (size_t i = 0; i < col.size(); i++)
	{
		w << col[i];
		if (i != (n - 1)) w << " ";
		else w << endl;
	}
	n = raw.size();
	for (size_t i = 0; i < raw.size(); i++)
	{
		w << raw[i];
		if (i != (n - 1)) w << " ";
		else w << endl;
	}
}
void SparseMatrix::save_compressed_matrix_with_rhs(double* b, std::string filename)
{
	save_compressed_matrix(filename);
	std::ofstream w(filename, std::ios_base::app);
	for (int i = 0; i < Nfull; i++)
	{
		w << b[i];
		if (i != (Nfull - 1)) w << " ";
		else w << endl;
	}
}



void SparseMatrix::read_compressed_matrix(std::string filename)
{
	std::ifstream read(filename);
	//std::istringstream iss;
	if (!read.good())
	{
		cout << "Error: no such a file '" << filename << "'" << endl;
		return;
	}
	else
	{
		//std::ostringstream oss;
		//oss << read.rdbuf();
		//iss.str(oss.str());
	}
	std::string line;
	//std::string symbol;
	double f;
	int i;
	//getline(iss, line);
	getline(read, line);
	std::stringstream ss;
	ss << line;
	ss >> Nfull; ss >> nval; ss >> nraw;

	resize(Nfull);

	//values
	getline(read, line);
	ss.clear();
	ss << line;
	while (ss >> f) 
		val.push_back(f);
		
	//columns
	getline(read, line);
	ss.clear();
	ss << line;
	while (ss >> i)
		col.push_back(i);

	//raws
	getline(read, line);
	ss.clear();
	ss << line;
	//because nraw is already allocated
	for (int r = 0; r < nraw; r++) 
	{
		ss >> i;
		raw[r] = i;
	}
}

void SparseMatrix::read_compressed_matrix_with_rhs(double* b, std::string filename)
{
	read_compressed_matrix(filename);

	std::ifstream read(filename);
	std::string line;

	getline(read, line); 
	getline(read, line); 
	getline(read, line); 
	getline(read, line);
	getline(read, line);

	std::stringstream ss;
	ss << line;

	for (int i = 0; i < Nfull; i++)
	{
		ss >> b[i];
	}
}

void SparseMatrix::read_full_matrix_with_rhs(int N, double* b, std::string filename)
{
	Nfull = N;
	resize(Nfull);
	std::ifstream read(filename);
	if (!read.good())
	{
		cout << "Error: no such a file '" << filename << "'" << endl;
		return;
	}

	for (int i = 0; i < N; i++)
	{
		std::string line;
		std::stringstream ss;
		getline(read, line);
		ss << line;
		double f = 0.0;
		std::map<int, double> m;
		for (int j = 0; j < N; j++)
		{
			ss >> f;
			if (f != 0.0)
			{
				m[j] = f;
			}
		} 
		add_line_with_map(m, i);
		if (b != nullptr)
		{
			ss >> b[i];
		}
	}


}

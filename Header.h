#pragma once
#include <iostream>
#include <vector>
#include "FromOuterSparse/SparseMatrix.h"
// Тип для хранения векторов
typedef std::vector<double> Vector;
// Тип для хранения матрицы
typedef std::vector<Vector> Matrix;

// Функция для умножения матрицы на вектор
Vector multiply(const Matrix& A, const Vector& x) {
    int n = A.size();
    Vector result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Функция для скалярного произведения векторов
double dot(const Vector& v1, const Vector& v2) {
    double result = 0.0;
    for (int i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Функция для вычисления нормы вектора
double norm(const Vector& v) {
    return sqrt(dot(v, v));
}

// Функция для вычитания векторов
Vector subtract(const Vector& v1, const Vector& v2) {
    Vector result(v1.size());
    for (int i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

// Функция для умножения вектора на скаляр
Vector multiply(const Vector& v, double scalar) {
    Vector result(v.size());
    for (int i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// Основная функция для итерационного метода
void iterativeMethod(const Matrix& A, const Vector& b, Vector& x0, int maxIterations, double tol) {
    int n = A.size();

    Vector r0 = subtract(b, multiply(A, x0)); // r^0 = b - A * x^0
    Vector r_tilde = r0;                      // r~ = r^0
    Vector p = r0;                             // p^0 = r^0
    Vector v(n, 0.0);                          // v^0 = 0
    Vector t(n, 0.0);                          // t^0 = 0
    Vector s(n, 0.0);                          // s^0 = 0
    double rho_old = 1.0;                      // rho^0 = 1
    double omega_old = 1.0;                    // omega^0 = 1
    double alpha = 1.0;
    double beta = 0.0;

    for (int k = 1; k <= maxIterations; ++k) {
        double rho_new = dot(r_tilde, r0); // rho^k = (r~, r^k-1)

        if (fabs(rho_new) < tol) {
            std::cout << "Solution converged after " << k << " iterations.\n";
            return;
        }

        if (k > 1) {
            beta = (rho_new / rho_old) * (alpha / omega_old); // beta^k = ...
            p = subtract(multiply(r0, beta), multiply(v, omega_old)); // p^k = r^k-1 + ...
        }

        v = multiply(A, p); // v^k = A * p^k
        alpha = rho_new / dot(r_tilde, v); // alpha^k = ...

        s = subtract(r0, multiply(v, alpha)); // s^k = r^k-1 - alpha^k * v^k

        if (norm(s) < tol) {
            x0 = subtract(x0, multiply(p, alpha)); // x^k = x^k-1 + alpha^k * p^k
            std::cout << "Solution converged after " << k << " iterations.\n";
            return;
        }

        t = multiply(A, s); // t^k = A * s^k
        double omega_new = dot(t, s) / dot(t, t); // omega^k = ...

        x0 = subtract(subtract(x0, multiply(p, alpha)), multiply(s, omega_new)); // x^k = ...
        r0 = subtract(s, multiply(t, omega_new)); // r^k = s^k - omega^k * t^k

        if (norm(r0) < tol) {
            std::cout << "Solution converged after " << k << " iterations.\n";
            return;
        }

        rho_old = rho_new;    // rho^k -> rho^k-1
        omega_old = omega_new; // omega^k -> omega^k-1
    }

    std::cout << "Maximum iterations reached.\n";
}


void run(SparseMatrix &SM, double *b, unsigned int N)
{
    Matrix A(N);
    for (int i = 0; i < N; i++)
        A[i].resize(N);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = SM.get_element(i, j);

    Vector B(N);
    Vector x0(N, 0.0);
    for (int i = 0; i < N; i++)
        B[i] = -b[i];


    int maxIterations = 1000; // Максимальное количество итераций
    double tolerance = 1e-6;  // Точность решения

    iterativeMethod(A, B, x0, maxIterations, tolerance);

    // Вывод решения
    std::cout << "Solution:\n";
    for (double xi : x0) {
        std::cout << xi << " " << std::endl;
    }
    std::cout << std::endl;
}
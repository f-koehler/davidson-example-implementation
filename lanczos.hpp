#ifndef LANCZOS_HPP_
#define LANCZOS_HPP_

#include "orthogonalization.hpp"
#include "random.hpp"
#include "types.hpp"
#include <iostream>

#include <Eigen/Eigenvalues>

template <typename Value>
auto lanczos(const Matrix<Value>& A, int m = 20)
{
    const auto rows     = A.rows();
    Matrix<Value> V     = Matrix<Value>::Zero(rows, m - 1);
    Vector<Value> alpha = Vector<Value>::Zero(m);
    Vector<Value> beta  = Vector<Value>::Zero(m - 1);

    V.col(0) = generate_random_unit_vector<Value>(rows);

    for(int j = 0; j < m - 2; ++j) {
        Vector<Value> w = A * V.col(j);
        alpha(j)        = w.dot(V.col(j));
        w               = w - alpha(j) * V.col(j);
        if(j > 0) w -= beta(j) * V.col(j - 1);

        beta(j + 1)  = std::sqrt(w.dot(w));
        if(beta(j+1) < 1e-11) return std::make_tuple(alpha, beta);

        V.col(j + 1) = w / beta(j + 1);

        orthonormalize_mgs(V, j + 2);
    }

    alpha(m - 1) = (A * V.col(m - 1)).dot(V.col(m - 1));

    return std::make_tuple(alpha, beta);
}

template <typename Value>
auto lanczos_lowest(const Matrix<Value>& A, int m = 20)
{
    auto result = lanczos(A, m);

    Eigen::SelfAdjointEigenSolver<Matrix<Value>> solver;
    solver.computeFromTridiagonal(std::get<0>(result), std::get<1>(result));

    auto& eigenvalues  = solver.eigenvalues();
    auto& eigenvectors = solver.eigenvectors();
    EigenSystem<Value, Value> sys(eigenvalues.rows());

    for(int i = 0; i < eigenvalues.rows(); ++i) {
        sys[i] = EigenPair<Value, Value>{eigenvalues[i], eigenvectors.col(i)};
    }

    sys.sort();
    return sys.minimal_eigenvalue_pair();
}

#endif

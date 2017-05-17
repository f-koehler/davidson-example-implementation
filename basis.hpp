#ifndef BASIS_HPP_
#define BASIS_HPP_

#include "types.hpp"

#include <Eigen/SVD>

template <typename Value>
void orthnormalize_gs(Matrix<Value>& V)
{
    const auto size = V.cols();
    for(std::size_t i = 1; i < size; ++i) {
        const auto v = V.col(i);
        for(std::size_t j = 0; j < i; ++j) {
            V.col(i) -= (v.dot(V.col(j)) / V.col(j).dot(V.col(j))) * V.col(j);
        }
    }

    for(std::size_t i = 0; i < size; ++i) {
        V.col(i) /= V.col(i).norm();
    }
}

template <typename Value>
void orthnormalize_mgs(Matrix<Value>& V)
{
    Matrix<Value> U = V;
    const auto size = V.cols();

    U.col(0) = V.col(0) / std::sqrt(V.col(0).dot(V.col(0)));
    for(std::size_t i = 1; i < size; ++i) {
        U.col(i) = V.col(i);
        for(std::size_t j = 0; j < i; ++j) {
            U.col(i) -= (U.col(i).dot(U.col(j)) / U.col(j).dot(U.col(j))) * U.col(j);
        }
        U.col(i) /= std::sqrt(U.col(i).dot(U.col(i)));
    }

    V = U;
}

template <typename Value>
Matrix<Value> generate_orthonormal_basis_matrix(int dimension, std::size_t basis_size)
{
    if(basis_size > dimension) {
        throw std::domain_error("basis_size must be smaller than or equal to dimension");
    }

    Matrix<Value> V = Matrix<Value>::Zero(dimension, basis_size);
    for(std::size_t i = 0; i < basis_size; ++i) {
        V(i, i) = 1.;
    }
    orthnormalize_mgs(V);
    return V;
}

#endif

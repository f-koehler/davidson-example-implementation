#ifndef ORTHOGONALIZATION_HPP_
#define ORTHOGONALIZATION_HPP_

#include "types.hpp"

template <typename Value>
void orthonormalize_cgs(Matrix<Value>& V)
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
void orthonormalize_mgs(Matrix<Value>& V)
{
    Matrix<Value> U = V;
    const auto size = V.cols();

    U.col(0) = V.col(0) / std::sqrt(V.col(0).dot(V.col(0)));
    for(std::size_t i = 1; i < size; ++i) {
        for(std::size_t j = 0; j < i; ++j) {
            U.col(i) -= (U.col(i).dot(U.col(j)) / U.col(j).dot(U.col(j))) * U.col(j);
        }
        U.col(i) /= std::sqrt(U.col(i).dot(U.col(i)));
    }

    V = U;
}

#endif

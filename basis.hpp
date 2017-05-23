#ifndef BASIS_HPP_
#define BASIS_HPP_

#include "types.hpp"
#include "orthogonalization.hpp"

template <typename Value>
bool is_orthonormal_basis_matrix(const Matrix<Value>& matrix,
                                 const typename IsComplex<Value>::FloatType& tolerance = 1e-12)
{
    const auto cols = matrix.cols();

    for(int i = 0; i < cols; ++i) {
        for(int j = 0; j < i; ++j) {
            if(std::sqrt(matrix.col(i).dot(matrix.col(j))) > tolerance) return false;
        }
        if(std::abs(std::sqrt(matrix.col(i).dot(matrix.col(i))) - 1) > tolerance) return false;
    }
    return true;
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
    orthonormalize_mgs(V);
    return V;
}

#endif

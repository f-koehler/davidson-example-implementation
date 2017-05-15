#ifndef BASIS_HPP_
#define BASIS_HPP_

#include "types.hpp"

template <typename Value>
Vector<Value> project(Vector<Value> u, Vector<Value> v)
{
    return (v.dot(u) / u.dot(u)) * v;
}

template <typename Value>
void orthonormalize(Basis<Value>& basis)
{
    const auto size = basis.size();
    for(std::size_t i = 1; i < size; ++i) {
        const auto v = basis[i];
        for(std::size_t j = 0; j < i; ++j) {
            basis[i] -= project(basis[j], v);
        }
    }

    for(std::size_t i = 0; i < size; ++i) {
        basis[i] /= basis[i].norm();
    }
}

template <typename Value>
Basis<Value> generate_orthonormal_basis(int dimension, std::size_t basis_size)
{
    Basis<Value> basis(basis_size, Vector<Value>::Zero(dimension));
    for(std::size_t i = 0; i < basis_size; ++i) {
        basis[i](i) = 1.;
    }
    return basis;
}

template <typename Value>
Matrix<Value> generate_orthonormal_basis_matrix(int dimension, std::size_t basis_size)
{
    if(basis_size > dimension) {
        throw std::domain_error("basis_size must be smaller than or equal to dimension");
    }

    Matrix<Value> V = Matrix<Value>::Zero(dimension, basis_size);
    for(std::size_t i = 0; i < basis_size; ++i) {
        V(i,i) = 1.;
    }
    return V;
}

#endif

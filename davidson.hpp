#ifndef DAVIDSON_HPP_
#define DAVIDSON_HPP_

#include <iomanip>
#include <iostream>

#include "basis.hpp"
#include "diagonalize.hpp"
#include "random.hpp"

template <typename Value>
auto compute_rayleigh_ritz_pairs_hermitian(const Matrix<Value>& A, const Matrix<Value>& V,
                                           int k = 0)
{
    // compute B=V^† * A * V
    const auto B = V.adjoint() * A * V;

    // diagonalize B
    auto eigen_system = compute_eigensystem_hermitian<Value>(B).sort();

    // compute the Rayleigh-Ritz vectors
    for(auto& pair : eigen_system) {
        pair.vec = V * pair.vec;
    }

    if(k) eigen_system.erase(eigen_system.begin() + k, eigen_system.end());

    return eigen_system;
}

template <typename Value>
auto apply_davidson_hermitian(const Matrix<Value>& A, int initial_space_size,
                              const typename IsComplex<Value>::FloatType& tol, int max_iter = 50)
{
    using Float = typename IsComplex<Value>::FloatType;

    // store the RR-Pair with the smallest eigenvalue
    EigenPair<Float, Value> smallest;

    // generate initial trial space
    auto P = generate_orthonormal_basis_matrix<Value>(A.rows(), initial_space_size);

    // construct diagonal matrix
    /* DiagonalMatrix<Value> D(A); */
    Vector<Value> D = A.diagonal();

    // construct identity matrix
    /* DiagonalMatrix<Value> I(A); */
    /* I.setIdentity(); */
    auto I = Vector<Value>(A.rows());
    for(int i = 0; i < A.rows(); ++i) I(i) = 1.;

    for(int j = 0; j < max_iter; ++j) {
        // compute all Rayleigh-Ritz-Pairs and find the smallest eigenvalue
        auto smallest = compute_rayleigh_ritz_pairs_hermitian(A, P, 1)[0];

        // compute the residual
        auto r = A * smallest.vec - smallest.val * smallest.vec;

        // exit if residual is small enough
        auto norm = r.norm();
        if(norm <= tol) return smallest;
        std::clog << "iter: " << j << "\n\tresidual: " << norm
                  << "\n\teigenval: " << std::setprecision(14) << smallest.val << "\n\n";

        // extend trial space
        P.conservativeResize(P.rows(), P.cols() + 1);
        P.col(P.cols() - 1) = (D - smallest.val * I).asDiagonal().inverse() * r;
        orthnormalize_mgs(P);
    }

    std::cerr << "reached max_iter without converging" << '\n';

    return smallest;
}

#endif

#ifndef DAVIDSON_HPP_
#define DAVIDSON_HPP_

#include <iomanip>
#include <iostream>

#include "basis.hpp"
#include "diagonalize.hpp"
#include "random.hpp"

/* template <typename Value> */
/* auto compute_rayleigh_ritz_pairs_hermitian(const Matrix<Value>& A, const Matrix<Value>& V, */
/*                                            int k = 0) */
/* { */
/*     // compute B=V^† * A * V */
/*     const auto B = V.adjoint() * A * V; */

/*     // diagonalize B */
/*     auto eigen_system = compute_eigensystem_hermitian<Value>(B).sort(); */

/*     // compute the Rayleigh-Ritz vectors */
/*     for(auto& pair : eigen_system) { */
/*         pair.vec = V * pair.vec; */
/*     } */

/*     if(k) eigen_system.erase(eigen_system.begin() + k, eigen_system.end()); */

/*     return eigen_system; */
/* } */

/* template <typename Value> */
/* auto apply_davidson_hermitian(const Matrix<Value>& A, int initial_space_size, */
/*                               const typename IsComplex<Value>::FloatType& tol, int max_iter = 50) */
/* { */
/*     using Float = typename IsComplex<Value>::FloatType; */

/*     // store the RR-Pair with the smallest eigenvalue */
/*     EigenPair<Float, Value> smallest; */

/*     // generate initial trial space */
/*     auto P = generate_orthonormal_basis_matrix<Value>(A.rows(), initial_space_size); */
/*     /1* auto P = generate_random_basis_matrix<Value>(A.rows(), initial_space_size); *1/ */

/*     // construct diagonal matrix */
/*     /1* DiagonalMatrix<Value> D(A); *1/ */
/*     Vector<Value> D = A.diagonal(); */

/*     // construct identity matrix */
/*     /1* DiagonalMatrix<Value> I(A); *1/ */
/*     /1* I.setIdentity(); *1/ */
/*     auto I = Vector<Value>(A.rows()); */
/*     for(int i = 0; i < A.rows(); ++i) I(i) = 1.; */

/*     for(int j = 0; j < max_iter; ++j) { */
/*         // compute all Rayleigh-Ritz-Pairs and find the smallest eigenvalue */
/*         auto smallest = compute_rayleigh_ritz_pairs_hermitian(A, P, 1)[0]; */

/*         // compute the residual */
/*         auto r = A * smallest.vec - smallest.val * smallest.vec; */

/*         // exit if residual is small enough */
/*         auto norm = r.norm(); */
/*         if(norm <= tol) return smallest; */
/*         std::clog << "iter: " << j << "\n\tresidual: " << norm */
/*                   << "\n\teigenval: " << std::setprecision(14) << smallest.val << "\n\n"; */

/*         // extend trial space */
/*         P.conservativeResize(P.rows(), P.cols() + 1); */
/*         P.col(P.cols() - 1) = (D - smallest.val * I).asDiagonal().inverse() * r; */
/*         orthonormalize_mgs(P); */
/*     } */

/*     std::cerr << "reached max_iter without converging" << '\n'; */

/*     return smallest; */
/* } */

template <typename Value>
auto apply_davidson_hermitian(const Matrix<Value>& A,
                              const typename IsComplex<Value>::FloatType& tol, int max_iter = 100)
{
    using Float = typename IsComplex<Value>::FloatType;
    const auto rows = A.rows();

    Matrix<Value> V = Matrix<Value>::Zero(rows, 1);
    Matrix<Value> M = Matrix<Value>::Zero(rows, 1);
    Matrix<Value> I = Matrix<Value>::Identity(rows, rows);

    // create random initial guess
    Vector<Value> t = generate_random_unit_vector<Value>(rows);

    for(int m = 1; m < max_iter + 1; ++m) {
        // modified Gram-Schmidt to orthogonalize t
        for(int i = 0; i < m - 1; ++i) t -= V.col(i).dot(t) * V.col(i);

        // add v_m to basis
        V.conservativeResize(rows, m);
        V.col(m - 1) = t / std::sqrt(t.dot(t));

        auto v_A = A * V.col(m - 1);

        // add new column to M
        M.conservativeResize(m, m);
        for(int i = 0; i < m; ++i) {
            M(i, m - 1) = V.col(i).dot(v_A);
            M(m - 1, i) = get_conjugate(M(i, m - 1));
        }

        // find the lowest eigen pair of the upper triangular matrix M
        Eigen::SelfAdjointEigenSolver<Matrix<Value>> solver;
        solver.compute(M);
        auto eval = solver.eigenvalues()(0);
        auto eval_idx = 0;
        for(int i = 1; i < m; ++i) {
            if(solver.eigenvalues()(i) < eval) {
                eval_idx = i;
                eval = solver.eigenvalues()(i);
            }
        }
        Vector<Value> evec = solver.eigenvectors().col(eval_idx);

        // make sure that eigenvector has length 1
        evec /= std::sqrt(evec.dot(evec));

        // compute residual
        auto u = V * evec;
        auto r = A * u - eval * u;
        auto r_norm = std::sqrt(r.dot(r));

        // check for convergence
        if(r_norm < tol) return EigenPair<Float, Value>{eval, evec};

        // log status
        std::clog << "iter: " << m << "\n\tresidual: " << r_norm
                  << "\n\teigenval: " << std::setprecision(14) << eval << "\n\n";

        // compute new t
        Matrix<Value> B = I - u * u.adjoint();

        std::cout << "\n\n" << B << "\n\n";
        if(m == 4) return EigenPair<Float, Value>{};

        B = B * (A - eval * I) * B;
        t = Eigen::HouseholderQR<Matrix<Value>>(B).solve(-r);
        /* t = B.inverse() * (-r); */
    }

    std::cerr << "reached maximum iteration without converging\n";
    return EigenPair<Float, Value>{};
}

#endif

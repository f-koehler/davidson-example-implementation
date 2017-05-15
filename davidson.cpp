#include <algorithm>
#include <iostream>
#include <tuple>
using namespace std;

#include <Eigen/Eigenvalues>

#include "basis.hpp"
#include "random.hpp"

template <typename Value>
auto compute_eigensystem(const Matrix<Value>& matrix)
{
    using Solver =
        typename std::conditional<IsComplex<Value>::value, Eigen::ComplexEigenSolver<Matrix<Value>>,
                                  Eigen::EigenSolver<Matrix<Value>>>::type;
    using Complex = typename IsComplex<Value>::ComplexType;

    Solver solver;
    solver.compute(matrix);

    const auto eigenvalues  = solver.eigenvalues();
    const auto eigenvectors = solver.eigenvectors();

    std::vector<EigenPair<Complex>> pairs(eigenvalues.rows());
    for(int i = 0; i < eigenvalues.rows(); ++i) {
        pairs[i].val = eigenvalues(i);
        pairs[i].vec = eigenvectors.col(i);
    }

    return pairs;
}

template <typename Value>
auto compute_eigensystem_hermitian(const Matrix<Value>& matrix)
{
    using Solver = Eigen::SelfAdjointEigenSolver<Matrix<Value>>;
    using Float  = typename IsComplex<Value>::FloatType;

    Solver solver;
    solver.compute(matrix);

    const auto eigenvalues  = solver.eigenvalues();
    const auto eigenvectors = solver.eigenvectors();

    EigenSystem<Float> pairs(eigenvalues.rows());
    for(int i = 0; i < eigenvalues.rows(); ++i) {
        pairs[i].val = eigenvalues(i);
        pairs[i].vec = eigenvectors.col(i);
    }

    pairs.sort();

    return pairs;
}

template <typename Value>
auto compute_rayleigh_ritz_pairs_hermitian(const Matrix<Value>& A, const Matrix<Value>& V)
{
    // compute B=V^† * A * V
    const auto B = V.adjoint() * A * V;

    // diagonalize B
    auto eigen_system = compute_eigensystem_hermitian<Value>(B);

    // compute the Rayleigh-Ritz vectors
    for(auto& pair : eigen_system) {
        pair.vec = V * pair.vec;
    }

    // return the eigen system
    return eigen_system;
}

template <typename Value>
auto apply_davidson_hermitian(const Matrix<Value>& A, int k,
                              const typename IsComplex<Value>::FloatType& val,
                              const typename IsComplex<Value>::FloatType& tol, int max_iter = 50)
{
    using Float = typename IsComplex<Value>::FloatType;

    // construct diagonal matrix
    Vector<Value> D = A.diagonal();

    // construct identity matrix
    auto I = Vector<Value>(A.rows());
    for(int i = 0; i < A.rows(); ++i) I(i) = 1.;

    // construct initial trial space
    auto P = generate_orthonormal_basis_matrix<double>(A.rows(), k);

    // compute all Rayleigh-Ritz-Pairs of the initial trial space
    auto pairs = compute_rayleigh_ritz_pairs_hermitian(A, P);

    // find pair matching desired eigenvalue best
    auto pos   = pairs.find_best_matching_pair(val);
    auto theta = pos->val;
    auto y     = pos->vec;
    pairs.clear();

    // compute residual vector
    Vector<Value> r = A * y - theta * y;

    // check for the unlikely case that the RR-eigenvalue is accurate
    if(r.norm() <= tol) return EigenPair<Value>{theta, y};

    // perform iterations
    for(int j = 0; j < max_iter; ++j) {
        // extend trial space
        P.conservativeResize(P.rows(), P.cols() + 1);
        P.col(P.cols() - 1) = (D - theta * I).asDiagonal().inverse() * r;
        orthonormalize(P);

        // compute all Rayleigh-Ritz-Pairs of the new trial space
        pairs = compute_rayleigh_ritz_pairs_hermitian(A, P);

        // find pair matching desired eigenvalue best
        pos   = pairs.find_best_matching_pair(val);
        theta = pos->val;
        y     = pos->vec;
        pairs.clear();

        // compute the residual
        r = A * y - theta * y;

        // exit if residual is small enough
        auto norm = r.norm();
        clog << "iter: " << j << "\tresidual: " << norm << "\teigenval: " << theta << '\n';
        if(norm <= tol) return EigenPair<double>{theta, y};
    }

    cerr << "reached max_iter without converging" << '\n';

    return EigenPair<double>{theta, y};
}

int main()
{
    const auto A = generate_random_hermitian_matrix<double>(1024, 1024);
    cout << apply_davidson_hermitian(A, 200, 1.1e4, 1e-11, 200) << '\n';
}

#ifndef DIAGONALIZE_HPP_
#define DIAGONALIZE_HPP_

#include <Eigen/Eigenvalues>

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

#endif

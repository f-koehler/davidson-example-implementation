#ifndef TYPES_HPP_
#define TYPES_HPP_

#define EIGEN_USE_MKL_ALL

#include <algorithm>
#include <cmath>
#include <complex>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <typeinfo>

#include <Eigen/Dense>

template <typename Value>
using Vector = Eigen::Matrix<Value, Eigen::Dynamic, 1>;

template <typename Value>
using Matrix = Eigen::Matrix<Value, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Value>
using DiagonalMatrix = Eigen::DiagonalMatrix<Value, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Value>
struct IsComplex {
    using FloatType             = Value;
    using ComplexType           = std::complex<Value>;
    static constexpr bool value = false;
};

template <typename Float>
struct IsComplex<std::complex<Float>> {
    using FloatType             = Float;
    using ComplexType           = std::complex<Float>;
    static constexpr bool value = true;
};

template <typename Value, typename VectorValue = Value>
struct EigenPair {
    Value val;
    Vector<VectorValue> vec;
};

template <typename Value, typename VectorValue = Value>
std::ostream& operator<<(std::ostream& strm, const EigenPair<VectorValue>& pair)
{
    strm << pair.val << " :\t" << pair.vec.transpose();
    return strm;
}

template <typename Value, typename VectorValue = Value>
struct EigenSystem : public std::vector<EigenPair<Value, VectorValue>> {
    using std::vector<EigenPair<Value, VectorValue>>::vector;

    auto& sort()
    {
        static_assert(!IsComplex<Value>::value, "complex numbers cannot be ordered");
        std::sort(this->begin(), this->end(),
                  [](const EigenPair<Value, VectorValue>& a,
                     const EigenPair<Value, VectorValue>& b) { return a.val < b.val; });
        return *this;
    }


    auto minimal_eigenvalue_pair()
    {
        return std::min_element(
            this->begin(), this->end(),
            [](const EigenPair<Value, VectorValue>& a, const EigenPair<Value, VectorValue>& b) {
                return a.val < b.val;
            });
    }
    auto minimal_eigenvalue_pair() const
    {
        return std::min_element(
            this->begin(), this->end(),
            [](const EigenPair<Value, VectorValue>& a, const EigenPair<Value, VectorValue>& b) {
                return a.val < b.val;
            });
    }

    auto find_best_matching_pair(const Value& eigenvalue)
    {
        return std::min_element(
            this->begin(), this->end(), [&eigenvalue](const EigenPair<Value, VectorValue>& a,
                                                      const EigenPair<Value, VectorValue>& b) {
                return std::norm(a.val - eigenvalue) < std::norm(b.val - eigenvalue);
            });
    }
    auto find_best_matching_pair(const Value& eigenvalue) const
    {
        return std::min_element(
            this->begin(), this->end(), [&eigenvalue](const EigenPair<Value, VectorValue>& a,
                                                      const EigenPair<Value, VectorValue>& b) {
                return std::norm(a.val - eigenvalue) < std::norm(b.val - eigenvalue);
            });
    }
};

#endif

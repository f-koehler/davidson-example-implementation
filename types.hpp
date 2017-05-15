#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <algorithm>
#include <complex>
#include <ostream>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>

template <typename Value>
using Vector = Eigen::Matrix<Value, Eigen::Dynamic, 1>;

template <typename Value>
using Matrix = Eigen::Matrix<Value, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Value>
using Basis = std::vector<Vector<Value>>;

template <typename Value>
struct IsComplex {
    using FloatType = Value;
    using ComplexType = std::complex<Value>;
    static constexpr bool value = false;
};

template <typename Float>
struct IsComplex<std::complex<Float>> {
    using FloatType = Float;
    using ComplexType = std::complex<Float>;
    static constexpr bool value = true;
};

template <typename Value>
struct EigenPair {
    Value val;
    Vector<Value> vec;
};

template <typename Value>
std::ostream& operator<<(std::ostream& strm, const EigenPair<Value>& pair)
{
    strm << pair.val << " :\t" << pair.vec.transpose();
    return strm;
}

template <typename Value>
struct EigenSystem : public std::vector<EigenPair<Value>> {
    using std::vector<EigenPair<Value>>::vector;

    void sort() {
        static_assert(!IsComplex<Value>::value, "complex numbers cannot be ordered");
        std::sort(
            this->begin(), this->end(),
            [](const EigenPair<Value>& a, const EigenPair<Value>& b) { return a.val < b.val; });
    }

    auto minimal_eigenvalue_pair() const
    {
        return *std::min_element(
            this->begin(), this->end(),
            [](const EigenPair<Value>& a, const EigenPair<Value>& b) { return a.val < b.val; });
    }
};

#endif

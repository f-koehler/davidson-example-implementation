#ifndef RANDOM_HPP_
#define RANDOM_HPP_

#include <random>

#include "types.hpp"

auto& get_random_number_generator()
{
    static std::mt19937_64 prng;
    return prng;
}

template <typename Value>
typename std::enable_if<!IsComplex<Value>::value, Vector<Value>>::type
generate_random_vector(int dimensions, Value min = -10., Value max = 10.)
{
    Vector<Value> vec(dimensions);
    std::uniform_real_distribution<Value> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < dimensions; ++i) {
        vec(i) = dist(prng);
    }
    return vec;
}

template <typename Value>
typename std::enable_if<IsComplex<Value>::value, Vector<Value>>::type
generate_random_vector(int dimensions, typename Value::value_type min = -10.,
                       typename Value::value_type max = 10.)
{
    Vector<Value> vec(dimensions);
    std::uniform_real_distribution<typename Value::value_type> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < dimensions; ++i) {
        vec(i) = Value{dist(prng), dist(prng)};
    }
    return vec;
}

template <typename Value>
typename std::enable_if<!IsComplex<Value>::value, Matrix<Value>>::type
generate_random_matrix(int rows, int cols, Value min = -10., Value max = 10.)
{
    Matrix<Value> matrix(rows, cols);
    std::uniform_real_distribution<Value> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix(i, j) = dist(prng);
        }
    }
    return matrix;
}

template <typename Value>
typename std::enable_if<IsComplex<Value>::value, Matrix<Value>>::type
generate_random_matrix(int rows, int cols, typename Value::value_type min = -10.,
                       typename Value::value_type max = 10.)
{
    Matrix<Value> matrix(rows, cols);
    std::uniform_real_distribution<typename Value::value_type> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            matrix(i, j) = Value{dist(prng), dist(prng)};
        }
    }
    return matrix;
}

template <typename Value>
typename std::enable_if<!IsComplex<Value>::value, Matrix<Value>>::type
generate_random_hermitian_matrix(int rows, int cols, Value min = -10., Value max = 10.)
{
    Matrix<Value> matrix(rows, cols);
    std::uniform_real_distribution<Value> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < i; ++j) {
            matrix(j, i) = matrix(i, j) = dist(prng);
        }
        matrix(i, i) = dist(prng);
    }
    return matrix;
}

template <typename Value>
typename std::enable_if<IsComplex<Value>::value, Matrix<Value>>::type
generate_random_hermitian_matrix(int rows, int cols,
                                 typename IsComplex<Value>::FloatType min = -10.,
                                 typename IsComplex<Value>::FloatType max = 10.)
{
    Matrix<Value> matrix(rows, cols);
    std::uniform_real_distribution<typename IsComplex<Value>::FloatType> dist(min, max);
    auto& prng = get_random_number_generator();
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < i; ++j) {
            matrix(i, j) = Value{dist(prng), dist(prng)};
            matrix(j, i) = std::conj(matrix(i, j));
        }
        matrix(i, i) = dist(prng);
    }
    return matrix;
}

#endif

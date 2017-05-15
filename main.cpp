#include <algorithm>
#include <iostream>
#include <tuple>
using namespace std;

#include "davidson.hpp"
#include "random.hpp"

int main()
{
    const auto A = generate_random_hermitian_matrix<double>(1024, 1024);
    cout << apply_davidson_hermitian(A, 200, 1.1e4, 1e-11, 200).val << '\n';
}

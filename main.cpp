#include <algorithm>
#include <iostream>
#include <tuple>
using namespace std;

#include "davidson.hpp"
#include "random.hpp"

int main()
{
    const auto A  = generate_random_hermitian_matrix<double>(400, 400);
    const auto es = compute_eigensystem_hermitian(A);
    cout << "result:\t" << setprecision(15) << apply_davidson_hermitian(A, 80, 400., 1e-11, 300).val << '\n';
}

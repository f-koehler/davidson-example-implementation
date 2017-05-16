#include <algorithm>
#include <iostream>
#include <tuple>
using namespace std;

#include "davidson.hpp"
#include "random.hpp"
#include "timing.hpp"

int main()
{
    const auto A = generate_random_hermitian_matrix<double>(2000, 2000);

    StopWatch watch_davidson;
    const auto result_davidson = apply_davidson_hermitian(A, 100, 513., 1e-11, 300);
    watch_davidson.stop();

    StopWatch watch_exact;
    const auto result_exact = compute_eigensystem_hermitian(A);
    watch_exact.stop();

    cout << "Davidson:\n\ttime:\t" << watch_davidson << "\n\tresult:\t" << result_davidson.val
         << '\n';
    cout << "Exact:\n\ttime:\t" << watch_exact << "\n";
}

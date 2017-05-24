#include <algorithm>
#include <iostream>
#include <tuple>
using namespace std;

#include "davidson.hpp"
#include "lanczos.hpp"
#include "random.hpp"
#include "timing.hpp"

int main()
{
    const auto A = generate_random_hermitian_matrix<double>(10, 10);

    StopWatch watch_davidson;
    const auto result_davidson = apply_davidson_hermitian(A, 1e-10, 500);
    watch_davidson.stop();

    StopWatch watch_lanczos;
    const auto result_lanczos = lanczos_lowest(A, 200);
    watch_lanczos.stop();

    StopWatch watch_full;
    const auto result_full = compute_eigensystem_hermitian(A).minimal_eigenvalue_pair();
    watch_full.stop();

    cout << "Davidson:\n\ttime:\t" << watch_davidson << "\n\tresult:\t" << setprecision(14)
         << result_davidson.val << '\n';

    cout << "Lanczos: \n\ttime:\t" << watch_lanczos << "\n\tresult:\t" << setprecision(14)
         << result_lanczos.val << '\n';

    cout << "Full:    \n\ttime:\t" << watch_full << "\n\tresult:\t" << setprecision(14)
         << result_full.val << '\n';
}

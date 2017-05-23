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
    const auto result_davidson = apply_davidson_hermitian(A, 10, 1e-10, 500);
    watch_davidson.stop();

    cout << "Davidson:\n\ttime:\t" << watch_davidson << "\n\tresult:\t" << setprecision(14)
         << result_davidson.val << '\n';
}

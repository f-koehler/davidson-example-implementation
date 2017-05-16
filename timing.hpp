#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <chrono>
#include <iomanip>
#include <ostream>

struct StopWatch {
    using Clock     = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    static Clock clock;

    TimePoint start, end;

    StopWatch() { start = clock.now(); }

    auto& stop()
    {
        end = clock.now();
        return *this;
    }
};
StopWatch::Clock StopWatch::clock = std::chrono::high_resolution_clock();

std::ostream& operator<<(std::ostream& strm, const StopWatch& watch)
{
    strm << std::chrono::duration_cast<std::chrono::milliseconds>(watch.end - watch.start).count()
         << "ms";
    return strm;
}

#endif

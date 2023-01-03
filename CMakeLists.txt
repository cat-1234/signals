cmake_minimum_required(VERSION 3.23)
project(signals)

set(CMAKE_CXX_STANDARD 23)

# set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

# find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -O3 -mavx -mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f -Wall -Wextra")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -O3 -march=native -Wall -Wextra")

add_executable(signals main.cpp)

set(benchmark_LIBRARIES "benchmark::benchmark")
target_link_libraries(signals PRIVATE ${benchmark_LIBRARIES})

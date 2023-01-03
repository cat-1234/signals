#pragma once

#include <vector>
#include <random>


void fill_random_1n1(std::vector<int>& array)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    auto dist = std::discrete_distribution<>{{50,50}};
    constexpr int values[]{1,-1};
    const auto n = array.size();
    for (std::size_t i = 0; i < n; ++i)
        array[i] = values[dist(rng)];
}


void fill_random_pos(std::vector<int>& array)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    constexpr auto max_value = 1'000;
    auto dist = std::uniform_int_distribution<>{1, max_value};
    const auto n = array.size();
    for (std::size_t i = 0; i < n; ++i)
        array[i] = dist(rng);
}


template <typename T, unsigned long N>
void fill_random_1n1(T(&array)[N])
{
    std::random_device rd;
    std::mt19937 rng(rd());
    auto dist = std::discrete_distribution<>{{50,50}};
    constexpr int values[]{1,-1};
    for (unsigned long i = 0; i < N; ++i)
        array[i] = values[dist(rng)];
}


template <typename T, unsigned long N>
void fill_random_pos(T(&array)[N])
{
    std::random_device rd;
    std::mt19937 rng(rd());
    constexpr auto max_value = 1'000;
    auto dist = std::uniform_int_distribution<>{1, max_value};
    for (unsigned long i = 0; i < N; ++i)
        array[i] = dist(rng);
}


template <typename T>
void fill_random_pos(T* array, int N)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    constexpr auto max_value = 1'000;
    auto dist = std::uniform_int_distribution<>{1, max_value};
    for (auto i = 0; i < N; ++i)
        array[i] = dist(rng);
}

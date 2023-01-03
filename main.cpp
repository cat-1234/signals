#include <cstdint>
#include <cstring>
#include <cassert>
#include <limits>
#include <concepts>
#include <type_traits>
#include <tuple>
#include <iostream>
#include <immintrin.h>

#include <benchmark/benchmark.h>

#include "random_fill.hpp" // includes vector


static_assert(sizeof(uint64_t) == 8); // may be smaller on some machines


using ind_size_t = uint8_t;
static_assert(std::is_unsigned_v<ind_size_t>);


constexpr std::size_t IND_SIZE = sizeof(ind_size_t)*8;
constexpr auto IND_T_MAX_SLL = ind_size_t{0b1} << (IND_SIZE - 1);
constexpr auto IND_T_MAX = std::numeric_limits<ind_size_t>::max();


template <typename T>
concept IsSigned = std::is_signed_v<T>;


struct trends_bitmap
{
    ind_size_t bitmaps[2];
    constexpr trends_bitmap() noexcept : bitmaps() {}
    [[gnu::always_inline]]
    constexpr void add_trend(int64_t dir) noexcept // dir is 0/1 for 1/-1 t (use int64_t to avoid assemly upcasting for idx on 64bit machine)
    {
        bitmaps[1 - dir] >>= 1;
        bitmaps[dir] = (bitmaps[dir] >> 1) | IND_T_MAX_SLL;
    }
};

auto trends = trends_bitmap{};


struct alignas(64) mm512_buffer
{
    static constexpr auto SIZE = 512/IND_SIZE;
    ind_size_t data[SIZE];
    constexpr mm512_buffer() noexcept : data()
    {
        for (std::size_t i = 0; i < SIZE; ++i)
            data[i] = IND_T_MAX;
    }
};


template <unsigned long N>
struct indicator
{
    ind_size_t bitmaps[2];
    template <IsSigned T>
    constexpr indicator(const T (&trends)[N]) noexcept : bitmaps()
    {
        static_assert(N <= IND_SIZE);
        for (unsigned long i = 0; i < N; ++i) // assumes trends only holds -1/+1 values
        {
            const auto idx = -(trends[i] >> 1); // arithmetic shift
            bitmaps[1 - idx] >>= 1;
            bitmaps[idx] = (bitmaps[idx] >> 1) | IND_T_MAX_SLL;
        }
    }
};

constexpr auto N_INDICATORS = 4;
constexpr auto cxpr_ind1 = indicator{{1, 1, 1, 1}};
constexpr auto cxpr_ind2 = indicator{{1, -1, -1, 1}};
constexpr auto cxpr_ind3 = indicator{{1, 1, 1, -1}};
constexpr auto cxpr_ind4 = indicator{{1, -1, 1, 1}};
constexpr auto indicators = std::tie(cxpr_ind1, cxpr_ind2, cxpr_ind3, cxpr_ind4);
static_assert(cxpr_ind1.bitmaps[0] == ind_size_t{0b1111} << (IND_SIZE - 4));


// google benchmark: ~4ns per price for one cxpr indicator cmp, ~7-8 for 4 indicator patterns
struct trends_indicator_cmp
{
    template <unsigned long M = 0>
    [[gnu::always_inline]]
    static constexpr int cmp()
    {
        if constexpr (M < N_INDICATORS)
        {
            const auto& idctr = std::get<M>(indicators);
            int curr = (
                        ((idctr.bitmaps[0] & trends.bitmaps[0]) ^ idctr.bitmaps[0]) ^ 
                        ((idctr.bitmaps[1] & trends.bitmaps[1]) ^ idctr.bitmaps[1])
                       ) == 0;
            int next = cmp<M + 1>();
            return curr + next;
        }
        return 0;
    }
};


template <unsigned long N>
struct buffer_loader
{
    static void load_buffers(__m512i* bitmaps, const void* buff1, const void* buff2)
    {
        bitmaps[0] = _mm512_load_si512((__m512i*)buff1);
        bitmaps[1] = _mm512_load_si512((__m512i*)buff2);
    }
};


template <>
struct buffer_loader<64>
{
    static void load_buffers(__m512i* bitmaps, const void* buff1, const void* buff2)
    {
        bitmaps[0] = _mm512_load_epi64(buff1);
        bitmaps[1] = _mm512_load_epi64(buff2);
    }
};


template <>
struct buffer_loader<32>
{
    static void load_buffers(__m512i* bitmaps, const void* buff1, const void* buff2)
    {
        bitmaps[0] = _mm512_load_epi32(buff1);
        bitmaps[1] = _mm512_load_epi32(buff2);
    }
};


struct mm_indicator
{
    __m512i bitmaps[2];
    static constexpr auto MAX_N_INDICATORS = 512/IND_SIZE;
    mm512_buffer buffers[2];
    short idx;
    constexpr auto add_indicators_to_buffers(){}
    template <IsSigned T, unsigned long N, IsSigned ...Ts, unsigned long ...M>
    constexpr auto add_indicators_to_buffers(const T(&front)[N], const Ts(&... indicators)[M])
    {
        static_assert(N <= IND_SIZE);
        ind_size_t ind_bits[2]{};
        for (std::size_t i = 0; i < N; ++i)
        {
            const auto bitmap_idx = -(front[i] >> 1);
            ind_bits[1-bitmap_idx] >>= 1;
            ind_bits[bitmap_idx] = (ind_bits[bitmap_idx] >> 1) | IND_T_MAX_SLL;
        }
        buffers[0].data[idx] = ind_bits[0];
        buffers[1].data[idx] = ind_bits[1];
        ++idx;
        add_indicators_to_buffers(indicators...);
    }
    template <IsSigned ...Ts, unsigned long ...M>
    mm_indicator(const Ts(&... indicators)[M]) : bitmaps{}, buffers{}, idx{}
    {
        static_assert(sizeof...(Ts) <= MAX_N_INDICATORS);
        add_indicators_to_buffers(indicators...);
        buffer_loader<IND_SIZE>::load_buffers(&bitmaps[0], &buffers[0].data[0], &buffers[1].data[0]);
    }
};


const auto m_mm_indicator = mm_indicator
{
    {1, 1, 1, 1},
    {1, -1, -1, 1},
    {1, 1, 1, -1},
    {1, -1, 1, 1}
};


// same performance as above (N = 32)
// const auto m_mm_indicator = mm_indicator
// {
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1},
//     {1, 1, 1, 1, 1, 1, 1, 1},
//     {1, 1, 1, 1, -1, -1, -1, -1},
//     {1, 1, 1, 1, -1, -1, 1},
//     {-1, -1, -1, -1, -1}
// };
                

template <unsigned long IndicatorSize>
struct mm_trends_indicator_cmp
{};


template <>
struct mm_trends_indicator_cmp<64>
{
    [[gnu::always_inline]]
    static auto cmp()
    {
        auto bcst_pos = _mm512_set1_epi64(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, m_mm_indicator.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi64_mask(and_pos, m_mm_indicator.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi64(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, m_mm_indicator.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi64_mask(and_neg, m_mm_indicator.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


template <>
struct mm_trends_indicator_cmp<32>
{
    [[gnu::always_inline]]
    static auto cmp()
    {
        auto bcst_pos = _mm512_set1_epi32(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, m_mm_indicator.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi32_mask(and_pos, m_mm_indicator.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi32(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, m_mm_indicator.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi32_mask(and_neg, m_mm_indicator.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


template <>
struct mm_trends_indicator_cmp<16>
{
    [[gnu::always_inline]]
    static auto cmp()
    {
        auto bcst_pos = _mm512_set1_epi16(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, m_mm_indicator.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi16_mask(and_pos, m_mm_indicator.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi16(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, m_mm_indicator.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi16_mask(and_neg, m_mm_indicator.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


template <>
struct mm_trends_indicator_cmp<8>
{
    [[gnu::always_inline]]
    static auto cmp()
    {
        auto bcst_pos = _mm512_set1_epi8(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, m_mm_indicator.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi8_mask(and_pos, m_mm_indicator.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi8(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, m_mm_indicator.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi8_mask(and_neg, m_mm_indicator.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcountll(matches);
    }
};


int64_t position = 0;


auto run_signals()
{
    constexpr auto i64_shift = 63;
    constexpr int prices[]{1, 2, 3, 4, 5, 4, 3, 5}; 
    position = 0;
    for (auto i = 1; i < std::size(prices); ++i)
    {
        int64_t diff = prices[i] - prices[i -1]; // use int64_t to avoid extra promotion to QWORD in assembly
        if (diff)
        {
            trends.add_trend(-(diff>>i64_shift));
            position += mm_trends_indicator_cmp<IND_SIZE>::cmp();
        }
    }
    return 1 - (position == 3);
}


static void BM_cmp(benchmark::State& state)
{
    const auto m_prices_size = state.range(0); // i64
    auto m_prices = new int[m_prices_size];
    fill_random_pos(m_prices, m_prices_size);
    constexpr auto i64_shift = sizeof(int64_t)*8 - 1;
    for (auto _ : state)
    {
        trends.bitmaps[0] = 0;
        trends.bitmaps[1] = 0;
        position = 0;
        for (auto i = 1; i < m_prices_size; ++i)
        {
            int64_t diff = m_prices[i] - m_prices[i - 1];
            if (diff)
            {
                trends.add_trend(-(diff>>i64_shift));
                position += mm_trends_indicator_cmp<IND_SIZE>::cmp();
            }
        }
    }
    state.SetComplexityN(m_prices_size);
//     std::cout << "position: " << position << std::endl;
}


BENCHMARK(BM_cmp)->RangeMultiplier(10)->Range(10, 10'000'000)->Complexity(benchmark::oN);
BENCHMARK_MAIN();


// int main(int ac, char** av)
// {
//     return run_signals();
// }

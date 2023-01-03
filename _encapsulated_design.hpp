#include <cstdint>
#include <cstring>
#include <limits>
#include <concepts>
#include <type_traits>
#include <vector>
#include <tuple>
#include <bitset>
#include <immintrin.h>

#include <benchmark/benchmark.h>

#include "random_fill.hpp"


using ind_size_t = uint16_t;
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
    // use int64_t to avoid extra assembly promotion to QWORD for idx
    [[gnu::always_inline]]
    constexpr void add_trend(int64_t dir) noexcept // dir is 0/1 for 1/-1 t
    {
        bitmaps[1 - dir] >>= 1;
        bitmaps[dir] = (bitmaps[dir] >> 1) | IND_T_MAX_SLL;
    }
};


struct vec_indicator
{
    ind_size_t bitmaps[2];
    template <IsSigned T>
    vec_indicator(const std::vector<T>& trends) : bitmaps()
    {
        const auto n = trends.size();
        assert(n <= IND_SIZE);
        for (auto i = 0; i < n; ++i) // assumes trends only holds -1/+1 values
        {
            auto idx = -(trends[i] >> 1); // arithmetic for signed type
            bitmaps[1 - idx] >>= 1;
            bitmaps[idx] = (bitmaps[idx] >> 1) | IND_T_MAX_SLL;
        }
    }
};


struct alignas(64) mm512_buffer
{
    static constexpr int SIZE = 512/IND_SIZE;
    ind_size_t data[SIZE];
    constexpr mm512_buffer() noexcept : data()
    {
        for (auto i = 0; i < SIZE; ++i)
            data[i] = IND_T_MAX;
    }
};

struct vec_mm_indicator
{
    static constexpr auto MAX_N_INDICATORS = 512/IND_SIZE;
    __m512i bitmaps[2];
    short idx;
    mm512_buffer buffers[2];
    template <IsSigned T>
    void add_indicators_to_buffers(const std::vector<std::vector<T>>& trends)
    {
        const auto n = trends.size();
        assert(n <= MAX_N_INDICATORS);
        for (auto i = 0; i < n; ++i)
        {
            const auto& indicator = trends[i];
            const auto size = indicator.size();
            assert(size <= IND_SIZE);
            ind_size_t ind_bits[2]{};
            for (auto j = 0; j < size; ++j)
            {
                const auto bitmap_idx = -(indicator[j] >> 1);
                ind_bits[1-bitmap_idx] >>= 1;
                ind_bits[bitmap_idx] = (ind_bits[bitmap_idx] >> 1) | IND_T_MAX_SLL;
            }
            buffers[0].data[idx] = ind_bits[0];
            buffers[1].data[idx] = ind_bits[1];
            ++idx;
        }
    }
    template <IsSigned T>
    vec_mm_indicator(const std::vector<std::vector<T>>& trends) : bitmaps{}, buffers{}, idx{}
    {
        add_indicators_to_buffers(trends);
        bitmaps[0] = _mm512_load_epi64(&buffers[0].data[0]);
        bitmaps[1] = _mm512_load_epi64(&buffers[1].data[0]);
    }
    vec_mm_indicator() = default;
};


struct vec_trends_indicator_cmp
{
    trends_bitmap trends;
    std::vector<vec_indicator> indicators;
    std::size_t size;
    constexpr vec_trends_indicator_cmp(std::vector<vec_indicator>&& indicators, std::size_t size)
    : trends(), indicators{std::move(indicators)}, size{size}
    {}
    constexpr auto cmp(int64_t dir) // maybe not cxpr
    {
        trends.add_trend(dir);
        int result = 0;
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto& idctr = indicators[i];
            result += (
                ((idctr.bitmaps[0] & trends.bitmaps[0]) ^ idctr.bitmaps[0]) ^
                ((idctr.bitmaps[1] & trends.bitmaps[1]) ^ idctr.bitmaps[1])
                    ) == 0;
        }
        return result;
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
    constexpr indicator() = default;
    explicit constexpr indicator(const indicator& other) noexcept : bitmaps()
    {
        bitmaps[0] = other.bitmaps[0];
        bitmaps[1] = other.bitmaps[1];
    }
    constexpr indicator(indicator&& other) noexcept : bitmaps()
    {
        bitmaps[0] = other.bitmaps[0];
        bitmaps[1] = other.bitmaps[1];
    }
};


template <unsigned long ...N>
struct trends_indicator_cmp
{
    static constexpr auto SIZE = sizeof...(N);
    trends_bitmap trends;
    std::tuple<indicator<N>...> indicators;
    constexpr trends_indicator_cmp(const indicator<N>&... _indicators)
        : trends(), indicators(std::tie(_indicators...)) {}
    template <unsigned long M>
    [[gnu::always_inline]]
    int cmp() const
    {
        if constexpr (M < SIZE)
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
    [[gnu::always_inline]]
    auto cmp(int64_t dir)
    {
        trends.add_trend(dir);
        return cmp<0>();
    }
};


// constexpr auto ind1 = indicator{{1, 1, 1}};
// constexpr auto ind2 = indicator{{1, -1, 1, 1}};
// constexpr auto cmp = trends_indicator_cmp{ind1, ind2};
// static_assert(std::is_same_v<decltype(std::get<0>(cmp.indicators)), decltype((ind1))>);
// static_assert(std::is_same_v<decltype(std::get<1>(cmp.indicators)), decltype((ind2))>);


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
        for (unsigned long i = 0; i < N; ++i)
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
    constexpr mm_indicator() = default;
    explicit constexpr mm_indicator(const mm_indicator& other) noexcept : bitmaps{}, buffers{}, idx{}
    {
        bitmaps[0] = other.bitmaps[0];
        bitmaps[1] = other.bitmaps[1];
    }
    constexpr mm_indicator(mm_indicator&& other) noexcept : bitmaps{}, buffers{}, idx{}
    {
        bitmaps[0] = other.bitmaps[0];
        bitmaps[1] = other.bitmaps[1];
    }
};


// template <typename T>
// concept MMIndicatorType = std::is_same_v<T, vec_mm_indicator> || std::is_same_v<T, mm_indicator>;


// template <unsigned long IndicatorSize, MMIndicatorType MMIndicator>
template <unsigned long IndicatorSize>
struct mm_trends_indicator_cmp
{};


// template <MMIndicatorType MMIndicator>
template <>
struct mm_trends_indicator_cmp<64>
{
    trends_bitmap trends;
    mm_indicator indicators;
    constexpr mm_trends_indicator_cmp(const mm_indicator& indicators) : trends(), indicators(indicators) {}
    [[gnu::always_inline]]
    auto cmp(int64_t dir)
    {
        trends.add_trend(dir);
        auto bcst_pos = _mm512_set1_epi64(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, indicators.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi64_mask(and_pos, indicators.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi64(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, indicators.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi64_mask(and_neg, indicators.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


// template <MMIndicatorType MMIndicator>
template <>
struct mm_trends_indicator_cmp<32>
{
    trends_bitmap trends;
    mm_indicator indicators;
    constexpr mm_trends_indicator_cmp(const mm_indicator& indicators) : trends(), indicators(indicators) {}
    [[gnu::always_inline]]
    auto cmp(int64_t dir)
    {
        trends.add_trend(dir);
        auto bcst_pos = _mm512_set1_epi32(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, indicators.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi32_mask(and_pos, indicators.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi32(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, indicators.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi32_mask(and_neg, indicators.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


// template <MMIndicatorType MMIndicator>
template <>
struct mm_trends_indicator_cmp<16>
{
    trends_bitmap trends;
    mm_indicator indicators;
    constexpr mm_trends_indicator_cmp(const mm_indicator& indicators) : trends(), indicators(indicators) {}
    auto cmp(int64_t dir)
    {
        trends.add_trend(dir);
        auto bcst_pos = _mm512_set1_epi16(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, indicators.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi16_mask(and_pos, indicators.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi16(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, indicators.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi16_mask(and_neg, indicators.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcount(matches);
    }
};


// template <MMIndicatorType MMIndicator>
template <>
struct mm_trends_indicator_cmp<8>
{
    trends_bitmap trends;
    mm_indicator indicators;
    constexpr mm_trends_indicator_cmp(const mm_indicator& indicators) : trends(), indicators(indicators) {}
    auto cmp(int64_t dir)
    {
        trends.add_trend(dir);
        auto bcst_pos = _mm512_set1_epi8(trends.bitmaps[0]);
        auto and_pos = _mm512_and_si512(bcst_pos, indicators.bitmaps[0]);
        auto mask_cmp_pos = _mm512_cmpeq_epi8_mask(and_pos, indicators.bitmaps[0]);
        auto bcst_neg = _mm512_set1_epi8(trends.bitmaps[1]);
        auto and_neg = _mm512_and_si512(bcst_neg, indicators.bitmaps[1]);
        auto mask_cmp_neg = _mm512_cmpeq_epi8_mask(and_neg, indicators.bitmaps[1]);
        auto matches = mask_cmp_pos & mask_cmp_neg;
        return __builtin_popcountll(matches);
    }
};


template <unsigned long ...N>
struct signals
{
    static constexpr auto i64_shift = sizeof(int64_t)*8 - 1;
    trends_indicator_cmp<N...> cmp;
    constexpr signals(const indicator<N>&... indicators) : cmp{indicators...} {}
    template <typename T, typename T1, typename T2>
    void process(const T* prices, T1& position, T2* positions, std::size_t M)
    {

        for (std::size_t i = 1; i < M; ++i)
        {
            int64_t diff = prices[i] - prices[i - 1];
            if (diff)
            {
                position += cmp.cmp(-(diff>>i64_shift));
            }
            // price_positions[i] = position;
        }
    }
};


struct mm_signals
{
    static constexpr auto i64_shift = sizeof(int64_t)*8 - 1;
    mm_trends_indicator_cmp<IND_SIZE> cmp;
    constexpr mm_signals(const mm_indicator& indicators) : cmp{std::move(indicators)} {}
    template <typename T, typename T1, typename T2>
    void process(T* prices, T1& position, T2* positions, const std::size_t N)
    {

        for (std::size_t i = 1; i < N; ++i)
        {
            int64_t diff = prices[i] - prices[i - 1];
            if (diff)
            {
                position += cmp.cmp(-(diff>>i64_shift));
            }
            // price_positions[i] = position;
        }
    }
};


static void BM_signals(benchmark::State& state)
{
    const auto m_mm_indicator = mm_indicator
    {
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, -1, -1, -1, -1},
        {1, 1, 1, 1, -1, -1, 1},
        {-1, -1, -1, -1, -1}
    };
    // constexpr auto ind1 = indicator{{1, 1, 1, 1, 1, 1, 1, 1}};
    // constexpr auto ind2 = indicator{{1, 1, -1, -1, -1, 1}};
    // constexpr auto ind3 = indicator{{1, -1, 1, -1, 1, -1}};
    // constexpr auto ind4 = indicator{{1, 1, 1, 1, -1, -1, -1, -1}};
    auto mm_s = mm_signals{m_mm_indicator};
    // auto s = signals{ind1};
    const auto prices_size = state.range(0);
    auto m_prices = new int[prices_size];
    fill_random_pos(m_prices, prices_size);
    auto m_positions = new int64_t[prices_size]{};
    int64_t m_position = 0;
    for (auto _ : state)
    {    
        mm_s.process(m_prices, m_position, m_positions, prices_size);
        benchmark::DoNotOptimize(m_position);
    }
    state.SetComplexityN(prices_size);
}


BENCHMARK(BM_signals)->RangeMultiplier(10)->Range(10, 10'000'000)->Complexity(benchmark::oN);
BENCHMARK_MAIN();

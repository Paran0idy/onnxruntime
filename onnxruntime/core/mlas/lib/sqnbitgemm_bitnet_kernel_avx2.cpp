/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"
#include "sqnbitgemm_bitnet_kernel_avx2.h"
#include <immintrin.h>
#include <stdint.h>
typedef float float_type;

#include "string.h"
#include <type_traits>

#define extract_low_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
#define extract_high_epi8_epi16(v) _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))
#define extract_low_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
#define extract_high_epi16_epi32(v) _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1))

template <int N>
struct SignedHalvingAdder {
    SignedHalvingAdder<N / 2> adder;
    __m256i lhs;

    inline void push(__m256i v, int k) {
        if (k < N / 2) {
            adder.push(v, k);
            if (k == N / 2 - 1) {
                lhs = adder.get();
            }
        } else {
            adder.push(v, k - N / 2);
            if (k == N - 1) {
                lhs = _mm256_avg_epu8(lhs, adder.get());
            }
        }
    }

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <>
struct SignedHalvingAdder<2> {
    __m256i lhs;

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs = v;
        } else {
            lhs = _mm256_avg_epu8(lhs, v);
        }
    }

    inline __m256i get() {
        return lhs;
    }

    inline __m256i get_low() {
        return extract_low_epi8_epi16(lhs);
    }

    inline __m256i get_high() {
        return extract_high_epi8_epi16(lhs);
    }
};

template <int N>
struct SignedWideningAdder {
    __m256i lhs_low;
    __m256i lhs_high;

    inline void push(__m256i v, int k) {
        if (k == 0) {
            lhs_low = extract_low_epi8_epi16(v);
            lhs_high = extract_high_epi8_epi16(v);
        } else {
            lhs_low = _mm256_add_epi16(lhs_low, extract_low_epi8_epi16(v));
            lhs_high = _mm256_add_epi16(lhs_high, extract_high_epi8_epi16(v));
        }
    }

    inline __m256i get_low() {
        return lhs_low;
    }

    inline __m256i get_high() {
        return lhs_high;
    }
};

template <bool FastAggregation, int ActK>
using SignedAdder = std::conditional_t<FastAggregation, SignedHalvingAdder<ActK>, SignedWideningAdder<ActK>>;

template <int K>
struct mylog2 {
    enum {
        value = 1 + mylog2<K / 2>::value
    };
};

template <>
struct mylog2<0> {
    enum {
        value = -1
    };
};

constexpr int get_bias_scale(int bits) {
    // The bias scale will be added to the first bit
    // 15 = (1/2 + 1 + 2 + 4) / (1/2)
    // 7 = (1/2 + 1 + 2) / (1/2)
    // 3 = (1/2 + 1) / (1/2)
    // 1 = (1/2) / (1/2)
    if (bits == 4) {
        return 15;
    } else if (bits == 3) {
        return 7;
    } else if (bits == 2) {
        return 3;
    } else if (bits == 1) {
        return 1;
    } else {
        return 0;
    }
}


// When FastAggregation is enabled, FastAggregationK = ActK
// zero_points is merged into scales to maintain API
template <bool has_scale, int K, int Bits, int ActK = 16, bool FastAggregation = false, bool ZeroPoint = false, bool OneScale = false>
inline int32_t tbl_g4_int8_float_update_impl(int32_t m, float_type* c, int8_t* lut, uint8_t* a, float_type* scales, float_type* lut_scales, float_type* lut_biases) {
    const __m128i vec_mask = _mm_set1_epi8(0x0f);
    __m128i vec_lut[K];

// #pragma unroll
    for (int k = 0; k < K; k++) {
        vec_lut[k] = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 16));
    }

    SignedAdder<FastAggregation, ActK> adder;
    for (int i = 0; i < m / 2; i += 16) {
        __m256 vec_c0, vec_c1, vec_c2, vec_c3;

        float_type partial_sum = (float_type)-0.0f;
// #pragma unroll
        for (int kk = 0; kk < K; kk += ActK) {
// #pragma unroll
            for (int k = 0; k < ActK; k++) {
                // (M // bm, KK / K / 4, bm / 16 / 2, K * 16)
                __m128i vec_as = _mm_loadu_si128(reinterpret_cast<__m128i*>(a + i * K + (kk + k) * 16));
                __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);
                __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);

                __m256i vec_lut_ = _mm256_set_m128i(vec_lut[kk + k], vec_lut[kk + k]);
                __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
                __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
                adder.push(vec_v, k);
            }

            __m256 vec_v_low_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_low()));
            __m256 vec_v_low_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_low()));
            __m256 vec_v_high_low = _mm256_cvtepi32_ps(extract_low_epi16_epi32(adder.get_high()));
            __m256 vec_v_high_high = _mm256_cvtepi32_ps(extract_high_epi16_epi32(adder.get_high()));

            float_type lut_s = lut_scales[kk / ActK];
            float_type lut_b = lut_biases[kk / ActK];

            partial_sum += lut_b;

            if (FastAggregation) {
                lut_s = lut_s * ActK;
                lut_b -= lut_s * (mylog2<ActK>::value / 4 * get_bias_scale(Bits));
            }

#define lut_fma(vs, ib) \
    ((ib) % Bits) ? (_mm256_mul_ps((vs),   _mm256_set1_ps(lut_s))) \
                  : (_mm256_fmadd_ps((vs), _mm256_set1_ps(lut_s), _mm256_set1_ps(lut_b)))
            if (kk == 0) {
                vec_c0 = lut_fma(vec_v_low_low,   (i / 4    ));
                vec_c1 = lut_fma(vec_v_low_high,  (i / 4 + 1));
                vec_c2 = lut_fma(vec_v_high_low,  (i / 4 + 2));
                vec_c3 = lut_fma(vec_v_high_high, (i / 4 + 3));
            } else {
                vec_c0 = _mm256_add_ps(vec_c0, lut_fma(vec_v_low_low,   (i / 4    )));
                vec_c1 = _mm256_add_ps(vec_c1, lut_fma(vec_v_low_high,  (i / 4 + 1)));
                vec_c2 = _mm256_add_ps(vec_c2, lut_fma(vec_v_high_low,  (i / 4 + 2)));
                vec_c3 = _mm256_add_ps(vec_c3, lut_fma(vec_v_high_high, (i / 4 + 3)));
            }
#undef lut_fma
        }

        if (ZeroPoint) {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16);
            vec_c0 = _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2));
            vec_c1 = _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8));
            vec_c2 = _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16));
            vec_c3 = _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24));
            __m256 vec_z0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 16 + 8);
            __m256 vec_z1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 16 + 8);
            __m256 vec_z2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 16 + 8);
            __m256 vec_z3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 16 + 8);
            partial_sum *= 2;
#define add_zero(cs, zs, ib) \
    ((ib) % Bits) ? ((cs)) \
                  : (_mm256_fmadd_ps((zs), _mm256_set1_ps(partial_sum), (cs)))
            _mm256_storeu_ps(c + i * 2,      add_zero(vec_c0, vec_z0, (i / 4    )));
            _mm256_storeu_ps(c + i * 2 + 8,  add_zero(vec_c1, vec_z1, (i / 4 + 1)));
            _mm256_storeu_ps(c + i * 2 + 16, add_zero(vec_c2, vec_z2, (i / 4 + 2)));
            _mm256_storeu_ps(c + i * 2 + 24, add_zero(vec_c3, vec_z3, (i / 4 + 3)));
#undef add_zero
        } else {
            __m256 vec_s0 = _mm256_loadu_ps(scales + ((i / 4    ) / Bits) * 8);
            __m256 vec_s1 = _mm256_loadu_ps(scales + ((i / 4 + 1) / Bits) * 8);
            __m256 vec_s2 = _mm256_loadu_ps(scales + ((i / 4 + 2) / Bits) * 8);
            __m256 vec_s3 = _mm256_loadu_ps(scales + ((i / 4 + 3) / Bits) * 8);
            _mm256_storeu_ps(c + i * 2,      _mm256_fmadd_ps(vec_c0, vec_s0, _mm256_loadu_ps(c + i * 2)));
            _mm256_storeu_ps(c + i * 2 + 8,  _mm256_fmadd_ps(vec_c1, vec_s1, _mm256_loadu_ps(c + i * 2 + 8)));
            _mm256_storeu_ps(c + i * 2 + 16, _mm256_fmadd_ps(vec_c2, vec_s2, _mm256_loadu_ps(c + i * 2 + 16)));
            _mm256_storeu_ps(c + i * 2 + 24, _mm256_fmadd_ps(vec_c3, vec_s3, _mm256_loadu_ps(c + i * 2 + 24)));
        }
    }
    return 0;
}


// TODO: refactor function naming


#define tbl_g4_int8_float_update(s, k, b, ak, fa, z, os)                                                                                                                                                   \
    int32_t tbl_g4_int8_float_update_s##s##_k##k##_b##b##_ak##ak##_fa##fa##_z##z##_os##os(int32_t m, void* c, int8_t* lut, uint8_t* a, void* scales, void* lut_scales, void* lut_biases) {  \
        return tbl_g4_int8_float_update_impl<s, k, b, ak, fa, z, os>(m, (float_type*)c, lut, a, (float_type*)scales, (float_type*)lut_scales, (float_type*)lut_biases);                                                                                        \
    }


#ifdef __cplusplus
extern "C" {
#endif

int32_t tbl_float_reset(int32_t m, void* c) {
    memset(c, 0, m * sizeof(float_type));
    return 0;
}

#ifdef __cplusplus
}
#endif

#include <algorithm>


static inline float _mm256_addv_ps(const __m256 v) {
    __m128 res = _mm256_extractf128_ps(v, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(v));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Current implementation requires (K * 4) == act_group_size and K >= 8
// s0 = -1, s1 = 1
// TODO: loop K
// Still preserve FastAggregationK althougth it's unused for compatibility
template <int FastAggregationK = 16, int Bits = 4>
inline int32_t lut_ctor_g4_int8_impl(int32_t act_k, int8_t* qlut, float_type* b, float_type* lut_scales, float_type* lut_biases) {
    __m256 vec_lut[16];
    float biases = 0.0;
    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    float scales = *lut_scales;
    float t_scales = scales ? 1.0f / scales : 0.0f;

    for (int k = 0; k < act_k / 32; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 32 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 32 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 32 + 2, vec_bi, 1);
        __m256 vec_b3 = _mm256_i32gather_ps(b + k * 32 + 3, vec_bi, 1);

// #pragma unroll
        for (int g = 1; g < 16; g += 2) {
            vec_lut[g] = vec_b0;
            if (g & 0b0010) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b1);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b1);
            }
            if (g & 0b0100) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b2);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b2);
            }
            if (g & 0b1000) {
                vec_lut[g] = _mm256_add_ps(vec_lut[g], vec_b3);
            } else {
                vec_lut[g] = _mm256_sub_ps(vec_lut[g], vec_b3);
            }
        }
// #pragma unroll
        for (int g = 0; g < 16; g += 2) {
            vec_lut[g] = -vec_lut[15 - g];
        }

        biases += _mm256_addv_ps(vec_lut[0]);

// #pragma unroll
        for (int g = 0; g < 16; ++g) {
            vec_lut[g] = _mm256_mul_ps(vec_lut[g], _mm256_set1_ps(t_scales));
        }

        __m256i vec_qlut[4];
        const __m256i shuf = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                              0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
// #pragma unroll
        for (int g = 0; g < 4; g += 1) {
            __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(vec_lut[g * 4 + 3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

            i0 = _mm256_packs_epi32(i0, i1);	         // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
            i2 = _mm256_packs_epi32(i2, i3);	         // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                                         // Convert int16 to int8
            i0 = _mm256_packs_epi16(i0, i2);	         // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7,  12, 13, 14, 15,  20, 21, 22, 23,  28, 29, 30, 31
            vec_qlut[g] = _mm256_shuffle_epi8(i0, shuf);  // 0, 8, 16, 24,  1, 9, 17, 25,  2, 10, 18, 26,  3, 11, 19, 27,  4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31
        }

        int32_t* qlut_i32 = reinterpret_cast<int32_t*>(qlut);
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 0 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 0);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 1 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 1);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 2 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 2);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 3 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 3);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 4 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 4);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 5 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 5);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 6 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 6);
        }
// #pragma unroll
        for (int g = 0; g < 4; ++g) {
            qlut_i32[k * 32 + 7 * 4 + g] = _mm256_extract_epi32(vec_qlut[g], 7);
        }
    }

    *lut_scales = scales;
    *lut_biases = biases;

    return 0;
}

#define lut_ctor(fak, bits)                                                                                                      \
    int32_t lut_ctor_g4_int8_k##fak##_b##bits(int32_t act_k, int8_t* qlut, void* b, void* lut_scales, void* lut_biases) {        \
        return lut_ctor_g4_int8_impl<fak, bits>(act_k, qlut, (float_type*)b, (float_type*)lut_scales, (float_type*)lut_biases);  \
    }

#ifdef __cplusplus
extern "C" {
#endif

int32_t partial_max_g4_int8_k8(void* lut_scales_, void* b_) {
    float_type* lut_scales = (float_type*)lut_scales_;
    float_type* b = (float_type*)b_;

    const __m256i vec_bi = _mm256_set_epi32(112, 96, 80, 64, 48, 32, 16, 0);
    __m256 vec_b0 = _mm256_i32gather_ps(b + 0, vec_bi, 1);
    __m256 vec_b1 = _mm256_i32gather_ps(b + 1, vec_bi, 1);
    __m256 vec_b2 = _mm256_i32gather_ps(b + 2, vec_bi, 1);
    __m256 vec_b3 = _mm256_i32gather_ps(b + 3, vec_bi, 1);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    __m256 vec_babs0 = _mm256_andnot_ps(vec_sign, vec_b0);
    __m256 vec_babs1 = _mm256_andnot_ps(vec_sign, vec_b1);
    __m256 vec_babs2 = _mm256_andnot_ps(vec_sign, vec_b2);
    __m256 vec_babs3 = _mm256_andnot_ps(vec_sign, vec_b3);
    __m256 abssum = _mm256_add_ps(_mm256_add_ps(vec_babs0, vec_babs1), _mm256_add_ps(vec_babs2, vec_babs3));
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(abssum, 1), _mm256_castps256_ps128(abssum));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    float scales = _mm_cvtss_f32(max4) / 127;
    *lut_scales = std::max(*lut_scales, scales);

    return 0;
}

int32_t partial_max_reset(void* lut_scales_) {
    float_type* lut_scales = (float_type*)lut_scales_;
    *lut_scales = 0.0;
    return 0;
}

#ifdef __cplusplus
}
#endif

tbl_g4_int8_float_update(true, 16, 2, 16, false, true, false)

lut_ctor(0, 2)

/**
 * @brief 2-bit Quantized GEMM Kernel (AVX2 Optimized)
 * 
 * This kernel performs 2-bit quantized matrix multiplication (C = A * B_quantized)
 * using look-up tables (LUTs) with dynamic dequantization and fused scaling/bias operations.
 * Optimized with AVX2 instructions and block-based computation for memory efficiency.
 * 
 * @tparam M Matrix A rows (must be multiple of 64)
 * @tparam K Matrix A columns/Matrix B rows (must be multiple of 64)
 * @param A            Pointer to input matrix A
 * @param LUT          Pointer to precomputed 2-bit quantization LUT
 * @param Scales       Pointer to quantization scale factors
 * @param LUT_Scales   Pointer to dynamic LUT scaling factors
 * @param LUT_Biases   Pointer to dynamic LUT bias terms
 * @param C            Pointer to output matrix C
 * @return int32_t     Always returns 0
 **/
template<size_t M, size_t K>
int32_t SQ2BitGemmKernel_CompInt8_avx2_impl(void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C) {
    // 32-byte aligned accumulator for intermediate results
    alignas(32) float CBits[M];
    alignas(32) float C_global[M / 2];
    tbl_float_reset(M, (&(CBits[0])));
    int32_t m = M / 64, k = K / 64;
    // K-dimension block loop
    for (int32_t k_outer = 0; k_outer < k; ++k_outer) {
      tbl_g4_int8_float_update_strue_k16_b2_ak16_fafalse_ztrue_osfalse(M, (&(CBits[0])), (&(((int8_t*)LUT)[(k_outer * 256)])), (&(((uint8_t*)A)[(k_outer * 8 * M)])), (&(((float*)Scales)[((k_outer >> 1) * M)])), (&(((float*)LUT_Scales)[k_outer])), (&(((float*)LUT_Biases)[k_outer])));
    }
    for (int32_t m_c_outer = 0; m_c_outer < m; ++m_c_outer) {
        const int32_t base_idx = m_c_outer * 64;
        const int32_t out_base = m_c_outer * 32;
        for (int i = 0; i < 32; ++i) {
            C_global[out_base + i] = (CBits[base_idx + i] * 0.5f) + CBits[base_idx + i + 8];
        }}
    for (int32_t m_inner_outer = 0; m_inner_outer < m; ++m_inner_outer) {
        const int32_t base = m_inner_outer * 32;
        for (int i = 0; i < 32; ++i) {
            ((float*)C)[base + i] = C_global[base + i];
        }
    }
    return 0;
  }

/**
 * @brief Quantization Look-Up Table (LUT) Generation Kernel
 * 
 * Generates 2-bit quantization LUTs by analyzing input matrix B statistics.
 * Computes per-block scaling factors and biases for quantization.
 * 
 * @tparam K Matrix B columns (must be multiple of 64)
 * @param B            Pointer to input matrix B
 * @param LUT_Scales   Output dynamic scaling factors
 * @param LUT_Biases   Output dynamic bias terms
 * @param QLUT         Output quantized LUT
 * @return int32_t     Always returns 0
 */
template <size_t K>
int32_t QuantizeARowLUT_CompInt8_impl(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT) {
    int32_t k = K / 64;
    // Phase 1: Compute per-block scaling factors
    for (int32_t kk_outer = 0; kk_outer < k; ++kk_outer) {
        // Initialize scaling factor for current block
        partial_max_reset((&(((float*)LUT_Scales)[kk_outer])));
        for (int32_t k_outer = 0; k_outer < 2; ++k_outer) {
            // Compute maximum absolute value and update scaling factor
            partial_max_g4_int8_k8((&(((float*)LUT_Scales)[kk_outer])), (&(((float*)B)[((kk_outer * 64) + (k_outer * 32))])));
        }
    }
    // Phase 2: LUT Generation
    for (int32_t k_outer_1 = 0; k_outer_1 < k; ++k_outer_1) {
        lut_ctor_g4_int8_k0_b2(64, (&(((int8_t*)QLUT)[(k_outer_1 * 256)])), (&(((float*)B)[(k_outer_1 * 64)])), (&(((float*)LUT_Scales)[k_outer_1])), (&(((float*)LUT_Biases)[k_outer_1])));
    }
    return 0;
}

size_t
Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
  // TODO: This code shall change according to T-Mac.
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

void SQ2BitGemmPackQuantBData(
  size_t /*N*/,
  size_t /*K*/,
  size_t /*BlkLen*/,
  MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
  const std::byte* /*QuantBDataBegin*/,
  std::byte* /*PackedQuantBDataBegin*/,
  MLAS_THREADPOOL* /*ThreadPool*/
) 
{
  // TODO: need implementation
}

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            // QuantData + Scale + ZeroPoint but LUT format
            const size_t PerGemmWorkspaceSize = M * BlockCountK * (Q8BlkLUTSize(BlkLen) + sizeof(float));
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
}

size_t
SQ2BitGemmKernel_CompInt8_avx2(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const float* QuantAZeroPoint,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    size_t ldc,
    const float* Bias
)
{
  // reference SQ4BitGemmKernel_CompInt8_avx2
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    MLAS_UNREFERENCED_PARAMETER(QuantBZeroPoint);
    MLAS_UNREFERENCED_PARAMETER(CountN);
    MLAS_UNREFERENCED_PARAMETER(BlockCountK);
    MLAS_UNREFERENCED_PARAMETER(ldc);
    MLAS_UNREFERENCED_PARAMETER(Bias);

    if (CountM == 128 && CountK == 4096) {
        SQ2BitGemmKernel_CompInt8_avx2_impl<128, 4096>(
            (void*)QuantBData,
            (void*)QuantA,
            (void*)QuantBScale,
            (void*)QuantAScale,
            (void*)QuantAZeroPoint,
            C);
    } else if (CountM == 256 && CountK == 4096) {
        SQ2BitGemmKernel_CompInt8_avx2_impl<256, 4096>(
            (void*)QuantBData,
            (void*)QuantA,
            (void*)QuantBScale,
            (void*)QuantAScale,
            (void*)QuantAZeroPoint,
            C);
    } else if (CountM == 1024 && CountK == 14436) {
        SQ2BitGemmKernel_CompInt8_avx2_impl<1024, 14336>(
            (void*)QuantBData,
            (void*)QuantA,
            (void*)QuantBScale,
            (void*)QuantAScale,
            (void*)QuantAZeroPoint,
            C);
    } else {
        ORT_ENFORCE(false, "Unsupported shape: CountM=", CountM, ", CountK=", CountK);

    }
    return 0;
}

void
QuantizeARow_CompInt8(
    size_t /*BlkLen*/,
    const float* /*A*/,
    size_t /*CountK*/,
    std::byte* /*QuantA*/
)
{
  // shall be similar to QuantizeARow_CompInt8_avx2 without blksum related code.
}


void
QuantizeARowLUT_CompInt8(
    size_t BlkLen,
    const float* A,
    size_t CountK,
    std::byte* QuantA,
    float* QuantAScale,
    float* QuantAZeroPoint
)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);

    if (CountK == 4096) {
        QuantizeARowLUT_CompInt8_impl<4096>(
            (void*)A,
            QuantAScale,
            QuantAZeroPoint,
            QuantA
        );
    } else if (CountK == 14336) {
        QuantizeARowLUT_CompInt8_impl<14336>(
            (void*)A,
            QuantAScale,
            QuantAZeroPoint,
            QuantA
        );
    } else {
        ORT_ENFORCE(false, "Unsupported CountK value. Supported values are 4096 and 14336.");
    }
}

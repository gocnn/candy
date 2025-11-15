#ifndef FLASH_API_H
#define FLASH_API_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stdbool.h>

    /**
     * Flash Attention Multi-Head Attention (MHA) Forward Pass
     *
     * This is the unified Flash Attention API that supports all features:
     * - Multiple data types (FP16/BF16)
     * - Multiple head dimensions (32, 64, 96, 128, 160, 192, 224, 256)
     * - Causal and non-causal attention
     * - Sliding window attention
     * - Variable-length sequence support
     * - ALiBi positional encoding
     * - Soft capping
     *
     * @param q_ptr             Query tensor pointer [batch, num_heads, seqlen_q, head_dim]
     * @param k_ptr             Key tensor pointer [batch, num_heads_k, seqlen_k, head_dim]
     * @param v_ptr             Value tensor pointer [batch, num_heads_k, seqlen_k, head_dim]
     * @param o_ptr             Output tensor pointer [batch, num_heads, seqlen_q, head_dim]
     * @param softmax_lse_ptr   Softmax LSE output pointer [batch, num_heads, seqlen_q] (optional)
     * @param alibi_slopes_ptr  ALiBi slopes pointer [num_heads] (optional, for ALiBi positional encoding)
     *
     * @param cu_seqlens_q_ptr  Cumulative sequence lengths for Query [batch+1] (variable-length sequences, optional)
     * @param cu_seqlens_k_ptr  Cumulative sequence lengths for Key [batch+1] (variable-length sequences, optional)
     *
     * @param q_batch_stride    Query batch stride (in elements, not bytes)
     * @param k_batch_stride    Key batch stride
     * @param v_batch_stride    Value batch stride
     * @param o_batch_stride    Output batch stride
     * @param alibi_slopes_batch_stride ALiBi slopes batch stride
     *
     * @param q_row_stride      Query row stride (seqlen dimension)
     * @param k_row_stride      Key row stride
     * @param v_row_stride      Value row stride
     * @param o_row_stride      Output row stride
     *
     * @param q_head_stride     Query head stride (num_heads dimension)
     * @param k_head_stride     Key head stride
     * @param v_head_stride     Value head stride
     * @param o_head_stride     Output head stride
     *
     * @param b                 Batch size
     * @param h                 Number of Query heads
     * @param h_k               Number of Key/Value heads (for GQA/MQA)
     * @param d                 Head dimension
     * @param d_rounded         Aligned head dimension (typically aligned to multiples of 8)
     * @param softmax_scale     Softmax scaling factor (typically 1/sqrt(head_dim))
     *
     * @param seqlen_q          Query sequence length
     * @param seqlen_k          Key sequence length
     * @param seqlen_q_rounded  Aligned Query sequence length (for performance optimization)
     * @param seqlen_k_rounded  Aligned Key sequence length (for performance optimization)
     *
     * @param is_bf16           Use BFloat16 data type (0=FP16, 1=BF16)
     * @param is_causal         Apply causal masking (0=no, 1=yes)
     * @param unpadded_lse      LSE output format (0=padded, 1=unpadded)
     *
     * @param window_size_left  Left window size (-1=unlimited)
     * @param window_size_right Right window size (-1=unlimited)
     *
     * @param softcap           Soft capping parameter (0.0=disabled)
     */
    void run_mha(
        void *q_ptr,
        void *k_ptr,
        void *v_ptr,
        void *o_ptr,
        void *softmax_lse_ptr,
        void *alibi_slopes_ptr,

        int32_t *cu_seqlens_q_ptr,
        int32_t *cu_seqlens_k_ptr,

        uint32_t q_batch_stride,
        uint32_t k_batch_stride,
        uint32_t v_batch_stride,
        uint32_t o_batch_stride,
        uint32_t alibi_slopes_batch_stride,

        uint32_t q_row_stride,
        uint32_t k_row_stride,
        uint32_t v_row_stride,
        uint32_t o_row_stride,

        uint32_t q_head_stride,
        uint32_t k_head_stride,
        uint32_t v_head_stride,
        uint32_t o_head_stride,

        uint32_t b,
        uint32_t h,
        uint32_t h_k,
        uint32_t d,
        uint32_t d_rounded,
        float softmax_scale,

        uint32_t seqlen_q,
        uint32_t seqlen_k,
        uint32_t seqlen_q_rounded,
        uint32_t seqlen_k_rounded,

        int is_bf16,
        int is_causal,
        int unpadded_lse,

        int window_size_left,
        int window_size_right,

        float softcap);

#ifdef __cplusplus
}
#endif

#endif // FLASH_API_H

#include "blis.h"


#ifdef __cplusplus
extern "C" {
#endif

void bli_dgemmsup2_rv_haswell_asm_6x8r
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    );

void bli_dgemmsup2_rv_armv8a_asm_8x6r
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    );

void bli_dgemmsup2_rv_armv8a_asm_8x6c
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    );

void bli_dgemmsup2_cv_armv8a_asm_8x6r
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    );

void bli_dgemmsup2_cv_armv8a_asm_8x6c
    (
     dim_t            m,
     dim_t            n,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a0, inc_t cs_a0,
     double *restrict b, inc_t rs_b0, inc_t cs_b0,
     double *restrict beta,
     double *restrict c, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    );

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

typedef typeof(&bli_dpackm_haswell_asm_8xk) l1mukr_t;
typedef typeof(&bli_dgemm_haswell_asm_6x8) ukr_dgemm_bulk_t;
typedef typeof(&bli_dgemmsup2_rv_haswell_asm_6x8r) ukr_dgemm_sup_t;

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)

typedef typeof(&bli_dpackm_armv8a_int_8xk) l1mukr_t;
typedef typeof(&bli_dgemm_armv8a_asm_8x6r) ukr_dgemm_bulk_t;
typedef typeof(&bli_dgemmsup2_rv_armv8a_asm_8x6r) ukr_dgemm_sup_t;

#endif

void bls_dgemm
    (
     dim_t m0,
     dim_t n0,
     dim_t k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a, inc_t cs_a,
     double *restrict b, inc_t rs_b, inc_t cs_b,
     double *restrict beta,
     double *restrict c, inc_t rs_c, inc_t cs_c,
     cntx_t *cntx,
     rntm_t *rntm,
     ukr_dgemm_bulk_t ukr_bulk,
     ukr_dgemm_sup_t  ukr_sup
    );

#ifdef __cplusplus
}
#endif

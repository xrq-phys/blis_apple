#include "blis.h"


#ifdef __cplusplus
extern "C" {
#endif

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

typedef typeof(&bli_dpackm_armv8a_int_8xk) l1mukr_t;
typedef typeof(&bli_dgemm_armv8a_asm_8x6r) ukr_dgemm_bulk_t;
typedef typeof(&bli_dgemmsup2_rv_armv8a_asm_8x6r) ukr_dgemm_sup_t;

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

void dgemm_finalize();

#ifdef __cplusplus
}
#endif


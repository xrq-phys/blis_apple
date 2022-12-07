#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)

#include "blis.h"
#include <assert.h>


// Label locality & misc.
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

/* Order of row-major DGEMM_8x6's execution in 2x2 blocks:
 *
 * +---+ +---+ +---+
 * | 0 | | 2 | | 4 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 1 | | 3 | | 5 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 6 | | 8 | | 10|
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 7 | | 9 | | 11|
 * +---+ +---+ +---+
 *
 */
#define DGEMM_8X6_MKER_LOOP(SUFFIX,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BE0,BE1,BE2,BE3,BE4,BE5,RSB,LOADNEXT,CADDR,RSC,LASTB,PRFC,PAADDR,PACKA,PBADDR,PACKB) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C00,C10,B0,A0) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C20,C30,B0,A1) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C01,C11,B1,A0) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C21,C31,B1,A1) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C02,C12,B2,A0) \
  DGEMM_STORE1V_ ##PACKA (A0,PAADDR,0) \
  DGEMM_LOAD1V_ ##LOADNEXT (A0,AADDR,ASHIFT) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C22,C32,B2,A1) \
  DGEMM_STORE1V_ ##PACKA (A1,PAADDR,16) \
  DGEMM_LOAD1V_ ##LOADNEXT (A1,AADDR,ASHIFT+16) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C40,C50,B0,A2) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C60,C70,B0,A3) \
  DGEMM_STORE1V_ ##PACKB (B0,PBADDR,0) \
  DGEMM_LOAD1V_G_ ##LOADNEXT (B0,BE0,BE1,RSB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C41,C51,B1,A2) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C61,C71,B1,A3) \
  DGEMM_STORE1V_ ##PACKB (B1,PBADDR,16) \
  DGEMM_LOAD1V_G_ ##LOADNEXT (B1,BE2,BE3,RSB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C42,C52,B2,A2) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C62,C72,B2,A3)

#define DGEMM_STORE1V_nopack(V,ADDR,SHIFT)
#define DGEMM_STORE1V_pack(V,ADDR,SHIFT) \
  DSTORE1V(V,ADDR,SHIFT)

// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DLOAD1V(V1,ADDR,IMM)

#define DGEMM_LOAD1V_G_noload(V1,ADDR0,ADDR1,ST)
#define DGEMM_LOAD1V_G_load(V1,ADDR0,ADDR1,ST) \
" ld1  {v"#V1".d}[0], ["#ADDR0"], "#ST" \n\t" \
" ld1  {v"#V1".d}[1], ["#ADDR1"], "#ST" \n\t"

// Interleaving prefetch or not.
#define GEMM_PRFC_FH_noload(CADDR)
#define GEMM_PRFC_LH_FWD_noload(CADDR,RSC,LASTB)
#define GEMM_PRFC_FH_load(CADDR) \
" prfm PLDL1KEEP, ["#CADDR"]           \n\t"
#define GEMM_PRFC_LH_FWD_load(CADDR,RSC,LASTB) \
" prfm PLDL1KEEP, ["#CADDR", "#LASTB"] \n\t" \
" add  "#CADDR", "#CADDR", "#RSC"      \n\t"

// For row-storage of C.
#define DLOADC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
  DLOAD1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
  DSTORE1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// For col-storage of C.
#define DLOADC_4V_C_FWD(C0,C1,C2,C3,CADDR,CSHIFT,CSC) \
  DLOAD4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"
#define DSTOREC_4V_C_FWD(C0,C1,C2,C3,CADDR,CSHIFT,CSC) \
  DSTORE4V(C0,C1,C2,C3,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#CSC" \n\t"


///////////// FOR bli_dgemmsup2_cv_armv8a_asm_8x6r_* /////////////
//
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define DGEMM_8X6_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BE0,BE1,BE2,BE3,BE4,BE5,RSB,LOADNEXT,PRFC,PACKA,PACKB) \
  DGEMM_8X6_MKER_LOOP(SUFFIX,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BE0,BE1,BE2,BE3,BE4,BE5,RSB,LOADNEXT,%[c_ld],%[rs_c],40,PRFC,%[a_p],PACKA,%[b_p],PACKB)
//
// Microkernel is defined here as:
#define DGEMM_8X6_MKER_LOOP_LOC_FWD(SUFFIX,A0,A1,A2,A3,B0,B1,B2,PRFC,PACKA,PACKB) \
  DGEMM_8X6_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,A3,B0,B1,B2,%[a],0,%[b_2],%[b_3],%[b_4],%[b_5],%[b_0],%[b_1],%[rs_b],load,PRFC,PACKA,PACKB) \
  DGEMM_STORE1V_ ##PACKB (B2,%[b_p],16*2) \
" add             %[b_p], %[b_p], #6*8   \n\t" /* 6x8-specific */ \
" ld1             {v"#B2".d}[0], [%[b_0]], %[rs_b] \n\t" \
" ld1             {v"#B2".d}[1], [%[b_1]], %[rs_b] \n\t" \
  DGEMM_STORE1V_ ##PACKA (A2,%[a_p],16*2) \
" ldr             q"#A2", [%[a], #16*2]  \n\t" \
  DGEMM_STORE1V_ ##PACKA (A3,%[a_p],16*3) \
" add             %[a_p], %[a_p], #8*8   \n\t" /* 6x8-specific */ \
" ldr             q"#A3", [%[a], #16*3]  \n\t" \
" add             %[a], %[a], %[cs_a]    \n\t"
//
///////////// END bli_dgemmsup2_cv_armv8a_asm_8x6r_* /////////////

#define GENDEF(PACKA,PACKB) \
void bli_dgemmsup2_cv_armv8a_asm_8x6r_ ## PACKA ## _ ## PACKB \
    ( \
     dim_t            m, \
     dim_t            n, \
     dim_t            k, \
     double *restrict alpha, \
     double *restrict a, inc_t rs_a0, inc_t cs_a0, \
     double *restrict b, inc_t rs_b0, inc_t cs_b0, \
     double *restrict beta, \
     double *restrict c, inc_t rs_c0, inc_t cs_c0, \
     auxinfo_t       *data, \
     cntx_t          *cntx, \
     double *restrict a_p, \
     double *restrict b_p \
    ) \
{ \
  const void* a_next = bli_auxinfo_next_a( data ); \
  const void* b_next = bli_auxinfo_next_b( data ); \
\
  /* Typecast local copies of integers in case dim_t and inc_t are a
   * different size than is expected by load instructions. */ \
  uint64_t k_mker = k / 4; \
  uint64_t k_left = k % 4; \
  uint64_t cs_a   = cs_a0; \
  uint64_t rs_b   = rs_b0; \
  uint64_t rs_c   = rs_c0; \
  uint64_t cs_c   = cs_c0; \
\
  /* Define all integer registers outside asm block & avoid "x*" clobbing */ \
  uint64_t c_ld; \
  double *b_0 = b, \
         *b_1 = b_0 + cs_b0, \
         *b_2 = b_1 + cs_b0, \
         *b_3 = b_2 + cs_b0, \
         *b_4 = b_3 + cs_b0, \
         *b_5 = b_4 + cs_b0; \
\
  __asm__ volatile \
  ( \
/* %[a]   : A address.
 * %[b]   : B address.
 * %[cs_a]: Column-skip of A.
 * %[rs_b]: Row-skip of B.
 * %[c]   : C address.
 * %[rs_c]: Row-skip of C.
 * %[cs_c]: Column-skip of C. */ \
\
 /* Multiply some address skips by sizeof(double). */ \
" lsl             %[cs_a], %[cs_a], #3            \n\t" \
" lsl             %[rs_b], %[rs_b], #3            \n\t" /* rs_b */ \
" lsl             %[rs_c], %[rs_c], #3            \n\t" /* rs_c */ \
" lsl             %[cs_c], %[cs_c], #3            \n\t" /* cs_c */ \
"                                                 \n\t" \
" mov             %[c_ld], %[c]                   \n\t" \
"                                                 \n\t" \
/* %[k_mker]: Number of 4-loops.
 * %[k_left]: Number of loops left. */ \
\
 /* Load from memory. */ \
LABEL(DLOAD_ABC_ ## PACKA ## _ ## PACKB) \
"                                                 \n\t" /* No-microkernel early return is a must */ \
" cmp             %[k_mker], #0                   \n\t" /*  to avoid out-of-boundary read. */ \
BEQ(DK_LEFT_LOOP_INIT_ ## PACKA ## _ ## PACKB) \
"                                                 \n\t" /* Load A. */ \
" ldr             q24, [%[a], #16*0]              \n\t" \
" ldr             q25, [%[a], #16*1]              \n\t" \
" ldr             q26, [%[a], #16*2]              \n\t" \
" ldr             q27, [%[a], #16*3]              \n\t" \
" add             %[a], %[a], %[cs_a]             \n\t" \
"                                                 \n\t" \
" ld1             {v28.d}[0], [%[b_0]], %[rs_b]   \n\t" /* Load B. */ \
" ld1             {v28.d}[1], [%[b_1]], %[rs_b]   \n\t" \
" ld1             {v29.d}[0], [%[b_2]], %[rs_b]   \n\t" \
" ld1             {v29.d}[1], [%[b_3]], %[rs_b]   \n\t" \
" ld1             {v30.d}[0], [%[b_4]], %[rs_b]   \n\t" \
" ld1             {v30.d}[1], [%[b_5]], %[rs_b]   \n\t" \
" ld1             {v31.d}[0], [%[b_0]], %[rs_b]   \n\t" \
" ld1             {v31.d}[1], [%[b_1]], %[rs_b]   \n\t" \
\
/* Start microkernel loop -- Special treatment for the very first loop. */ \
" subs            %[k_mker], %[k_mker], #1        \n\t" /* Set count before final replica. */ \
DGEMM_8X6_MKER_LOOP_LOC_FWD(INIT,24,25,26,27,28,29,30,load,PACKA,PACKB) /* Prefetch C 1-4/8. */ \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,31,28,29,load,PACKA,PACKB) /* Prefetch C 5-8/8. */ \
/* Branch early to avoid reading excess mem. */ \
BEQ(DFIN_MKER_LOOP_ ## PACKA ## _ ## PACKB) \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,30,31,28,noload,PACKA,PACKB) \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,29,30,31,noload,PACKA,PACKB) \
/* Start microkernel loop. */ \
LABEL(DK_MKER_LOOP_ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" /* Set count before final replica. */ \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,28,29,30,noload,PACKA,PACKB) \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,31,28,29,noload,PACKA,PACKB) \
/* Branch early to avoid reading excess mem. */ \
BEQ(DFIN_MKER_LOOP_ ## PACKA ## _ ## PACKB) \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,30,31,28,noload,PACKA,PACKB) \
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,29,30,31,noload,PACKA,PACKB) \
BRANCH(DK_MKER_LOOP_ ## PACKA ## _ ## PACKB) \
\
/* Final microkernel loop. */ \
LABEL(DFIN_MKER_LOOP_ ## PACKA ## _ ## PACKB) \
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,30,31,28,%[a],0,%[b_2],%[b_3],%[b_4],%[b_5],%[b_0],%[b_1],%[rs_b],load,noload,PACKA,PACKB) \
DGEMM_STORE1V_ ##PACKB (28,%[b_p],16*2) \
DGEMM_STORE1V_ ##PACKA (26,%[a_p],16*2) \
DGEMM_STORE1V_ ##PACKA (27,%[a_p],16*3) \
 "add             %[b_p], %[b_p], #6*8            \n\t" /* 6x8-specific */ \
 "add             %[a_p], %[a_p], #8*8            \n\t" /* 6x8-specific */ \
" ldr             q26, [%[a], #16*2]              \n\t" \
" ldr             q27, [%[a], #16*3]              \n\t" \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,29,30,31,xzr,-1,xzr,xzr,xzr,xzr,xzr,xzr,xzr,noload,noload,PACKA,PACKB) \
DGEMM_STORE1V_ ##PACKB (31,%[b_p],16*2) \
DGEMM_STORE1V_ ##PACKA (26,%[a_p],16*2) \
DGEMM_STORE1V_ ##PACKA (27,%[a_p],16*3) \
 "add             %[b_p], %[b_p], #6*8            \n\t" /* 6x8-specific */ \
 "add             %[a_p], %[a_p], #8*8            \n\t" /* 6x8-specific */ \
\
/* Loops left behind microkernels. */ \
LABEL(DK_LEFT_LOOP_ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" /* End of exec. */ \
BEQ(DWRITE_MEM_PREP_ ## PACKA ## _ ## PACKB) \
" ldr             q24, [%[a], #16*0]              \n\t" /* Load A col. */ \
" ldr             q25, [%[a], #16*1]              \n\t" \
" ldr             q26, [%[a], #16*2]              \n\t" \
" ldr             q27, [%[a], #16*3]              \n\t" \
" add             %[a], %[a], %[cs_a]             \n\t" \
" ld1             {v28.d}[0], [%[b_0]], %[rs_b]   \n\t" /* Load B row. */ \
" ld1             {v28.d}[1], [%[b_1]], %[rs_b]   \n\t" \
" ld1             {v29.d}[0], [%[b_2]], %[rs_b]   \n\t" \
" ld1             {v29.d}[1], [%[b_3]], %[rs_b]   \n\t" \
" ld1             {v30.d}[0], [%[b_4]], %[rs_b]   \n\t" \
" ld1             {v30.d}[1], [%[b_5]], %[rs_b]   \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,28,29,30,xzr,-1,xzr,xzr,xzr,xzr,xzr,xzr,xzr,noload,noload,PACKA,PACKB) \
DGEMM_STORE1V_ ##PACKB (30,%[b_p],16*2) \
DGEMM_STORE1V_ ##PACKA (26,%[a_p],16*2) \
DGEMM_STORE1V_ ##PACKA (27,%[a_p],16*3) \
 "add             %[b_p], %[b_p], #6*8            \n\t" /* 6x8-specific */ \
 "add             %[a_p], %[a_p], #8*8            \n\t" /* 6x8-specific */ \
BRANCH(DK_LEFT_LOOP_ ## PACKA ## _ ## PACKB) \
\
/* Initial loop without microkernels. */ \
LABEL(DK_LEFT_LOOP_INIT_ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" /* End of exec. */ \
BEQ(DCLEAR_CCOLS_ ## PACKA ## _ ## PACKB) \
" ldr             q24, [%[a], #16*0]              \n\t" /* Load A col. */ \
" ldr             q25, [%[a], #16*1]              \n\t" \
" ldr             q26, [%[a], #16*2]              \n\t" \
" ldr             q27, [%[a], #16*3]              \n\t" \
" add             %[a], %[a], %[cs_a]             \n\t" \
" ld1             {v28.d}[0], [%[b_0]], %[rs_b]   \n\t" /* Load B row. */ \
" ld1             {v28.d}[1], [%[b_1]], %[rs_b]   \n\t" \
" ld1             {v29.d}[0], [%[b_2]], %[rs_b]   \n\t" \
" ld1             {v29.d}[1], [%[b_3]], %[rs_b]   \n\t" \
" ld1             {v30.d}[0], [%[b_4]], %[rs_b]   \n\t" \
" ld1             {v30.d}[1], [%[b_5]], %[rs_b]   \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(INIT,24,25,26,27,28,29,30,xzr,-1,xzr,xzr,xzr,xzr,xzr,xzr,xzr,noload,noload,PACKA,PACKB) \
DGEMM_STORE1V_ ##PACKB (30,%[b_p],16*2) \
DGEMM_STORE1V_ ##PACKA (26,%[a_p],16*2) \
DGEMM_STORE1V_ ##PACKA (27,%[a_p],16*3) \
 "add             %[b_p], %[b_p], #6*8            \n\t" /* 6x8-specific */ \
 "add             %[a_p], %[a_p], #8*8            \n\t" /* 6x8-specific */ \
BRANCH(DK_LEFT_LOOP_ ## PACKA ## _ ## PACKB) \
\
/* No loop at all. Clear rows separately. */ \
LABEL(DCLEAR_CCOLS_ ## PACKA ## _ ## PACKB) \
CLEAR8V(0,1,2,3,4,5,6,7) \
CLEAR8V(8,9,10,11,12,13,14,15) \
CLEAR8V(16,17,18,19,20,21,22,23) \
\
/* Scale and write to memory. */ \
LABEL(DWRITE_MEM_PREP_ ## PACKA ## _ ## PACKB) \
" ld1r            {v24.2d}, [%[alpha]]            \n\t" /* Load alpha & beta. */ \
" ld1r            {v25.2d}, [%[beta]]             \n\t" \
"                                                 \n\t" \
LABEL(DPREFETCH_ABNEXT_ ## PACKA ## _ ## PACKB) \
" prfm            PLDL1KEEP, [%[a_next], 64*0]    \n\t" /* Do not know cache line size,             */ \
" prfm            PLDL1KEEP, [%[a_next], 64*1]    \n\t" /*  issue some number of prfm instructions  */ \
" prfm            PLDL1KEEP, [%[a_next], 64*2]    \n\t" /*  to try to activate hardware prefetcher. */ \
" prfm            PLDL1KEEP, [%[b_next], 64*0]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*1]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*2]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*3]    \n\t" \
"                                                 \n\t" \
" fmov            d26, #1.0                       \n\t" \
" fcmp            d24, d26                        \n\t" \
BEQ(DUNIT_ALPHA_ ## PACKA ## _ ## PACKB)                                        \
DSCALE8V(0,1,2,3,4,5,6,7,24,0)                          \
DSCALE8V(8,9,10,11,12,13,14,15,24,0)                    \
DSCALE8V(16,17,18,19,20,21,22,23,24,0)                  \
LABEL(DUNIT_ALPHA_ ## PACKA ## _ ## PACKB)                                      \
"                                                 \n\t" \
" cmp             %[cs_c], #8                     \n\t" \
" mov             %[c_ld], %[c]                   \n\t" /* C address for loading. */ \
"                                                 \n\t" \
BNE(DWRITE_MEM_C_ ## PACKA ## _ ## PACKB) \
\
/* Row-major C-storage. */ \
LABEL(DWRITE_MEM_R_ ## PACKA ## _ ## PACKB) \
" fcmp            d25, #0.0                       \n\t" /* Sets conditional flag whether *beta == 0. */ \
"                                                 \n\t" /* This conditional flag will be used        */ \
"                                                 \n\t" /*  multiple times for skipping load.        */ \
\
/* Row 0 & 1: */ \
BEQ(DZERO_BETA_R_0_1_ ## PACKA ## _ ## PACKB) \
DLOADC_3V_R_FWD(26,27,28,%[c_ld],0,%[rs_c]) \
DLOADC_3V_R_FWD(29,30,31,%[c_ld],0,%[rs_c]) \
DSCALEA2V(0,1,26,27,25,0) \
DSCALEA2V(2,3,28,29,25,0) \
DSCALEA2V(4,5,30,31,25,0) \
LABEL(DZERO_BETA_R_0_1_ ## PACKA ## _ ## PACKB) \
DSTOREC_3V_R_FWD(0,1,2,%[c],0,%[rs_c]) \
DSTOREC_3V_R_FWD(3,4,5,%[c],0,%[rs_c]) \
/* Row 2 & 3 & 4 & 5: */ \
BEQ(DZERO_BETA_R_2_3_4_5_ ## PACKA ## _ ## PACKB) \
DLOADC_3V_R_FWD(26,27,28,%[c_ld],0,%[rs_c]) \
DLOADC_3V_R_FWD(29,30,31,%[c_ld],0,%[rs_c]) \
DLOADC_3V_R_FWD(0,1,2,%[c_ld],0,%[rs_c]) \
DLOADC_3V_R_FWD(3,4,5,%[c_ld],0,%[rs_c]) \
DSCALEA4V(6,7,8,9,26,27,28,29,25,0) \
DSCALEA4V(10,11,12,13,30,31,0,1,25,0) \
DSCALEA4V(14,15,16,17,2,3,4,5,25,0) \
LABEL(DZERO_BETA_R_2_3_4_5_ ## PACKA ## _ ## PACKB) \
DSTOREC_3V_R_FWD(6,7,8,%[c],0,%[rs_c]) \
DSTOREC_3V_R_FWD(9,10,11,%[c],0,%[rs_c]) \
DSTOREC_3V_R_FWD(12,13,14,%[c],0,%[rs_c]) \
DSTOREC_3V_R_FWD(15,16,17,%[c],0,%[rs_c]) \
/* Row 6 & 7 */ \
BEQ(DZERO_BETA_R_6_7_ ## PACKA ## _ ## PACKB) \
DLOADC_3V_R_FWD(26,27,28,%[c_ld],0,%[rs_c]) \
DLOADC_3V_R_FWD(29,30,31,%[c_ld],0,%[rs_c]) \
DSCALEA2V(18,19,26,27,25,0) \
DSCALEA2V(20,21,28,29,25,0) \
DSCALEA2V(22,23,30,31,25,0) \
LABEL(DZERO_BETA_R_6_7_ ## PACKA ## _ ## PACKB) \
DSTOREC_3V_R_FWD(18,19,20,%[c],0,%[rs_c]) \
DSTOREC_3V_R_FWD(21,22,23,%[c],0,%[rs_c]) \
BRANCH(DEND_WRITE_MEM_ ## PACKA ## _ ## PACKB) \
\
/* Column-major C-storage. */ \
LABEL(DWRITE_MEM_C_ ## PACKA ## _ ## PACKB) \
" ld1r            {v30.2d}, [%[beta]]             \n\t" /* Reload beta. */ \
" fcmp            d30, #0.0                       \n\t" /* Sets conditional flag whether *beta == 0. */ \
"                                                 \n\t" /* This conditional flag will be used */ \
"                                                 \n\t" /*  multiple times for skipping load. */ \
/* In-register transpose,
 *  do transposition in row-order. */ \
" trn1            v24.2d, v0.2d, v3.2d            \n\t" /* Row 0-1. */ \
" trn2            v25.2d, v0.2d, v3.2d            \n\t" \
" trn1            v26.2d, v1.2d, v4.2d            \n\t" \
" trn2            v27.2d, v1.2d, v4.2d            \n\t" \
" trn1            v28.2d, v2.2d, v5.2d            \n\t" \
" trn2            v29.2d, v2.2d, v5.2d            \n\t" \
"                                                 \n\t" \
" trn1            v0.2d, v6.2d, v9.2d             \n\t" /* Row 2-3. */ \
" trn2            v1.2d, v6.2d, v9.2d             \n\t" \
" trn1            v2.2d, v7.2d, v10.2d            \n\t" \
" trn2            v3.2d, v7.2d, v10.2d            \n\t" \
" trn1            v4.2d, v8.2d, v11.2d            \n\t" \
" trn2            v5.2d, v8.2d, v11.2d            \n\t" \
"                                                 \n\t" \
" trn1            v6.2d, v12.2d, v15.2d           \n\t" /* Row 4-5. */ \
" trn2            v7.2d, v12.2d, v15.2d           \n\t" \
" trn1            v8.2d, v13.2d, v16.2d           \n\t" \
" trn2            v9.2d, v13.2d, v16.2d           \n\t" \
" trn1            v10.2d, v14.2d, v17.2d          \n\t" \
" trn2            v11.2d, v14.2d, v17.2d          \n\t" \
"                                                 \n\t" \
" trn1            v12.2d, v18.2d, v21.2d          \n\t" /* Row 4-5. */ \
" trn2            v13.2d, v18.2d, v21.2d          \n\t" \
" trn1            v14.2d, v19.2d, v22.2d          \n\t" \
" trn2            v15.2d, v19.2d, v22.2d          \n\t" \
" trn1            v16.2d, v20.2d, v23.2d          \n\t" \
" trn2            v17.2d, v20.2d, v23.2d          \n\t" \
"                                                 \n\t" \
BEQ(ZERO_BETA_R_0_1_ ## PACKA ## _ ## PACKB) \
DLOADC_4V_C_FWD(18,19,20,21,%[c_ld],0,%[cs_c]) \
DSCALEA4V(24,0,6,12,18,19,20,21,30,0) \
DLOADC_4V_C_FWD(18,19,20,21,%[c_ld],0,%[cs_c]) \
DSCALEA4V(25,1,7,13,18,19,20,21,30,0) \
LABEL(ZERO_BETA_R_0_1_ ## PACKA ## _ ## PACKB) \
DSTOREC_4V_C_FWD(24,0,6,12,%[c],0,%[cs_c]) \
DSTOREC_4V_C_FWD(25,1,7,13,%[c],0,%[cs_c]) \
BEQ(ZERO_BETA_R_2_3_4_5_ ## PACKA ## _ ## PACKB) \
DLOADC_4V_C_FWD(18,19,20,21,%[c_ld],0,%[cs_c]) \
DLOADC_4V_C_FWD(22,23,24,25,%[c_ld],0,%[cs_c]) \
DSCALEA8V(26,2,8,14,27,3,9,15,18,19,20,21,22,23,24,25,30,0) \
DLOADC_4V_C_FWD(18,19,20,21,%[c_ld],0,%[cs_c]) \
DLOADC_4V_C_FWD(22,23,24,25,%[c_ld],0,%[cs_c]) \
DSCALEA8V(28,4,10,16,29,5,11,17,18,19,20,21,22,23,24,25,30,0) \
LABEL(ZERO_BETA_R_2_3_4_5_ ## PACKA ## _ ## PACKB) \
DSTOREC_4V_C_FWD(26,2,8,14,%[c],0,%[cs_c]) \
DSTOREC_4V_C_FWD(27,3,9,15,%[c],0,%[cs_c]) \
DSTOREC_4V_C_FWD(28,4,10,16,%[c],0,%[cs_c]) \
DSTOREC_4V_C_FWD(29,5,11,17,%[c],0,%[cs_c]) \
/* Done. */ \
LABEL(DEND_WRITE_MEM_ ## PACKA ## _ ## PACKB) \
: [a]      "+r" (a), \
  [cs_a]   "+r" (cs_a), \
  [b_0]    "+r" (b_0), \
  [b_1]    "+r" (b_1), \
  [b_2]    "+r" (b_2), \
  [b_3]    "+r" (b_3), \
  [b_4]    "+r" (b_4), \
  [b_5]    "+r" (b_5), \
  [rs_b]   "+r" (rs_b), \
  [c]      "+r" (c), \
  [c_ld]   "+r" (c_ld), \
  [rs_c]   "+r" (rs_c), \
  [cs_c]   "+r" (cs_c), \
  [k_mker] "+r" (k_mker), \
  [k_left] "+r" (k_left), \
  [alpha]  "+r" (alpha), \
  [beta]   "+r" (beta), \
  [a_next] "+r" (a_next), \
  [b_next] "+r" (b_next), \
  [a_p]    "+r" (a_p), \
  [b_p]    "+r" (b_p) \
: \
: /* Clobber all vector registers. */ \
  "v0","v1","v2","v3","v4","v5","v6","v7", \
  "v8","v9","v10","v11","v12","v13","v14","v15", \
  "v16","v17","v18","v19", \
  "v20","v21","v22","v23", \
  "v24","v25","v26","v27", \
  "v28","v29","v30","v31" \
  ); \
\
}

GENDEF(pack,pack)
GENDEF(pack,nopack)
GENDEF(nopack,pack)
#undef GENDEF

#if 0
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
    )
{
    assert( m == 8 );
    assert( n == 6 );
    assert( rs_a0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );

    switch ( !!pack_a << 1 | !!pack_b ) {
    case ( 1 << 1 | 1 ):
        bli_dgemmsup2_cv_armv8a_asm_8x6r_pack_pack
            ( m, n, k,
              alpha,
              a, rs_a0, cs_a0,
              b, rs_b0, cs_b0,
              beta,
              c, rs_c0, cs_c0,
              data, cntx, a_p, b_p
            ); break;
    case ( 1 << 1 | 0 ):
        bli_dgemmsup2_cv_armv8a_asm_8x6r_pack_nopack
            ( m, n, k,
              alpha,
              a, rs_a0, cs_a0,
              b, rs_b0, cs_b0,
              beta,
              c, rs_c0, cs_c0,
              data, cntx, a_p, b_p
            ); break;
    case ( 0 << 1 | 1 ):
        bli_dgemmsup2_cv_armv8a_asm_8x6r_nopack_pack
            ( m, n, k,
              alpha,
              a, rs_a0, cs_a0,
              b, rs_b0, cs_b0,
              beta,
              c, rs_c0, cs_c0,
              data, cntx, a_p, b_p
            ); break;
    default:
        assert( 0 ); break;
    }
}
#endif

#endif


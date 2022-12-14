#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)

#include "blis.h"
#include <assert.h>


// Label locality & misc.
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

#define DGEMM_MUL_1LINE(OP,DR0,DR1,DR2,SR0,SR1,SR2,VCOEF,IDX) \
 #OP " v"#DR0".2d, v"#SR0".2d, v"#VCOEF".d["#IDX"] \n\t" \
 #OP " v"#DR1".2d, v"#SR1".2d, v"#VCOEF".d["#IDX"] \n\t" \
 #OP " v"#DR2".2d, v"#SR2".2d, v"#VCOEF".d["#IDX"] \n\t"

#define DGEMM_8X6_MKER_LOOP_1(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C00,C01,C02,B0,B1,B2,A0,0)

#define DGEMM_8X6_MKER_LOOP_2(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_1(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C10,C11,C12,B0,B1,B2,A0,1)

#define DGEMM_8X6_MKER_LOOP_3(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_2(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C20,C21,C22,B0,B1,B2,A1,0)

#define DGEMM_8X6_MKER_LOOP_4(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_3(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C30,C31,C32,B0,B1,B2,A1,1)

#define DGEMM_8X6_MKER_LOOP_5(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_4(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C40,C41,C42,B0,B1,B2,A2,0)

#define DGEMM_8X6_MKER_LOOP_6(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_5(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C50,C51,C52,B0,B1,B2,A2,1)

#define DGEMM_8X6_MKER_LOOP_7(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_6(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C60,C61,C62,B0,B1,B2,A3,0)

#define DGEMM_8X6_MKER_LOOP_8(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_7(OP,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1LINE(OP,C70,C71,C72,B0,B1,B2,A3,1)

#define DGEMM_8X6_MKER_LOOP_LOC(THISM,OP,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_ ## THISM (OP,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2)

#define DGEMM_LOAD_ACOL_LOC_1(A0,A1,A2,A3) \
 "ldr  d"#A0", [%[a], #0*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_2(A0,A1,A2,A3) \
 "ldr  q"#A0", [%[a], #0*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_3(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_2(A0,A1,A2,A3) \
 "ldr  d"#A1", [%[a], #1*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_4(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_2(A0,A1,A2,A3) \
 "ldr  q"#A1", [%[a], #1*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_5(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_4(A0,A1,A2,A3) \
 "ldr  d"#A2", [%[a], #2*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_6(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_4(A0,A1,A2,A3) \
 "ldr  q"#A2", [%[a], #2*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_7(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_6(A0,A1,A2,A3) \
 "ldr  d"#A3", [%[a], #3*16]  \n\t"

#define DGEMM_LOAD_ACOL_LOC_8(A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_LOC_6(A0,A1,A2,A3) \
 "ldr  q"#A3", [%[a], #3*16]  \n\t"

#define DGEMM_LOAD_BROW_LOC(B0,B1,B2) \
 "ld1  {v"#B0".d}[0], [%[b_0]], %[rs_b]  \n\t" \
 "ld1  {v"#B0".d}[1], [%[b_1]], %[rs_b]  \n\t" \
 "ld1  {v"#B1".d}[0], [%[b_2]], %[rs_b]  \n\t" \
 "ld1  {v"#B1".d}[1], [%[b_3]], %[rs_b]  \n\t" \
 "ld1  {v"#B2".d}[0], [%[b_4]], %[rs_b]  \n\t" \
 "ld1  {v"#B2".d}[1], [%[b_5]], %[rs_b]  \n\t"

#define CLOAD_1ROW(C0,C1,C2,CADDR) \
 "ldr  q"#C0", ["#CADDR", 0*16]  \n\t" \
 "ldr  q"#C1", ["#CADDR", 1*16]  \n\t" \
 "ldr  q"#C2", ["#CADDR", 2*16]  \n\t"

#define CSCALE_1ROW(DV0,DV1,DV2,SV0,SV1,SV2,VBETA,IDX) \
 "fmla  v"#DV0".2d, v"#SV0".2d, v"#VBETA".d["#IDX"]  \n\t" \
 "fmla  v"#DV1".2d, v"#SV1".2d, v"#VBETA".d["#IDX"]  \n\t" \
 "fmla  v"#DV2".2d, v"#SV2".2d, v"#VBETA".d["#IDX"]  \n\t"

#define CSTORE_1ROW(C0,C1,C2,CADDR) \
 "str  q"#C0", ["#CADDR", 0*16]  \n\t" \
 "str  q"#C1", ["#CADDR", 1*16]  \n\t" \
 "str  q"#C2", ["#CADDR", 2*16]  \n\t"

#define CIO_UNIT_1ROW(DV0,DV1,DV2,SV0,SV1,SV2,VBETA,IDX,CADDR,RSC) \
  CLOAD_1ROW(SV0,SV1,SV2,CADDR) \
  CSCALE_1ROW(DV0,DV1,DV2,SV0,SV1,SV2,VBETA,IDX) \
  CSTORE_1ROW(DV0,DV1,DV2,CADDR) \
 "add  "#CADDR", "#CADDR", "#RSC"  \n\t"

#define CIO_UNIT_1COL(DV0,DV1,DV2,SV0,SV1,SV2,VBETA,IDX,CADDR,CSC) \
 "mov  %[c_ld], "#CADDR"  \n\t" \
 "ld1  {v"#SV0".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "ld1  {v"#SV0".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "ld1  {v"#SV1".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "ld1  {v"#SV1".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "ld1  {v"#SV2".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "ld1  {v"#SV2".d}[1], [%[c_ld]], "#CSC"  \n\t" \
  CSCALE_1ROW(DV0,DV1,DV2,SV0,SV1,SV2,VBETA,IDX) \
 "mov  %[c_ld], "#CADDR"  \n\t" \
 "st1  {v"#DV0".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#DV0".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#DV1".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#DV1".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#DV2".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#DV2".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "add  "#CADDR", "#CADDR", #8  \n\t"

#define CSTORE_1(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C00,C01,C02,V0,V1,V2,VBETA,IDX,CADDR,LDC)

#define CSTORE_2(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_1(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C10,C11,C12,V3,V4,V5,VBETA,IDX,CADDR,LDC)

#define CSTORE_3(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_2(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C20,C21,C22,V0,V1,V2,VBETA,IDX,CADDR,LDC)

#define CSTORE_4(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_3(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C30,C31,C32,V3,V4,V5,VBETA,IDX,CADDR,LDC)

#define CSTORE_5(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_4(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C40,C41,C42,V0,V1,V2,VBETA,IDX,CADDR,LDC)

#define CSTORE_6(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_5(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C50,C51,C52,V3,V4,V5,VBETA,IDX,CADDR,LDC)

#define CSTORE_7(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_6(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C60,C61,C62,V0,V1,V2,VBETA,IDX,CADDR,LDC)

#define CSTORE_8(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_7(SCHEMA,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C70,C71,C72,V3,V4,V5,VBETA,IDX,CADDR,LDC)

#define PACKA_STORE_FWD_nopack(A0,A1,A2,A3,PAADDR)
#define PACKA_STORE_FWD_pack(A0,A1,A2,A3,PAADDR) \
 "str  q"#A0", ["#PAADDR", #0*16]  \n\t" \
 "str  q"#A1", ["#PAADDR", #1*16]  \n\t" \
 "str  q"#A2", ["#PAADDR", #2*16]  \n\t" \
 "str  q"#A3", ["#PAADDR", #3*16]  \n\t" \
 "add  "#PAADDR", "#PAADDR", #4*16  \n\t"

#define PACKB_STORE_FWD_nopack(B0,B1,B2,PBADDR)
#define PACKB_STORE_FWD_pack(B0,B1,B2,PBADDR) \
 "str  q"#B0", ["#PBADDR", #0*16]  \n\t" \
 "str  q"#B1", ["#PBADDR", #1*16]  \n\t" \
 "str  q"#B2", ["#PBADDR", #2*16]  \n\t" \
 "add  "#PBADDR", "#PBADDR", #3*16  \n\t"


#define GENDECL(THISM,PACKA,PACKB) \
void bli_dgemmsup2_cv_armv8a_asm_## THISM ##x6r_ ## PACKA ## _ ## PACKB \
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
    )

#define GENDEF(THISM,PACKA,PACKB) \
  GENDECL(THISM,PACKA,PACKB) \
{ \
  const void* a_next = bli_auxinfo_next_a( data ); \
  const void* b_next = bli_auxinfo_next_b( data ); \
\
  /* Typecast local copies of integers in case dim_t and inc_t are a
   * different size than is expected by load instructions. */ \
  uint64_t k_mker = k / 4; \
  uint64_t k_left = k % 4; \
  uint64_t cs_a   = cs_a0 << 3; \
  uint64_t rs_b   = rs_b0 << 3; \
  uint64_t rs_c   = rs_c0 << 3; \
  uint64_t cs_c   = cs_c0 << 3; \
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
" cmp             %[k_mker], #0                   \n\t" \
BEQ(DK_LEFT_LOOP_INIT_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(28,29,30) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmul,24,25,26,27,28,29,30) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (28,29,30,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(31,28,29) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,31,28,29) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (31,28,29,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(30,31,28) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,30,31,28) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (30,31,28,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(29,30,31) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_MKER_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(28,29,30) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,28,29,30) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (28,29,30,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(31,28,29) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,31,28,29) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (31,28,29,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(30,31,28) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,30,31,28) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (30,31,28,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(29,30,31) \
" add             %[a], %[a], %[cs_a]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
BRANCH(DK_MKER_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_LEFT_LOOP_INIT_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(CLEAR_CROWS_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(29,30,31) \
" add             %[a], %[a], %[cs_a]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmul,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
LABEL(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(DWRITE_MEM_PREP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
DGEMM_LOAD_ACOL_LOC_ ## THISM (24,25,26,27) \
DGEMM_LOAD_BROW_LOC(29,30,31) \
" add             %[a], %[a], %[cs_a]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
BRANCH(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(CLEAR_CROWS_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
CLEAR8V(0,1,2,3,4,5,6,7) \
CLEAR8V(8,9,10,11,12,13,14,15) \
CLEAR8V(16,17,18,19,20,21,22,23) \
\
LABEL(DWRITE_MEM_PREP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" ld1r            {v30.2d}, [%[alpha]]            \n\t" /* Load alpha & beta. */ \
" ld1r            {v31.2d}, [%[beta]]             \n\t" \
"                                                 \n\t" \
LABEL(DPREFETCH_ABNEXT_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" prfm            PLDL1KEEP, [%[a_next], 64*0]    \n\t" \
" prfm            PLDL1KEEP, [%[a_next], 64*1]    \n\t" \
" prfm            PLDL1KEEP, [%[a_next], 64*2]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*0]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*1]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*2]    \n\t" \
" prfm            PLDL1KEEP, [%[b_next], 64*3]    \n\t" \
"                                                 \n\t" \
" fmov            d26, #1.0                       \n\t" \
" fcmp            d30, d26                        \n\t" \
BEQ(DUNIT_ALPHA_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
DSCALE8V(0,1,2,3,4,5,6,7,30,0)         \
DSCALE8V(8,9,10,11,12,13,14,15,30,0)   \
DSCALE8V(16,17,18,19,20,21,22,23,30,0) \
LABEL(DUNIT_ALPHA_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
"                                                 \n\t" \
" cmp             %[cs_c], #8                     \n\t" \
"                                                 \n\t" \
BNE(DWRITE_MEM_C_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
CSTORE_## THISM (ROW,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,%[c],%[rs_c],24,25,26,27,28,29,31,0) \
BRANCH(DEND_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DWRITE_MEM_C_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
CSTORE_## THISM (COL,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,%[c],%[cs_c],24,25,26,27,28,29,31,0) \
LABEL(DEND_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
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

// GENDEF(1,pack,pack)
// GENDEF(2,pack,pack)
// GENDEF(3,pack,pack)
// GENDEF(4,pack,pack)
// GENDEF(5,pack,pack)
// GENDEF(6,pack,pack)
// GENDEF(7,pack,pack)
GENDECL(8,pack,pack);

GENDEF(1,pack,nopack)
GENDEF(2,pack,nopack)
GENDEF(3,pack,nopack)
GENDEF(4,pack,nopack)
GENDEF(5,pack,nopack)
GENDEF(6,pack,nopack)
GENDEF(7,pack,nopack)
GENDECL(8,pack,nopack);

// GENDEF(1,nopack,pack)
// GENDEF(2,nopack,pack)
// GENDEF(3,nopack,pack)
// GENDEF(4,nopack,pack)
// GENDEF(5,nopack,pack)
// GENDEF(6,nopack,pack)
// GENDEF(7,nopack,pack)
GENDECL(8,nopack,pack);

GENDEF(1,nopack,nopack)
GENDEF(2,nopack,nopack)
GENDEF(3,nopack,nopack)
GENDEF(4,nopack,nopack)
GENDEF(5,nopack,nopack)
GENDEF(6,nopack,nopack)
GENDEF(7,nopack,nopack)
GENDECL(8,nopack,nopack);

#undef GENDEF
#undef GENDECL


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
    if ( n != 6 )
    {
        if ( m == 8 )
        {
            bli_dgemmsup2_cv_armv8a_asm_8x6c
                ( m, n, k,
                  alpha,
                  a, rs_a0, cs_a0,
                  b, rs_b0, cs_b0,
                  beta,
                  c, rs_c0, cs_c0,
                  data, cntx,
                  a_p, pack_a,
                  b_p, pack_b );
            return ;
        }
        else
        {
            // Static C scratchpad for this.
            static double c_t[ 6*8 ];
            static double one = 1.0;
            static double zero = 0.0;
            // Caller ensures p_a has enough space. Now do the packing.
            l1mukr_t dpackm = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_MRXK_KER, cntx );

            dpackm
                ( BLIS_NO_CONJUGATE, BLIS_PACKED_COLUMNS, 
                  m, k, k,
                  alpha,
                  a, rs_a0, cs_a0,
                  a_p, 8,
                  cntx );

            bli_dgemmsup2_cv_armv8a_asm_8x6c
                ( 8, n, k, &one,
                  a_p, 1, 8,
                  b, rs_b0, cs_b0, &zero,
                  c_t, 1, 8,
                  data, cntx,
                  a_p, 0,
                  b_p, 0 // pack_b
                );

            // Unpack result C.
            for ( int i = 0; i < m; ++i )
                for ( int j = 0; j < n; ++j )
                    c[ i * rs_c0 + j * cs_c0 ] =
                        c[ i * rs_c0 + j * cs_c0 ] * *beta +
                        c_t[ i + j * 8 ];

            return ;
        }
    }

#ifdef DEBUG
    assert( m <= 8 );
    assert( rs_a0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif

    switch ( !!pack_a << 9 | !!pack_b << 8 | m ) {
#define EXPAND_CASE_BASE( M, PA, PB, PACKA, PACKB ) \
    case ( PA << 9 | PB << 8 | M ): \
        bli_dgemmsup2_cv_armv8a_asm_ ## M ## x6r_## PACKA ##_## PACKB \
            ( m, n, k, \
              alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, \
              beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p \
            ); break;
#define EXPAND_CASE1( M ) \
      EXPAND_CASE_BASE( M, 1, 1, pack, pack ) \
      EXPAND_CASE_BASE( M, 1, 0, pack, nopack ) \
      EXPAND_CASE_BASE( M, 0, 1, nopack, pack ) \
      EXPAND_CASE_BASE( M, 0, 0, nopack, nopack )
    EXPAND_CASE1(8)

    // Final row-block. B tiles'll never be reused.
#define EXPAND_CASE2( M ) \
    case ( 1 << 9 | 1 << 8 | M ): \
      EXPAND_CASE_BASE( M, 1, 0, pack, nopack ) \
    case ( 0 << 9 | 1 << 8 | M ): \
      EXPAND_CASE_BASE( M, 0, 0, nopack, nopack )
    EXPAND_CASE2(7)
    EXPAND_CASE2(6)
    EXPAND_CASE2(5)
    EXPAND_CASE2(4)
    EXPAND_CASE2(3)
    EXPAND_CASE2(2)
    EXPAND_CASE2(1)

    default:
#ifdef DEBUG
        assert( 0 );
#endif
        break;
    }
}

#endif


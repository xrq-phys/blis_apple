#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)

#include <assert.h>
#include "blis.h"


// Label locality & misc.
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

#define DGEMM_MUL_1COL(OP,DR0,DR1,DR2,DR3,SR0,SR1,SR2,SR3,VCOEF,IDX) \
 #OP " v"#DR0".2d, v"#SR0".2d, v"#VCOEF".d["#IDX"] \n\t" \
 #OP " v"#DR1".2d, v"#SR1".2d, v"#VCOEF".d["#IDX"] \n\t" \
 #OP " v"#DR2".2d, v"#SR2".2d, v"#VCOEF".d["#IDX"] \n\t" \
 #OP " v"#DR3".2d, v"#SR3".2d, v"#VCOEF".d["#IDX"] \n\t"

#define DGEMM_8X6_MKER_LOOP_1(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C00,C10,C20,C30,A0,A1,A2,A3,B0,0)

#define DGEMM_8X6_MKER_LOOP_2(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_1(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C01,C11,C21,C31,A0,A1,A2,A3,B0,1)

#define DGEMM_8X6_MKER_LOOP_3(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_2(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C02,C12,C22,C32,A0,A1,A2,A3,B1,0)

#define DGEMM_8X6_MKER_LOOP_4(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_3(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C03,C13,C23,C33,A0,A1,A2,A3,B1,1)

#define DGEMM_8X6_MKER_LOOP_5(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_4(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C04,C14,C24,C34,A0,A1,A2,A3,B2,0)

#define DGEMM_8X6_MKER_LOOP_6(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_5(OP,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_MUL_1COL(OP,C05,C15,C25,C35,A0,A1,A2,A3,B2,1)

#define DGEMM_8X6_MKER_LOOP_LOC(THISN,OP,A0,A1,A2,A3,B0,B1,B2) \
  DGEMM_8X6_MKER_LOOP_ ## THISN (OP,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2)

#define DGEMM_LOAD_BROW_1(B0,B1,B2,BADDR) \
 "ldr  d"#B0", ["#BADDR", #8*0]  \n\t"

#define DGEMM_LOAD_BROW_2(B0,B1,B2,BADDR) \
 "ldr  q"#B0", ["#BADDR", #8*0]  \n\t"

#define DGEMM_LOAD_BROW_3(B0,B1,B2,BADDR) \
  DGEMM_LOAD_BROW_2(B0,B1,B2,BADDR) \
 "ldr  d"#B1", ["#BADDR", #8*2]  \n\t"

#define DGEMM_LOAD_BROW_4(B0,B1,B2,BADDR) \
  DGEMM_LOAD_BROW_2(B0,B1,B2,BADDR) \
 "ldr  q"#B1", ["#BADDR", #8*2]  \n\t"

#define DGEMM_LOAD_BROW_5(B0,B1,B2,BADDR) \
  DGEMM_LOAD_BROW_4(B0,B1,B2,BADDR) \
 "ldr  d"#B2", ["#BADDR", #8*4]  \n\t"

#define DGEMM_LOAD_BROW_6(B0,B1,B2,BADDR) \
  DGEMM_LOAD_BROW_4(B0,B1,B2,BADDR) \
 "ldr  q"#B2", ["#BADDR", #8*4]  \n\t"

#define DGEMM_LOAD_ACOL_LOC(A0,A1,A2,A3) \
 "ld1  {v"#A0".d}[0], [%[a_0]], %[cs_a]  \n\t" \
 "ld1  {v"#A0".d}[1], [%[a_1]], %[cs_a]  \n\t" \
 "ld1  {v"#A1".d}[0], [%[a_2]], %[cs_a]  \n\t" \
 "ld1  {v"#A1".d}[1], [%[a_3]], %[cs_a]  \n\t" \
 "ld1  {v"#A2".d}[0], [%[a_4]], %[cs_a]  \n\t" \
 "ld1  {v"#A2".d}[1], [%[a_5]], %[cs_a]  \n\t" \
 "ld1  {v"#A3".d}[0], [%[a_6]], %[cs_a]  \n\t" \
 "ld1  {v"#A3".d}[1], [%[a_7]], %[cs_a]  \n\t"

#define CSCALE_1COL(DV0,DV1,DV2,DV3,SV0,SV1,SV2,SV3,VBETA,IDX) \
 "fmla  v"#DV0".2d, v"#SV0".2d, v"#VBETA".d["#IDX"]  \n\t" \
 "fmla  v"#DV1".2d, v"#SV1".2d, v"#VBETA".d["#IDX"]  \n\t" \
 "fmla  v"#DV2".2d, v"#SV2".2d, v"#VBETA".d["#IDX"]  \n\t" \
 "fmla  v"#DV3".2d, v"#SV3".2d, v"#VBETA".d["#IDX"]  \n\t"

#define CIO_UNIT_1COL(DV0,DV1,DV2,DV3,SV0,SV1,SV2,SV3,VBETA,IDX,CADDR,CSC) \
 "ldr  q"#SV0", ["#CADDR", 0*16]  \n\t" \
 "ldr  q"#SV1", ["#CADDR", 1*16]  \n\t" \
 "ldr  q"#SV2", ["#CADDR", 2*16]  \n\t" \
 "ldr  q"#SV3", ["#CADDR", 3*16]  \n\t" \
  CSCALE_1COL(DV0,DV1,DV2,DV3,SV0,SV1,SV2,SV3,VBETA,IDX) \
 "str  q"#DV0", ["#CADDR", 0*16]  \n\t" \
 "str  q"#DV1", ["#CADDR", 1*16]  \n\t" \
 "str  q"#DV2", ["#CADDR", 2*16]  \n\t" \
 "str  q"#DV3", ["#CADDR", 3*16]  \n\t" \
 "add  "#CADDR", "#CADDR", "#CSC"  \n\t"

#define CIO_UNIT_1ROW(DV0,DV1,DV2,DV3,SV0,SV1,SV2,SV3,VBETA,IDX,CADDR,RSC) \
 "mov  %[c_ld], "#CADDR"  \n\t" \
 "ld1  {v"#SV0".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV0".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV1".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV1".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV2".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV2".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV3".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "ld1  {v"#SV3".d}[1], [%[c_ld]], "#RSC"  \n\t" \
  CSCALE_1COL(DV0,DV1,DV2,DV3,SV0,SV1,SV2,SV3,VBETA,IDX) \
 "mov  %[c_ld], "#CADDR"  \n\t" \
 "st1  {v"#DV0".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV0".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV1".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV1".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV2".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV2".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV3".d}[0], [%[c_ld]], "#RSC"  \n\t" \
 "st1  {v"#DV3".d}[1], [%[c_ld]], "#RSC"  \n\t" \
 "add  "#CADDR", "#CADDR", #8  \n\t"

#define CSTORE_1(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C00,C10,C20,C30,V0,V1,V2,V3,VBETA,IDX,CADDR,LDC)

#define CSTORE_2(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_1(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C01,C11,C21,C31,V0,V1,V2,V3,VBETA,IDX,CADDR,LDC)

#define CSTORE_3(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_2(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C02,C12,C22,C32,C00,C10,C20,C30,VBETA,IDX,CADDR,LDC)

#define CSTORE_4(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_3(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C03,C13,C23,C33,C01,C11,C21,C31,VBETA,IDX,CADDR,LDC)

#define CSTORE_5(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_4(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C04,C14,C24,C34,V0,V1,V2,V3,VBETA,IDX,CADDR,LDC)

#define CSTORE_6(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CSTORE_5(SCHEMA,C00,C01,C02,C03,C04,C05,C10,C11,C12,C13,C14,C15,C20,C21,C22,C23,C24,C25,C30,C31,C32,C33,C34,C35,CADDR,LDC,V0,V1,V2,V3,V4,V5,VBETA,IDX) \
  CIO_UNIT_1 ## SCHEMA (C05,C15,C25,C35,C02,C12,C22,C32,VBETA,IDX,CADDR,LDC)

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


#define GENDECL(THISN,PACKA,PACKB) \
void bli_dgemmsup2_rv_armv8a_asm_8x## THISN ##c_ ## PACKA ## _ ## PACKB \
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

#define GENDEF(THISN,PACKA,PACKB) \
  GENDECL(THISN,PACKA,PACKB) \
{ \
  const void* a_next = bli_auxinfo_next_a( data ); \
  const void* b_next = bli_auxinfo_next_b( data ); \
\
  /* Typecast local copies of integers in case dim_t and inc_t are a
   * different size than is expected by load instructions. */ \
  uint64_t k_mker = k / 5; \
  uint64_t k_left = k % 5; \
  uint64_t cs_a   = cs_a0 << 3; \
  uint64_t rs_b   = rs_b0 << 3; \
  uint64_t rs_c   = rs_c0 << 3; \
  uint64_t cs_c   = cs_c0 << 3; \
\
  /* Define all integer registers outside asm block & avoid "x*" clobbing */ \
  uint64_t c_ld; \
  double *a_0 = a, \
         *a_1 = a_0 + rs_a0, \
         *a_2 = a_1 + rs_a0, \
         *a_3 = a_2 + rs_a0, \
         *a_4 = a_3 + rs_a0, \
         *a_5 = a_4 + rs_a0, \
         *a_6 = a_5 + rs_a0, \
         *a_7 = a_6 + rs_a0; \
\
  __asm__ volatile \
  ( \
" cmp             %[k_mker], #0                   \n\t" \
BEQ(DK_LEFT_LOOP_INIT_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
DGEMM_LOAD_ACOL_LOC(24,25,26,27) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmul,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(28,24,25,26) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,28,24,25,26,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (28,24,25,26,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(27,28,24,25) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,27,28,24,25,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (27,28,24,25,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(26,27,28,24) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,26,27,28,24,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (26,27,28,24,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(25,26,27,28) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,25,26,27,28,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (25,26,27,28,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_MKER_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
DGEMM_LOAD_ACOL_LOC(24,25,26,27) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(28,24,25,26) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,28,24,25,26,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (28,24,25,26,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(27,28,24,25) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,27,28,24,25,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (27,28,24,25,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(26,27,28,24) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,26,27,28,24,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (26,27,28,24,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
DGEMM_LOAD_ACOL_LOC(25,26,27,28) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,25,26,27,28,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (25,26,27,28,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
BRANCH(DK_MKER_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_LEFT_LOOP_INIT_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(CLEAR_CROWS_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
DGEMM_LOAD_ACOL_LOC(24,25,26,27) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmul,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
\
LABEL(DK_LEFT_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(DWRITE_MEM_PREP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
DGEMM_LOAD_ACOL_LOC(24,25,26,27) \
DGEMM_LOAD_BROW_ ## THISN (29,30,31,%[b]) \
" add             %[b], %[b], %[rs_b]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_8X6_MKER_LOOP_LOC(THISN,fmla,24,25,26,27,29,30,31) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
BRANCH(DK_LEFT_LOOP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(CLEAR_CROWS_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
CLEAR8V(0,1,2,3,4,5,6,7) \
CLEAR8V(8,9,10,11,12,13,14,15) \
CLEAR8V(16,17,18,19,20,21,22,23) \
\
LABEL(DWRITE_MEM_PREP_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
" ld1r            {v30.2d}, [%[alpha]]            \n\t" /* Load alpha & beta. */ \
" ld1r            {v31.2d}, [%[beta]]             \n\t" \
"                                                 \n\t" \
LABEL(DPREFETCH_ABNEXT_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
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
BEQ(DUNIT_ALPHA_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
DSCALE8V(0,1,2,3,4,5,6,7,30,0)         \
DSCALE8V(8,9,10,11,12,13,14,15,30,0)   \
DSCALE8V(16,17,18,19,20,21,22,23,30,0) \
LABEL(DUNIT_ALPHA_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
"                                                 \n\t" \
" cmp             %[rs_c], #8                     \n\t" \
"                                                 \n\t" \
BNE(DWRITE_MEM_R_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
\
CSTORE_## THISN (COL,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,%[c],%[cs_c],24,25,26,27,28,29,31,0) \
BRANCH(DEND_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DWRITE_MEM_R_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
CSTORE_## THISN (ROW,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,%[c],%[rs_c],24,25,26,27,28,29,31,0) \
LABEL(DEND_ ## THISN ## _ ## PACKA ## _ ## PACKB) \
: [a_0]    "+r" (a_0), \
  [a_1]    "+r" (a_1), \
  [a_2]    "+r" (a_2), \
  [a_3]    "+r" (a_3), \
  [a_4]    "+r" (a_4), \
  [a_5]    "+r" (a_5), \
  [a_6]    "+r" (a_6), \
  [a_7]    "+r" (a_7), \
  [cs_a]   "+r" (cs_a), \
  [b]      "+r" (b), \
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
// // GENDECL(6,pack,pack);

// GENDEF(1,pack,nopack)
// GENDEF(2,pack,nopack)
// GENDEF(3,pack,nopack)
// GENDEF(4,pack,nopack)
// GENDEF(5,pack,nopack)
// // GENDECL(6,pack,nopack);

GENDEF(1,nopack,pack)
GENDEF(2,nopack,pack)
GENDEF(3,nopack,pack)
GENDEF(4,nopack,pack)
GENDEF(5,nopack,pack)
// GENDECL(6,nopack,pack);

GENDEF(1,nopack,nopack)
GENDEF(2,nopack,nopack)
GENDEF(3,nopack,nopack)
GENDEF(4,nopack,nopack)
GENDEF(5,nopack,nopack)

#undef GENDEF
#undef GENDECL


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
    )
{
#ifdef DEBUG
    assert( m == 8 );
    assert( n <= 6 );
    assert( cs_b0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif

    switch ( !!pack_a << 9 | !!pack_b << 8 | n ) {
    // Final column-block. A tiles'll never be reused.
#define EXPAND_CASE(N) \
    case ( 0 << 9 | 1 << 8 | N ): \
    case ( 1 << 9 | 1 << 8 | N ): \
        bli_dgemmsup2_rv_armv8a_asm_8x ## N ## c_nopack_pack \
            ( m, n, k, \
              alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, \
              beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p \
            ); break; \
    case ( 0 << 9 | 0 << 8 | N ): \
    case ( 1 << 9 | 0 << 8 | N ): \
        bli_dgemmsup2_rv_armv8a_asm_8x ## N ## c_nopack_nopack \
            ( m, n, k, \
              alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, \
              beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p \
            ); break;
    // EXPAND_CASE(6) // Forbidden w/o _pack_* case instantiation.
    EXPAND_CASE(5)
    EXPAND_CASE(4)
    EXPAND_CASE(3)
    EXPAND_CASE(2)
    EXPAND_CASE(1)
    default:
#ifdef DEBUG
        assert( 0 );
#endif
        break;
    }
}

#endif


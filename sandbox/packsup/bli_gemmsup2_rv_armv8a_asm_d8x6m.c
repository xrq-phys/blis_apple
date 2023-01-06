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

#define DGEMM_LOAD_ACOL_1(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A0".d}[0], ["#AEL0"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_2(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_1(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A0".d}[1], ["#AEL1"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_3(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_2(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A1".d}[0], ["#AEL2"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_4(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_3(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A1".d}[1], ["#AEL3"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_5(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_4(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A2".d}[0], ["#AEL4"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_6(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_5(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A2".d}[1], ["#AEL5"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_7(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_6(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A3".d}[0], ["#AEL6"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_8(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
  DGEMM_LOAD_ACOL_7(A0,A1,A2,A3,AEL0,AEL1,AEL2,AEL3,AEL4,AEL5,AEL6,AEL7,CSA) \
 "ld1  {v"#A3".d}[1], ["#AEL7"], "#CSA"  \n\t"

#define DGEMM_LOAD_ACOL_LOC(THISM,A0,A1,A2,A3) \
  DGEMM_LOAD_ACOL_ ## THISM (A0,A1,A2,A3,%[a_0],%[a_1],%[a_2],%[a_3],%[a_4],%[a_5],%[a_6],%[a_7],%[cs_a])

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
 "st1  {v"#SV0".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#SV0".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#SV1".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#SV1".d}[1], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#SV2".d}[0], [%[c_ld]], "#CSC"  \n\t" \
 "st1  {v"#SV2".d}[1], [%[c_ld]], "#CSC"  \n\t" \
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

#define PRFA_LOC \
 "prfm PLDL1STRM, [%[a_next]] \n\t" \
 "add  %[a_next], %[a_next], %[cs_a2] \n\t"


#define GENDECL(THISM,PACKA,PACKB) \
BLIS_INLINE void bli_dgemmsup2_rv_armv8a_asm_## THISM ##x6_ ## PACKA ## _ ## PACKB \
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
  uint64_t cs_a_next = bls_aux_ls_ext_next( data ); \
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
BEQ(DK_LEFT_LOOP_INIT_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,24,25,26,27) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmul,24,25,26,27,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,28,24,25,26) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,28,24,25,26,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (28,24,25,26,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,27,28,24,25) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,27,28,24,25,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (27,28,24,25,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,26,27,28,24) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,26,27,28,24,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (26,27,28,24,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,25,26,27,28) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,25,26,27,28,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (25,26,27,28,%[a_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_MKER_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" subs            %[k_mker], %[k_mker], #1        \n\t" \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,24,25,26,27) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,28,24,25,26) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,28,24,25,26,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (28,24,25,26,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,27,28,24,25) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,27,28,24,25,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (27,28,24,25,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,26,27,28,24) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,26,27,28,24,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (26,27,28,24,%[a_p]) \
\
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,25,26,27,28) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,25,26,27,28,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (25,26,27,28,%[a_p]) \
\
BEQ(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
BRANCH(DK_MKER_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
\
LABEL(DK_LEFT_LOOP_INIT_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(CLEAR_CROWS_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,24,25,26,27) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmul,24,25,26,27,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
\
LABEL(DK_LEFT_LOOP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" cmp             %[k_left], #0                   \n\t" \
BEQ(DWRITE_MEM_PREP_ ## THISM ## _ ## PACKA ## _ ## PACKB) \
" ldr             q29, [%[b], #16*0]              \n\t" \
" ldr             q30, [%[b], #16*1]              \n\t" \
" ldr             q31, [%[b], #16*2]              \n\t" \
" add             %[b], %[b], %[rs_b]             \n\t" \
" sub             %[k_left], %[k_left], #1        \n\t" \
DGEMM_LOAD_ACOL_LOC(THISM,24,25,26,27) PRFA_LOC \
DGEMM_8X6_MKER_LOOP_LOC(THISM,fmla,24,25,26,27,29,30,31) \
PACKB_STORE_FWD_ ## PACKB (29,30,31,%[b_p]) \
PACKA_STORE_FWD_ ## PACKA (24,25,26,27,%[a_p]) \
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
: [a_0]    "+r" (a_0), \
  [a_1]    "+r" (a_1), \
  [a_2]    "+r" (a_2), \
  [a_3]    "+r" (a_3), \
  [a_4]    "+r" (a_4), \
  [a_5]    "+r" (a_5), \
  [a_6]    "+r" (a_6), \
  [a_7]    "+r" (a_7), \
  [cs_a]   "+r" (cs_a), \
  [cs_a2]  "+r" (cs_a_next), \
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
// GENDEF(6,pack,pack)
// GENDEF(7,pack,pack)
// GENDECL(8,pack,pack);

GENDEF(1,pack,nopack)
GENDEF(2,pack,nopack)
GENDEF(3,pack,nopack)
GENDEF(4,pack,nopack)
GENDEF(5,pack,nopack)
GENDEF(6,pack,nopack)
GENDEF(7,pack,nopack)
// GENDECL(8,pack,nopack);

// GENDEF(1,nopack,pack)
// GENDEF(2,nopack,pack)
// GENDEF(3,nopack,pack)
// GENDEF(4,nopack,pack)
// GENDEF(5,nopack,pack)
// GENDEF(6,nopack,pack)
// GENDEF(7,nopack,pack)
// GENDECL(8,nopack,pack);

GENDEF(1,nopack,nopack)
GENDEF(2,nopack,nopack)
GENDEF(3,nopack,nopack)
GENDEF(4,nopack,nopack)
GENDEF(5,nopack,nopack)
GENDEF(6,nopack,nopack)
GENDEF(7,nopack,nopack)
// GENDEF(8,nopack,nopack);

// Clear all macros.
#undef DGEMM_MUL_1LINE
#undef DGEMM_8X6_MKER_LOOP_1
#undef DGEMM_8X6_MKER_LOOP_2
#undef DGEMM_8X6_MKER_LOOP_3
#undef DGEMM_8X6_MKER_LOOP_4
#undef DGEMM_8X6_MKER_LOOP_5
#undef DGEMM_8X6_MKER_LOOP_6
#undef DGEMM_8X6_MKER_LOOP_7
#undef DGEMM_8X6_MKER_LOOP_8
#undef DGEMM_8X6_MKER_LOOP_LOC
#undef DGEMM_LOAD_ACOL_1
#undef DGEMM_LOAD_ACOL_2
#undef DGEMM_LOAD_ACOL_3
#undef DGEMM_LOAD_ACOL_4
#undef DGEMM_LOAD_ACOL_5
#undef DGEMM_LOAD_ACOL_6
#undef DGEMM_LOAD_ACOL_7
#undef DGEMM_LOAD_ACOL_8
#undef DGEMM_LOAD_ACOL_LOC
#undef CLOAD_1ROW
#undef CSCALE_1ROW
#undef CSTORE_1ROW
#undef CIO_UNIT_1ROW
#undef CIO_UNIT_1COL
#undef CSTORE_1
#undef CSTORE_2
#undef CSTORE_3
#undef CSTORE_4
#undef CSTORE_5
#undef CSTORE_6
#undef CSTORE_7
#undef CSTORE_8
#undef PACKA_STORE_FWD_nopack
#undef PACKA_STORE_FWD_pack
#undef PACKB_STORE_FWD_nopack
#undef PACKB_STORE_FWD_pack
#undef GENDECL
#undef GENDEF

// Other sizes.
#include "bli_gemmsup2_rv_armv8a_asm_d8x6m.cin"
#include "bli_gemmsup2_rv_armv8a_asm_d8x5m.cin"

static double zero = 0.0;
static double c_t[8*6];

// Edge cases.
#define EXPAND_CASE_UKer( MN, PA, PB, M, N, PACKA, PACKB ) \
    case ( PA << 9 | PB << 8 | MN ): \
        bli_dgemmsup2_rv_armv8a_asm_ ## M ## x## N ##_## PACKA ##_ ## PACKB \
            ( M, N, k, \
              alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, \
              beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p \
            ); break;

#define GENDEF_Milliker(N,PACKA) \
BLIS_INLINE void bli_dgemmsup2_rv_armv8a_asm_8x ## N ## m_ ## PACKA \
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
     double *restrict b_p, int pack_b \
    ) \
{ \
    inc_t ps_a_p    = bls_aux_ps_ext_p   ( data ); \
    inc_t ps_a      = bls_aux_ps_ext     ( data ); \
    inc_t cs_a_next = bls_aux_ls_ext_next( data ); \
    const void *next_a = bli_auxinfo_next_a( data ); \
    const void *next_b = bli_auxinfo_next_b( data ); \
\
    bli_auxinfo_set_next_b( b_p, data ); \
    bls_aux_set_ls_ext_next( cs_a0, data ); \
\
    if ( m >= 8 && pack_b ) \
    { \
        if ( m == 8 ) \
        { \
            bli_auxinfo_set_next_a( next_a, data ); \
            bli_auxinfo_set_next_b( next_b, data ); \
            bls_aux_set_ls_ext_next( cs_a_next, data ); \
        } \
        else \
            bli_auxinfo_set_next_a( a + bli_max(ps_a, 128), data ); \
\
        bli_dgemmsup2_rv_armv8a_asm_8x ## N ##_ ## PACKA ## _pack \
        ( 8, N, k, alpha, \
          a, rs_a0, cs_a0, \
          b, rs_b0, cs_b0, beta, \
          c, rs_c0, cs_c0, \
          data, cntx, a_p, b_p ); \
\
        m -= 8; \
        a += ps_a; \
        a_p += ps_a_p; \
        c += 8 * rs_c0; \
        b = b_p; \
        rs_b0 = 6; \
    } \
\
    for ( ; m >= 8; m -= 8 ) \
    { \
        if ( m == 8 ) \
        { \
            bli_auxinfo_set_next_a( next_a, data ); \
            bli_auxinfo_set_next_b( next_b, data ); \
            bls_aux_set_ls_ext_next( cs_a_next, data ); \
        } \
        else \
            bli_auxinfo_set_next_a( a + bli_max(ps_a, 128), data ); \
\
        /* Optionally call bulk kernel. */ \
        if ( a == a_p && b == b_p && N > 3 ) \
            bli_dgemm_armv8a_asm_8x6r \
            ( 8, N, k, alpha, a, b, beta, \
              c, rs_c0, cs_c0, data, cntx ); \
        else \
            bli_dgemmsup2_rv_armv8a_asm_8x ## N ##_ ## PACKA ## _nopack \
            ( 8, N, k, alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p ); \
\
        a += ps_a; \
        a_p += ps_a_p; \
        c += 8 * rs_c0; \
    } \
\
    double *c_orig = c; \
    inc_t rs_c_orig = rs_c0; \
    inc_t cs_c_orig = cs_c0; \
    double *beta_orig = beta; \
\
    if ( N != 6 ) \
    { \
        c = c_t; \
        rs_c0 = 6; \
        cs_c0 = 1; \
        beta = &zero; \
    } \
\
    bli_auxinfo_set_next_a( next_a, data ); \
    bli_auxinfo_set_next_b( next_b, data ); \
    bls_aux_set_ls_ext_next( cs_a_next, data ); \
\
    switch ( m ) \
    { \
        EXPAND_CASE_UKer( 7, 0, 0, 7, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 6, 0, 0, 6, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 5, 0, 0, 5, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 4, 0, 0, 4, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 3, 0, 0, 3, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 2, 0, 0, 2, 6, PACKA, nopack ) \
        EXPAND_CASE_UKer( 1, 0, 0, 1, 6, PACKA, nopack ) \
    } \
\
    if ( N != 6 ) \
        for ( int i = 0; i < m; ++i ) \
            for ( int j = 0; j < n; ++j ) \
                c_orig[ i * rs_c_orig + j * cs_c_orig ] = \
                    c_orig[ i * rs_c_orig + j * cs_c_orig ] * *beta_orig + \
                        c_t[ i * 6 + j ]; \
}

GENDEF_Milliker(6,pack)

GENDEF_Milliker(6,nopack)
GENDEF_Milliker(5,nopack)
GENDEF_Milliker(4,nopack)
GENDEF_Milliker(3,nopack)
GENDEF_Milliker(2,nopack)
GENDEF_Milliker(1,nopack)

#undef GENDEF_Milliker
#undef EXPAND_CASE_UKer

void bli_dgemmsup2_rv_armv8a_asm_8x6m
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
    assert( n <= 6 );
    assert( cs_b0 == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif

    switch ( !!pack_a << 9 | n ) {
#define EXPAND_CASE_Milliker( N, PA, PACKA ) \
    case ( PA << 9 | N ): \
        bli_dgemmsup2_rv_armv8a_asm_8x## N ##m_## PACKA \
            ( m, N, k, \
              alpha, \
              a, rs_a0, cs_a0, \
              b, rs_b0, cs_b0, \
              beta, \
              c, rs_c0, cs_c0, \
              data, cntx, a_p, b_p, pack_b \
            ); break;
    EXPAND_CASE_Milliker( 6, 1, pack )
    EXPAND_CASE_Milliker( 6, 0, nopack )
    case ( 1 << 9 | 5 ): EXPAND_CASE_Milliker( 5, 0, nopack )
    case ( 1 << 9 | 4 ): EXPAND_CASE_Milliker( 4, 0, nopack )
    case ( 1 << 9 | 3 ): EXPAND_CASE_Milliker( 3, 0, nopack )
    case ( 1 << 9 | 2 ): EXPAND_CASE_Milliker( 2, 0, nopack )
    case ( 1 << 9 | 1 ): EXPAND_CASE_Milliker( 1, 0, nopack )
    default:
#ifdef DEBUG
        assert( 0 );
#endif
        break;
    }
}

#endif


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include "blis.h"
#include <assert.h>


#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
// vpmovsxbq, vmaskmovpd was not defined.
#define vpmovsxbq(_0, _1) INSTR_(vpmovsxbq, _0, _1)
#define vmaskmovpd(_0, _1, _2) INSTR_(vmaskmovpd, _0, _1, _2)


#define DGEMM_6X8_NANOKER_ALINE_1(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR), ymm(A0_)) \
    INST (ymm(A0_), B0, C00) \
    INST (ymm(A0_), B1, C01) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR)))

#define DGEMM_6X8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR        ), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA, 1), ymm(A1_)) \
    INST (ymm(A0_), B0, C00) \
    INST (ymm(A0_), B1, C01) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C10) \
    INST (ymm(A1_), B1, C11) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR)))

#define DGEMM_6X8_NANOKER_ALINE_3(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_6X8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    INST (ymm(A0_), B0, C20) \
    INST (ymm(A0_), B1, C21) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 2*8)))

#define DGEMM_6X8_NANOKER_ALINE_4(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_6X8_NANOKER_ALINE_2(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 2), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA3,1), ymm(A1_)) \
    INST (ymm(A0_), B0, C20) \
    INST (ymm(A0_), B1, C21) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C30) \
    INST (ymm(A1_), B1, C31) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 2*8)))

#define DGEMM_6X8_NANOKER_ALINE_5(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_6X8_NANOKER_ALINE_4(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    INST (ymm(A0_), B0, C40) \
    INST (ymm(A0_), B1, C41) \
    PACK_ ##PACKA (vmovsd(xmm(A0_), mem(PAADDR, 4*8)))

#define DGEMM_6X8_NANOKER_ALINE_6(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    DGEMM_6X8_NANOKER_ALINE_4(INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    vbroadcastsd(mem(AADDR, RSA, 4), ymm(A0_)) \
    vbroadcastsd(mem(AADDR, RSA5,1), ymm(A1_)) \
    INST (ymm(A0_), B0, C40) \
    INST (ymm(A0_), B1, C41) \
    PACK_ ##PACKA (vunpcklpd(xmm(A1_), xmm(A0_), xmm(A0_))) \
    INST (ymm(A1_), B0, C50) \
    INST (ymm(A1_), B1, C51) \
    PACK_ ##PACKA (vmovapd(xmm(A0_), mem(PAADDR, 4*8)))

#define DGEMM_6X8_NANOKER(M,INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,BADDR,RSB,PAADDR,PBADDR,PACKA,PACKB) \
    vmovupd(mem(BADDR    ), B0) \
    vmovupd(mem(BADDR, 32), B1) \
    DGEMM_6X8_NANOKER_ALINE_## M (INST,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,A0_,A1_,B0,B1,AADDR,RSA,RSA3,RSA5,CSA,PAADDR,PACKA) \
    add(CSA, AADDR) \
    add(RSB, BADDR) \
    PACK_ ##PACKA (add(imm(6*8), PAADDR)) \
    PACK_ ##PACKB (vmovapd(B0, mem(PBADDR    ))) \
    PACK_ ##PACKB (vmovapd(B1, mem(PBADDR, 32))) \
    PACK_ ##PACKB (add(imm(8*8), PBADDR))

#define PACK_nopack(_1)
#define PACK_pack(_1) _1

#define DGEMM_6X8_NANOKER_LOC(M,INST,RSA,RSA3,RSA5,CSA,RSB,PACKA,PACKB) \
    DGEMM_6X8_NANOKER(M,INST,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,ymm12,ymm13,ymm14,ymm15,0,1,ymm2,ymm3,rax,RSA,RSA3,RSA5,CSA,rbx,RSB,rcx,rdx,PACKA,PACKB)

// No in-reg transpose support.
#define C1ROW_BETA_FWD(VFH,VLH,VBETA,CADDR,CSC) \
    vfmadd231pd(mem(CADDR), VBETA, VFH) \
    vmovupd(VFH, mem(CADDR)) \
    vfmadd231pd(mem(CADDR, 32), VBETA, VLH) \
    vmovupd(VLH, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define C1ROW_FWD(VFH,VLH,CADDR,CSC) \
    vmovupd(VFH, mem(CADDR)) \
    vmovupd(VLH, mem(CADDR, 32)) \
    add(CSC, CADDR)

#define CSTORE_LOC_1(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm4, ymm5, ymm3, CADDR, RSC)
#define CSTORE_LOC_2(CADDR,RSC) \
    CSTORE_LOC_1(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm6, ymm7, ymm3, CADDR, RSC)
#define CSTORE_LOC_3(CADDR,RSC) \
    CSTORE_LOC_2(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm8, ymm9, ymm3, CADDR, RSC)
#define CSTORE_LOC_4(CADDR,RSC) \
    CSTORE_LOC_3(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm10, ymm11, ymm3, CADDR, RSC)
#define CSTORE_LOC_5(CADDR,RSC) \
    CSTORE_LOC_4(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm12, ymm13, ymm3, CADDR, RSC)
#define CSTORE_LOC_6(CADDR,RSC) \
    CSTORE_LOC_5(CADDR,RSC) \
    C1ROW_BETA_FWD(ymm14, ymm15, ymm3, CADDR, RSC)

#define CSTOREZB_LOC_1(CADDR,RSC) \
    C1ROW_FWD(ymm4, ymm5, CADDR, RSC)
#define CSTOREZB_LOC_2(CADDR,RSC) \
    CSTOREZB_LOC_1(CADDR,RSC) \
    C1ROW_FWD(ymm6, ymm7, CADDR, RSC)
#define CSTOREZB_LOC_3(CADDR,RSC) \
    CSTOREZB_LOC_2(CADDR,RSC) \
    C1ROW_FWD(ymm8, ymm9, CADDR, RSC)
#define CSTOREZB_LOC_4(CADDR,RSC) \
    CSTOREZB_LOC_3(CADDR,RSC) \
    C1ROW_FWD(ymm10, ymm11, CADDR, RSC)
#define CSTOREZB_LOC_5(CADDR,RSC) \
    CSTOREZB_LOC_4(CADDR,RSC) \
    C1ROW_FWD(ymm12, ymm13, CADDR, RSC)
#define CSTOREZB_LOC_6(CADDR,RSC) \
    CSTOREZB_LOC_5(CADDR,RSC) \
    C1ROW_FWD(ymm14, ymm15, CADDR, RSC)

#define DTRANSPOSE_4X4(C0,C1,C2,C3,V0_,V1_,V2_,V3_) /* C0-3: In row, out col; V0-3: Scratch. */ \
    vunpcklpd(C1, C0, ymm(V0_)) /* Unpack bientries into V0: { C0[0], C1[0], C0[2], C1[2] } */ \
    vunpckhpd(C1, C0, ymm(V1_)) /* Unpack bientries into V1: { C0[1], C1[1], C0[3], C1[3] } */ \
    vunpcklpd(C3, C2, ymm(V2_)) /* Unpack bientries into V2: { C2[0], C3[0], C2[2], C3[2] } */ \
    vunpckhpd(C3, C2, ymm(V3_)) /* Unpack bientries into V3: { C2[1], C3[1], C2[3], C3[3] } */ \
    vinsertf128(imm(0x1), xmm(V2_), ymm(V0_), C0) /* Insert first half of V2 to last half of { C0 <- V0 } */ \
    vinsertf128(imm(0x1), xmm(V3_), ymm(V1_), C1) /* Insert first half of V3 to last half of { C1 <- V1 } */ \
    vperm2f128(imm(0x31), ymm(V2_), ymm(V0_), C2) /* Shuffle last half of V0 to first half of { C2 <- V2 } */ \
    vperm2f128(imm(0x31), ymm(V3_), ymm(V1_), C3) /* Shuffle last half of V1 to first half of { C3 <- V3 } */

#define CCOL_4V_nz(C0,C1,C2,C3,VBETA,CADDR,CSC,CSC3) \
    vfmadd231pd(mem(CADDR), VBETA, C0) /* Column 0, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 1), VBETA, C1) /* Column 1, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC, 2), VBETA, C2) /* Column 2, 0:4 */ \
    vfmadd231pd(mem(CADDR, CSC3, 1), VBETA, C3) /* Column 3, 0:4 */ \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

#define CCOL_4V_z(C0,C1,C2,C3,VBETA,CADDR,CSC,CSC3) \
    vmovupd(C0, mem(CADDR)) \
    vmovupd(C1, mem(CADDR, CSC, 1)) \
    vmovupd(C2, mem(CADDR, CSC, 2)) \
    vmovupd(C3, mem(CADDR, CSC3, 1))

#define BETA_nz(_1) _1
#define BETA_z(_1)

// In-reg transpose needs implementing case-by-case.
#define CSTORECOL_LOC_5(BETA,C00,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,CADDR,CADDR4,CSC,CSC3) \
    /* Here: ymm0-2 used additionally as scratch. Don't occupy them outside. */ \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(C00,C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(C01,C11,C21,C31,0,1,2,C50_) \
    /* Store block 1. */ \
    CCOL_4V_ ## BETA (C00,C10,C20,C30,ymm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    vpermpd(imm(0b01010101), ymm(C40_), ymm0) \
    vpermpd(imm(0b10101010), ymm(C40_), ymm1) \
    vpermpd(imm(0b11111111), ymm(C40_), ymm2) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm(C40_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd(xmm(C40_), mem(CADDR4)) \
    vmovsd(xmm0, mem(CADDR4, CSC, 1)) \
    vmovsd(xmm1, mem(CADDR4, CSC, 2)) \
    vmovsd(xmm2, mem(CADDR4, CSC3, 1)) \
    lea(mem(CADDR4, CSC, 4), CADDR4) \
    /* Store block 2. */ \
    CCOL_4V_ ## BETA (C01,C11,C21,C31,ymm(VBETA_),CADDR,CSC,CSC3) \
    vpermpd(imm(0b01010101), ymm(C41_), ymm0) \
    vpermpd(imm(0b10101010), ymm(C41_), ymm1) \
    vpermpd(imm(0b11111111), ymm(C41_), ymm2) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm(C41_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA ( vmovsd(mem(CADDR4, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA ( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd(xmm(C41_), mem(CADDR4)) \
    vmovsd(xmm0, mem(CADDR4, CSC, 1)) \
    vmovsd(xmm1, mem(CADDR4, CSC, 2)) \
    vmovsd(xmm2, mem(CADDR4, CSC3, 1))

#define CSTORECOL_LOC_4(BETA,C00,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(C00,C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(C01,C11,C21,C31,0,1,2,C50_) \
    CCOL_4V_ ## BETA (C00,C10,C20,C30,ymm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    CCOL_4V_ ## BETA (C01,C11,C21,C31,ymm(VBETA_),CADDR,CSC,CSC3)

#define CSTORECOL_LOC_3(BETA,C00,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_4X4(C00,C10,C20,C30,0,1,2,C50_) \
    DTRANSPOSE_4X4(C01,C11,C21,C31,0,1,2,C50_) \
    /* Prepare mask */ \
    mov(imm(0b00000000111111111111111111111111), CADDR4) \
    vmovd(CADDR4, xmm(C50_)) \
    vpmovsxbq(xmm(C50_), ymm(C50_)) \
    /* Store block 1. */ \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR), ymm(C50_), ymm0) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 1), ymm(C50_), ymm1) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 2), ymm(C50_), ymm2) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC3, 1), ymm(C50_), ymm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(ymm0, ymm(VBETA_), C00) ) \
    BETA_ ## BETA( vfmadd231pd(ymm1, ymm(VBETA_), C10) ) \
    BETA_ ## BETA( vfmadd231pd(ymm2, ymm(VBETA_), C20) ) \
    BETA_ ## BETA( vfmadd231pd(ymm(C51_), ymm(VBETA_), C30) ) \
    vmaskmovpd(C00, ymm(C50_), mem(CADDR)) \
    vmaskmovpd(C10, ymm(C50_), mem(CADDR, CSC, 1)) \
    vmaskmovpd(C20, ymm(C50_), mem(CADDR, CSC, 2)) \
    vmaskmovpd(C30, ymm(C50_), mem(CADDR, CSC3, 1)) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    /* Store block 2. */ \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR), ymm(C50_), ymm0) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 1), ymm(C50_), ymm1) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC, 2), ymm(C50_), ymm2) ) \
    BETA_ ## BETA( vmaskmovpd(mem(CADDR, CSC3, 1), ymm(C50_), ymm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(ymm0, ymm(VBETA_), C01) ) \
    BETA_ ## BETA( vfmadd231pd(ymm1, ymm(VBETA_), C11) ) \
    BETA_ ## BETA( vfmadd231pd(ymm2, ymm(VBETA_), C21) ) \
    BETA_ ## BETA( vfmadd231pd(ymm(C51_), ymm(VBETA_), C31) ) \
    vmaskmovpd(C01, ymm(C50_), mem(CADDR)) \
    vmaskmovpd(C11, ymm(C50_), mem(CADDR, CSC, 1)) \
    vmaskmovpd(C21, ymm(C50_), mem(CADDR, CSC, 2)) \
    vmaskmovpd(C31, ymm(C50_), mem(CADDR, CSC3, 1))

#define DTRANSPOSE_2X4(D0_,D1_,D2_,D3_,S0,S1) /* 2X4 row-stored (2 ymm) into col-stored (4 xmm). */ \
    vunpcklpd(S1, S0, ymm(D0_)) /* Unpack bientries into D0: { S0[0], S1[0], S0[2], S1[2] } */ \
    vunpckhpd(S1, S0, ymm(D1_)) /* Unpack bientries into D1: { S0[1], S1[1], S0[3], S1[3] } */ \
    vextractf128(imm(0x1), ymm(D0_), xmm(D2_)) /* Extract last half of D0 into D2 */ \
    vextractf128(imm(0x1), ymm(D1_), xmm(D3_)) /* Extract last half of D1 into D3 */

#define CSTORECOL_LOC_2(BETA,C00,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,CADDR,CADDR4,CSC,CSC3) \
    /* Transpose bulk part. */ \
    DTRANSPOSE_2X4(0,1,2,C50_,C00,C10) \
    CCOL_4V_ ## BETA (xmm0,xmm1,xmm2,xmm(C50_),xmm(VBETA_),CADDR,CSC,CSC3) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    DTRANSPOSE_2X4(0,1,2,C50_,C01,C11) \
    CCOL_4V_ ## BETA (xmm0,xmm1,xmm2,xmm(C50_),xmm(VBETA_),CADDR,CSC,CSC3)

#define CSTORECOL_LOC_1(BETA,C00,C01,C10,C11,C20,C21,C30,C31,C40_,C41_,C50_,C51_,VBETA_,CADDR,CADDR4,CSC,CSC3) \
    /* Block 1. */ \
    vpermpd(imm(0b01010101), C00, ymm0) \
    vpermpd(imm(0b10101010), C00, ymm1) \
    vpermpd(imm(0b11111111), C00, ymm2) \
    BETA_ ## BETA( vmovsd(mem(CADDR), xmm(C40_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 1), xmm(C41_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C41_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd( xmm0, mem(CADDR, CSC, 1) ) \
    vmovsd( xmm1, mem(CADDR, CSC, 2) ) \
    vmovsd( xmm2, mem(CADDR, CSC3, 1) ) \
    vmovapd(C00, ymm0) \
    BETA_ ## BETA( vfmadd231pd(xmm(C40_), xmm(VBETA_), xmm0) ) \
    vmovsd( xmm0, mem(CADDR) ) \
    lea(mem(CADDR, CSC, 4), CADDR) \
    /* Block 2. */ \
    vpermpd(imm(0b01010101), C01, ymm0) \
    vpermpd(imm(0b10101010), C01, ymm1) \
    vpermpd(imm(0b11111111), C01, ymm2) \
    BETA_ ## BETA( vmovsd(mem(CADDR), xmm(C40_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 1), xmm(C41_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC, 2), xmm(C50_)) ) \
    BETA_ ## BETA( vmovsd(mem(CADDR, CSC3, 1), xmm(C51_)) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C41_), xmm(VBETA_), xmm0) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C50_), xmm(VBETA_), xmm1) ) \
    BETA_ ## BETA( vfmadd231pd(xmm(C51_), xmm(VBETA_), xmm2) ) \
    vmovsd( xmm0, mem(CADDR, CSC, 1) ) \
    vmovsd( xmm1, mem(CADDR, CSC, 2) ) \
    vmovsd( xmm2, mem(CADDR, CSC3, 1) ) \
    vmovapd(C01, ymm0) \
    BETA_ ## BETA( vfmadd231pd(xmm(C40_), xmm(VBETA_), xmm0) ) \
    vmovsd( xmm0, mem(CADDR) ) \

#define GENDECL(M,PACKA,PACKB) \
void bli_dgemmsup2_rv_haswell_asm_## M ##x8r_ ## PACKA ## _ ## PACKB \
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

#define GENDEF(M,PACKA,PACKB) \
    GENDECL(M,PACKA,PACKB) \
{ \
    const void* a_next = bli_auxinfo_next_a( data ); \
    const void* b_next = bli_auxinfo_next_b( data ); \
    uint64_t cs_a_next = bli_auxinfo_ps_a( data ); /* Borrow the space. */ \
\
    /* Typecast local copies of integers in case dim_t and inc_t are a
     * different size than is expected by load instructions. */ \
    uint64_t k_mker = k / 4; \
    uint64_t k_left = k % 4; \
    uint64_t rs_a   = rs_a0; \
    uint64_t cs_a   = cs_a0; \
    uint64_t rs_b   = rs_b0; \
    uint64_t rs_c   = rs_c0; \
    uint64_t cs_c   = cs_c0; \
\
    begin_asm() \
\
    mov(var(k_mker), rsi) \
    mov(var(k_left), r14) \
    mov(var(a), rax) \
    mov(var(b), rbx) \
    mov(var(a_p), rcx) \
    mov(var(b_p), rdx) \
    mov(var(rs_a), rdi) \
    lea(mem(, rdi, 8), rdi) /* rs_a *= sizeof(double) */ \
    mov(var(cs_a), r10) \
    lea(mem(, r10, 8), r10) /* cs_a *= sizeof(double) */ \
    mov(var(rs_b), r11) \
    lea(mem(, r11, 8), r11) /* rs_b *= sizeof(double) */ \
    lea(mem(rdi, rdi, 2), r13) /* r13 = 3*rs_a; */ \
    lea(mem(r13, rdi, 2), r15) /* r15 = 5*rs_a; */ \
    mov(var(a_next), r8) \
    mov(var(cs_a2), r9) /* r9 = cs_a_next; */ \
    lea(mem(, r9, 8), r9) /* cs_a_next *= sizeof(double) */ \
    lea(mem(r9, r9, 2), r12) /* r12 = 3*cs_a_next; */ \
\
    /* TODO: Prefetch C. */ \
\
    test(rsi, rsi) \
    je(.DEMPTYALL) \
\
    label(.DK_4LOOP_INIT) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vmulpd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    je(.DK_LEFT_LOOP_PREP) \
\
    label(.DK_4LOOP) \
        prefetch(0, mem(r8, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 1, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r9, 2, 5*8)) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        prefetch(0, mem(r8, r12, 1, 5*8)) \
        lea(mem(r8, r9, 4), r8) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(rsi) \
    jne(.DK_4LOOP) \
\
    label(.DK_LEFT_LOOP_PREP) \
    test(r14, r14) \
    je(.DWRITEMEM_PREP) \
    jmp(.DK_LEFT_LOOP) \
\
    label(.DEMPTYALL) \
        vzeroall() \
\
    label(.DK_LEFT_LOOP) \
        prefetch(0, mem(r8, 5*8)) \
        add(r9, r8) \
        DGEMM_6X8_NANOKER_LOC(M,vfmadd231pd,rdi,r13,r15,r10,r11,PACKA,PACKB) \
        dec(r14) \
    jne(.DK_LEFT_LOOP) \
\
    label(.DWRITEMEM_PREP) \
\
    mov(var(alpha), rax) \
    mov(var(beta), rbx) \
    /* mov(var(a_next), rcx) */ \
    mov(var(b_next), rdx) \
    vbroadcastsd(mem(rax), ymm0) \
    vbroadcastsd(mem(rbx), ymm3) \
\
        vmulpd(ymm0, ymm4, ymm4) \
        vmulpd(ymm0, ymm5, ymm5) \
        vmulpd(ymm0, ymm6, ymm6) \
        vmulpd(ymm0, ymm7, ymm7) \
        vmulpd(ymm0, ymm8, ymm8) \
        vmulpd(ymm0, ymm9, ymm9) \
        vmulpd(ymm0, ymm10, ymm10) \
        vmulpd(ymm0, ymm11, ymm11) \
        vmulpd(ymm0, ymm12, ymm12) \
        vmulpd(ymm0, ymm13, ymm13) \
        vmulpd(ymm0, ymm14, ymm14) \
        vmulpd(ymm0, ymm15, ymm15) \
\
        /* prefetch(0, mem(rcx)) */ \
        /* prefetch(0, mem(rcx, 1*64)) \
         * prefetch(0, mem(rcx, 2*64)) */ \
        prefetch(0, mem(rdx)) \
        /* prefetch(0, mem(rdx, 1*64)) \
         * prefetch(0, mem(rdx, 2*64)) \
         * prefetch(0, mem(rdx, 3*64)) */ \
\
    mov(var(c), rcx) /* load c */ \
    mov(var(rs_c), rdi) /* load rs_c */ \
    lea(mem(, rdi, 8), rdi) /* rdi = rs_c * sizeof(double) */ \
    mov(var(cs_c), rsi) /* load cs_c */ \
    lea(mem(, rsi, 8), rsi) /* rsi = cs_c * sizeof(double) */ \
    lea(mem(rcx, rdi, 4), r14) /* load address of c + 4*rs_c; */ \
    lea(mem(rsi, rsi, 2), r13) /* r13 = 3*cs_c; */ \
\
    /* now avoid loading C if beta == 0 */ \
\
    vxorpd(ymm0, ymm0, ymm0) /* set ymm0 to zero. */ \
    vucomisd(xmm0, xmm3) /* set ZF if beta == 0. */ \
    je(.DBETAZERO) /* if ZF = 1, jump to beta == 0 case */ \
\
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTORED) /* jump to column storage case */ \
\
            CSTORE_LOC_## M (rcx, rdi) \
            jmp(.DDONE) /* jump to end. */ \
\
        label(.DCOLSTORED) \
\
            CSTORECOL_LOC_## M (nz,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,12,13,14,15,3,rcx,r14,rsi,r13) \
            jmp(.DDONE) /* jump to end. */ \
\
    label(.DBETAZERO) \
\
        cmp(imm(8), rdi) /* set ZF if (8*rs_c) == 8. */ \
        jz(.DCOLSTOREDBZ) /* jump to column storage case */ \
\
            CSTOREZB_LOC_## M (rcx, rdi) \
            jmp(.DDONE) /* jump to end. */ \
\
        label(.DCOLSTOREDBZ) /* jump to column storage case */ \
\
            CSTORECOL_LOC_## M (z,ymm4,ymm5,ymm6,ymm7,ymm8,ymm9,ymm10,ymm11,12,13,14,15,3,rcx,r14,rsi,r13) \
\
    label(.DDONE) \
\
    end_asm( \
    :: \
      [a]      "m" (a), \
      [rs_a]   "m" (rs_a), \
      [cs_a]   "m" (cs_a), \
      [cs_a2]  "m" (cs_a_next), \
      [b]      "m" (b), \
      [rs_b]   "m" (rs_b), \
      [c]      "m" (c), \
      [rs_c]   "m" (rs_c), \
      [cs_c]   "m" (cs_c), \
      [k_mker] "m" (k_mker), \
      [k_left] "m" (k_left), \
      [alpha]  "m" (alpha), \
      [beta]   "m" (beta), \
      [a_next] "m" (a_next), \
      [b_next] "m" (b_next), \
      [a_p]    "m" (a_p), \
      [b_p]    "m" (b_p) \
    : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", \
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", \
      "xmm0", "xmm1", "xmm2", "xmm3", \
      "xmm4", "xmm5", "xmm6", "xmm7", \
      "xmm8", "xmm9", "xmm10", "xmm11", \
      "xmm12", "xmm13", "xmm14", "xmm15", \
      "memory" \
    ) \
\
}

GENDECL(6,pack,pack);

GENDEF(1,pack,nopack)
GENDEF(2,pack,nopack)
GENDEF(3,pack,nopack)
GENDEF(4,pack,nopack)
GENDEF(5,pack,nopack)
GENDECL(6,pack,nopack);

GENDECL(6,nopack,pack);

GENDEF(1,nopack,nopack)
GENDEF(2,nopack,nopack)
GENDEF(3,nopack,nopack)
GENDEF(4,nopack,nopack)
GENDEF(5,nopack,nopack)
GENDECL(6,nopack,nopack);
#undef GENDEF


void bli_dgemmsup2_rv_haswell_asm_6x8r
    (
     dim_t            m,
     dim_t            n0,
     dim_t            k,
     double *restrict alpha,
     double *restrict a, inc_t rs_a, inc_t cs_a,
     double *restrict b, inc_t rs_b, inc_t cs_b,
     double *restrict beta0,
     double *restrict c0, inc_t rs_c0, inc_t cs_c0,
     auxinfo_t       *data,
     cntx_t          *cntx,
     double *restrict a_p, int pack_a,
     double *restrict b_p, int pack_b
    )
{
    bool use_ct = false;
    dim_t n = n0;
    if ( n0 < 8 )
    {
        if ( m == 6 )
        {
            bli_dgemmsup2_rv_haswell_asm_6x8rn
                ( m, n0, k,
                  alpha,
                  a, rs_a, cs_a,
                  b, rs_b, cs_b,
                  beta0,
                  c0, rs_c0, cs_c0,
                  data, cntx,
                  a_p, pack_a,
                  b_p, pack_b );
            return ;
        }
        else
        {
            double one = 1.0;
            // Caller ensures p_a has enough space. Now do the packing.
            l1mukr_t dpackm = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_PACKM_NRXK_KER, cntx );

            if ( b != b_p )
                dpackm
                    ( BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS, 
                      n0, k, k,
                      &one,
                      b, cs_b, rs_b,
                      b_p, 8,
                      cntx );

            n = 8;
            b = b_p;
            rs_b = 8;
            cs_b = 1;
            pack_b = false;
            use_ct = true;
        }
    }
#ifdef DEBUG
    assert( n == 8 );
    assert( cs_b == 1 );
    assert( rs_c0 == 1 ||
            cs_c0 == 1 );
#endif
    // Process CT scratch space.
    static double ct[6 * 8];
    double zero = 0.0;
    double *c; inc_t rs_c, cs_c;
    double *beta;
    if ( use_ct )
    {
        c = ct;
        rs_c = 8;
        cs_c = 1;
        beta = &zero;
    }
    else
    {
        c = c0;
        rs_c = rs_c0,
        cs_c = cs_c0;
        beta = beta0;
    }

    switch ( !!pack_a << 9 | !!pack_b << 8 | m ) {
#define EXPAND_CASE_BASE( M, PA, PB, PACKA, PACKB ) \
    case ( PA << 9 | PB << 8 | M ): \
        bli_dgemmsup2_rv_haswell_asm_ ## M ## x8r_## PACKA ##_## PACKB \
            ( m, n, k, \
              alpha, \
              a, rs_a, cs_a, \
              b, rs_b, cs_b, \
              beta, \
              c, rs_c, cs_c, \
              data, cntx, a_p, b_p \
            ); break;
#define EXPAND_CASE1( M ) \
      EXPAND_CASE_BASE( M, 1, 1, pack, pack ) \
      EXPAND_CASE_BASE( M, 1, 0, pack, nopack ) \
      EXPAND_CASE_BASE( M, 0, 1, nopack, pack ) \
      EXPAND_CASE_BASE( M, 0, 0, nopack, nopack )
    EXPAND_CASE1(6)

    // Final row-block. B tiles'll never be reused.
#define EXPAND_CASE2( M ) \
    case ( 1 << 9 | 1 << 8 | M ): \
      EXPAND_CASE_BASE( M, 1, 0, pack, nopack ) \
    case ( 0 << 9 | 1 << 8 | M ): \
      EXPAND_CASE_BASE( M, 0, 0, nopack, nopack )
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

    if ( c == ct )
    {
        for ( dim_t j = 0; j < n0; ++j )
            for ( dim_t i = 0; i < m; ++i )
                c0[i * rs_c0 + j * cs_c0] =
                    c0[i * rs_c0 + j * cs_c0] * *beta0 + ct[i * 8 + j];
    }
}

#endif
